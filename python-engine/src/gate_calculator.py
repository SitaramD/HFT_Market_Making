"""
gate_calculator.py — SITARAM HFT
=================================
Post-session Jenkins gate evaluator.

Usage (standalone):
    python gate_calculator.py --session SES_20260413_120657

Usage (from engine.py shutdown):
    from gate_calculator import calculate_and_write_gates
    calculate_and_write_gates(session_id, start_time, end_time)

Gates evaluated (11 total):
    1.  sharpe_gte_3              Annualised Sharpe ≥ 3.0
    2.  max_dd_lt_5pct            Max drawdown < 5 %
    3.  ic_gt_0_02                Mean IC > 0.02
    4.  fill_rate_gt_60pct        Fill rate > 60 %   (quotes filled / quotes sent)
    5.  inventory_mean_reversion_gt_70pct   % of ticks where |inventory| < 50 % of max
    6.  adverse_selection_lt_30pct          % of fills with negative realised PnL < 30 %
    7.  quote_to_trade_ratio_ok   QTR ≥ 1 (at least 1 quote per fill)
    8.  spread_minimum_ok         Average spread ≥ 0.10 USDT
    9.  ic_stability_ok           IC stddev / mean < 2.0  (stable IC)
   10.  fee_adjusted_pnl_positive Final cumulative PnL > 0
   11.  daily_loss_limit_respected  No single-day PnL < daily_loss_limit

All pass/fail values are written back to live_BTCUSDT_master.json
and the session status is updated to COMPLETED.
"""

import os
import json
import math
import logging
import argparse
import psycopg2
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────────
##MASTER_JSON = os.getenv("MASTER_JSON", r"E:\Binance\live_BTCUSDT_master.json")
MASTER_JSON = os.getenv("MASTER_JSON", "/mnt/sessions/live_BTCUSDT_master.json")
DB_HOST     = os.getenv("TIMESCALEDB_HOST",     "timescaledb")
DB_PORT     = int(os.getenv("TIMESCALEDB_PORT", "5432"))
DB_NAME     = os.getenv("TIMESCALEDB_DB",       "sitaram")
DB_USER     = os.getenv("TIMESCALEDB_USER",     "sitaram_user")
DB_PASS     = os.getenv("TIMESCALEDB_PASSWORD", "sitaram_secure_2026")

# Gate thresholds — must match Run-15 validated params
SHARPE_TARGET          = float(os.getenv("SHARPE_TARGET",          "3.0"))
MAX_DD_PCT             = float(os.getenv("MAX_DD_PCT",             "5.0"))
IC_MIN                 = float(os.getenv("IC_MIN",                 "0.02"))
FILL_RATE_MIN_PCT      = float(os.getenv("FILL_RATE_MIN_PCT",      "60.0"))
INV_REVERSION_MIN_PCT  = float(os.getenv("INV_REVERSION_MIN_PCT",  "70.0"))
ADVERSE_SEL_MAX_PCT    = float(os.getenv("ADVERSE_SEL_MAX_PCT",    "30.0"))
QTR_MIN                = float(os.getenv("QTR_MIN",                "1.0"))
SPREAD_MIN_USDT        = float(os.getenv("SPREAD_MIN_USDT",        "0.10"))
IC_STABILITY_MAX_RATIO = float(os.getenv("IC_STABILITY_MAX_RATIO", "2.0"))
MAX_INVENTORY_BTC      = float(os.getenv("MAX_INVENTORY_BTC",      "0.1"))

TICKS_PER_YEAR = 429_000   # ~500ms ticks × 6.5 hours × 252 trading days (approx)

log = logging.getLogger("gate_calculator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


# ── DB connection ───────────────────────────────────────────────────────────────
def _connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        connect_timeout=10
    )


# ── Individual gate calculations ────────────────────────────────────────────────

def _gate_sharpe(cur, session_id):
    """Annualised Sharpe = mean(pnl) / std(pnl) * sqrt(TICKS_PER_YEAR)"""
    cur.execute("""
        SELECT
            AVG(realized_pnl)          AS mean_pnl,
            STDDEV(realized_pnl)       AS std_pnl,
            COUNT(*)                   AS n
        FROM trades
        WHERE session_id = %s
    """, (session_id,))
    row = cur.fetchone()
    if not row or row[2] < 2 or (row[1] or 0) == 0:
        return None, {"reason": "insufficient trades", "n": row[2] if row else 0}
    mean_pnl, std_pnl, n = row
    ##sharpe = (mean_pnl / std_pnl) * math.sqrt(TICKS_PER_YEAR)
    sharpe = (float(mean_pnl) / float(std_pnl)) * math.sqrt(TICKS_PER_YEAR)
    return sharpe >= SHARPE_TARGET, {
        "value": round(sharpe, 4),
        "mean_pnl": round(mean_pnl, 6),
        "std_pnl":  round(std_pnl, 6),
        "n_fills":  n,
        "target":   SHARPE_TARGET
    }


def _gate_max_dd(cur, session_id):
    """Max drawdown < MAX_DD_PCT% of peak cumulative PnL"""
    cur.execute("""
        SELECT cumulative_pnl
        FROM trades
        WHERE session_id = %s
        ORDER BY trade_time ASC
    """, (session_id,))
    rows = [r[0] for r in cur.fetchall()]
    if len(rows) < 2:
        return None, {"reason": "insufficient trades"}

    peak = rows[0]
    max_dd = 0.0
    for pnl in rows:
        if pnl > peak:
            peak = pnl
        dd = (peak - pnl) / abs(peak) * 100 if peak != 0 else 0
        if dd > max_dd:
            max_dd = dd

    return max_dd < MAX_DD_PCT, {
        "value_pct": round(max_dd, 4),
        "peak_pnl":  round(peak, 4),
        "target_pct": MAX_DD_PCT
    }


def _gate_ic(cur, session_id):
    """Mean IC > IC_MIN across all feature_metrics rows for this session"""
    cur.execute("""
        SELECT
            AVG(ic_value)    AS mean_ic,
            COUNT(*)         AS n
        FROM feature_metrics
        WHERE session_id = %s
    """, (session_id,))
    row = cur.fetchone()
    if not row or row[1] == 0:
        return None, {"reason": "no feature_metrics rows"}
    mean_ic, n = row
    if mean_ic is None:
        return None, {"reason": "ic_value all NULL"}
    return float(mean_ic) > IC_MIN, {
        "value": round(float(mean_ic), 6),
        "n_rows": n,
        "target": IC_MIN
    }


def _gate_fill_rate(cur, session_id):
    """Fill rate = fills / quotes_sent * 100  > FILL_RATE_MIN_PCT"""
    cur.execute("""
        SELECT
            COUNT(*)           AS fills,
            SUM(quotes_sent)   AS total_quotes
        FROM trades
        WHERE session_id = %s
    """, (session_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        return None, {"reason": "no trades"}
    fills       = row[0]
    total_quotes = row[1] or 0
    if total_quotes == 0:
        return None, {"reason": "quotes_sent column is 0"}
    rate = fills / total_quotes * 100
    return rate > FILL_RATE_MIN_PCT, {
        "value_pct":    round(rate, 4),
        "fills":        fills,
        "quotes_sent":  total_quotes,
        "target_pct":   FILL_RATE_MIN_PCT
    }


def _gate_inventory_reversion(cur, session_id):
    """% of ticks where |inventory_after| < 50% of MAX_INVENTORY_BTC > INV_REVERSION_MIN_PCT"""
    half_max = MAX_INVENTORY_BTC * 0.5
    cur.execute("""
        SELECT
            COUNT(*)                                              AS total,
            SUM(CASE WHEN ABS(inventory_after) < %s THEN 1 ELSE 0 END) AS near_flat
        FROM trades
        WHERE session_id = %s
    """, (half_max, session_id))
    row = cur.fetchone()
    if not row or not row[0]:
        return None, {"reason": "no trades"}
    total, near_flat = row
    rate = (near_flat / total) * 100 if total > 0 else 0
    return rate > INV_REVERSION_MIN_PCT, {
        "value_pct":   round(rate, 4),
        "near_flat":   near_flat,
        "total_fills": total,
        "threshold_btc": half_max,
        "target_pct":  INV_REVERSION_MIN_PCT
    }


def _gate_adverse_selection(cur, session_id):
    """% of fills with negative realised PnL < ADVERSE_SEL_MAX_PCT"""
    cur.execute("""
        SELECT
            COUNT(*)                                           AS total,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) AS adverse
        FROM trades
        WHERE session_id = %s
    """, (session_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        return None, {"reason": "no trades"}
    total, adverse = row
    rate = (adverse / total) * 100 if total > 0 else 0
    return rate < ADVERSE_SEL_MAX_PCT, {
        "value_pct":  round(rate, 4),
        "adverse":    adverse,
        "total":      total,
        "target_pct": ADVERSE_SEL_MAX_PCT
    }


def _gate_qtr(cur, session_id):
    """Quote-to-trade ratio = quotes_sent / fills  ≥ QTR_MIN"""
    cur.execute("""
        SELECT
            COUNT(*)           AS fills,
            SUM(quotes_sent)   AS total_quotes
        FROM trades
        WHERE session_id = %s
    """, (session_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        return None, {"reason": "no trades"}
    fills, total_quotes = row
    if fills == 0:
        return None, {"reason": "zero fills"}
    qtr = (total_quotes or 0) / fills
    return qtr >= QTR_MIN, {
        "value":       round(qtr, 4),
        "fills":       fills,
        "quotes_sent": total_quotes,
        "target":      QTR_MIN
    }


def _gate_spread(cur, session_id):
    """Average spread ≥ SPREAD_MIN_USDT"""
    cur.execute("""
        SELECT AVG(spread), COUNT(*)
        FROM order_book_ticks
        WHERE session_id = %s
    """, (session_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        # fallback — try feature_metrics
        cur.execute("""
            SELECT AVG(spread_usdt), COUNT(*)
            FROM feature_metrics
            WHERE session_id = %s
        """, (session_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return None, {"reason": "no spread data"}
    avg_spread, n = row
    return float(avg_spread) >= SPREAD_MIN_USDT, {
        "value_usdt": round(float(avg_spread), 6),
        "n_rows":     n,
        "target_usdt": SPREAD_MIN_USDT
    }


def _gate_ic_stability(cur, session_id):
    """IC stability: stddev(IC) / mean(IC) < IC_STABILITY_MAX_RATIO"""
    cur.execute("""
        SELECT
            AVG(ic_value)    AS mean_ic,
            STDDEV(ic_value) AS std_ic,
            COUNT(*)         AS n
        FROM feature_metrics
        WHERE session_id = %s
    """, (session_id,))
    row = cur.fetchone()
    if not row or row[2] < 2:
        return None, {"reason": "insufficient feature_metrics rows"}
    mean_ic, std_ic, n = row
    if mean_ic is None or mean_ic == 0:
        return None, {"reason": "mean IC is zero"}
    ratio = abs(float(std_ic or 0)) / abs(float(mean_ic))
    return ratio < IC_STABILITY_MAX_RATIO, {
        "value_ratio": round(ratio, 4),
        "mean_ic":     round(float(mean_ic), 6),
        "std_ic":      round(float(std_ic or 0), 6),
        "n_rows":      n,
        "target_ratio": IC_STABILITY_MAX_RATIO
    }


def _gate_fee_adjusted_pnl(cur, session_id):
    """Final cumulative PnL > 0"""
    cur.execute("""
        SELECT cumulative_pnl
        FROM trades
        WHERE session_id = %s
        ORDER BY trade_time DESC
        LIMIT 1
    """, (session_id,))
    row = cur.fetchone()
    if not row:
        return None, {"reason": "no trades"}
    final_pnl = float(row[0])
    return final_pnl > 0, {
        "final_pnl_usdt": round(final_pnl, 4)
    }


def _gate_daily_loss_limit(session_data):
    """
    No single-day PnL < daily_loss_limit_usdt (from session parameters).
    Uses daily_pnl dict already in the JSON — no DB query needed.
    """
    daily_loss_limit = session_data.get("input_parameters", {}).get(
        "daily_loss_limit_usdt", -2000.0
    )
    daily_pnl = session_data.get("daily_pnl", {})
    if not daily_pnl:
        return None, {"reason": "no daily_pnl data"}

    worst_day = None
    worst_pnl = float("inf")
    for day, pnl in daily_pnl.items():
        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_day = day

    passed = worst_pnl >= daily_loss_limit
    return passed, {
        "worst_day":        worst_day,
        "worst_day_pnl":    round(worst_pnl, 4),
        "daily_loss_limit": daily_loss_limit,
        "all_days":         {k: round(v, 4) for k, v in daily_pnl.items()}
    }


# ── Main calculator ─────────────────────────────────────────────────────────────

def calculate_gates(session_id: str, session_data: dict) -> dict:
    """
    Run all 11 gates for session_id.
    Returns dict: gate_name -> {"pass": bool|None, "details": {...}}
    """
    results = {}

    # Gate 11 — daily loss limit (JSON only, no DB)
    passed, details = _gate_daily_loss_limit(session_data)
    results["daily_loss_limit_respected"] = {"pass": passed, "details": details}
    log.info(f"  [11] daily_loss_limit_respected = {passed}  {details}")

    # Gates 1–10 require TimescaleDB
    try:
        conn = _connect()
        cur  = conn.cursor()
        log.info(f"Connected to TimescaleDB — evaluating gates for {session_id}")

        db_gates = [
            ("sharpe_gte_3",                        _gate_sharpe),
            ("max_dd_lt_5pct",                      _gate_max_dd),
            ("ic_gt_0_02",                          _gate_ic),
            ("fill_rate_gt_60pct",                  _gate_fill_rate),
            ("inventory_mean_reversion_gt_70pct",   _gate_inventory_reversion),
            ("adverse_selection_lt_30pct",           _gate_adverse_selection),
            ("quote_to_trade_ratio_ok",             _gate_qtr),
            ("spread_minimum_ok",                   _gate_spread),
            ("ic_stability_ok",                     _gate_ic_stability),
            ("fee_adjusted_pnl_positive",           _gate_fee_adjusted_pnl),
        ]

        for gate_name, fn in db_gates:
            try:
                passed, details = fn(cur, session_id)
                results[gate_name] = {"pass": passed, "details": details}
                status = "✅ PASS" if passed else ("⚠️  NULL" if passed is None else "❌ FAIL")
                log.info(f"  [{gate_name}] {status}  {details}")
            except Exception as e:
                log.warning(f"  [{gate_name}] ERROR: {e}")
                results[gate_name] = {"pass": None, "details": {"error": str(e)}}

        cur.close()
        conn.close()

    except Exception as e:
        log.error(f"TimescaleDB connection failed: {e}")
        log.warning("DB-dependent gates will be null. Only daily_loss_limit evaluated.")
        for gate_name in [
            "sharpe_gte_3", "max_dd_lt_5pct", "ic_gt_0_02",
            "fill_rate_gt_60pct", "inventory_mean_reversion_gt_70pct",
            "adverse_selection_lt_30pct", "quote_to_trade_ratio_ok",
            "spread_minimum_ok", "ic_stability_ok", "fee_adjusted_pnl_positive"
        ]:
            results[gate_name] = {"pass": None, "details": {"error": str(e)}}

    return results


def calculate_and_write_gates(session_id: str,
                               start_time: datetime = None,
                               end_time: datetime = None) -> dict:
    """
    High-level entry point called from engine.py shutdown().

    1. Loads the master JSON
    2. Finds the session by session_id
    3. Calculates all 11 gates
    4. Writes gate pass/fail + details back to JSON
    5. Sets session status = COMPLETED and writes end_time + duration
    6. Saves the JSON
    7. Returns the gate results dict
    """
    # Load master JSON
    try:
        with open(MASTER_JSON, "r", encoding="utf-8") as f:
            master = json.load(f)
    except FileNotFoundError:
        log.error(f"Master JSON not found: {MASTER_JSON}")
        return {}
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return {}

    # Find session
    session_data = None
    session_idx  = None
    for i, s in enumerate(master.get("sessions", [])):
        if s.get("session_id") == session_id:
            session_data = s
            session_idx  = i
            break

    if session_data is None:
        log.error(f"Session {session_id} not found in master JSON")
        return {}

    log.info(f"Calculating gates for session {session_id}")
    gate_results = calculate_gates(session_id, session_data)

    # Summarise
    passed  = sum(1 for g in gate_results.values() if g["pass"] is True)
    failed  = sum(1 for g in gate_results.values() if g["pass"] is False)
    unknown = sum(1 for g in gate_results.values() if g["pass"] is None)
    total   = len(gate_results)

    log.info(f"Gates: {passed}/{total} passed, {failed} failed, {unknown} unknown/null")

    # Build gate summary for JSON (simple pass/fail + details)
    jenkins_gates_update = {}
    gate_details_update  = {}
    for gate_name, result in gate_results.items():
        jenkins_gates_update[gate_name] = result["pass"]
        gate_details_update[gate_name]  = result["details"]

    # Write end_time and duration
    now = end_time or datetime.now(timezone.utc)
    start = start_time
    if start is None:
        raw = session_data.get("start_time")
        if raw:
            start = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    duration_sec = round((now - start).total_seconds(), 1) if start else None

    # Update session
    master["sessions"][session_idx]["status"]         = "COMPLETED"
    master["sessions"][session_idx]["end_time"]       = now.isoformat()
    master["sessions"][session_idx]["duration_sec"]   = duration_sec
    master["sessions"][session_idx]["jenkins_gates"]  = jenkins_gates_update
    master["sessions"][session_idx]["gate_details"]   = gate_details_update
    master["sessions"][session_idx]["results"] = {
        "gates_passed":  passed,
        "gates_failed":  failed,
        "gates_unknown": unknown,
        "gates_total":   total,
        "all_passed":    passed == total,
        "evaluated_at":  now.isoformat()
    }

    # Save JSON
    try:
        with open(MASTER_JSON, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, default=str)
        log.info(f"Master JSON updated — session {session_id} marked COMPLETED")
        log.info(f"File: {MASTER_JSON}")
    except Exception as e:
        log.error(f"Failed to write master JSON: {e}")

    # Print final summary
    print("\n" + "=" * 60)
    print(f"  SITARAM GATE RESULTS — {session_id}")
    print("=" * 60)
    for gate_name, result in gate_results.items():
        status = "✅ PASS" if result["pass"] is True else \
                 ("❌ FAIL" if result["pass"] is False else "⚠️  NULL")
        print(f"  {status}  {gate_name}")
    print("=" * 60)
    print(f"  {passed}/{total} gates passed")
    print("=" * 60 + "\n")

    return gate_results


# ── Standalone CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SITARAM Gate Calculator")
    parser.add_argument(
        "--session", required=True,
        help="Session ID e.g. SES_20260413_120657"
    )
    parser.add_argument(
        "--master-json", default=MASTER_JSON,
        help="Path to live_BTCUSDT_master.json"
    )
    args = parser.parse_args()

    MASTER_JSON = args.master_json
    calculate_and_write_gates(args.session)
