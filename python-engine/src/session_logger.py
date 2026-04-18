"""
session_logger.py  —  SITARAM Live Session Master File Writer
=============================================================================
Writes every live trading session to:
    E:\\Binance\\live_BTCUSDT_master.json

One master file. Append forever. Feed to Claude AI for strategy analysis.

JSON Structure per session:
  - session_id          : unique identifier
  - status              : RUNNING | COMPLETED | HALTED
  - start_time / end_time / duration_sec
  - input_parameters    : all 15 AS model parameters
  - results             : final metrics (PnL, Sharpe, fills, inventory, etc.)
  - daily_pnl           : per-day breakdown matching Excel tracker
  - jenkins_gates       : pass/fail for all 11 validation gates
  - halt_events         : every halt with reason + timestamp
  - stall_events        : every data feed stall

Usage:
  from session_logger import SessionLogger
  logger = SessionLogger()
  logger.open_session()
  logger.record_daily_pnl("2026-04-06", 1738.0)
  logger.update_jenkins_gates({"sharpe": True, "max_dd": True, ...})
  logger.close_session(results_dict)

Called by engine.py on start and shutdown.
Also called by agent.py to update gates and daily PnL during the session.
=============================================================================
"""

import os
import json
import time
import logging
import fcntl          # file lock — prevents concurrent writes corrupting JSON
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("session-logger")

# =============================================================================
# PATH CONFIG
# =============================================================================
MASTER_FILE = os.getenv(
    "MASTER_JSON_PATH",
    "/mnt/sessions/live_BTCUSDT_master.json"   # Docker path (E:\Binance\ mounted here)
)

# =============================================================================
# ALL 11 JENKINS GATE DEFINITIONS
# Keys match Jenkins pipeline gate names exactly.
# =============================================================================
JENKINS_GATES = [
    "sharpe_gte_3",           # Sharpe ≥ 3.0
    "max_dd_lt_5pct",         # MaxDD < 5%
    "ic_gt_0_02",             # IC > 0.02
    "fill_rate_gt_60pct",     # Fill rate > 60%
    "inventory_mean_reversion_gt_70pct",  # Inventory mean reversion > 70%
    "adverse_selection_lt_30pct",         # Adverse selection rate < 30%
    "quote_to_trade_ratio_ok",            # Quote-to-trade ratio within bounds
    "spread_minimum_ok",                  # Spread ≥ minimum tick
    "ic_stability_ok",                    # IC stable across time windows
    "fee_adjusted_pnl_positive",          # Fee-adjusted P&L > 0
    "daily_loss_limit_respected",         # No daily loss limit breach
]

# =============================================================================
# MASTER FILE I/O  (thread-safe, file-locked)
# =============================================================================
def _load_master() -> dict:
    """Load master JSON. Creates fresh file if not found."""
    if not os.path.exists(MASTER_FILE):
        _init_master_file()
    try:
        with open(MASTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Master file read error: {e} — reinitialising")
        _init_master_file()
        with open(MASTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def _save_master(data: dict):
    """Save master JSON atomically with file lock."""
    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
    tmp = MASTER_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2, ensure_ascii=False)
            fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(tmp, MASTER_FILE)   # atomic replace — no partial writes
    except Exception as e:
        log.error(f"Master file write error: {e}")


def _init_master_file():
    """Create fresh master file with schema header."""
    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
    skeleton = {
        "_schema_version": "1.0",
        "_description":    "SITARAM HFT Live Trading Master File — BTC/USDT Bybit Spot",
        "_model":          "Avellaneda-Stoikov",
        "_created":        datetime.now(timezone.utc).isoformat(),
        "_purpose":        "Single master file for all live sessions. Feed to Claude AI for analysis.",
        "_fields_guide": {
            "session_id":       "Unique session ID — SES_YYYYMMDD_HHMMSS",
            "status":           "RUNNING | COMPLETED | HALTED",
            "input_parameters": "All AS model parameters active in this session",
            "results":          "Final metrics written on shutdown",
            "daily_pnl":        "Per-day PnL breakdown — date: amount",
            "jenkins_gates":    "Pass/fail for all 11 validation gates",
            "halt_events":      "Each halt: reason, time, inventory, pnl",
            "stall_events":     "Each feed stall: duration, last tick time",
        },
        "sessions": []
    }
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(skeleton, f, indent=2, ensure_ascii=False)
    log.info(f"Master file initialised: {MASTER_FILE}")


# =============================================================================
# SESSION LOGGER CLASS
# =============================================================================
class SessionLogger:
    """
    One instance per engine run.
    Call open_session() on startup, close_session() on shutdown.
    Call record_daily_pnl() and update_jenkins_gates() during the session.
    """

    def __init__(self, params: dict):
        """
        params: dict of all input parameters from engine.py
                (AS_GAMMA, MAX_INVENTORY, DAILY_LOSS_LIMIT, etc.)
        """
        self.session_id  = datetime.now(timezone.utc).strftime("SES_%Y%m%d_%H%M%S")
        self.params      = params
        self._start_time = time.time()
        self._opened     = False

    # ------------------------------------------------------------------
    # OPEN SESSION — call on engine startup
    # ------------------------------------------------------------------
    def open_session(self):
        """Write initial session record with RUNNING status."""
        master = _load_master()

        record = {
            "session_id":   self.session_id,
            "status":       "RUNNING",
            "symbol":       "BTCUSDT",
            "exchange":     "Bybit Spot",
            "mode":         "Paper Trading",
            "model":        "Avellaneda-Stoikov",
            "run_ref":      "Run-15",

            # Timestamps
            "start_time":   datetime.now(timezone.utc).isoformat(),
            "end_time":     None,
            "duration_sec": None,

            # All 15 input parameters
            "input_parameters": {
                "half_spread_usdt":       self.params.get("HALF_SPREAD",       0.15),
                "as_gamma":               self.params.get("AS_GAMMA",          2.0),
                "max_inventory_btc":      self.params.get("MAX_INVENTORY",     0.10),
                "quote_size_btc":         self.params.get("QUOTE_SIZE",        0.01),
                "tick_size":              self.params.get("TICK_SIZE",         0.1),
                "maker_fee":              self.params.get("MAKER_FEE",        -0.0001),
                "taker_fee":              self.params.get("TAKER_FEE",         0.0005),
                "initial_capital_usdt":   self.params.get("INITIAL_CAP",      100000.0),
                "daily_loss_limit_usdt":  self.params.get("DAILY_LOSS_LIM",   -2000.0),
                "regime_vol_threshold":   self.params.get("REGIME_VOL_THRESHOLD", 6e-5),
                "regime_vol_window":      self.params.get("REGIME_VOL_WINDOW", 50),
                "signal_weights": {
                    "obi":   0.50,
                    "micro": 0.35,
                    "vol":   0.15,
                },
                "data_stall_threshold_sec": self.params.get("DATA_STALL_SEC", 30),
                "simulated_latency_ms":   self.params.get("SIMULATED_LATENCY_MS", 10),
            },

            # Results — filled on close_session()
            "results": None,

            # Daily PnL — keyed by date string "YYYY-MM-DD"
            # Updated live by record_daily_pnl()
            "daily_pnl": {},

            # Jenkins gates — all start as None (not yet evaluated)
            "jenkins_gates": {g: None for g in JENKINS_GATES},

            # Events
            "halt_events":  [],
            "stall_events": [],

            # Running totals updated live
            "_live": {
                "last_updated":     datetime.now(timezone.utc).isoformat(),
                "cumulative_pnl":   0.0,
                "total_trades":     0,
                "inventory_btc":    0.0,
                "regime":           "normal",
                "data_feed_active": True,
                "latency_min_ms":   0.0,
                "latency_avg_ms":   0.0,
                "latency_max_ms":   0.0,
            }
        }

        master["sessions"].append(record)
        _save_master(master)
        self._opened = True
        log.info(f"Session opened: {self.session_id} → {MASTER_FILE}")
        return self.session_id

    # ------------------------------------------------------------------
    # RECORD DAILY PnL — call once per day or on each day boundary
    # ------------------------------------------------------------------
    def record_daily_pnl(self, date_str: str, pnl_usdt: float):
        """
        date_str: "YYYY-MM-DD"
        pnl_usdt: realised PnL for that day
        """
        master = _load_master()
        session = self._find_session(master)
        if session is None:
            return
        session["daily_pnl"][date_str] = round(pnl_usdt, 4)
        # Guard: _live may have been removed by close_session() already
        if "_live" in session:
            session["_live"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        _save_master(master)
        log.info(f"Daily PnL recorded: {date_str} = {pnl_usdt:+.2f} USDT")

    # ------------------------------------------------------------------
    # UPDATE LIVE METRICS — call periodically (every 60s)
    # ------------------------------------------------------------------
    def update_live_metrics(self, cumulative_pnl: float, total_trades: int,
                             inventory_btc: float, regime: str,
                             data_feed_active: bool,
                             latency_min_ms: float = 0.0,
                             latency_avg_ms: float = 0.0,
                             latency_max_ms: float = 0.0):
        """Snapshot of current live state — gives Claude mid-session visibility."""
        master = _load_master()
        session = self._find_session(master)
        if session is None:
            return
        session["_live"] = {
            "last_updated":     datetime.now(timezone.utc).isoformat(),
            "cumulative_pnl":   round(cumulative_pnl, 4),
            "total_trades":     total_trades,
            "inventory_btc":    round(inventory_btc, 6),
            "regime":           regime,
            "data_feed_active": data_feed_active,
            "latency_min_ms":   round(latency_min_ms, 2),
            "latency_avg_ms":   round(latency_avg_ms, 2),
            "latency_max_ms":   round(latency_max_ms, 2),
        }
        _save_master(master)

    # ------------------------------------------------------------------
    # UPDATE JENKINS GATES — call when agent.py evaluates gates
    # ------------------------------------------------------------------
    def update_jenkins_gates(self, gate_results: dict):
        """
        gate_results: dict of gate_name → True/False/None
        Example:
            {
                "sharpe_gte_3": True,
                "max_dd_lt_5pct": True,
                "ic_gt_0_02": False,
                ...
            }
        """
        master = _load_master()
        session = self._find_session(master)
        if session is None:
            return
        for gate, passed in gate_results.items():
            if gate in JENKINS_GATES:
                session["jenkins_gates"][gate] = passed

        # Summary counts
        evaluated = {k: v for k, v in session["jenkins_gates"].items()
                     if v is not None}
        session["_gates_summary"] = {
            "total":    len(JENKINS_GATES),
            "passed":   sum(1 for v in evaluated.values() if v is True),
            "failed":   sum(1 for v in evaluated.values() if v is False),
            "pending":  len(JENKINS_GATES) - len(evaluated),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_master(master)
        log.info(f"Jenkins gates updated: "
                 f"{session['_gates_summary']['passed']}/{len(JENKINS_GATES)} passed")

    # ------------------------------------------------------------------
    # APPEND HALT EVENT — call from engine._trigger_halt()
    # ------------------------------------------------------------------
    def append_halt_event(self, reason: str, inventory: float, pnl: float, mid: float):
        master = _load_master()
        session = self._find_session(master)
        if session is None:
            return
        session["halt_events"].append({
            "time":      datetime.now(timezone.utc).isoformat(),
            "reason":    reason,
            "inventory_btc": round(inventory, 6),
            "pnl_usdt":  round(pnl, 4),
            "mid_price": round(mid, 1),
        })
        session["status"] = "HALTED"
        _save_master(master)
        log.warning(f"Halt event logged: {reason}")

    # ------------------------------------------------------------------
    # APPEND STALL EVENT — call from engine stall watchdog
    # ------------------------------------------------------------------
    def append_stall_event(self, stall_sec: float, last_tick_ts_ms: float):
        master = _load_master()
        session = self._find_session(master)
        if session is None:
            return
        session["stall_events"].append({
            "time":          datetime.now(timezone.utc).isoformat(),
            "stall_sec":     round(stall_sec, 1),
            "last_tick_utc": datetime.fromtimestamp(
                last_tick_ts_ms / 1000, tz=timezone.utc).isoformat()
                if last_tick_ts_ms > 0 else "unknown",
        })
        _save_master(master)

    # ------------------------------------------------------------------
    # CLOSE SESSION — call on engine shutdown (Ctrl+C or graceful stop)
    # ------------------------------------------------------------------
    def close_session(self, final_metrics: dict):
        """
        final_metrics should include at minimum:
            cumulative_pnl_usdt, daily_pnl_usdt, total_trades,
            final_inventory_btc, sharpe, fill_rate, max_drawdown_pct,
            ic_value, adverse_selection_rate, uptime_sec, halt_reason
        """
        if not self._opened:
            log.warning("close_session() called but session was never opened")
            return

        master  = _load_master()
        session = self._find_session(master)
        if session is None:
            log.error(f"Session {self.session_id} not found in master file")
            return

        duration = int(time.time() - self._start_time)

        # Compute daily PnL total from breakdown for cross-check
        daily_total = sum(session["daily_pnl"].values())

        session["status"]       = "COMPLETED" if not session["halt_events"] else "HALTED"
        session["end_time"]     = datetime.now(timezone.utc).isoformat()
        session["duration_sec"] = duration
        session["duration_human"] = _fmt_duration(duration)

        session["results"] = {
            # Core performance
            "cumulative_pnl_usdt":    round(final_metrics.get("cumulative_pnl_usdt", 0), 4),
            "daily_pnl_total_usdt":   round(daily_total, 4),
            "sharpe_ratio":           round(final_metrics.get("sharpe", 0), 4),
            "max_drawdown_pct":       round(final_metrics.get("max_drawdown_pct", 0), 4),

            # Execution quality
            "total_trades":           final_metrics.get("total_trades", 0),
            "fill_rate_pct":          round(final_metrics.get("fill_rate", 0) * 100, 2),
            "adverse_selection_rate": round(final_metrics.get("adverse_selection_rate", 0), 4),
            "avg_spread_usdt":        round(final_metrics.get("avg_spread_usdt", 0), 4),

            # Latency — tick-to-fill end-to-end (milliseconds)
            "latency_min_ms":         round(final_metrics.get("latency_min_ms", 0.0), 2),
            "latency_avg_ms":         round(final_metrics.get("latency_avg_ms", 0.0), 2),
            "latency_max_ms":         round(final_metrics.get("latency_max_ms", 0.0), 2),
            "latency_fill_count":     final_metrics.get("latency_fill_count", 0),

            # Signal quality
            "ic_value":               round(final_metrics.get("ic_value", 0), 6),
            "avg_obi":                round(final_metrics.get("avg_obi", 0), 4),
            "avg_composite_signal":   round(final_metrics.get("avg_composite_signal", 0), 4),

            # Inventory
            "final_inventory_btc":    round(final_metrics.get("final_inventory_btc", 0), 6),
            "max_inventory_btc":      round(final_metrics.get("max_inventory_btc", 0), 6),
            "inventory_mean_reversion_pct": round(
                final_metrics.get("inventory_mean_reversion_pct", 0), 2),

            # Session meta
            "uptime_sec":             final_metrics.get("uptime_sec", duration),
            "halt_count":             len(session["halt_events"]),
            "stall_count":            len(session["stall_events"]),
            "halt_reason":            final_metrics.get("halt_reason", "MANUAL_STOP"),
        }

        # Clean up live snapshot — no longer needed
        session.pop("_live", None)

        _save_master(master)
        log.info(f"Session {self.session_id} closed → {MASTER_FILE} "
                 f"| PnL: {session['results']['cumulative_pnl_usdt']:+.2f} USDT "
                 f"| Duration: {session['duration_human']}")

    # ------------------------------------------------------------------
    # INTERNAL: find this session in the master sessions list
    # ------------------------------------------------------------------
    def _find_session(self, master: dict) -> Optional[dict]:
        for s in master.get("sessions", []):
            if s.get("session_id") == self.session_id:
                return s
        log.error(f"Session {self.session_id} not found in master")
        return None


# =============================================================================
# HELPERS
# =============================================================================
def _fmt_duration(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}h {m}m {s}s"


# =============================================================================
# QUERY HELPERS — for Claude AI analysis
# =============================================================================
def get_all_sessions() -> list:
    """Return full sessions list for Claude AI analysis."""
    return _load_master().get("sessions", [])

def get_session_summary() -> list:
    """Return lightweight summary of all sessions — no event arrays."""
    sessions = get_all_sessions()
    summary = []
    for s in sessions:
        r = s.get("results") or {}
        summary.append({
            "session_id":        s["session_id"],
            "status":            s["status"],
            "start_time":        s["start_time"],
            "end_time":          s.get("end_time"),
            "duration_human":    s.get("duration_human"),
            "cumulative_pnl":    r.get("cumulative_pnl_usdt"),
            "sharpe":            r.get("sharpe_ratio"),
            "max_dd_pct":        r.get("max_drawdown_pct"),
            "fill_rate_pct":     r.get("fill_rate_pct"),
            "ic_value":          r.get("ic_value"),
            "total_trades":      r.get("total_trades"),
            "halt_count":        r.get("halt_count", 0),
            "gates_passed":      s.get("_gates_summary", {}).get("passed"),
            "gates_total":       s.get("_gates_summary", {}).get("total"),
            "daily_pnl":         s.get("daily_pnl", {}),
        })
    return summary

def get_latest_session() -> Optional[dict]:
    """Return the most recent session record."""
    sessions = get_all_sessions()
    return sessions[-1] if sessions else None

def get_sessions_by_status(status: str) -> list:
    """Filter sessions by status: RUNNING | COMPLETED | HALTED"""
    return [s for s in get_all_sessions() if s.get("status") == status]
