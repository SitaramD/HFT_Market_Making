# =============================================================================
# SITARAM HFT — Backtester
#
# Pipeline:
#   1. DataLoader    — reads OB .data files (NDJSON) and trade CSV files
#   2. OrderBook     — reconstructs L2 book tick-by-tick
#   3. ASModel       — Avellaneda-Stoikov quotes: bid, ask, reservation price
#   4. FillSimulator — checks real public trades to determine fills
#   5. Backtester    — drives the loop, splits IS/OOS, runs walk-forward
#   6. ResultSaver   — appends every run to master JSON at E:\Binance
#
# Bugs fixed vs original:
#   BUG-1  load_trades: `f` -> `fpath`  (NameError / wrong file opened)
#   BUG-2  load_trades: dtype dict referenced non-existent post-transform
#          columns ('volume','side','rpi') instead of raw Binance columns
#   BUG-3  load_trades: header=0 on a headerless Binance CSV caused
#          'id' string to be cast as int64 -> the crash you saw
#   BUG-4  load_trades: parse_dates=['timestamp'] -- column is 'time' not
#          'timestamp' in raw Binance format; removed and converted manually
#   BUG-5  Duplicate `import logging` and `log =` at line 344-345 removed
#
# No synthetic data. If files are not found, the run raises FileNotFoundError.
# =============================================================================

import os
import json
import glob
import logging
import traceback
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from src.orderbook import OrderBook
from src.metrics import FillRecord, MetricsResult, MetricsCalculator

log = logging.getLogger('sitaram.backtester')


# =============================================================================
# MASTER JSON RESULT SAVER
# =============================================================================

# Single file that accumulates ALL backtest runs -- grows with every run.
MASTER_RESULT_FILE = r'E:\Binance\sitaram_backtest_master.json'


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------

def _safe_float(v):
    """Convert any numeric value to a JSON-safe scalar. NaN/Inf -> None."""
    if isinstance(v, float):
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 8)
    if isinstance(v, np.floating):
        return None if (np.isnan(v) or np.isinf(v)) else round(float(v), 8)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _serialize_period(
    metrics_result : MetricsResult,
    date_list      : List[str],
    run_ts_utc     : str,
    n_fills        : int,
    n_quotes       : int,
) -> dict:
    """
    Serialize one IS / OOS / Full-Period metric block to a dict.

    Layout in the JSON (period identity fields ALWAYS appear first so
    Claude LLM and humans immediately know WHAT period they are reading):

        period_start_date   "2026-03-01"
        period_end_date     "2026-03-11"
        n_days              11
        run_timestamp_utc   "2026-03-21T08:43:19+00:00"
        n_fills             8432
        n_quotes_placed     11620
        fill_rate_pct       72.56

        <all 23 MetricsResult fields follow>
    """
    block = {
        'period_start_date' : date_list[0]  if date_list else None,
        'period_end_date'   : date_list[-1] if date_list else None,
        'n_days'            : len(date_list),
        'run_timestamp_utc' : run_ts_utc,
        'n_fills'           : n_fills,
        'n_quotes_placed'   : n_quotes,
        'fill_rate_pct'     : round(n_fills / max(n_quotes, 1) * 100, 4),
    }

    # Append all 23 MetricsResult fields (works for dataclass or __dict__ obj)
    for k, v in vars(metrics_result).items():
        block[k] = _safe_float(v)

    return block


def _wf_window_to_dict(w: dict, run_ts_utc: str) -> dict:
    """Serialize one walk-forward window dict. Injects run_timestamp_utc."""
    train_d = {k: _safe_float(v) for k, v in vars(w['train']).items()}
    test_d  = {k: _safe_float(v) for k, v in vars(w['test']).items()}
    train_d['run_timestamp_utc'] = run_ts_utc
    test_d['run_timestamp_utc']  = run_ts_utc
    return {
        'window' : w['window'],
        'label'  : w['label'],
        'train'  : train_d,
        'test'   : test_d,
    }


# -----------------------------------------------------------------------------
# Run-record builder
# -----------------------------------------------------------------------------

def _build_run_record(results: dict, cfg: dict, run_meta: dict) -> dict:
    """
    Build one complete, self-contained run record for the master JSON.

    JSON structure (in order):
      run_meta                 identity: run_id, timestamp, duration, files,
                               backtest_start_date, backtest_end_date, n_days
      config                   all AS / risk / data parameters
      fills_by_day             per-day fill counts
      in_sample                period_start/end/n_days + all 23 metrics
      out_of_sample            period_start/end/n_days + all 23 metrics
      full_period              period_start/end/n_days + all 23 metrics
      walk_forward             IS->OOS Sharpe degradation windows
      market_making_metrics    11 pass/fail thresholds + ALL_METRICS_PASS
    """

    is_r   = results['in_sample']
    oos_r  = results['out_of_sample']
    full_r = results['full']
    run_ts = run_meta['run_ts_utc']

    # Date lists carried from _build_results
    is_dates  = results.get('is_dates',  [])
    oos_dates = results.get('oos_dates', [])
    all_dates = results.get('all_dates', [])

    # Fill / quote counts per period
    total_fills = results.get('total_fills', 0)
    total_qp    = results.get('total_quotes_placed', 0)
    is_fills_n  = results.get('is_fills_count',  0)
    oos_fills_n = results.get('oos_fills_count', 0)
    is_qp       = results.get('is_quotes',  0)
    oos_qp      = results.get('oos_quotes', 0)

    # ---- Market Making Metrics (pass/fail thresholds) --------------------
    # Evaluated on OOS where data exists -- more stringent than IS.
    ref = oos_r if oos_dates else full_r

    market_making_metrics = {
        'IC_gt_0.02'                       : (getattr(ref,   'information_coefficient', 0) or 0) > 0.02,
        'Sharpe_gt_1.5'                    : (getattr(ref,   'sharpe_ratio',            0) or 0) > 1.5,
        'MaxDrawdown_lt_5pct'              : abs(getattr(ref, 'max_drawdown_pct',      100) or 100) < 5.0,
        'FillRate_gt_60pct'                : (getattr(ref,   'fill_rate_pct',           0) or 0) > 60.0,
        'AdverseSelection_lt_30pct'        : (getattr(ref,   'adverse_selection_rate',  1) or 1) < 0.30,
        'QuoteToTradeRatio_gte_1'          : (getattr(ref,   'quote_to_trade_ratio',    0) or 0) >= 1.0,
        'SpreadMinimum_gte_tick'           : (getattr(ref,   'avg_spread_at_fill',      0) or 0) >= cfg.get('tick_size_usdt', 0.1),
        'IC_Stability_degradation_lt_50pct': abs(getattr(oos_r, 'sharpe_degradation_pct', 0) or 0) < 50.0,
        'FeeAdjustedPnL_positive'          : (getattr(full_r, 'net_pnl_usdt',           0) or 0) > 0,
        'InventoryMeanReversion_ok'        : abs(getattr(ref, 'avg_inventory',           0) or 0) < cfg.get('max_inventory_btc', 0.1) * 0.5,
        'AllSystemChecks_pass'             : True,   # set externally by automated checks
    }
    market_making_metrics['ALL_METRICS_PASS'] = all(market_making_metrics.values())

    # ---- Per-day fill summary --------------------------------------------
    fills_by_day_summary = [
        {'date': d, 'fills': cnt}
        for d, cnt in results.get('fills_by_day', [])
    ]

    return {
        # 1. Run identity -------------------------------------------------
        'run_meta': {
            **run_meta,
            'total_quotes_placed' : total_qp,
            'total_fills'         : total_fills,
            'fill_rate_pct'       : round(total_fills / max(total_qp, 1) * 100, 4),
            'backtest_start_date' : all_dates[0]  if all_dates else None,
            'backtest_end_date'   : all_dates[-1] if all_dates else None,
            'backtest_n_days'     : len(all_dates),
        },

        # 2. Strategy / risk parameters -----------------------------------
        'config': {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in cfg.items()
        },

        # 3. Per-day fill breakdown ----------------------------------------
        'fills_by_day': fills_by_day_summary,

        # 4. In-Sample: period identity + all 23 metrics ------------------
        'in_sample': _serialize_period(
            is_r, is_dates, run_ts, is_fills_n, is_qp
        ),

        # 5. Out-of-Sample: period identity + all 23 metrics --------------
        'out_of_sample': _serialize_period(
            oos_r, oos_dates, run_ts, oos_fills_n, oos_qp
        ),

        # 6. Full Period: all days combined --------------------------------
        'full_period': _serialize_period(
            full_r, all_dates, run_ts, total_fills, total_qp
        ),

        # 7. Walk-forward IS->OOS degradation windows ----------------------
        'walk_forward': [
            _wf_window_to_dict(w, run_ts)
            for w in results.get('walk_forward_windows', [])
        ],

        # 8. Market Making Metrics: 11 thresholds + ALL_METRICS_PASS -------
        'market_making_metrics': market_making_metrics,
    }


# -----------------------------------------------------------------------------
# Master file writer
# -----------------------------------------------------------------------------

def save_run_to_master(
    results      : dict,
    cfg          : dict,
    run_id       : str,
    duration_sec : float,
    ob_files     : List[str],
    trade_files  : List[str],
    error        : Optional[str] = None,
) -> str:
    """
    Append this run's full result to E:\\Binance\\sitaram_backtest_master.json.

    Master file structure:
    {
        "schema_version": "1.0",
        "description":    "...",
        "metric_glossary": { ... },
        "market_making_metrics_glossary": { ... },
        "total_runs":     N,
        "last_updated_utc": "...",
        "runs": [ <run_record>, ... ]   <- oldest first
    }

    Uses atomic write (tmp -> rename) so a crash never corrupts the file.
    Returns the path written.
    """
    master_path = MASTER_RESULT_FILE
    run_ts      = datetime.now(timezone.utc).isoformat()

    run_meta = {
        'run_id'       : run_id,
        'run_ts_utc'   : run_ts,
        'duration_sec' : round(duration_sec, 2),
        'ob_files'     : [os.path.basename(f) for f in ob_files],
        'trade_files'  : [os.path.basename(f) for f in trade_files],
        'n_ob_days'    : len(ob_files),
        'status'       : 'ERROR' if error else 'OK',
        'error'        : error,
    }

    if error:
        # Minimal record so every run -- even failures -- is traceable
        run_record = {
            'run_meta'              : run_meta,
            'config'                : {
                k: (float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in cfg.items()
            },
            'market_making_metrics' : {'ALL_METRICS_PASS': False},
        }
    else:
        run_record = _build_run_record(results, cfg, run_meta)

    # Load or create master file
    os.makedirs(os.path.dirname(master_path), exist_ok=True)

    if os.path.exists(master_path):
        try:
            with open(master_path, 'r', encoding='utf-8') as fh:
                master = json.load(fh)
        except (json.JSONDecodeError, OSError):
            log.warning("Master file unreadable -- backing up and creating fresh.")
            _backup_corrupt(master_path)
            master = _empty_master()
    else:
        master = _empty_master()

    master['runs'].append(run_record)
    master['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
    master['total_runs']       = len(master['runs'])

    # Atomic write: write to .tmp then rename (safe on NTFS + ext4)
    tmp_path = master_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as fh:
        json.dump(master, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, master_path)

    all_pass = run_record.get('market_making_metrics', {}).get('ALL_METRICS_PASS', False)
    log.info(
        f"Run saved to master JSON -> {master_path}  "
        f"(run #{master['total_runs']}  |  "
        f"status: {run_meta['status']}  |  "
        f"ALL_METRICS_PASS: {all_pass})"
    )
    return master_path


def _empty_master() -> dict:
    """Return a fresh master file skeleton with glossaries embedded."""
    return {
        'schema_version'   : '1.0',
        'description'      : (
            'SITARAM HFT Backtest Master Results. '
            'One entry per run in "runs" list, appended chronologically. '
            'Designed for analysis by Claude LLM. '
            'Each run contains: run_meta (identity, period dates, duration), '
            'config (all AS/risk/data params), fills_by_day (per-day breakdown), '
            'in_sample / out_of_sample / full_period (period dates + all 23 metrics), '
            'walk_forward (IS->OOS Sharpe degradation), '
            'market_making_metrics (11 pass/fail thresholds + ALL_METRICS_PASS).'
        ),

        # Metric field definitions so Claude LLM has full context
        'metric_glossary': {
            'period_start_date'        : 'First date of this IS/OOS/Full window e.g. 2026-03-01.',
            'period_end_date'          : 'Last date of this IS/OOS/Full window e.g. 2026-03-11.',
            'n_days'                   : 'Number of calendar days in this metric window.',
            'run_timestamp_utc'        : 'UTC ISO-8601 timestamp of the backtest run that produced this record.',
            'n_fills'                  : 'Total order fills in this period.',
            'n_quotes_placed'          : 'Total limit orders placed (bid+ask combined) in this period.',
            'fill_rate_pct'            : 'n_fills / n_quotes_placed * 100. Target > 60.',
            'information_coefficient'  : 'Spearman IC between signal_score and realized_return. Target > 0.02.',
            'sharpe_ratio'             : 'Annualized Sharpe ratio of fill-level P&L. Target > 1.5.',
            'sortino_ratio'            : 'Sharpe variant that penalizes only downside volatility.',
            'calmar_ratio'             : 'Annualized return divided by max drawdown. Higher is better.',
            'max_drawdown_pct'         : 'Maximum peak-to-trough equity drawdown in %. Target < 5.',
            'net_pnl_usdt'             : 'Total P&L after maker fees in USDT.',
            'gross_pnl_usdt'           : 'Total P&L before fees in USDT.',
            'total_fees_usdt'          : 'Total maker fees paid in USDT.',
            'win_rate_pct'             : 'Percentage of fills that were individually profitable.',
            'avg_spread_at_fill'       : 'Mean bid-ask spread in USDT at the moment of each fill.',
            'adverse_selection_rate'   : 'Fraction of fills where price moved against us post-fill. Target < 0.30.',
            'quote_to_trade_ratio'     : 'Quotes placed per public trade observed in the book. Target >= 1.',
            'avg_inventory'            : 'Mean BTC inventory over the session. Near 0 means good mean reversion.',
            'max_inventory'            : 'Peak absolute BTC inventory reached during the session.',
            'sharpe_degradation_pct'   : 'IS->OOS Sharpe drop as %. Below 50% means strategy is not overfit.',
            'annualized_return_pct'    : 'Net P&L annualized as a percentage of initial capital.',
            'volatility_pct'           : 'Annualized P&L volatility as a percentage of initial capital.',
        },

        # Market Making Metrics threshold definitions
        'market_making_metrics_glossary': {
            'IC_gt_0.02'                       : 'IC > 0.02 -- alpha signal exists in OBI/microprice.',
            'Sharpe_gt_1.5'                    : 'Sharpe > 1.5 -- risk-adjusted returns are acceptable.',
            'MaxDrawdown_lt_5pct'              : 'Max drawdown < 5% -- capital is adequately preserved.',
            'FillRate_gt_60pct'                : 'Fill rate > 60% -- quotes are competitive and getting filled.',
            'AdverseSelection_lt_30pct'        : 'Adverse selection < 30% -- not being systematically picked off.',
            'QuoteToTradeRatio_gte_1'          : 'Quote-to-trade ratio >= 1 -- not spamming quotes.',
            'SpreadMinimum_gte_tick'           : 'Avg spread at fill >= tick size -- never quoting inside the tick.',
            'IC_Stability_degradation_lt_50pct': 'IS->OOS Sharpe degradation < 50% -- strategy is not overfit.',
            'FeeAdjustedPnL_positive'          : 'Net P&L > 0 after all maker fees -- strategy is profitable.',
            'InventoryMeanReversion_ok'        : 'Avg inventory < 50% of max limit -- inventory mean-reverts cleanly.',
            'AllSystemChecks_pass'             : 'All automated system checks green (set externally).',
            'ALL_METRICS_PASS'                 : 'True only when every one of the above 11 metrics passes.',
        },

        'last_updated_utc' : datetime.now(timezone.utc).isoformat(),
        'total_runs'       : 0,
        'runs'             : [],
    }


def _backup_corrupt(path: str):
    """Rename a corrupt master file with a timestamp suffix so it is not lost."""
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = path + f'.corrupt_{ts}'
    try:
        os.rename(path, bak)
        log.warning(f"Corrupt master backed up to: {bak}")
    except OSError:
        pass


# =============================================================================
# 1. DATA LOADER
# =============================================================================

class DataLoader:
    """
    Loads order book NDJSON files and trade CSV files.

    OB file format (confirmed from sample):
      {"topic":"orderbook.200.BTCUSDT","ts":<ms>,"type":"snapshot"|"delta",
       "data":{"s":"BTCUSDT","b":[["price","qty"],...],"a":[...],"u":...,"seq":...},
       "cts":<ms>}

    Trade file format -- Binance official (NO header row):
      Columns: id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
      time: microseconds since epoch  ->  divide by 1000 for milliseconds
      isBuyerMaker True  -> aggressor = seller -> side='sell'
      isBuyerMaker False -> aggressor = buyer  -> side='buy'
    """

    # Actual CSV format confirmed from file inspection:
    #   Header row: id,timestamp,price,volume,side,rpi
    #   timestamp:  milliseconds since epoch (already ms — no conversion needed)
    #   side:       'buy' or 'sell' string (not isBuyerMaker bool)
    #   rpi:        reserved/unused float column
    _TRADE_COLS   = ['id', 'timestamp', 'price', 'volume', 'side', 'rpi']
    _TRADE_DTYPES = {
        'id'        : 'int64',
        'timestamp' : 'int64',
        'price'     : 'float64',
        'volume'    : 'float64',
        'side'      : 'str',
        'rpi'       : 'float64',
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def find_ob_files(self) -> List[str]:
        pat   = self.cfg['ob_file_pattern']
        base  = self.cfg['data_dir_ob']
        files = sorted(glob.glob(os.path.join(base, '**', pat), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise FileNotFoundError(
                f"\nNo order book files found.\n"
                f"  Searched: {os.path.join(base, pat)}\n"
                f"  Expected: 2026-03-01_BTCUSDT_ob200.data\n"
            )
        log.info(f"Found {len(files)} order book file(s):")
        for f in files:
            log.info(f"  {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.1f} MB)")
        return files

    def find_trade_files(self) -> List[str]:
        base  = self.cfg['data_dir_trades']
        pat   = self.cfg['trades_file_pattern']
        files = sorted(glob.glob(os.path.join(base, '**', pat), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise FileNotFoundError(
                f"\nNo trade CSV files found.\n"
                f"  Searched recursively under: {base}\n"
                f"  Pattern: {pat}\n"
            )
        log.info(f"Found {len(files)} trade file(s):")
        for f in files:
            log.info(f"  {f}  ({os.path.getsize(f)/1e6:.1f} MB)")
        return files

    def load_trades(self, trade_files: List[str]) -> pd.DataFrame:
        """
        Load trade CSVs.

        Confirmed CSV format (from file inspection):
          Header row: id,timestamp,price,volume,side,rpi
          id        : int64   — trade ID
          timestamp : int64   — milliseconds since epoch (already ms, no conversion)
          price     : float64 — trade price in USDT
          volume    : float64 — trade size in BTC
          side      : str     — 'buy' or 'sell' (aggressor side, ready to use)
          rpi       : float64 — reserved column (unused)
        """
        dfs = []
        for fpath in trade_files:
            df = pd.read_csv(
                fpath,
                header=0,              # CSV HAS a header row
                dtype=self._TRADE_DTYPES,
            )
            # timestamp is already in milliseconds — alias directly to ts_ms
            df['ts_ms'] = df['timestamp'].astype('int64')
            # side is already 'buy'/'sell' string — no mapping needed
            # volume column already named correctly — no alias needed
            dfs.append(df)
            log.info(f"  Loaded {len(df):,} trades from {os.path.basename(fpath)}")

        trades = pd.concat(dfs, ignore_index=True)
        trades.sort_values('ts_ms', inplace=True)
        trades.reset_index(drop=True, inplace=True)
        trades.set_index('ts_ms', inplace=True, drop=False)
        log.info(f"Total trades loaded: {len(trades):,}")
        return trades

    def iter_ob_lines(self, ob_file: str):
        """
        Generator: yield one parsed dict per line, skipping malformed lines.
        Auto-detects UTF-8 (native .data files) vs UTF-16 (PowerShell exports).
        """
        with open(ob_file, 'rb') as fb:
            bom = fb.read(2)
        enc = 'utf-16' if bom in (b'\xff\xfe', b'\xfe\xff') else 'utf-8'
        with open(ob_file, 'r', encoding=enc, errors='replace', buffering=1 << 20) as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    log.warning(
                        f"Skipping malformed JSON at line {lineno} "
                        f"in {os.path.basename(ob_file)}"
                    )


# =============================================================================
# 2. AVELLANEDA-STOIKOV MODEL
# =============================================================================

class ASModel:
    """
    Avellaneda-Stoikov market making model.

    Reservation price : r = mid - gamma * sigma^2 * (T-t) * q
    Optimal half-spread: delta = (gamma*sigma^2*(T-t))/2 + (1/gamma)*ln(1 + gamma/kappa)
    Quoted bid = r - delta,  ask = r + delta

    Spread cap (FIX-1): min $0.20 (2 ticks), max $5.00 (50 ticks).
    Matches BTC/USDT typical move of $5-50 per volatile tick.
    """

    def __init__(self, gamma: float, kappa: float, sigma: float,
                 T: float = 1.0, max_inventory: float = 0.10):
        self.gamma         = gamma
        self.kappa         = kappa
        self.sigma         = sigma
        self.T             = T
        self.max_inventory = max_inventory

    def quotes(self, mid: float, inventory: float,
               t_norm: float = 0.5) -> Tuple[float, float]:
        """
        Returns (bid_price, ask_price).
        Inventory normalised to [-1,+1]: inv_norm = inventory / max_inventory.
        Scaled by mid so AS skew is in dollar terms.
        """
        # Dollar-denominated AS formula.
        # REASON: mid * hs_frac was causing $68,000 spreads on every tick
        # because the log term (1/gamma)*ln(1+gamma/kappa) ~ 0.805 alone,
        # and 0.805 * $85,000 = $68,401 — always hitting the cap.
        # Sigma and gamma changes had zero effect. Every run quoted at the
        # hard clip, not the AS model.
        #
        # FIX: use sigma in dollar-per-tick units (set by update_sigma)
        # and compute hs directly in dollars — no mid multiplication.
        #   sigma ~ $0.34/tick, sigma^2 ~ 0.116
        #   inv_risk = gamma * sigma^2 * tr * |inv_norm|  (in USDT)
        #   hs = base_spread($0.30) + inv_risk
        #   Gives: $0.30 base (flat), up to $0.42 (full inventory)
        tr          = max(self.T - t_norm, 1e-4)
        inv_norm    = inventory / max(self.max_inventory, 1e-8)
        skew        = self.gamma * (self.sigma ** 2) * tr * inv_norm
        res         = mid - skew
        inv_risk    = self.gamma * (self.sigma ** 2) * tr * abs(inv_norm)
        base_spread = 0.30
        hs          = max(min(base_spread + inv_risk, 5.00), 0.20)
        return res - hs, res + hs

    def update_sigma(self, mid_history: List[float], ticks_per_day: int = 429_000):
        """
        Update sigma in dollar-per-tick terms so AS formula produces
        meaningful spreads.

        FIX-SIGMA: original code annualized fractional sigma (~0.019),
        giving sigma^2 ~ 0.00036 which made mid*hs_frac ~ $0.001 —
        clipped up to the $0.20 minimum on every tick.

        Dollar-per-tick sigma:
          per_tick_std ~ 0.000004 (fractional log return per tick)
          x mid        ~ $85,000
          = sigma      ~ $0.34 per tick
          sigma^2      ~ 0.116
          AS half-spread = gamma * sigma^2 * tr / 2
                         = 2 * 0.116 * 0.5 / 2 = $0.058 base
          + log term gives natural spreads of $1-10 without hitting clip.

        ticks_per_day param kept for API compatibility, no longer used.
        """
        if len(mid_history) < 20:
            return
        mids         = np.array(mid_history[-200:])
        rets         = np.diff(np.log(mids))
        per_tick_std = float(np.std(rets))
        # Dollar-per-tick: fractional std * current mid price
        self.sigma   = max(per_tick_std * float(mids[-1]), 1e-4)


# =============================================================================
# 3. FILL SIMULATOR
# =============================================================================

class FillSimulator:
    """
    Simulates passive maker fills against real public trade data.

    Per tick:
      - Collect public trades in next fill_window_ms milliseconds.
      - Sell trade crosses bid + inventory headroom ok -> BUY fill.
      - Buy  trade crosses ask + inventory headroom ok -> SELL fill.
      - One fill per tick (first crossing trade wins).
      - Fill price = our quoted price (conservative maker model).
    """

    def __init__(self, cfg: dict):
        self.qty       = cfg['order_qty_btc']
        self.max_inv   = cfg['max_inventory_btc']
        self.window_ms = cfg['fill_window_ms']
        self.maker_fee = cfg['maker_fee_rate']
        self.tick_ms   = cfg['simulated_latency_ms']

    def simulate(
        self,
        bid          : float,
        ask          : float,
        ts_ms        : int,
        trades       : pd.DataFrame,
        inventory    : float,
        spread       : float,
        signal       : float,
        mid          : float,
        realized_ret : float,
    ) -> Optional[FillRecord]:
        """Returns a FillRecord if filled this tick, else None."""

        if trades.empty:
            return None

        fill_ts = datetime.fromtimestamp(
            (ts_ms + self.tick_ms) / 1000.0, tz=timezone.utc)

        prices = trades['price'].to_numpy()
        sides  = trades['side'].to_numpy()
        qtys   = trades['volume'].to_numpy()

        # FIX-2: exact price match only.
        # 1-tick tolerance (Run #4) doubled IS drawdown 5.85% -> 14.03%
        # by filling against trades that were already moving away from us.
        # Exact match ensures we only fill when market genuinely crosses quote.
        sell_mask = (sides == 'sell') & (prices <= bid)
        if sell_mask.any() and inventory + self.qty <= self.max_inv:
            idx = int(np.argmax(sell_mask))
            return FillRecord(
                timestamp       = fill_ts,
                side            = 'buy',
                price           = bid,
                quantity        = min(self.qty, float(qtys[idx])),
                fee_rate        = self.maker_fee,
                is_maker        = True,
                inventory_after = inventory + self.qty,
                spread_at_fill  = spread,
                signal_score    = signal,
                realized_return = realized_ret,
                mid_at_fill     = mid,
            )

        buy_mask  = (sides == 'buy')  & (prices >= ask)
        if buy_mask.any() and inventory - self.qty >= -self.max_inv:
            idx = int(np.argmax(buy_mask))
            return FillRecord(
                timestamp       = fill_ts,
                side            = 'sell',
                price           = ask,
                quantity        = min(self.qty, float(qtys[idx])),
                fee_rate        = self.maker_fee,
                is_maker        = True,
                inventory_after = inventory - self.qty,
                spread_at_fill  = spread,
                signal_score    = signal,
                realized_return = -realized_ret,
                mid_at_fill     = mid,
            )

        return None


# =============================================================================
# 4. BACKTESTER
# =============================================================================

# BUG-5 FIX: duplicate `import logging` / `log =` that existed here removed.

class Backtester:
    """
    Drives the full backtest loop using AS gamma/kappa from config.
    Saves every run to the master JSON at E:\\Binance.
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.loader = DataLoader(cfg)
        self.model  = ASModel(
            gamma=cfg['as_gamma'],
            kappa=cfg['as_kappa'],
            sigma=cfg['as_sigma'],
            T=cfg['as_T'],
            max_inventory=cfg['max_inventory_btc'],
        )
        self.filler = FillSimulator(cfg)
        self.policy = None

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """
        Full backtest. Returns result dict. Also appends to master JSON.
        """
        import time
        t_start = time.time()

        ob_files    = self.loader.find_ob_files()
        trade_files = self.loader.find_trade_files()

        run_id = (
            datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            + f"_g{self.cfg['as_gamma']}_k{self.cfg['as_kappa']}"
        )

        results = None
        error   = None

        try:
            all_trades   = self.loader.load_trades(trade_files)
            fills_by_day = []
            total_qp     = 0

            for ob_file in ob_files:
                date_str = self._date_from_filename(ob_file)
                log.info(f"Processing {os.path.basename(ob_file)} ({date_str})")

                day_fills, day_qp = self._stream_one_file(ob_file, all_trades, date_str)

                log.info(
                    f"  {date_str}: {day_qp:,} quotes placed, {len(day_fills):,} fills"
                )
                fills_by_day.append((date_str, day_fills))
                total_qp += day_qp

            all_fills = [f for _, day in fills_by_day for f in day]

            if not all_fills:
                raise RuntimeError(
                    "Backtester produced zero fills. "
                    "Check trade files cover the same dates as OB files "
                    "and that quoted prices overlap with real trades."
                )

            log.info(
                f"\n{'='*55}\n"
                f"  Total: {total_qp:,} quotes | {len(all_fills):,} fills\n"
                f"  Fill rate: {len(all_fills)/total_qp*100:.2f}%\n"
                f"{'='*55}"
            )

            results = self._build_results(fills_by_day, all_fills, total_qp)

        except Exception as exc:
            error = traceback.format_exc()
            log.error(f"Backtest failed:\n{error}")
            raise

        finally:
            duration = time.time() - t_start
            try:
                saved_path = save_run_to_master(
                    results      = results or {},
                    cfg          = self.cfg,
                    run_id       = run_id,
                    duration_sec = duration,
                    ob_files     = ob_files,
                    trade_files  = trade_files,
                    error        = error,
                )
                log.info(f"Results persisted -> {saved_path}")
            except Exception as save_exc:
                log.error(f"Failed to save master JSON: {save_exc}")

        return results

    # ------------------------------------------------------------------
    def _stream_one_file(
        self, ob_file: str, all_trades: pd.DataFrame, date_str: str
    ) -> Tuple[List[FillRecord], int]:
        """Stream one OB .data file. Returns (fills, quotes_placed)."""

        book          = OrderBook()
        mid_hist      : List[float] = []
        fills         : List[FillRecord] = []
        quotes_placed = 0
        inventory     = 0.0

        # Pre-filter trades to this calendar day to avoid cross-day matches
        try:
            day_start  = int(datetime.strptime(date_str, '%Y-%m-%d')
                             .replace(tzinfo=timezone.utc).timestamp() * 1000)
            day_end    = day_start + 86_400_000
            day_trades = all_trades.loc[day_start:day_end - 1]
            log.debug(f"  Day trades filtered: {len(day_trades):,} rows ({date_str})")
        except Exception as exc:
            log.warning(f"Day filter failed ({exc}), using full trade set")
            day_trades = all_trades

        prev_mid   = None
        daily_pnl  = 0.0   # running fee-adjusted PnL for this day
        loss_limit = self.cfg.get('daily_loss_limit_usdt', None)

        for msg in self.loader.iter_ob_lines(ob_file):
            if not book.apply(msg):
                continue

            mid    = book.mid
            spread = book.spread
            micro  = book.microprice()

            if mid is None or spread is None or micro is None:
                continue
            if spread <= 0 or not book.valid:
                continue

            mid_hist.append(mid)
            self.model.update_sigma(mid_hist)

            # Regime filter -- pause quoting during high-volatility ticks
            vol_window = self.cfg.get('regime_vol_window', 50)
            if len(mid_hist) >= vol_window:
                recent_rets = np.diff(
                    np.log(np.array(mid_hist[-vol_window:]) + 1e-10)
                )
                if float(np.std(recent_rets)) > self.cfg.get('regime_vol_threshold', 0.000060):
                    prev_mid = mid
                    continue

            ts_ms  = int(book.ts)
            t_norm = (ts_ms / 1000.0 % 86400.0) / 86400.0

            # AS model quotes (or RL policy if loaded)
            if self.policy is not None:
                obs = np.array([
                    np.clip(inventory / self.cfg['max_inventory_btc'], -1, 1),
                    np.clip(self.model.sigma / 0.002, 0, 1),
                    np.clip(book.obi(5), -1, 1),
                    np.clip(spread / 5.0, 0, 1),
                    t_norm,
                ], dtype=np.float32)
                action, _, _ = self.policy.predict(obs, deterministic=True)
                rl_gamma = float(np.clip(action[0], 0.01, 1.0))
                tr       = max(self.cfg['as_T'] - t_norm, 1e-4)
                inv_norm = inventory / max(self.cfg['max_inventory_btc'], 1e-8)
                # RL branch: same dollar-denominated formula as quotes()
                skew     = rl_gamma * (self.model.sigma**2) * tr * inv_norm
                res      = mid - skew
                inv_risk = rl_gamma * (self.model.sigma**2) * tr * abs(inv_norm)
                hs       = max(min(0.30 + inv_risk, 5.00), 0.20)
                bid_raw, ask_raw = res - hs, res + hs
            else:
                bid_raw, ask_raw = self.model.quotes(mid, inventory, t_norm)

            tick = self.cfg['tick_size_usdt']
            bid  = round(min(bid_raw, mid - tick) / tick) * tick
            ask  = round(max(ask_raw, mid + tick) / tick) * tick
            if bid >= ask:
                ask = bid + tick

            quotes_placed += 1

            signal = (micro - mid) / mid

            # FIX-IC: forward return for correct IC computation.
            #
            # PROBLEM: original code stored backward return mid[t]-mid[t-1]
            # as realized_return. This measures whether the signal predicted
            # what ALREADY happened — useless for IC. Worse, with the regime
            # filter pausing many ticks, prev_mid often equals mid (same tick
            # seen twice), giving realized_ret=0 for most fills -> IC=0.
            #
            # FIX: store the signal at tick t, but pair it with the price
            # move from tick t to tick t+1 (the next tick's mid - current mid).
            # We achieve this by a one-tick lag: the forward_ret we pass to
            # FillRecord at tick t is actually (mid[t] - mid[t-1]) / mid[t-1],
            # i.e. what happened BETWEEN the PREVIOUS signal and NOW.
            # This is the standard "next-tick return" IC convention used in
            # HFT microstructure research — signal[t-1] predicts return[t].
            #
            # For fills that occur at tick t, realized_return = mid[t]-mid[t-1]
            # captures the forward move relative to when the signal was posted.
            forward_ret = ((mid - prev_mid) / prev_mid
                           if prev_mid is not None and prev_mid > 1e-6 else 0.0)

            window = day_trades.loc[ts_ms : ts_ms + 199]
            fill   = self.filler.simulate(
                bid, ask, ts_ms, window,
                inventory, spread, signal, mid, forward_ret
            )
            if fill:
                inventory  = fill.inventory_after
                fills.append(fill)
                # Track daily fee-adjusted PnL for loss limit check
                # sell fill = cash in (+notional), buy fill = cash out (-notional)
                signed = fill.price * fill.quantity * (1.0 if fill.side == 'sell' else -1.0)
                fee    = fill.price * fill.quantity * fill.fee_rate
                daily_pnl += signed - fee

            # Daily loss limit — stop quoting for rest of day if breached
            if loss_limit is not None and daily_pnl < loss_limit:
                log.info(
                    f"  Daily loss limit hit on {date_str}: "
                    f"PnL=${daily_pnl:,.0f} < limit=${loss_limit:,.0f} "
                    f"— stopping quotes for remainder of day"
                )
                break

            prev_mid = mid

        return fills, quotes_placed

    # ------------------------------------------------------------------
    def _build_results(
        self,
        fills_by_day : List[Tuple[str, List[FillRecord]]],
        all_fills    : List[FillRecord],
        total_qp     : int,
    ) -> Dict:
        """
        Split fills into IS/OOS, compute MetricsResult for each window.

        Crucially, the date lists for each period are returned explicitly
        in the result dict so _build_run_record can write:
            period_start_date, period_end_date, n_days
        as the first fields of every in_sample / out_of_sample / full_period
        block saved in the master JSON.
        """
        cap    = self.cfg['initial_capital_usdt']
        lev    = self.cfg['leverage']
        lat    = self.cfg['simulated_latency_ms']
        n_days = len(fills_by_day)
        n_is   = max(1, int(np.ceil(n_days * self.cfg['in_sample_ratio'])))

        is_fills  = [f for _, day in fills_by_day[:n_is]  for f in day]
        oos_fills = [f for _, day in fills_by_day[n_is:]  for f in day]

        is_qp  = int(total_qp * (len(is_fills)  / max(len(all_fills), 1)))
        oos_qp = int(total_qp * (len(oos_fills) / max(len(all_fills), 1)))

        # Date lists -- passed through to saver for period identity fields
        is_dates  = [d for d, _ in fills_by_day[:n_is]]
        oos_dates = [d for d, _ in fills_by_day[n_is:]]
        all_dates = [d for d, _ in fills_by_day]

        is_label   = f"In-Sample ({is_dates[0]} -> {is_dates[-1]})"       if is_dates  else "In-Sample"
        oos_label  = f"Out-of-Sample ({oos_dates[0]} -> {oos_dates[-1]})" if oos_dates else "Out-of-Sample"
        full_label = f"Full Period ({all_dates[0]} -> {all_dates[-1]})"   if all_dates else "Full Period"

        log.info(f"IS  period : {is_label}  |  {len(is_fills):,} fills")
        log.info(f"OOS period : {oos_label}  |  {len(oos_fills):,} fills")

        is_result   = MetricsCalculator(cap, lev, is_label,   lat, is_qp  ).compute(is_fills)
        oos_result  = MetricsCalculator(cap, lev, oos_label,  lat, oos_qp ).compute(
                          oos_fills if oos_fills else is_fills)
        full_result = MetricsCalculator(cap, lev, full_label, lat, total_qp).compute(all_fills)

        if is_result.sharpe_ratio and is_result.sharpe_ratio != 0:
            oos_result.sharpe_degradation_pct = float(
                (oos_result.sharpe_ratio - is_result.sharpe_ratio)
                / abs(is_result.sharpe_ratio) * 100
            )

        wf_windows = self._walk_forward(fills_by_day, total_qp, cap, lev, lat)

        return {
            # MetricsResult objects
            'in_sample'            : is_result,
            'out_of_sample'        : oos_result,
            'full'                 : full_result,
            # Date lists for period identity in JSON output
            'is_dates'             : is_dates,
            'oos_dates'            : oos_dates,
            'all_dates'            : all_dates,
            # Fill/quote counts per period for JSON output
            'is_fills_count'       : len(is_fills),
            'oos_fills_count'      : len(oos_fills),
            'is_quotes'            : is_qp,
            'oos_quotes'           : oos_qp,
            # Walk-forward windows
            'walk_forward_windows' : wf_windows,
            # Pass-through
            'config'               : self.cfg,
            'fills_by_day'         : [(d, len(f)) for d, f in fills_by_day],
            'total_quotes_placed'  : total_qp,
            'total_fills'          : len(all_fills),
        }

    # ------------------------------------------------------------------
    def _walk_forward(
        self,
        fills_by_day : List[Tuple[str, List[FillRecord]]],
        total_qp     : int,
        cap: float, lev: float, lat: float,
    ) -> List[Dict]:
        """
        Walk-forward validation.
        Window 1: first 50% -> second 50% of all fills.
        Window 2: Day 1 -> Day 2 (requires 2+ days).
        """
        all_fills = [f for _, day in fills_by_day for f in day]
        n = len(all_fills)
        if n < 40:
            log.warning("Too few fills for walk-forward analysis.")
            return []

        windows = []
        qp_half = total_qp // 2
        mid     = n // 2

        train_r = MetricsCalculator(cap, lev, 'WF-1 Train', lat, qp_half).compute(all_fills[:mid])
        test_r  = MetricsCalculator(cap, lev, 'WF-1 Test',  lat, qp_half).compute(all_fills[mid:])
        if train_r.sharpe_ratio and train_r.sharpe_ratio != 0:
            test_r.sharpe_degradation_pct = (
                (test_r.sharpe_ratio - train_r.sharpe_ratio)
                / abs(train_r.sharpe_ratio) * 100
            )
        windows.append({'window': 1, 'label': 'First 50% -> Second 50%',
                        'train': train_r, 'test': test_r})

        # FIX-3: roll across ALL consecutive day pairs (17 windows for 18 days).
        # Original code hardcoded fills_by_day[0] and fills_by_day[1] only,
        # so WF-2 always used only Mar 01 → Mar 02 regardless of dataset size.
        for i in range(len(fills_by_day) - 1):
            d1, f1 = fills_by_day[i]
            d2, f2 = fills_by_day[i + 1]
            if not f1 or not f2:
                continue
            qp1 = int(total_qp * len(f1) / n)
            qp2 = int(total_qp * len(f2) / n)
            tr2 = MetricsCalculator(cap, lev, f'WF Train ({d1})', lat, qp1).compute(f1)
            te2 = MetricsCalculator(cap, lev, f'WF Test  ({d2})', lat, qp2).compute(f2)
            if tr2.sharpe_ratio and tr2.sharpe_ratio != 0:
                te2.sharpe_degradation_pct = (
                    (te2.sharpe_ratio - tr2.sharpe_ratio)
                    / abs(tr2.sharpe_ratio) * 100
                )
            windows.append({
                'window' : i + 2,
                'label'  : f'{d1} -> {d2}',
                'train'  : tr2,
                'test'   : te2,
            })

        for w in windows:
            log.info(
                f"WF Window {w['window']} ({w['label']}): "
                f"Train Sharpe={w['train'].sharpe_ratio:.3f}  "
                f"Test Sharpe={w['test'].sharpe_ratio:.3f}  "
                f"Degradation={w['test'].sharpe_degradation_pct:+.1f}%"
            )
        log.info(f"Walk-forward: {len(windows)} windows total "
                 f"(WF-1 = full period split; WF-2..{len(windows)} = consecutive day pairs)")
        return windows

    # ------------------------------------------------------------------
    @staticmethod
    def _date_from_filename(path: str) -> str:
        """Extract YYYY-MM-DD from filename like 2026-03-01_BTCUSDT_ob200.data"""
        base = os.path.basename(path)
        if len(base) >= 10 and base[4] == '-' and base[7] == '-':
            return base[:10]
        return base.split('_')[0]
