# =============================================================================
# SITARAM HFT — Backtester
#
# Pipeline:
#   1. DataLoader   — reads OB .data files (NDJSON) and trade CSV files
#   2. OrderBook    — reconstructs L2 book tick-by-tick
#   3. ASModel      — Avellaneda-Stoikov quotes: bid, ask, reservation price
#   4. FillSimulator— checks real public trades to determine fills
#   5. Backtester   — drives the loop, splits IS/OOS, runs walk-forward
#
# No synthetic data. If files are not found, the run raises FileNotFoundError.
# =============================================================================

import os
import json
import glob
import logging
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from src.orderbook import OrderBook
from src.metrics import FillRecord, MetricsResult, MetricsCalculator

log = logging.getLogger('sitaram.backtester')


# =============================================================================
# 1. DATA LOADER
# =============================================================================

class DataLoader:
    """
    Loads order book NDJSON files and trade CSV files.

    OB file format  (confirmed from sample):
      One JSON object per line.
      {"topic":"orderbook.200.BTCUSDT","ts":<ms>,"type":"snapshot"|"delta",
       "data":{"s":"BTCUSDT","b":[["price","qty"],...],"a":[...],"u":...,"seq":...},
       "cts":<ms>}

    Trade file format (confirmed from sample):
      CSV with header: id,timestamp,price,volume,side,rpi
      timestamp: milliseconds in scientific notation (e.g. 1.77232E+12)
      side: lowercase 'buy' or 'sell'
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def find_ob_files(self) -> List[str]:
        path = os.path.join(self.cfg['data_dir_ob'], self.cfg['ob_file_pattern'])
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(
                f"\n❌ No order book files found.\n"
                f"   Searched: {path}\n"
                f"   Expected files like: 2026-03-01_BTCUSDT_ob200.data\n"
            )
        log.info(f"Found {len(files)} order book file(s):")
        for f in files:
            log.info(f"  {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.1f} MB)")
        return files

    def find_trade_files(self) -> List[str]:
        base = self.cfg['data_dir_trades']
        pat  = self.cfg['trades_file_pattern']

        # Search recursively: handles both flat files AND nested folders
        # E.g. E:\Binance\March\BTCUSDT_2026-03-01.csv\BTCUSDT_2026-03-01.csv
        all_matches = sorted(glob.glob(os.path.join(base, '**', pat), recursive=True))

        # Filter to actual FILES only — exclude folders that happen to end in .csv
        files = [f for f in all_matches if os.path.isfile(f)]

        if not files:
            raise FileNotFoundError(
                f"\n❌ No trade CSV files found.\n"
                f"   Searched recursively under: {base}\n"
                f"   Pattern: {pat}\n"
                f"   Expected structure:\n"
                f"     {base}\\BTCUSDT_2026-03-01.csv\\BTCUSDT_2026-03-01.csv\n"
            )
        log.info(f"Found {len(files)} trade file(s):")
        for f in files:
            log.info(f"  {f}  ({os.path.getsize(f)/1e6:.1f} MB)")
        return files

    def load_trades(self, trade_files: List[str]) -> pd.DataFrame:
        """
        Load all trade CSVs into one sorted DataFrame.
        timestamp column: scientific notation ms → int64
        """
        dfs = []
        for fpath in trade_files:
            df = pd.read_csv(
                fpath,
                dtype={'id': 'int64', 'price': 'float64',
                       'volume': 'float64', 'side': 'str', 'rpi': 'int8'},
                float_precision='high',
            )
            # timestamp may come as float in scientific notation
            df['ts_ms'] = df['timestamp'].astype(float).astype('int64')
            dfs.append(df)
            log.info(f"  Loaded {len(df):,} trades from {os.path.basename(fpath)}")

        trades = pd.concat(dfs, ignore_index=True)
        trades.sort_values('ts_ms', inplace=True)
        trades.reset_index(drop=True, inplace=True)
        log.info(f"Total trades loaded: {len(trades):,}")
        return trades

    def iter_ob_lines(self, ob_file: str):
        """
        Generator: yield one parsed dict per line, skipping malformed lines.
        Auto-detects encoding: UTF-8 for native .data files,
        UTF-16 for files exported via PowerShell.
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

    Reservation price:
        r = mid - γ · σ² · (T - t) · q

    Optimal half-spread:
        δ = (γ · σ² · (T - t)) / 2 + (1/γ) · ln(1 + γ/κ)

    Quoted bid  = r - δ
    Quoted ask  = r + δ

    γ (gamma)  = risk aversion
    κ (kappa)  = order arrival rate
    σ (sigma)  = price volatility (updated dynamically)
    T          = session horizon (normalized to 1.0)
    t          = current normalized time within session
    q          = current inventory in BTC
    """

    def __init__(self, gamma: float, kappa: float, sigma: float, T: float = 1.0):
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.T     = T

    def quotes(self, mid: float, inventory: float,
               t_norm: float = 0.5) -> Tuple[float, float]:
        """
        Returns (bid_price, ask_price).
        t_norm: normalized time within session [0, 1].
        """
        tr  = max(self.T - t_norm, 1e-4)
        res = mid - self.gamma * (self.sigma ** 2) * tr * inventory
        hs  = (self.gamma * self.sigma ** 2 * tr) / 2.0 + \
              (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return res - hs, res + hs

    def update_sigma(self, mid_history: List[float]):
        """Dynamically update σ from realized mid-price log returns."""
        if len(mid_history) < 20:
            return
        mids = np.array(mid_history[-200:])
        rets = np.diff(np.log(mids))
        sigma = float(np.std(rets))
        self.sigma = max(sigma, 1e-8)


# =============================================================================
# 3. FILL SIMULATOR
# =============================================================================

class FillSimulator:
    """
    Simulates order fills by checking real public trade data.

    Logic per tick:
      - Look at all public trades in the next `fill_window_ms` milliseconds.
      - If a Sell trade crosses our bid AND inventory limit allows → BUY fill.
      - If a Buy  trade crosses our ask AND inventory limit allows → SELL fill.
      - Only one fill per tick (first crossing trade wins).

    This is a conservative passive fill model:
      - We only fill as makers (limit orders).
      - We never cross the spread to fill.
      - Fill price = our quoted price, not the trade price.
    """

    def __init__(self, cfg: dict):
        self.qty        = cfg['order_qty_btc']
        self.max_inv    = cfg['max_inventory_btc']
        self.window_ms  = cfg['fill_window_ms']
        self.maker_fee  = cfg['maker_fee_rate']
        self.tick_ms    = cfg['simulated_latency_ms']

    def simulate(
        self,
        bid:        float,
        ask:        float,
        ts_ms:      int,
        trades:     pd.DataFrame,
        inventory:  float,
        spread:     float,
        signal:     float,
        mid:        float,
        mid_next:   float,
    ) -> Optional[FillRecord]:
        """
        Returns one FillRecord if a fill occurs, else None.

        trades: pre-filtered to the relevant time window externally
                for performance (done in Backtester.stream_fills).
        """
        # Realized return for IC: how did mid move in the next tick?
        realized_ret = (mid_next - mid) / mid if mid > 1e-6 else 0.0

        fill_ts = datetime.fromtimestamp(
            (ts_ms + self.tick_ms) / 1000.0, tz=timezone.utc)

        for _, tr in trades.iterrows():
            t_side  = str(tr['side']).lower()
            t_price = float(tr['price'])
            t_qty   = float(tr['volume'])

            # Passive maker fill logic:
            # Sell aggressor within 0.05% of our bid fills our resting bid.
            # Buy  aggressor within 0.05% of our ask fills our resting ask.
            # Standard exchange behaviour: aggressive order sweeps our limit.
            if t_side == 'sell' and t_price <= bid * 1.0005:
                if inventory + self.qty <= self.max_inv:
                    return FillRecord(
                        timestamp       = fill_ts,
                        side            = 'buy',
                        price           = bid,
                        quantity        = min(self.qty, t_qty),
                        fee_rate        = self.maker_fee,
                        is_maker        = True,
                        inventory_after = inventory + self.qty,
                        spread_at_fill  = spread,
                        signal_score    = signal,
                        realized_return = realized_ret,
                        mid_at_fill     = mid,
                    )

            elif t_side == 'buy' and t_price >= ask * 0.9995:
                if inventory - self.qty >= -self.max_inv:
                    return FillRecord(
                        timestamp       = fill_ts,
                        side            = 'sell',
                        price           = ask,
                        quantity        = min(self.qty, t_qty),
                        fee_rate        = self.maker_fee,
                        is_maker        = True,
                        inventory_after = inventory - self.qty,
                        spread_at_fill  = spread,
                        signal_score    = signal,
                        realized_return = -realized_ret,
                        mid_at_fill     = mid,
                    )

        return None   # No fill this tick


# =============================================================================
# 4. BACKTESTER
# =============================================================================

class RLBacktester:
    """
    Drives the full backtest loop.

    For each OB file (one per day):
      - Reconstruct L2 book tick by tick
      - Compute AS model quotes
      - Simulate fills against real public trades
      - Collect FillRecord objects

    Then:
      - Split fills into IS and OOS by date (file boundary = natural split)
      - Compute all 23 metrics for IS, OOS, and full period
      - Run walk-forward windows
    """

    def __init__(self, cfg: dict, policy_path: str = None):
        self.cfg    = cfg
        self.loader = DataLoader(cfg)
        self.filler = FillSimulator(cfg)
        self.policy = None
        self.gamma_kappa_log = []

        # Load RL policy if provided
        if policy_path and os.path.exists(policy_path):
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.rl_agent import PPOPolicy
            self.policy = PPOPolicy(obs_dim=5, act_dim=2)
            self.policy.load(policy_path)
            log.info(f'RL policy loaded from {policy_path}')
        else:
            log.warning('No policy path — using fixed AS gamma/kappa from config')

        # Fallback fixed model
        self.model = ASModel(
            gamma=cfg['as_gamma'],
            kappa=cfg['as_kappa'],
            sigma=cfg['as_sigma'],
            T=cfg['as_T'],
        )

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """
        Full backtest. Returns result dict with keys:
          'in_sample', 'out_of_sample', 'full',
          'walk_forward_windows', 'config',
          'fills_by_day'  (list of (date_str, fills) for debugging)
        """
        ob_files     = self.loader.find_ob_files()
        trade_files  = self.loader.find_trade_files()
        all_trades   = self.loader.load_trades(trade_files)

        fills_by_day = []
        total_qp     = 0

        for ob_file in ob_files:
            date_str = self._date_from_filename(ob_file)
            log.info(f"─── Processing {os.path.basename(ob_file)} ({date_str}) ───")

            day_fills, day_qp = self._stream_one_file(
                ob_file, all_trades, date_str)

            log.info(
                f"  {date_str}: {day_qp:,} quotes placed, "
                f"{len(day_fills):,} fills"
            )
            fills_by_day.append((date_str, day_fills))
            total_qp += day_qp

        # Flatten all fills in chronological order
        all_fills = [f for _, day in fills_by_day for f in day]

        if not all_fills:
            raise RuntimeError(
                "Backtester produced zero fills. "
                "Check that your trade files cover the same dates as your OB files, "
                "and that price levels in your quotes overlap with real trades."
            )

        log.info(
            f"\n{'='*55}\n"
            f"  Total: {total_qp:,} quotes | {len(all_fills):,} fills\n"
            f"  Fill rate: {len(all_fills)/total_qp*100:.2f}%\n"
            f"{'='*55}"
        )

        return self._build_results(fills_by_day, all_fills, total_qp)

    # ------------------------------------------------------------------
    def _stream_one_file(
        self, ob_file: str, all_trades: pd.DataFrame, date_str: str
    ) -> Tuple[List[FillRecord], int]:
        """
        Stream one OB file. Returns (fills, quotes_placed).
        """
        book      = OrderBook()
        mid_hist  : List[float] = []
        fills     : List[FillRecord] = []
        quotes_placed = 0
        inventory     = 0.0

        # Pre-filter trades to this date's range for speed
        # We'll narrow further by timestamp inside the loop
        day_trades = all_trades  # full slice; narrowed per-tick below

        prev_mid = None

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

            ts_ms  = int(book.ts)
            t_norm = (ts_ms / 1000.0 % 86400.0) / 86400.0

            # AS model quotes — use RL policy if available
            if self.policy is not None:
                from src.orderbook import OrderBook as _OB
                sigma_norm = self.model.sigma / 0.002
                obi_val    = book.obi(5)
                spread_norm = spread / 5.0
                obs = np.array([
                    np.clip(inventory / self.cfg['max_inventory_btc'], -1, 1),
                    np.clip(sigma_norm, 0, 1),
                    np.clip(obi_val, -1, 1),
                    np.clip(spread_norm, 0, 1),
                    t_norm,
                ], dtype=np.float32)
                action, _, _ = self.policy.predict(obs, deterministic=True)
                rl_gamma = float(np.clip(action[0], 0.01, 1.0))
                rl_kappa = float(np.clip(action[1], 0.5,  5.0))
                self.gamma_kappa_log.append((rl_gamma, rl_kappa))
                # Compute AS quotes with RL-chosen gamma/kappa
                tr   = max(self.cfg['as_T'] - t_norm, 1e-4)
                res  = mid - rl_gamma * (self.model.sigma**2) * tr * inventory
                hs   = (rl_gamma * self.model.sigma**2 * tr) / 2.0 +                        (1.0 / rl_gamma) * np.log(1.0 + rl_gamma / rl_kappa)
                tick = self.cfg['tick_size_usdt']
                hs   = min(hs, 5 * tick)
                bid_raw = res - hs
                ask_raw = res + hs
            else:
                bid_raw, ask_raw = self.model.quotes(mid, inventory, t_norm)

            # Enforce minimum 1-tick spread around mid
            tick = self.cfg['tick_size_usdt']
            bid  = min(bid_raw, mid - tick)
            ask  = max(ask_raw, mid + tick)

            # Round to tick size
            bid = round(bid / tick) * tick
            ask = round(ask / tick) * tick

            if bid >= ask:
                ask = bid + tick

            quotes_placed += 1

            # Signal: normalized microprice deviation from mid
            signal = (micro - mid) / mid

            # Mid in next tick for realized return
            mid_next = mid_hist[-1] if len(mid_hist) > 1 else mid

            # Get trades in fill window (10s bucket alignment for CSV resolution)
            bucket_ms = (ts_ms // 10000) * 10000
            t_end     = bucket_ms + 10000
            window = day_trades[
                (day_trades['ts_ms'] >= bucket_ms) &
                (day_trades['ts_ms'] <  t_end)
            ]

            fill = self.filler.simulate(
                bid, ask, ts_ms, window,
                inventory, spread, signal, mid, mid_next
            )

            if fill:
                inventory = fill.inventory_after
                fills.append(fill)

        return fills, quotes_placed

    # ------------------------------------------------------------------
    def _build_results(
        self,
        fills_by_day: List[Tuple[str, List[FillRecord]]],
        all_fills:    List[FillRecord],
        total_qp:     int,
    ) -> Dict:
        """
        Split fills into IS/OOS by day boundary, compute all metric sets.

        With 2 days of data:
          IS  = Day 1 (Mar 01) fills
          OOS = Day 2 (Mar 02) fills
        """
        cap     = self.cfg['initial_capital_usdt']
        lev     = self.cfg['leverage']
        lat     = self.cfg['simulated_latency_ms']
        n_days  = len(fills_by_day)
        n_is    = max(1, int(np.ceil(n_days * self.cfg['in_sample_ratio'])))

        is_fills  = [f for _, day in fills_by_day[:n_is]  for f in day]
        oos_fills = [f for _, day in fills_by_day[n_is:]  for f in day]

        is_qp  = int(total_qp * (len(is_fills)  / max(len(all_fills), 1)))
        oos_qp = int(total_qp * (len(oos_fills) / max(len(all_fills), 1)))

        log.info(f"IS  fills: {len(is_fills):,}  ({fills_by_day[0][0]})")
        if oos_fills:
            log.info(f"OOS fills: {len(oos_fills):,}  ({fills_by_day[n_is][0] if n_is < n_days else 'N/A'})")

        is_result = MetricsCalculator(
            cap, lev, 'In-Sample (Day 1: Mar 01)', lat, is_qp
        ).compute(is_fills)

        oos_result = MetricsCalculator(
            cap, lev, 'Out-of-Sample (Day 2: Mar 02)', lat, oos_qp
        ).compute(oos_fills if oos_fills else is_fills)

        full_result = MetricsCalculator(
            cap, lev, 'Full Period (Mar 01–02)', lat, total_qp
        ).compute(all_fills)

        # Metric 23: Sharpe degradation IS → OOS
        if is_result.sharpe_ratio != 0:
            oos_result.sharpe_degradation_pct = float(
                (oos_result.sharpe_ratio - is_result.sharpe_ratio)
                / abs(is_result.sharpe_ratio) * 100
            )

        wf_windows = self._walk_forward(fills_by_day, total_qp, cap, lev, lat)

        return {
            'gamma_kappa_log':      self.gamma_kappa_log,
            'in_sample':            is_result,
            'out_of_sample':        oos_result,
            'full':                 full_result,
            'walk_forward_windows': wf_windows,
            'config':               self.cfg,
            'fills_by_day':         [(d, len(f)) for d, f in fills_by_day],
            'total_quotes_placed':  total_qp,
            'total_fills':          len(all_fills),
        }

    # ------------------------------------------------------------------
    def _walk_forward(
        self,
        fills_by_day: List[Tuple[str, List[FillRecord]]],
        total_qp:     int,
        cap: float, lev: float, lat: float,
    ) -> List[Dict]:
        """
        Walk-forward validation.

        With 2 days:
          Window 1: Train on first 50% of Day 1, Test on second 50% of Day 1
          Window 2: Train on all of Day 1, Test on Day 2

        With more days the windows expand naturally.
        Auto-determines split points from fill timestamps.
        """
        all_fills = [f for _, day in fills_by_day for f in day]
        n = len(all_fills)
        if n < 40:
            log.warning("Too few fills for walk-forward analysis.")
            return []

        windows = []

        # Window 1: first half train / second half test (intra-day)
        mid = n // 2
        w1_train = all_fills[:mid]
        w1_test  = all_fills[mid:]
        qp_half  = total_qp // 2

        train_r = MetricsCalculator(cap, lev, 'WF-1 Train', lat, qp_half).compute(w1_train)
        test_r  = MetricsCalculator(cap, lev, 'WF-1 Test',  lat, qp_half).compute(w1_test)
        if train_r.sharpe_ratio != 0:
            test_r.sharpe_degradation_pct = (
                (test_r.sharpe_ratio - train_r.sharpe_ratio)
                / abs(train_r.sharpe_ratio) * 100
            )
        windows.append({'window': 1, 'label': 'First 50% → Second 50%',
                        'train': train_r, 'test': test_r})

        # Window 2: Day 1 train / Day 2 test (if 2+ days)
        if len(fills_by_day) >= 2:
            day1_fills = fills_by_day[0][1]
            day2_fills = fills_by_day[1][1]
            if day1_fills and day2_fills:
                qp1 = int(total_qp * len(day1_fills) / n)
                qp2 = int(total_qp * len(day2_fills) / n)
                tr2 = MetricsCalculator(cap, lev, 'WF-2 Train (Mar 01)', lat, qp1).compute(day1_fills)
                te2 = MetricsCalculator(cap, lev, 'WF-2 Test  (Mar 02)', lat, qp2).compute(day2_fills)
                if tr2.sharpe_ratio != 0:
                    te2.sharpe_degradation_pct = (
                        (te2.sharpe_ratio - tr2.sharpe_ratio)
                        / abs(tr2.sharpe_ratio) * 100
                    )
                windows.append({'window': 2, 'label': 'Day 1 (Mar 01) → Day 2 (Mar 02)',
                                'train': tr2, 'test': te2})

        for w in windows:
            log.info(
                f"WF Window {w['window']} ({w['label']}): "
                f"Train Sharpe={w['train'].sharpe_ratio:.3f}  "
                f"Test Sharpe={w['test'].sharpe_ratio:.3f}  "
                f"Degradation={w['test'].sharpe_degradation_pct:+.1f}%"
            )

        return windows

    #  ------------------------------------------------------------------
    @staticmethod
    def _date_from_filename(path: str) -> str:
        """Extract date string from filename like 2026-03-01_BTCUSDT_ob200.data"""
        base = os.path.basename(path)
        # Try YYYY-MM-DD prefix
        if len(base) >= 10 and base[4] == '-' and base[7] == '-':
            return base[:10]
        return base.split('_')[0]