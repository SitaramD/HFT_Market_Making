# =============================================================================
# SITARAM HFT — RL Environment
# Step 1 of 5
#
# A Gymnasium-compatible environment that replays real order book data
# tick by tick. The RL agent observes market state and outputs gamma/kappa
# for the AS model. The environment computes the reward from real fills.
#
# Compatible with stable-baselines3 AND our custom numpy PPO.
# =============================================================================

import json
import os
import numpy as np
from typing import List, Tuple, Optional, Dict

from src.orderbook  import OrderBook
from src.backtester import ASModel, FillSimulator, DataLoader
from src.config     import CONFIG


# =============================================================================
# NORMALISATION CONSTANTS
# Computed from typical BTC/USDT market microstructure.
# Used to keep all state variables in [-1, 1] range for stable RL training.
# =============================================================================
NORM = {
    'inventory':    CONFIG['max_inventory_btc'],   # ±0.01 BTC → ±1
    'vol':          0.002,                          # typical 200-tick std
    'spread':       5.0,                            # USDT, clips at 5
    'obi':          1.0,                            # already [-1, 1]
    't_norm':       1.0,                            # already [0, 1]
}

# Action bounds — RL outputs raw values, clipped to these
GAMMA_MIN, GAMMA_MAX = 0.01, 1.00
KAPPA_MIN, KAPPA_MAX = 0.50, 5.00

# Reward shaping constants
LAMBDA_INV = 0.100    # penalty per unit of inventory²  (scaled for max_inv=0.10)
LAMBDA_ADV = 0.010    # penalty per adverse fill (USDT)
LAMBDA_DD  = 1.000    # penalty per unit of drawdown beyond 5%
DD_LIMIT   = 0.05     # 5% max drawdown threshold


# =============================================================================
# MARKET MAKING ENVIRONMENT
# =============================================================================

class MarketMakingEnv:
    """
    Replays one full day of OB + trade data as an RL environment.

    observation_space: Box(5,)  — [inventory, vol, obi, spread, t_norm]
    action_space:      Box(2,)  — [gamma, kappa]  both continuous

    Each step() = one 200ms OB tick.
    reset()     = rewind to start of the day's data.

    Compatible with stable-baselines3 (if installed) and our numpy PPO.
    """

    OBS_DIM = 5
    ACT_DIM = 2

    def __init__(self, ob_file: str, trades_df, cfg: dict = CONFIG):
        self.ob_file  = ob_file
        self.cfg      = cfg

        # Pre-load all OB messages into memory for fast reset/replay
        self._ob_msgs: List[dict] = self._load_ob(ob_file)

        # ── Fast trade lookup via sorted numpy arrays ──────────────────
        # Convert once at init — O(log n) per tick via searchsorted
        # instead of O(n) pandas boolean scan on every tick.
        trades_sorted       = trades_df.sort_values('ts_ms').reset_index(drop=True)
        self._trade_ts      = trades_sorted['ts_ms'].to_numpy(dtype=np.int64)
        self._trade_price   = trades_sorted['price'].to_numpy(dtype=np.float64)
        self._trade_volume  = trades_sorted['volume'].to_numpy(dtype=np.float64)
        self._trade_side    = trades_sorted['side'].to_numpy()
        # Keep a sliceable DataFrame view aligned to the sorted arrays
        self._trades_sorted = trades_sorted

        # Components
        self.filler = FillSimulator(cfg)

        # Episode state — reset on each reset()
        self._reset_state()

    # ------------------------------------------------------------------
    # GYM INTERFACE
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset to start of day. Returns initial observation."""
        self._reset_state()
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action [gamma, kappa] at current tick.
        Returns (obs, reward, done, info).
        """
        # Clip action to valid range
        gamma = float(np.clip(action[0], GAMMA_MIN, GAMMA_MAX))
        kappa = float(np.clip(action[1], KAPPA_MIN, KAPPA_MAX))

        # Advance to next valid tick
        fill_info = self._advance_tick(gamma, kappa)

        obs    = self._obs()
        reward = self._reward(fill_info)
        done   = self._cursor >= len(self._ob_msgs)

        info = {
            'tick':       self._cursor,
            'mid':        self._mid,
            'bid':        fill_info.get('bid', 0),
            'ask':        fill_info.get('ask', 0),
            'gamma':      gamma,
            'kappa':      kappa,
            'sigma':      self._sigma,
            'inventory':  self._inventory,
            'fill':       fill_info.get('fill', False),
            'fill_side':  fill_info.get('side', None),
            'fill_pnl':   fill_info.get('pnl', 0.0),
            'reward':     reward,
            'equity':     self._equity,
            'drawdown':   self._drawdown_pct(),
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # INTERNAL STATE MANAGEMENT
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._book      = OrderBook()
        self._cursor    = 0
        self._inventory = 0.0
        self._mid       = 0.0
        self._sigma     = CONFIG['as_sigma']
        self._mid_hist  : List[float] = []
        self._equity    = CONFIG['initial_capital_usdt']
        self._peak_eq   = CONFIG['initial_capital_usdt']
        self._total_pnl = 0.0
        self._quotes_placed = 0
        self._fills_count   = 0

        # Advance past invalid ticks to find first valid book state
        self._seek_first_valid()

    def _seek_first_valid(self):
        """Consume messages until we have a valid book (snapshot applied)."""
        while self._cursor < len(self._ob_msgs):
            msg = self._ob_msgs[self._cursor]
            self._cursor += 1
            if self._book.apply(msg) and self._book.valid:
                self._mid = self._book.mid
                self._mid_hist.append(self._mid)
                return

    def _advance_tick(self, gamma: float, kappa: float) -> dict:
        """
        Process the next valid OB tick with given gamma/kappa.
        Returns fill info dict.
        """
        result = {'fill': False, 'bid': 0, 'ask': 0,
                  'side': None, 'pnl': 0.0}

        # Find next valid book tick
        while self._cursor < len(self._ob_msgs):
            msg = self._ob_msgs[self._cursor]
            self._cursor += 1
            if self._book.apply(msg) and self._book.valid:
                break
        else:
            return result  # end of data

        mid    = self._book.mid
        spread = self._book.spread
        micro  = self._book.microprice()
        ts_ms  = int(self._book.ts)

        if not (mid and spread and micro):
            return result

        # Update mid history and sigma
        self._mid = mid
        self._mid_hist.append(mid)
        self._update_sigma()

        # AS model quotes using RL-chosen gamma/kappa
        t_norm   = (ts_ms / 1000.0 % 86400.0) / 86400.0
        tr       = max(self.cfg['as_T'] - t_norm, 1e-4)
        res      = mid - gamma * (self._sigma ** 2) * tr * self._inventory
        hs       = (gamma * self._sigma ** 2 * tr) / 2.0 +                    (1.0 / gamma) * np.log(1.0 + gamma / kappa)
        # Cap half-spread at 5 ticks to ensure quotes stay near market
        tick     = self.cfg['tick_size_usdt']
        hs       = min(hs, 5 * tick)
        bid_r    = res - hs
        ask_r    = res + hs

        bid  = round(min(bid_r, mid - tick) / tick) * tick
        ask  = round(max(ask_r, mid + tick) / tick) * tick
        if bid >= ask:
            ask = bid + tick

        self._quotes_placed += 1
        result['bid'] = bid
        result['ask'] = ask

        # Get trades in fill window — O(log n) via searchsorted
        # Trade timestamps have 10s resolution in source CSV.
        # Align OB ts to 10s bucket so windows match trade ts granularity.
        bucket_ms = (ts_ms // 10000) * 10000
        t_end     = bucket_ms + 10000  # 10-second window
        lo        = np.searchsorted(self._trade_ts, bucket_ms, side='left')
        hi        = np.searchsorted(self._trade_ts, t_end,     side='left')
        window    = self._trades_sorted.iloc[lo:hi]

        signal   = (micro - mid) / mid if mid > 0 else 0.0
        mid_next = self._mid_hist[-1]

        fill = self.filler.simulate(
            bid, ask, ts_ms, window,
            self._inventory, spread, signal, mid, mid_next
        )

        if fill:
            self._inventory = fill.inventory_after
            self._fills_count += 1

            # Fee-adjusted PnL from this fill
            notional = fill.price * fill.quantity
            signed   = notional if fill.side == 'sell' else -notional
            fee      = notional * fill.fee_rate
            pnl      = signed - fee

            self._total_pnl += pnl
            self._equity     = CONFIG['initial_capital_usdt'] + self._total_pnl
            self._peak_eq    = max(self._peak_eq, self._equity)

            result.update({
                'fill':     True,
                'side':     fill.side,
                'pnl':      pnl,
                'adv':      fill.realized_return < 0 if fill.side == 'buy'
                            else fill.realized_return > 0,
            })

        return result

    def _update_sigma(self):
        if len(self._mid_hist) >= 20:
            mids  = np.array(self._mid_hist[-200:])
            rets  = np.diff(np.log(mids + 1e-10))
            sigma = float(np.std(rets))
            self._sigma = max(sigma, 1e-8)

    def _drawdown_pct(self) -> float:
        if self._peak_eq <= 0:
            return 0.0
        return (self._peak_eq - self._equity) / self._peak_eq

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        """
        Returns normalised state vector: [inventory, vol, obi, spread, t_norm]
        All values clipped to [-1, 1] for stable training.
        """
        ts_ms  = int(self._book.ts) if self._book.ts else 0
        t_norm = (ts_ms / 1000.0 % 86400.0) / 86400.0

        obs = np.array([
            np.clip(self._inventory / NORM['inventory'],  -1, 1),
            np.clip(self._sigma     / NORM['vol'],         0, 1),
            np.clip(self._book.obi(5) if self._book.ready else 0, -1, 1),
            np.clip((self._book.spread or 0) / NORM['spread'], 0, 1),
            t_norm,
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def _reward(self, fill_info: dict) -> float:
        """
        Reward = fee-adjusted PnL this tick
               - λ_inv  × inventory²
               - λ_adv  × adverse_selection (if fill was adversely selected)
               - λ_dd   × excess drawdown beyond 5%
        """
        pnl = fill_info.get('pnl', 0.0)

        # Inventory penalty: quadratic, symmetric
        inv_penalty = LAMBDA_INV * (self._inventory ** 2)

        # Adverse selection penalty: only on fills that moved against us
        adv_penalty = 0.0
        if fill_info.get('fill') and fill_info.get('adv', False):
            adv_penalty = LAMBDA_ADV * abs(fill_info.get('pnl', 0.0))

        # Drawdown penalty: only kicks in beyond 5% limit
        dd = self._drawdown_pct()
        dd_penalty = LAMBDA_DD * max(0.0, dd - DD_LIMIT)

        reward = pnl - inv_penalty - adv_penalty - dd_penalty
        return float(reward)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ob(ob_file: str) -> List[dict]:
        """Load all OB messages into memory. ~429K lines per day."""
        with open(ob_file, 'rb') as fb:
            bom = fb.read(2)
        enc = 'utf-16' if bom in (b'\xff\xfe', b'\xfe\xff') else 'utf-8'

        msgs = []
        with open(ob_file, 'r', encoding=enc, errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return msgs

    @property
    def n_ticks(self) -> int:
        return len(self._ob_msgs)

    @property
    def quotes_placed(self) -> int:
        return self._quotes_placed

    @property
    def fills_count(self) -> int:
        return self._fills_count
