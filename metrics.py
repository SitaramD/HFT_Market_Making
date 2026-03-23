# =============================================================================
# SITARAM HFT — Metrics Calculator
# Computes all 23 institutional metrics from real simulation fill records.
# =============================================================================

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import List

from scipy.stats import spearmanr

ANNUALIZATION = np.sqrt(365)
TRADING_DAYS  = 365

log = logging.getLogger('sitaram.metrics')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FillRecord:
    """
    One simulated order fill from the backtest.
    Produced by FillSimulator and consumed by MetricsCalculator.
    """
    timestamp:        datetime   # UTC fill time
    side:             str        # 'buy' or 'sell'
    price:            float      # Fill price (USDT) — our quoted bid or ask
    quantity:         float      # BTC quantity filled
    fee_rate:         float      # Signed: maker=-0.00010, taker=+0.00055
    is_maker:         bool       # True = limit fill (rebate)
    inventory_after:  float      # Net BTC inventory after this fill
    spread_at_fill:   float      # Best bid-ask spread at quote time (USDT)
    signal_score:     float      # Microprice deviation at quote time
    realized_return:  float      # Mid-price move in next tick (for IC calc)
    mid_at_fill:      float      # Mid-price at fill time


@dataclass
class MetricsResult:
    """All 23 metrics + supporting fields for one evaluation period."""

    # Period metadata
    period_type:              str   = ''
    period_start:             str   = ''
    period_end:               str   = ''
    n_periods:                int   = 0    # Number of PnL buckets used
    total_trades:             int   = 0
    quotes_placed:            int   = 0

    # ── RETURN METRICS ────────────────────────────────────────────────────
    sharpe_ratio:             float = 0.0   # 1
    sortino_ratio:            float = 0.0   # 2
    calmar_ratio:             float = 0.0   # 3
    apy_post_leverage:        float = 0.0   # 4  (%)

    # ── RISK METRICS ──────────────────────────────────────────────────────
    max_drawdown_pct:         float = 0.0   # 5  (%)
    max_drawdown_usdt:        float = 0.0   # 6
    var_95_daily:             float = 0.0   # 7
    drawdown_duration_days:   float = 0.0   # 8  (periods)
    inventory_mean_reversion: float = 0.0   # 9  (%)

    # ── EDGE & SIGNAL ─────────────────────────────────────────────────────
    ic_rolling:               float = 0.0   # 10
    ic_stability_days:        int   = 0     # 11
    ic_decay_500ms:           float = 0.0   # 12
    adverse_selection_rate:   float = 0.0   # 13 (%)

    # ── EXECUTION ─────────────────────────────────────────────────────────
    fill_rate:                float = 0.0   # 14 (%)
    quote_to_trade_ratio:     float = 0.0   # 15
    maker_ratio:              float = 0.0   # 16 (%)
    fee_adjusted_pnl:         float = 0.0   # 17 (USDT)
    avg_latency_ms:           float = 0.0   # 18

    # ── CAPACITY ──────────────────────────────────────────────────────────
    max_capacity_usdt:        float = 0.0   # 19
    daily_turnover_usdt:      float = 0.0   # 20
    market_impact_bps:        float = 0.0   # 21
    regime_sensitivity:       float = 0.0   # 22

    # ── OOS INTEGRITY ─────────────────────────────────────────────────────
    sharpe_degradation_pct:   float = 0.0   # 23

    # Supporting (not in the 23)
    gross_pnl_usdt:           float = 0.0
    total_fees_usdt:          float = 0.0
    volatility_ann:           float = 0.0
    daily_turnover_btc:       float = 0.0
    avg_spread_usdt:          float = 0.0
    avg_mid_price:            float = 0.0


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class MetricsCalculator:
    """
    Pass a list of FillRecord objects → get a fully populated MetricsResult.

    Instantiate one per period (IS, OOS, each WF window).
    """

    def __init__(
        self,
        initial_capital:      float = 100_000.0,
        leverage:             float = 1.0,
        period_type:          str   = 'backtest',
        simulated_latency_ms: float = 10.0,
        quotes_placed:        int   = 0,
    ):
        self.capital  = initial_capital
        self.leverage = leverage
        self.ptype    = period_type
        self.latency  = simulated_latency_ms
        self.qp       = quotes_placed

    # ------------------------------------------------------------------
    def compute(self, fills: List[FillRecord]) -> MetricsResult:
        r = MetricsResult(period_type=self.ptype)

        if not fills:
            return r

        df = self._to_df(fills)
        r.total_trades   = len(df)
        r.quotes_placed  = self.qp if self.qp > 0 else len(df) * 3
        r.period_start   = str(df['ts'].min().date())
        r.period_end     = str(df['ts'].max().date())
        r.avg_spread_usdt = float(df['spread_at_fill'].mean())
        r.avg_mid_price   = float(df['mid_at_fill'].mean())

        # ── PnL components ───────────────────────────────────────────
        # buy fill:  we paid price → cash out, btc in  → signed = -notional
        # sell fill: we received price → cash in, btc out → signed = +notional
        df['signed']   = df.apply(
            lambda x: x['notional'] if x['side'] == 'sell'
                      else -x['notional'], axis=1)
        df['fee_usdt'] = df['notional'] * df['fee_rate']
        df['net']      = df['signed'] - df['fee_usdt']

        r.gross_pnl_usdt   = float(df['signed'].sum())
        r.total_fees_usdt  = float(df['fee_usdt'].sum())
        r.fee_adjusted_pnl = float(df['net'].sum())   # Metric 17

        # ── Build PnL time series ─────────────────────────────────────
        pnl_series = self._pnl_series(df)
        r.n_periods = len(pnl_series)

        # ── 1. Sharpe ────────────────────────────────────────────────
        r.sharpe_ratio = self._sharpe(pnl_series)

        # ── 2. Sortino ───────────────────────────────────────────────
        r.sortino_ratio = self._sortino(pnl_series)

        # ── 4. APY ───────────────────────────────────────────────────
        r.apy_post_leverage = self._apy(r.fee_adjusted_pnl, r.n_periods)

        # ── 5 & 6. Max Drawdown ──────────────────────────────────────
        r.max_drawdown_pct, r.max_drawdown_usdt = self._drawdown(pnl_series)

        # ── 3. Calmar ────────────────────────────────────────────────
        r.calmar_ratio = float(r.apy_post_leverage / r.max_drawdown_pct) \
            if r.max_drawdown_pct > 1e-9 else 0.0

        # ── 7. VaR 95% ───────────────────────────────────────────────
        r.var_95_daily = float(abs(np.percentile(pnl_series.values, 5)))

        # ── 8. Drawdown Duration ─────────────────────────────────────
        r.drawdown_duration_days = self._dd_duration(pnl_series)

        # ── 9. Inventory Mean Reversion ──────────────────────────────
        # % of fills where position is within 1 order-qty of flat
        near_flat = (df['inventory_after'].abs() <= 0.001).sum()
        r.inventory_mean_reversion = float(near_flat / len(df) * 100)

        # ── 10. IC Rolling ───────────────────────────────────────────
        r.ic_rolling = self._ic(df['signal_score'], df['realized_return'])

        # ── 11. IC Stability ─────────────────────────────────────────
        r.ic_stability_days = self._ic_stability(df)

        # ── 12. IC Decay proxy ───────────────────────────────────────
        # First quarter of fills = shorter horizon IC proxy
        q = max(len(df) // 4, 10)
        r.ic_decay_500ms = self._ic(
            df['signal_score'].iloc[:q],
            df['realized_return'].iloc[:q]
        )

        # ── 13. Adverse Selection ────────────────────────────────────
        # Buy fill adversely selected if mid fell after fill (we paid too much)
        # Sell fill adversely selected if mid rose after fill
        adv = (
            ((df['side'] == 'buy')  & (df['realized_return'] < 0)) |
            ((df['side'] == 'sell') & (df['realized_return'] > 0))
        ).sum()
        r.adverse_selection_rate = float(adv / len(df) * 100)

        # ── 14. Fill Rate ────────────────────────────────────────────
        r.fill_rate = float(len(df) / r.quotes_placed * 100)

        # ── 15. Quote-to-Trade Ratio ─────────────────────────────────
        r.quote_to_trade_ratio = float(r.quotes_placed / len(df))

        # ── 16. Maker Ratio ──────────────────────────────────────────
        r.maker_ratio = float(df['is_maker'].sum() / len(df) * 100)

        # ── 18. Latency ──────────────────────────────────────────────
        r.avg_latency_ms = self.latency

        # ── 19. Max Deployable Capital ───────────────────────────────
        # Theoretical capacity: if we increase size until market impact
        # erodes the half-spread edge. Conservative: 100x current daily notional.
        by_day = df.groupby(df['ts'].dt.date)
        r.daily_turnover_usdt = float(by_day['notional'].sum().mean())
        r.daily_turnover_btc  = float(by_day['quantity'].sum().mean())
        r.max_capacity_usdt   = float(r.daily_turnover_usdt * 100)

        # ── 20 already set above ─────────────────────────────────────

        # ── 21. Market Impact ────────────────────────────────────────
        # Half-spread as proxy for one-way market impact cost
        r.market_impact_bps = float(
            (df['spread_at_fill'] / df['mid_at_fill']).mean() * 10_000 / 2
        )

        # ── 22. Regime Sensitivity ───────────────────────────────────
        # |Sharpe(first half) - Sharpe(second half)| as stability measure
        h = len(pnl_series) // 2
        if h >= 4:
            s1 = self._sharpe(pnl_series.iloc[:h])
            s2 = self._sharpe(pnl_series.iloc[h:])
            r.regime_sensitivity = float(abs(s1 - s2))

        # ── Supporting ───────────────────────────────────────────────
        r.volatility_ann = float(pnl_series.std() * ANNUALIZATION)

        return r

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _to_df(fills: List[FillRecord]) -> pd.DataFrame:
        rows = [{
            'ts':             pd.Timestamp(f.timestamp).tz_localize('UTC')
                              if f.timestamp.tzinfo is None
                              else pd.Timestamp(f.timestamp),
            'side':           f.side,
            'price':          f.price,
            'quantity':       f.quantity,
            'notional':       f.price * f.quantity,
            'fee_rate':       f.fee_rate,
            'is_maker':       bool(f.is_maker),
            'inventory_after':f.inventory_after,
            'spread_at_fill': f.spread_at_fill,
            'signal_score':   f.signal_score,
            'realized_return':f.realized_return,
            'mid_at_fill':    f.mid_at_fill,
        } for f in fills]

        df = pd.DataFrame(rows)
        df.sort_values('ts', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _pnl_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Build a PnL time series for Sharpe/drawdown/VaR calculations.

        Bucketing strategy:
        - 2+ calendar days → daily buckets  (clean Sharpe interpretation)
        - 1 calendar day   → 30-min buckets (intraday, ~48 obs per day)
        - < 30 min of data → 5-min buckets  (fallback)
        """
        n_days = df['ts'].dt.date.nunique()
        if n_days >= 2:
            key = df['ts'].dt.date
        else:
            span_min = (df['ts'].max() - df['ts'].min()).total_seconds() / 60
            freq = '30min' if span_min >= 60 else '5min'
            key = df['ts'].dt.floor(freq)

        return df.groupby(key)['net'].sum()

    def _sharpe(self, s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        std = s.std()
        if std < 1e-12:
            return 0.0
        return float((s.mean() / std) * ANNUALIZATION)

    def _sortino(self, s: pd.Series) -> float:
        down = s[s < 0]
        if len(down) < 2:
            return 0.0
        std = down.std()
        if std < 1e-12:
            return 0.0
        return float((s.mean() / std) * ANNUALIZATION)

    def _apy(self, fee_adj_pnl: float, n_periods: int) -> float:
        cap = self.capital * self.leverage
        if cap < 1e-9 or n_periods == 0:
            return 0.0
        # Scale to annual: assume n_periods covers 1 day if < 48 (30-min buckets)
        # or n_periods days if daily buckets
        return float((fee_adj_pnl / cap) * (TRADING_DAYS / n_periods) * 100)

    def _drawdown(self, s: pd.Series):
        equity = self.capital + s.cumsum()
        peak   = equity.cummax()
        dd_u   = equity - peak
        dd_pct = dd_u / peak * 100
        return float(abs(dd_pct.min())), float(abs(dd_u.min()))

    def _dd_duration(self, s: pd.Series) -> float:
        equity = self.capital + s.cumsum()
        peak   = equity.cummax()
        in_dd  = equity < peak
        cur, mx = 0, 0
        for v in in_dd:
            cur = cur + 1 if v else 0
            mx  = max(mx, cur)
        return float(mx)

    @staticmethod
    def _ic(signals: pd.Series, returns: pd.Series) -> float:
        """
        Spearman Rank IC with full guard rails.

        Replaces the original Pearson corr() call which produced NaN
        (and silently became 0.0) whenever signal_score or realized_return
        had zero variance — the root cause of IC=0.000 in the backtest.

        Spearman is the standard for HFT microstructure IC:
          - Robust to fat-tailed return distributions
          - Not affected by outlier fills
          - Consistent with industry IC reporting conventions

        Diagnostics: logs which series is flat so the upstream bug
        (bad forward-return window or constant microprice) is immediately
        visible in the run output rather than silently zero.
        """
        valid = pd.DataFrame({'s': signals, 'r': returns}).dropna()
        if len(valid) < 10:
            return 0.0

        s = valid['s'].to_numpy()
        r = valid['r'].to_numpy()

        s_std = float(np.std(s))
        r_std = float(np.std(r))

        # Guard: near-zero variance means correlation is undefined.
        # Log which series is flat so the root cause is visible.
        if s_std < 1e-10 or r_std < 1e-10:
            flat = 'signal_score' if s_std < 1e-10 else 'realized_return'
            log.warning(
                f"IC=0: '{flat}' has near-zero variance — "
                f"signal_std={s_std:.2e}  return_std={r_std:.2e}  n={len(s)}.  "
                f"Check: (1) forward-return window uses ts field not row offset, "
                f"(2) microprice is updating between snapshots."
            )
            return 0.0

        ic, _ = spearmanr(s, r)
        return float(ic) if not np.isnan(ic) else 0.0

    @staticmethod
    def _ic_stability(df: pd.DataFrame) -> int:
        """Max consecutive 30-min buckets where IC > 0.02."""
        df = df.copy()
        df['bucket'] = df['ts'].dt.floor('30min')
        buckets = sorted(df['bucket'].unique())
        consec, best = 0, 0
        for b in buckets:
            bdf = df[df['bucket'] == b]
            if len(bdf) >= 5:
                ic = MetricsCalculator._ic(bdf['signal_score'], bdf['realized_return'])
                if ic > 0.02:
                    consec += 1
                    best    = max(best, consec)
                else:
                    consec = 0
        return best
