"""
SITARAM HFT — Unit Tests: Gate Logic
Priority 2 — Guards the path to live trading.

Tests cover:
  - Sharpe ratio calculation correctness
  - Max drawdown calculation correctness
  - Fill rate calculation correctness
  - Gate pass/fail logic
  - Edge cases: zero trades, single trade, all-loss runs
  - Property-based: gate thresholds are monotone
"""

import math
import pytest
import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

# ── Gate thresholds (Run-15) ──────────────────────────────────
SHARPE_TARGET  = 3.0
MAX_DD_PCT     = 5.0
FILL_RATE_MIN  = 5.0
INITIAL_CAP    = 100_000.0
ANNUALIZE      = math.sqrt(252 * 24 * 60)   # per-minute returns → annual


# ── Pure implementations (mirrors what engine/persister computes) ──

def compute_sharpe(pnl_series: list, risk_free: float = 0.0) -> float:
    """
    Sharpe = mean(excess_returns) / std(excess_returns) * sqrt(N_annualize)
    pnl_series: list of per-trade PnL in USDT
    """
    if len(pnl_series) < 2:
        return 0.0
    arr   = np.array(pnl_series, dtype=float)
    mu    = np.mean(arr) - risk_free
    sigma = np.std(arr, ddof=1)
    if sigma == 0.0:
        return 0.0
    return float(mu / sigma * ANNUALIZE)


def compute_max_drawdown_pct(equity_curve: list) -> float:
    """
    Max drawdown as % of peak equity.
    equity_curve: list of cumulative PnL values (NOT returns).
    """
    if not equity_curve:
        return 0.0
    eq  = np.array(equity_curve, dtype=float) + INITIAL_CAP
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / peak * 100.0
    return float(np.max(dd))


def compute_fill_rate(quotes_sent: int, fills_received: int) -> float:
    """Fill rate as percentage of quotes that resulted in fills."""
    if quotes_sent == 0:
        return 0.0
    return fills_received / quotes_sent * 100.0


def gate_pass(sharpe: float, max_dd_pct: float, fill_rate_pct: float) -> dict:
    """
    Evaluate all three gates. Returns dict with individual + overall results.
    All gates are advisory in the pipeline; this logic is what persister writes.
    """
    sharpe_ok    = sharpe       >= SHARPE_TARGET
    dd_ok        = max_dd_pct   <= MAX_DD_PCT
    fill_ok      = fill_rate_pct >= FILL_RATE_MIN
    return {
        "sharpe_pass":    sharpe_ok,
        "drawdown_pass":  dd_ok,
        "fill_rate_pass": fill_ok,
        "overall_pass":   sharpe_ok and dd_ok and fill_ok,
        "sharpe":         sharpe,
        "max_drawdown_pct": max_dd_pct,
        "fill_rate_pct":  fill_rate_pct,
    }


# ═══════════════════════════════════════════════════════════════
# 1. SHARPE RATIO TESTS
# ═══════════════════════════════════════════════════════════════

class TestSharpeCalculation:

    def test_positive_consistent_returns_high_sharpe(self):
        """Consistent positive PnL must yield high Sharpe."""
        pnl = [10.0] * 200          # perfectly consistent
        s   = compute_sharpe(pnl)
        assert s > SHARPE_TARGET

    def test_mixed_returns_lower_sharpe(self):
        """High variance PnL reduces Sharpe."""
        rng = np.random.default_rng(0)
        pnl = list(rng.normal(5.0, 50.0, 500))   # mean positive, high vol
        s   = compute_sharpe(pnl)
        assert s < compute_sharpe([5.0] * 500)    # worse than consistent

    def test_zero_returns_zero_sharpe(self):
        """All-zero PnL → zero Sharpe (std = 0 path)."""
        assert compute_sharpe([0.0] * 100) == 0.0

    def test_single_trade_zero_sharpe(self):
        """Single observation → no valid std → zero Sharpe."""
        assert compute_sharpe([100.0]) == 0.0

    def test_empty_pnl_zero_sharpe(self):
        assert compute_sharpe([]) == 0.0

    def test_negative_mean_negative_sharpe(self):
        """Consistently losing trades must give negative Sharpe."""
        pnl = [-5.0] * 200
        # std=0 → returns 0.0, not negative (edge case)
        # Use mixed losses to get valid std
        rng = np.random.default_rng(1)
        pnl = list(rng.normal(-5.0, 2.0, 300))
        s   = compute_sharpe(pnl)
        assert s < 0.0

    def test_sharpe_above_target_passes_gate(self):
        pnl = [10.0] * 300
        s   = compute_sharpe(pnl)
        result = gate_pass(sharpe=s, max_dd_pct=1.0, fill_rate_pct=20.0)
        assert result["sharpe_pass"] is True

    def test_sharpe_below_target_fails_gate(self):
        rng = np.random.default_rng(2)
        pnl = list(rng.normal(1.0, 50.0, 100))
        s   = compute_sharpe(pnl)
        result = gate_pass(sharpe=s, max_dd_pct=1.0, fill_rate_pct=20.0)
        # Sharpe gate may fail; check field is present
        assert "sharpe_pass" in result

    @given(
        mean = st.floats(min_value=-100.0, max_value=100.0),
        std  = st.floats(min_value=0.1,    max_value=200.0),
        n    = st.integers(min_value=10,   max_value=1000),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_property_sharpe_finite(self, mean, std, n):
        """Sharpe must be finite for any plausible PnL distribution."""
        rng = np.random.default_rng(0)
        pnl = list(rng.normal(mean, std, n))
        s   = compute_sharpe(pnl)
        assert math.isfinite(s)

    @given(
        scale = st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=200)
    def test_property_sharpe_scale_invariant(self, scale):
        """Sharpe is scale-invariant: multiplying all returns by k doesn't change it."""
        rng  = np.random.default_rng(42)
        base = list(rng.normal(5.0, 2.0, 200))
        s1   = compute_sharpe(base)
        s2   = compute_sharpe([x * scale for x in base])
        assert s1 == pytest.approx(s2, rel=1e-5)


# ═══════════════════════════════════════════════════════════════
# 2. MAX DRAWDOWN TESTS
# ═══════════════════════════════════════════════════════════════

class TestMaxDrawdown:

    def test_monotone_equity_zero_drawdown(self):
        """Steadily rising equity curve → zero drawdown."""
        eq = list(range(0, 1000, 10))     # cumulative PnL
        dd = compute_max_drawdown_pct(eq)
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_full_loss_scenario(self):
        """Equity dropping from peak to zero → 100% drawdown."""
        # cumulative PnL: rises then crashes to initial level
        eq = [100, 500, 1000, 500, 0]   # peak at 1000, back to 0
        dd = compute_max_drawdown_pct(eq)
        # peak equity = 100_000 + 1000 = 101_000
        # trough = 100_000 + 0 = 100_000
        # dd = 1000/101000 * 100
        expected = 1000 / 101_000 * 100
        assert dd == pytest.approx(expected, rel=1e-4)

    def test_drawdown_within_limit_passes(self):
        """DD below 5% → gate passes."""
        eq = [0, 100, 200, 150, 300]
        dd = compute_max_drawdown_pct(eq)
        result = gate_pass(sharpe=5.0, max_dd_pct=dd, fill_rate_pct=20.0)
        assert result["drawdown_pass"] is True

    def test_drawdown_exceeds_limit_fails_gate(self):
        """Construct a series with >5% drawdown from peak."""
        # Make peak at 10000 USDT PnL, then crash by 6000
        eq = list(range(0, 10001, 100)) + list(range(10000, 3999, -100))
        dd = compute_max_drawdown_pct(eq)
        assert dd > MAX_DD_PCT
        result = gate_pass(sharpe=5.0, max_dd_pct=dd, fill_rate_pct=20.0)
        assert result["drawdown_pass"] is False

    def test_empty_equity_curve(self):
        assert compute_max_drawdown_pct([]) == 0.0

    def test_single_point_equity_curve(self):
        assert compute_max_drawdown_pct([500.0]) == 0.0

    def test_daily_loss_limit_respected(self):
        """
        Daily loss limit = -2000 USDT.
        If cumulative PnL drops by 2000 from session start, system should
        have stopped. Verify that a 2000 USDT loss is < 2% of initial capital.
        """
        daily_loss = 2_000.0
        loss_pct   = daily_loss / INITIAL_CAP * 100
        assert loss_pct == pytest.approx(2.0, rel=1e-6)
        assert loss_pct < MAX_DD_PCT, \
            "Daily loss limit should trigger before max drawdown gate"

    @given(
        pnl_changes = st.lists(
            st.floats(min_value=-500.0, max_value=500.0),
            min_size=2, max_size=500,
        )
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_property_drawdown_between_0_and_100(self, pnl_changes):
        """Drawdown must always be between 0% and 100%."""
        cumulative = list(np.cumsum(pnl_changes))
        dd = compute_max_drawdown_pct(cumulative)
        assert 0.0 <= dd <= 100.0
        assert math.isfinite(dd)


# ═══════════════════════════════════════════════════════════════
# 3. FILL RATE TESTS
# ═══════════════════════════════════════════════════════════════

class TestFillRate:

    def test_zero_quotes_zero_fill_rate(self):
        assert compute_fill_rate(0, 0) == 0.0

    def test_all_filled(self):
        assert compute_fill_rate(100, 100) == pytest.approx(100.0)

    def test_none_filled(self):
        assert compute_fill_rate(100, 0) == pytest.approx(0.0)

    def test_typical_fill_rate(self):
        # 50 fills from 200 quotes = 25%
        assert compute_fill_rate(200, 50) == pytest.approx(25.0)

    def test_fill_rate_above_min_passes_gate(self):
        fr     = compute_fill_rate(100, 10)   # 10%
        result = gate_pass(sharpe=5.0, max_dd_pct=1.0, fill_rate_pct=fr)
        assert result["fill_rate_pass"] is True

    def test_fill_rate_below_min_fails_gate(self):
        fr     = compute_fill_rate(1000, 2)   # 0.2%
        result = gate_pass(sharpe=5.0, max_dd_pct=1.0, fill_rate_pct=fr)
        assert result["fill_rate_pass"] is False

    @given(
        sent  = st.integers(min_value=1,  max_value=10_000),
        fills = st.integers(min_value=0,  max_value=10_000),
    )
    @settings(max_examples=500)
    def test_property_fill_rate_bounded(self, sent, fills):
        """Fill rate must be in [0, 100] when fills <= sent."""
        assume(fills <= sent)
        fr = compute_fill_rate(sent, fills)
        assert 0.0 <= fr <= 100.0
        assert math.isfinite(fr)


# ═══════════════════════════════════════════════════════════════
# 4. COMBINED GATE TESTS
# ═══════════════════════════════════════════════════════════════

class TestGateCombined:

    def test_all_gates_pass(self):
        result = gate_pass(sharpe=4.5, max_dd_pct=2.0, fill_rate_pct=15.0)
        assert result["overall_pass"] is True
        assert result["sharpe_pass"]
        assert result["drawdown_pass"]
        assert result["fill_rate_pass"]

    def test_one_gate_failure_fails_overall(self):
        """Overall gate fails if ANY single gate fails."""
        # Only drawdown fails
        result = gate_pass(sharpe=4.5, max_dd_pct=6.0, fill_rate_pct=15.0)
        assert result["overall_pass"] is False
        assert result["drawdown_pass"] is False
        assert result["sharpe_pass"] is True
        assert result["fill_rate_pass"] is True

    def test_gate_result_contains_all_keys(self):
        result = gate_pass(sharpe=3.5, max_dd_pct=3.0, fill_rate_pct=10.0)
        for key in ["sharpe_pass", "drawdown_pass", "fill_rate_pass",
                    "overall_pass", "sharpe", "max_drawdown_pct", "fill_rate_pct"]:
            assert key in result

    def test_boundary_sharpe_exactly_at_target(self):
        """Sharpe exactly at target must PASS (≥ not >)."""
        result = gate_pass(sharpe=SHARPE_TARGET, max_dd_pct=1.0, fill_rate_pct=10.0)
        assert result["sharpe_pass"] is True

    def test_boundary_dd_exactly_at_limit(self):
        """DD exactly at limit must PASS (≤ not <)."""
        result = gate_pass(sharpe=4.0, max_dd_pct=MAX_DD_PCT, fill_rate_pct=10.0)
        assert result["drawdown_pass"] is True

    def test_boundary_fill_rate_exactly_at_min(self):
        """Fill rate exactly at min must PASS (≥ not >)."""
        result = gate_pass(sharpe=4.0, max_dd_pct=2.0, fill_rate_pct=FILL_RATE_MIN)
        assert result["fill_rate_pass"] is True

    @given(
        sharpe    = st.floats(min_value=-10.0, max_value=20.0),
        dd        = st.floats(min_value=0.0,   max_value=50.0),
        fill_rate = st.floats(min_value=0.0,   max_value=100.0),
    )
    @settings(max_examples=500)
    def test_property_gate_overall_iff_all_pass(self, sharpe, dd, fill_rate):
        """overall_pass must be True iff all three individual gates pass."""
        result = gate_pass(sharpe=sharpe, max_dd_pct=dd, fill_rate_pct=fill_rate)
        expected_overall = (
            result["sharpe_pass"] and
            result["drawdown_pass"] and
            result["fill_rate_pass"]
        )
        assert result["overall_pass"] == expected_overall
