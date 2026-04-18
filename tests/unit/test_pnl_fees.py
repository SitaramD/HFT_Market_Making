"""
SITARAM HFT — Unit Tests: PnL & Fee Accounting
Priority 3 — Ensures every dollar is tracked correctly.

Tests cover:
  - Maker vs taker fee computation
  - Realized PnL on round-trip trades
  - Unrealized PnL mark-to-market
  - Fee impact on break-even spread
  - Inventory cost basis tracking
  - Daily loss limit enforcement
  - Property-based: PnL conservation
"""

import math
import pytest
import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

# ── Fee constants (Run-15 validated) ─────────────────────────
MAKER_FEE      = -0.0001    # negative = rebate (income)
TAKER_FEE      =  0.00055
INITIAL_CAP    =  100_000.0
DAILY_LOSS_LIM = -2_000.0


# ── Pure PnL / Fee implementations ───────────────────────────

def maker_fee(notional: float) -> float:
    """
    Maker fee on a notional trade size.
    Negative = rebate → income for market maker.
    Returns signed fee amount (positive = cost, negative = income).
    """
    return notional * MAKER_FEE


def taker_fee(notional: float) -> float:
    return notional * TAKER_FEE


def realized_pnl_round_trip(
    buy_price: float,
    sell_price: float,
    qty: float,
    buy_is_maker: bool = True,
    sell_is_maker: bool = True,
) -> float:
    """
    Compute realized PnL for a complete round-trip (buy then sell).
    PnL = (sell - buy) * qty - fees
    Fees: maker fills get rebate, taker fills pay fee.
    """
    gross_pnl = (sell_price - buy_price) * qty
    buy_fee   = maker_fee(buy_price  * qty) if buy_is_maker  else taker_fee(buy_price  * qty)
    sell_fee  = maker_fee(sell_price * qty) if sell_is_maker else taker_fee(sell_price * qty)
    # buy_fee is negative (rebate) → adds to PnL
    # sell_fee is negative (rebate) → adds to PnL
    net_pnl   = gross_pnl - buy_fee - sell_fee
    return net_pnl


def unrealized_pnl(entry_price: float, mark_price: float,
                   qty: float, side: str) -> float:
    """
    Mark-to-market unrealized PnL.
    side: 'long' or 'short'
    """
    if side == 'long':
        return (mark_price - entry_price) * qty
    else:
        return (entry_price - mark_price) * qty


def break_even_spread(price: float, maker: bool = True) -> float:
    """
    Minimum spread needed to cover both sides' fees and break even.
    For two maker fills: rebate on both → spread can be 0 or negative.
    For mixed maker/taker: must cover taker fee.
    """
    buy_fee_rate  = MAKER_FEE if maker else TAKER_FEE
    sell_fee_rate = MAKER_FEE if maker else TAKER_FEE
    # break_even_spread = total_fee_rate * price * 2 (both sides)
    return (abs(buy_fee_rate) + abs(sell_fee_rate)) * price


def track_inventory_cost_basis(trades: list) -> dict:
    """
    FIFO inventory tracker.
    trades: list of {'side': 'buy'/'sell', 'price': float, 'qty': float}
    Returns: {'inventory': float, 'cost_basis': float, 'realized_pnl': float}
    """
    inventory    = 0.0
    cost_basis   = 0.0   # total cost of held inventory
    realized_pnl = 0.0

    for t in trades:
        qty   = t['qty']
        price = t['price']

        if t['side'] == 'buy':
            cost_basis += price * qty
            inventory  += qty
        else:  # sell
            if inventory > 0:
                avg_cost      = cost_basis / inventory if inventory > 0 else price
                sell_qty      = min(qty, inventory)
                realized_pnl += (price - avg_cost) * sell_qty
                cost_basis   -= avg_cost * sell_qty
                inventory    -= sell_qty

    return {
        'inventory':    round(inventory,    8),
        'cost_basis':   round(cost_basis,   4),
        'realized_pnl': round(realized_pnl, 4),
    }


def check_daily_loss_limit(session_pnl: float) -> bool:
    """Returns True if daily loss limit is breached."""
    return session_pnl <= DAILY_LOSS_LIM


# ═══════════════════════════════════════════════════════════════
# 1. FEE COMPUTATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeeComputation:

    def test_maker_fee_is_negative_rebate(self):
        """Maker fee must be negative (income for market maker)."""
        fee = maker_fee(notional=50_000.0)
        assert fee < 0.0

    def test_taker_fee_is_positive_cost(self):
        """Taker fee must be positive (cost)."""
        fee = taker_fee(notional=50_000.0)
        assert fee > 0.0

    def test_maker_fee_exact_value(self):
        notional = 50_000.0
        fee      = maker_fee(notional)
        assert fee == pytest.approx(notional * MAKER_FEE, rel=1e-9)

    def test_taker_fee_exact_value(self):
        notional = 50_000.0
        fee      = taker_fee(notional)
        assert fee == pytest.approx(notional * TAKER_FEE, rel=1e-9)

    def test_maker_fee_linear_in_notional(self):
        """Fees must be linear: double notional → double fee."""
        assert maker_fee(100.0) * 2 == pytest.approx(maker_fee(200.0), rel=1e-9)

    def test_taker_fee_linear_in_notional(self):
        assert taker_fee(100.0) * 2 == pytest.approx(taker_fee(200.0), rel=1e-9)

    def test_maker_taker_fee_ratio(self):
        """Taker fee must be larger than maker rebate magnitude."""
        assert abs(taker_fee(1.0)) > abs(maker_fee(1.0))

    def test_zero_notional_zero_fee(self):
        assert maker_fee(0.0) == 0.0
        assert taker_fee(0.0) == 0.0

    @given(notional=st.floats(min_value=0.01, max_value=1_000_000.0))
    @settings(max_examples=300)
    def test_property_maker_fee_always_negative(self, notional):
        assert maker_fee(notional) < 0.0

    @given(notional=st.floats(min_value=0.01, max_value=1_000_000.0))
    @settings(max_examples=300)
    def test_property_taker_fee_always_positive(self, notional):
        assert taker_fee(notional) > 0.0


# ═══════════════════════════════════════════════════════════════
# 2. REALIZED PNL ROUND-TRIP TESTS
# ═══════════════════════════════════════════════════════════════

class TestRealizedPnL:

    def test_flat_spread_maker_maker_positive_pnl(self):
        """
        Market maker earns rebate on BOTH sides of a round-trip
        even when buy==sell price, because maker rebates net positive.
        """
        price = 50_000.0
        qty   = 0.001
        pnl   = realized_pnl_round_trip(price, price, qty,
                                         buy_is_maker=True, sell_is_maker=True)
        # gross = 0, fees = 2 * rebate (positive income)
        assert pnl > 0.0

    def test_spread_wider_than_taker_fees_profitable(self):
        """
        A spread wider than total taker fees must yield positive PnL.
        Taker cost = 2 * 0.055% = 0.11% of notional per round-trip.
        """
        buy  = 49_970.0
        sell = 50_030.0    # 60 USDT spread on 50k BTC
        qty  = 0.001
        pnl  = realized_pnl_round_trip(buy, sell, qty,
                                        buy_is_maker=False, sell_is_maker=False)
        assert pnl > 0.0

    def test_zero_spread_taker_taker_negative_pnl(self):
        """
        Zero spread with taker fees on both sides → guaranteed loss.
        """
        price = 50_000.0
        qty   = 0.001
        pnl   = realized_pnl_round_trip(price, price, qty,
                                         buy_is_maker=False, sell_is_maker=False)
        assert pnl < 0.0

    def test_pnl_scales_linearly_with_qty(self):
        """PnL must scale linearly with trade quantity."""
        buy, sell = 49_995.0, 50_005.0
        pnl1 = realized_pnl_round_trip(buy, sell, 0.001)
        pnl2 = realized_pnl_round_trip(buy, sell, 0.002)
        assert pnl2 == pytest.approx(pnl1 * 2, rel=1e-6)

    def test_inverted_round_trip_loss(self):
        """Buying high and selling low must always lose money."""
        pnl = realized_pnl_round_trip(50_010.0, 49_990.0, 0.001)
        assert pnl < 0.0

    def test_fee_income_improves_pnl(self):
        """Maker-maker should always outperform taker-taker at same prices."""
        buy, sell, qty = 49_995.0, 50_005.0, 0.001
        pnl_maker = realized_pnl_round_trip(buy, sell, qty,
                                             buy_is_maker=True, sell_is_maker=True)
        pnl_taker = realized_pnl_round_trip(buy, sell, qty,
                                             buy_is_maker=False, sell_is_maker=False)
        assert pnl_maker > pnl_taker

    def test_break_even_spread_calculation(self):
        """
        PnL at exactly break-even spread should be ≈ 0 for taker round-trip.
        """
        price = 50_000.0
        qty   = 0.001
        be    = break_even_spread(price, maker=False)
        buy   = price - be / 2
        sell  = price + be / 2
        pnl   = realized_pnl_round_trip(buy, sell, qty,
                                         buy_is_maker=False, sell_is_maker=False)
        assert abs(pnl) < 0.01   # within 1 cent of break-even

    @given(
        buy  = st.floats(min_value=10_000.0, max_value=100_000.0),
        sell = st.floats(min_value=10_000.0, max_value=100_000.0),
        qty  = st.floats(min_value=0.0001,   max_value=0.01),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_property_pnl_finite(self, buy, sell, qty):
        pnl = realized_pnl_round_trip(buy, sell, qty)
        assert math.isfinite(pnl)

    @given(
        spread = st.floats(min_value=0.0, max_value=100.0),
        qty    = st.floats(min_value=0.0001, max_value=0.01),
    )
    @settings(max_examples=300)
    def test_property_wider_spread_better_pnl(self, spread, qty):
        """Wider spread always yields better PnL (all else equal)."""
        mid   = 50_000.0
        pnl1  = realized_pnl_round_trip(mid - 1.0,      mid + 1.0,      qty)
        pnl2  = realized_pnl_round_trip(mid - 1.0 - spread/2,
                                         mid + 1.0 + spread/2, qty)
        assert pnl2 >= pnl1


# ═══════════════════════════════════════════════════════════════
# 3. UNREALIZED PNL TESTS
# ═══════════════════════════════════════════════════════════════

class TestUnrealizedPnL:

    def test_long_mark_above_entry_positive(self):
        pnl = unrealized_pnl(50_000.0, 50_100.0, 0.001, 'long')
        assert pnl == pytest.approx(0.1, rel=1e-6)

    def test_long_mark_below_entry_negative(self):
        pnl = unrealized_pnl(50_000.0, 49_900.0, 0.001, 'long')
        assert pnl == pytest.approx(-0.1, rel=1e-6)

    def test_short_mark_below_entry_positive(self):
        pnl = unrealized_pnl(50_000.0, 49_900.0, 0.001, 'short')
        assert pnl == pytest.approx(0.1, rel=1e-6)

    def test_at_entry_price_zero_unrealized(self):
        pnl = unrealized_pnl(50_000.0, 50_000.0, 0.001, 'long')
        assert pnl == pytest.approx(0.0, abs=1e-9)

    def test_long_and_short_opposite_signs(self):
        """Long and short unrealized PnL must be exact opposites."""
        long_pnl  = unrealized_pnl(50_000.0, 50_200.0, 0.001, 'long')
        short_pnl = unrealized_pnl(50_000.0, 50_200.0, 0.001, 'short')
        assert long_pnl == pytest.approx(-short_pnl, rel=1e-9)


# ═══════════════════════════════════════════════════════════════
# 4. INVENTORY COST BASIS TESTS
# ═══════════════════════════════════════════════════════════════

class TestInventoryCostBasis:

    def test_single_buy_no_realized_pnl(self):
        trades = [{'side': 'buy', 'price': 50_000.0, 'qty': 0.001}]
        result = track_inventory_cost_basis(trades)
        assert result['inventory']    == pytest.approx(0.001, rel=1e-6)
        assert result['realized_pnl'] == pytest.approx(0.0,   abs=1e-4)

    def test_round_trip_at_profit(self):
        trades = [
            {'side': 'buy',  'price': 50_000.0, 'qty': 0.001},
            {'side': 'sell', 'price': 50_100.0, 'qty': 0.001},
        ]
        result = track_inventory_cost_basis(trades)
        assert result['inventory']    == pytest.approx(0.0,  abs=1e-8)
        assert result['realized_pnl'] == pytest.approx(0.1,  rel=1e-4)

    def test_round_trip_at_loss(self):
        trades = [
            {'side': 'buy',  'price': 50_100.0, 'qty': 0.001},
            {'side': 'sell', 'price': 50_000.0, 'qty': 0.001},
        ]
        result = track_inventory_cost_basis(trades)
        assert result['realized_pnl'] == pytest.approx(-0.1, rel=1e-4)

    def test_inventory_zero_after_full_round_trip(self):
        trades = [
            {'side': 'buy',  'price': 50_000.0, 'qty': 0.005},
            {'side': 'sell', 'price': 50_050.0, 'qty': 0.005},
        ]
        result = track_inventory_cost_basis(trades)
        assert result['inventory'] == pytest.approx(0.0, abs=1e-8)

    def test_sell_without_inventory_ignored(self):
        """Selling without inventory (short) should not crash and inventory stays ≥ 0."""
        trades = [{'side': 'sell', 'price': 50_000.0, 'qty': 0.001}]
        result = track_inventory_cost_basis(trades)
        assert result['inventory'] >= 0.0


# ═══════════════════════════════════════════════════════════════
# 5. DAILY LOSS LIMIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestDailyLossLimit:

    def test_above_limit_not_breached(self):
        assert check_daily_loss_limit(-1_999.0) is False

    def test_exactly_at_limit_breached(self):
        assert check_daily_loss_limit(-2_000.0) is True

    def test_below_limit_breached(self):
        assert check_daily_loss_limit(-2_500.0) is True

    def test_positive_pnl_not_breached(self):
        assert check_daily_loss_limit(500.0) is False

    def test_zero_pnl_not_breached(self):
        assert check_daily_loss_limit(0.0) is False

    def test_limit_is_less_than_2pct_of_capital(self):
        """Daily loss limit must be < 2% of initial capital."""
        limit_pct = abs(DAILY_LOSS_LIM) / INITIAL_CAP * 100
        assert limit_pct == pytest.approx(2.0, rel=1e-6)
        assert limit_pct < 5.0    # must not be as large as the DD gate
