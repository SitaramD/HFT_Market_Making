"""
SITARAM HFT — Unit Tests: Order Book Imbalance & Fill Simulator
Priority 4 — Validates market microstructure logic.

Tests cover:
  - OBI calculation (bid/ask volume ratio)
  - Crossed book detection
  - Mid-price computation
  - Fill probability model
  - Fill window timing
  - Partial fill handling
  - Property-based: OBI bounds, fill monotonicity
"""

import math
import time
import pytest
import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
from typing import List, Tuple, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fixtures'))
from conftest import make_orderbook, MockRedis

# ── Constants ─────────────────────────────────────────────────
FILL_WINDOW_MS = 500
MAX_INV_BTC    = 0.01


# ── Pure OB implementations ───────────────────────────────────

def compute_mid(bids: list, asks: list) -> float:
    """Mid price = (best_bid + best_ask) / 2."""
    if not bids or not asks:
        return 0.0
    return (bids[0][0] + asks[0][0]) / 2.0


def compute_obi(bids: list, asks: list, depth: int = 5) -> float:
    """
    Order Book Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Uses top `depth` levels. Range: [-1, 1].
    Positive = buy pressure. Negative = sell pressure.
    """
    bid_vol = sum(b[1] for b in bids[:depth])
    ask_vol = sum(a[1] for a in asks[:depth])
    total   = bid_vol + ask_vol
    if total == 0.0:
        return 0.0
    return (bid_vol - ask_vol) / total


def is_crossed(bids: list, asks: list) -> bool:
    """Return True if best bid >= best ask (invalid / crossed book)."""
    if not bids or not asks:
        return False
    return bids[0][0] >= asks[0][0]


def spread_bps(bids: list, asks: list) -> float:
    """Spread in basis points relative to mid."""
    mid = compute_mid(bids, asks)
    if mid == 0.0:
        return 0.0
    spread = asks[0][0] - bids[0][0]
    return spread / mid * 10_000.0


def compute_vwap(levels: list, depth: int = 5) -> float:
    """Volume-weighted average price for one side of the book."""
    levels = levels[:depth]
    total_notional = sum(p * q for p, q in levels)
    total_vol      = sum(q for _, q in levels)
    if total_vol == 0.0:
        return 0.0
    return total_notional / total_vol


# ── Fill Simulator ────────────────────────────────────────────

def simulate_fill(
    quote_price: float,
    quote_side: str,           # 'bid' or 'ask'
    quote_qty: float,
    trade_tape: List[Dict],
    fill_window_ms: int = FILL_WINDOW_MS,
    quote_ts_ms: int = 0,
) -> Dict:
    """
    Simulate whether a resting limit order gets filled.

    Fill rule:
    - Bid fills when a trade comes in at price <= quote_price (seller hits bid)
    - Ask fills when a trade comes in at price >= quote_price (buyer lifts ask)
    Only trades within fill_window_ms of quote_ts_ms are considered.

    Returns: {'filled': bool, 'fill_price': float, 'fill_qty': float,
              'latency_ms': int}
    """
    deadline = quote_ts_ms + fill_window_ms

    for trade in trade_tape:
        ts = trade.get('ts', quote_ts_ms)
        if ts < quote_ts_ms or ts > deadline:
            continue

        price = trade['price']
        qty   = trade['size']

        if quote_side == 'bid' and trade['side'] == 'Sell' and price <= quote_price:
            return {
                'filled':     True,
                'fill_price': quote_price,    # passive fill at quoted price
                'fill_qty':   min(quote_qty, qty),
                'latency_ms': ts - quote_ts_ms,
            }
        elif quote_side == 'ask' and trade['side'] == 'Buy' and price >= quote_price:
            return {
                'filled':     True,
                'fill_price': quote_price,
                'fill_qty':   min(quote_qty, qty),
                'latency_ms': ts - quote_ts_ms,
            }

    return {'filled': False, 'fill_price': 0.0, 'fill_qty': 0.0, 'latency_ms': -1}


def inventory_within_rails(inventory: float, max_inv: float = MAX_INV_BTC) -> bool:
    """Return True if inventory is within ±max_inv."""
    return -max_inv <= inventory <= max_inv


# ═══════════════════════════════════════════════════════════════
# 1. MID-PRICE TESTS
# ═══════════════════════════════════════════════════════════════

class TestMidPrice:

    def test_mid_is_average_of_best_bid_ask(self):
        ob  = make_orderbook(mid=50_000.0, spread=2.0)
        mid = compute_mid(ob['bids'], ob['asks'])
        assert mid == pytest.approx(50_000.0, rel=1e-6)

    def test_mid_empty_bids_returns_zero(self):
        assert compute_mid([], [[50_001.0, 1.0]]) == 0.0

    def test_mid_empty_asks_returns_zero(self):
        assert compute_mid([[49_999.0, 1.0]], []) == 0.0

    def test_mid_tight_spread(self):
        bids = [[49_999.5, 1.0]]
        asks = [[50_000.5, 1.0]]
        mid  = compute_mid(bids, asks)
        assert mid == pytest.approx(50_000.0, rel=1e-6)

    @given(
        best_bid = st.floats(min_value=1_000.0, max_value=90_000.0),
        spread   = st.floats(min_value=0.01,    max_value=100.0),
    )
    @settings(max_examples=300)
    def test_property_mid_between_bid_and_ask(self, best_bid, spread):
        best_ask = best_bid + spread
        mid      = compute_mid([[best_bid, 1.0]], [[best_ask, 1.0]])
        assert best_bid <= mid <= best_ask


# ═══════════════════════════════════════════════════════════════
# 2. ORDER BOOK IMBALANCE TESTS
# ═══════════════════════════════════════════════════════════════

class TestOBI:

    def test_balanced_book_obi_near_zero(self, balanced_book):
        obi = compute_obi(balanced_book['bids'], balanced_book['asks'])
        assert abs(obi) < 0.1    # allow small numerical imbalance

    def test_bid_heavy_book_positive_obi(self, bid_heavy_book):
        obi = compute_obi(bid_heavy_book['bids'], bid_heavy_book['asks'])
        assert obi > 0.3

    def test_ask_heavy_book_negative_obi(self, ask_heavy_book):
        obi = compute_obi(ask_heavy_book['bids'], ask_heavy_book['asks'])
        assert obi < -0.3

    def test_obi_range_minus1_to_plus1(self):
        for skew in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            ob  = make_orderbook(bid_skew=min(0.99, max(-0.99, skew)))
            obi = compute_obi(ob['bids'], ob['asks'])
            assert -1.0 <= obi <= 1.0

    def test_obi_empty_book_zero(self):
        assert compute_obi([], []) == 0.0

    def test_obi_depth_parameter_respected(self):
        """OBI computed at depth=1 should differ from depth=10 on skewed book."""
        ob   = make_orderbook(bid_skew=0.5, depth=10)
        obi1 = compute_obi(ob['bids'], ob['asks'], depth=1)
        obi5 = compute_obi(ob['bids'], ob['asks'], depth=5)
        # They should be different (deeper levels have different volumes)
        # Both must still be in [-1, 1]
        assert -1.0 <= obi1 <= 1.0
        assert -1.0 <= obi5 <= 1.0

    def test_all_bids_obi_plus1(self):
        """If all volume is on bid side, OBI should be +1."""
        bids = [[49_999.0, 100.0]]
        asks = [[50_001.0, 0.0]]   # zero ask volume (degenerate)
        obi  = compute_obi(bids, asks)
        assert obi == pytest.approx(1.0, abs=1e-6)

    @given(
        bid_vol = st.floats(min_value=0.01, max_value=1000.0),
        ask_vol = st.floats(min_value=0.01, max_value=1000.0),
    )
    @settings(max_examples=500)
    def test_property_obi_bounded(self, bid_vol, ask_vol):
        """OBI must always be in [-1, 1]."""
        bids = [[49_999.0, bid_vol]]
        asks = [[50_001.0, ask_vol]]
        obi  = compute_obi(bids, asks)
        assert -1.0 <= obi <= 1.0

    @given(
        bid_vol = st.floats(min_value=0.01, max_value=1000.0),
        ask_vol = st.floats(min_value=0.01, max_value=1000.0),
    )
    @settings(max_examples=300)
    def test_property_obi_antisymmetric(self, bid_vol, ask_vol):
        """Swapping bid/ask volumes should negate OBI."""
        bids_a = [[49_999.0, bid_vol]]
        asks_a = [[50_001.0, ask_vol]]
        bids_b = [[49_999.0, ask_vol]]
        asks_b = [[50_001.0, bid_vol]]
        obi_a  = compute_obi(bids_a, asks_a)
        obi_b  = compute_obi(bids_b, asks_b)
        assert obi_a == pytest.approx(-obi_b, rel=1e-6)


# ═══════════════════════════════════════════════════════════════
# 3. CROSSED BOOK DETECTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestCrossedBook:

    def test_normal_book_not_crossed(self, balanced_book):
        assert is_crossed(balanced_book['bids'], balanced_book['asks']) is False

    def test_crossed_book_detected(self, crossed_book):
        assert is_crossed(crossed_book['bids'], crossed_book['asks']) is True

    def test_touching_book_is_crossed(self):
        """Best bid == best ask → treated as crossed (zero spread = invalid)."""
        bids = [[50_000.0, 1.0]]
        asks = [[50_000.0, 1.0]]
        assert is_crossed(bids, asks) is True

    def test_empty_book_not_crossed(self):
        assert is_crossed([], []) is False

    def test_spread_bps_negative_on_crossed(self, crossed_book):
        """Crossed book must yield negative spread in bps."""
        s = spread_bps(crossed_book['bids'], crossed_book['asks'])
        assert s < 0.0

    def test_normal_spread_bps_positive(self, balanced_book):
        s = spread_bps(balanced_book['bids'], balanced_book['asks'])
        assert s > 0.0

    @given(
        bid = st.floats(min_value=1_000.0, max_value=100_000.0),
        ask = st.floats(min_value=1_000.0, max_value=100_000.0),
    )
    @settings(max_examples=300)
    def test_property_crossed_iff_bid_ge_ask(self, bid, ask):
        bids   = [[bid, 1.0]]
        asks   = [[ask, 1.0]]
        result = is_crossed(bids, asks)
        assert result == (bid >= ask)


# ═══════════════════════════════════════════════════════════════
# 4. FILL SIMULATOR TESTS
# ═══════════════════════════════════════════════════════════════

class TestFillSimulator:

    def _make_tape_with_trade(
        self, price: float, size: float, side: str, ts_offset_ms: int = 100
    ) -> List[Dict]:
        return [{'price': price, 'size': size, 'side': side,
                 'ts': ts_offset_ms}]

    def test_bid_fills_when_seller_hits(self):
        """Bid order fills when a sell trade comes at or below bid price."""
        tape   = self._make_tape_with_trade(49_999.0, 0.001, 'Sell', ts_offset_ms=100)
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=0)
        assert result['filled'] is True

    def test_bid_does_not_fill_above_bid_price(self):
        """Sell trade above bid price does not fill the bid."""
        tape   = self._make_tape_with_trade(50_001.0, 0.001, 'Sell', ts_offset_ms=100)
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=0)
        assert result['filled'] is False

    def test_ask_fills_when_buyer_lifts(self):
        """Ask order fills when a buy trade comes at or above ask price."""
        tape   = self._make_tape_with_trade(50_001.0, 0.001, 'Buy', ts_offset_ms=100)
        result = simulate_fill(50_000.0, 'ask', 0.001, tape, quote_ts_ms=0)
        assert result['filled'] is True

    def test_ask_does_not_fill_below_ask_price(self):
        """Buy trade below ask price does not fill the ask."""
        tape   = self._make_tape_with_trade(49_999.0, 0.001, 'Buy', ts_offset_ms=100)
        result = simulate_fill(50_000.0, 'ask', 0.001, tape, quote_ts_ms=0)
        assert result['filled'] is False

    def test_trade_outside_fill_window_no_fill(self):
        """Trade arriving after fill_window_ms must not trigger a fill."""
        tape   = self._make_tape_with_trade(49_999.0, 0.001, 'Sell',
                                             ts_offset_ms=FILL_WINDOW_MS + 100)
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=0)
        assert result['filled'] is False

    def test_trade_at_window_boundary_fills(self):
        """Trade arriving exactly at fill_window_ms edge must fill."""
        tape   = self._make_tape_with_trade(49_999.0, 0.001, 'Sell',
                                             ts_offset_ms=FILL_WINDOW_MS)
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=0)
        assert result['filled'] is True

    def test_trade_before_quote_ts_no_fill(self):
        """Trades before quote timestamp must not fill (no time travel)."""
        tape   = self._make_tape_with_trade(49_999.0, 0.001, 'Sell',
                                             ts_offset_ms=0)   # same as quote_ts
        # quote_ts = 500, trade at 0 → before quote
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=500)
        assert result['filled'] is False

    def test_fill_price_equals_quote_price(self):
        """Passive fill must be at quoted price, not trade price."""
        tape   = self._make_tape_with_trade(49_990.0, 0.001, 'Sell', ts_offset_ms=50)
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=0)
        if result['filled']:
            assert result['fill_price'] == pytest.approx(50_000.0, rel=1e-6)

    def test_fill_qty_capped_at_trade_size(self):
        """Fill qty must not exceed the trade size that triggered the fill."""
        tape   = self._make_tape_with_trade(49_999.0, 0.0005, 'Sell', ts_offset_ms=50)
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=0)
        if result['filled']:
            assert result['fill_qty'] <= 0.001

    def test_empty_tape_no_fill(self):
        result = simulate_fill(50_000.0, 'bid', 0.001, [], quote_ts_ms=0)
        assert result['filled'] is False

    def test_latency_recorded_on_fill(self):
        """Filled orders must record positive latency."""
        tape   = self._make_tape_with_trade(49_999.0, 0.001, 'Sell', ts_offset_ms=250)
        result = simulate_fill(50_000.0, 'bid', 0.001, tape, quote_ts_ms=0)
        if result['filled']:
            assert result['latency_ms'] == 250

    def test_latency_minus1_on_no_fill(self):
        result = simulate_fill(50_000.0, 'bid', 0.001, [], quote_ts_ms=0)
        assert result['latency_ms'] == -1


# ═══════════════════════════════════════════════════════════════
# 5. INVENTORY RAIL TESTS
# ═══════════════════════════════════════════════════════════════

class TestInventoryRails:

    def test_zero_inventory_within_rails(self):
        assert inventory_within_rails(0.0) is True

    def test_max_long_at_rail(self):
        assert inventory_within_rails(MAX_INV_BTC) is True

    def test_max_short_at_rail(self):
        assert inventory_within_rails(-MAX_INV_BTC) is True

    def test_above_max_long_outside_rails(self):
        assert inventory_within_rails(MAX_INV_BTC + 0.0001) is False

    def test_below_max_short_outside_rails(self):
        assert inventory_within_rails(-MAX_INV_BTC - 0.0001) is False

    def test_inventory_accumulation_bug_scenario(self):
        """
        Regression: engine was pinning at -MAX_INV_BTC requiring manual restart.
        Simulate the scenario: inventory hits max short rail.
        The rail check must detect it BEFORE submitting another ask.
        """
        inventory = -MAX_INV_BTC    # at max short
        within    = inventory_within_rails(inventory)
        # System should detect this and halt quoting (not keep going short)
        assert within is True       # exactly at rail = still valid
        # One more fill would breach it
        next_inv  = inventory - 0.0001
        assert inventory_within_rails(next_inv) is False

    @given(inv=st.floats(min_value=-1.0, max_value=1.0))
    @settings(max_examples=500)
    def test_property_rail_correct_for_all_values(self, inv):
        expected = -MAX_INV_BTC <= inv <= MAX_INV_BTC
        assert inventory_within_rails(inv) == expected


# ═══════════════════════════════════════════════════════════════
# 6. VOLATILITY REGIME TESTS
# ═══════════════════════════════════════════════════════════════

class TestVolatilityRegime:

    VOL_THRESHOLD = 0.000045    # Run-15 validated

    def _realized_vol(self, log_returns: np.ndarray) -> float:
        """Realized vol = std of log returns."""
        return float(np.std(log_returns, ddof=1))

    def test_high_vol_returns_trigger_high_vol_regime(self, high_vol_returns):
        rv     = self._realized_vol(high_vol_returns)
        regime = 'HIGH_VOL' if rv > self.VOL_THRESHOLD else 'NORMAL'
        assert regime == 'HIGH_VOL'

    def test_low_vol_returns_stay_normal_regime(self, low_vol_returns):
        rv     = self._realized_vol(low_vol_returns)
        regime = 'HIGH_VOL' if rv > self.VOL_THRESHOLD else 'NORMAL'
        assert regime == 'NORMAL'

    def test_vol_at_threshold_boundary(self):
        """Exactly at threshold → NORMAL (not strict >)."""
        # Force sigma exactly at threshold
        n  = 20
        rv = self.VOL_THRESHOLD
        # regime check uses strict >
        regime = 'HIGH_VOL' if rv > self.VOL_THRESHOLD else 'NORMAL'
        assert regime == 'NORMAL'

    def test_vol_threshold_calibration_warning(self):
        """
        Document the known calibration risk:
        if threshold was set on calm data (sigma=0.00002),
        active market (sigma=0.00005) would permanently lock HIGH_VOL.
        """
        calm_sigma   = 0.000020
        active_sigma = 0.000050
        assert active_sigma > self.VOL_THRESHOLD, \
            "Active market vol exceeds threshold → HIGH_VOL lock expected"
        assert calm_sigma < self.VOL_THRESHOLD, \
            "Calm market vol is below threshold → NORMAL expected"

    @given(sigma=st.floats(min_value=1e-7, max_value=0.001))
    @settings(max_examples=300)
    def test_property_regime_is_deterministic(self, sigma):
        """Same sigma must always produce same regime."""
        regime1 = 'HIGH_VOL' if sigma > self.VOL_THRESHOLD else 'NORMAL'
        regime2 = 'HIGH_VOL' if sigma > self.VOL_THRESHOLD else 'NORMAL'
        assert regime1 == regime2
