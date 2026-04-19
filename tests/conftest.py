"""
SITARAM HFT — Test Fixtures
Shared constants, synthetic market data, and mock objects
used across unit and integration test suites.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Tuple

# ── Strategy Parameters (Run-15 validated) ───────────────────
AS_GAMMA        = 2.0        # risk aversion
AS_KAPPA        = 0.5        # order arrival rate
MAX_INV_BTC     = 0.01       # max inventory rails ±0.01 BTC
MAKER_FEE       = -0.0001    # rebate (negative = income)
TAKER_FEE       =  0.00055   # taker fee
FILL_WINDOW_MS  =  500       # fill window in milliseconds
INITIAL_CAPITAL =  100_000.0 # USDT
DAILY_LOSS_LIMIT= -2_000.0   # USDT

# Gate thresholds
SHARPE_TARGET   = 3.0
MAX_DD_PCT      = 5.0
FILL_RATE_MIN   = 5.0

# Regime
VOL_THRESHOLD   = 0.000045   # sigma threshold for HIGH_VOL regime

# ── Synthetic Order Book Helpers ──────────────────────────────

def make_orderbook(
    mid: float = 50_000.0,
    spread: float = 1.0,
    depth: int = 10,
    bid_skew: float = 0.0,   # positive = more bid volume (buy pressure)
) -> Dict:
    """
    Build a synthetic L2 order book dict with `depth` levels each side.
    bid_skew shifts volume imbalance: 0 = balanced, +1 = all bids, -1 = all asks.
    """
    best_bid = mid - spread / 2
    best_ask = mid + spread / 2

    bids, asks = [], []
    for i in range(depth):
        bid_price = round(best_bid - i * 0.5, 2)
        ask_price = round(best_ask + i * 0.5, 2)
        base_vol = 1.0 - i * 0.08          # volume tapers with depth
        bid_vol  = max(0.01, base_vol * (1 + bid_skew))
        ask_vol  = max(0.01, base_vol * (1 - bid_skew))
        bids.append([bid_price, round(bid_vol, 4)])
        asks.append([ask_price, round(ask_vol, 4)])

    return {
        "bids": bids,
        "asks": asks,
        "mid":  mid,
        "ts":   1_713_500_000_000,   # fixed ms timestamp
    }


def make_trade_tape(
    n: int = 100,
    mid: float = 50_000.0,
    vol_pct: float = 0.001,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate a synthetic public trade tape of `n` trades.
    vol_pct: price std as fraction of mid.
    """
    rng = np.random.default_rng(seed)
    prices = mid + rng.normal(0, mid * vol_pct, n)
    sizes  = rng.uniform(0.0001, 0.05, n)
    sides  = rng.choice(["Buy", "Sell"], n)
    ts_base = 1_713_500_000_000
    return [
        {
            "price": round(float(p), 2),
            "size":  round(float(s), 6),
            "side":  side,
            "ts":    ts_base + i * 100,
        }
        for i, (p, s, side) in enumerate(zip(prices, sizes, sides))
    ]


def make_volatility_window(
    n: int = 20,
    sigma: float = 0.00003,
    seed: int = 42,
) -> np.ndarray:
    """
    Return an array of log-return mid-price observations
    with specified realized volatility (sigma per tick).
    """
    rng = np.random.default_rng(seed)
    mid0 = 50_000.0
    log_returns = rng.normal(0, sigma, n)
    return log_returns


# ── Mock Redis ────────────────────────────────────────────────

class MockRedis:
    """In-memory Redis mock that covers the keys used by the engine."""

    def __init__(self):
        self._store: Dict[str, str] = {}

    def set(self, key, value, ex=None):
        self._store[key] = str(value)

    def get(self, key):
        v = self._store.get(key)
        return v.encode() if v is not None else None

    def hset(self, name, mapping=None, **kwargs):
        m = mapping or kwargs
        for k, v in m.items():
            self._store[f"{name}:{k}"] = str(v)

    def hget(self, name, key):
        v = self._store.get(f"{name}:{key}")
        return v.encode() if v is not None else None

    def hgetall(self, name):
        prefix = f"{name}:"
        return {
            k[len(prefix):].encode(): v.encode()
            for k, v in self._store.items()
            if k.startswith(prefix)
        }

    def ping(self):
        return True

    def flushdb(self):
        self._store.clear()


# ── Pytest Fixtures ───────────────────────────────────────────

@pytest.fixture
def mock_redis():
    return MockRedis()


@pytest.fixture
def balanced_book():
    return make_orderbook(mid=50_000.0, spread=1.0, bid_skew=0.0)


@pytest.fixture
def bid_heavy_book():
    """Strong buy pressure — imbalance should be positive."""
    return make_orderbook(mid=50_000.0, spread=1.0, bid_skew=0.8)


@pytest.fixture
def ask_heavy_book():
    """Strong sell pressure — imbalance should be negative."""
    return make_orderbook(mid=50_000.0, spread=1.0, bid_skew=-0.8)


@pytest.fixture
def crossed_book():
    """Crossed order book — best bid > best ask (invalid state)."""
    return {
        "bids": [[50_001.0, 1.0], [50_000.5, 0.5]],
        "asks": [[49_999.0, 1.0], [49_998.5, 0.5]],
        "mid":  50_000.0,
        "ts":   1_713_500_000_000,
    }


@pytest.fixture
def trade_tape():
    return make_trade_tape(n=200, mid=50_000.0, vol_pct=0.001)


@pytest.fixture
def high_vol_returns():
    """Returns above VOL_THRESHOLD — should trigger HIGH_VOL regime."""
    return make_volatility_window(n=20, sigma=0.00006)


@pytest.fixture
def low_vol_returns():
    """Returns below VOL_THRESHOLD — should stay NORMAL regime."""
    return make_volatility_window(n=20, sigma=0.00002)
