# =============================================================================
# SITARAM HFT — Order Book Reconstructor
#
# Parses Bybit WebSocket NDJSON format exactly as observed in the data files:
#   {"topic":"orderbook.200.BTCUSDT","ts":1772323200336,"type":"snapshot",
#    "data":{"s":"BTCUSDT","b":[["price","qty"],...],"a":[...],"u":...,"seq":...},
#    "cts":...}
#
# type="snapshot" → full book replacement (first message per file)
# type="delta"    → incremental update (qty="0" means remove that level)
#
# ts field = exchange timestamp in milliseconds
# Price and quantity are strings → converted to float on insert
# =============================================================================

from sortedcontainers import SortedDict
from typing import Optional, Tuple


class OrderBook:
    """
    Maintains a live L2 order book by applying snapshot + delta messages.

    Internal storage:
      bids: SortedDict keyed by -price (so highest bid = first key)
      asks: SortedDict keyed by +price (so lowest ask = first key)

    Call apply(msg_dict) for each parsed JSON line.
    Returns True when the book is valid and ready to quote from.
    """

    __slots__ = ('bids', 'asks', 'ts', 'cts', 'seq', 'ready')

    def __init__(self):
        self.bids:  SortedDict = SortedDict()   # {-price: qty}
        self.asks:  SortedDict = SortedDict()   # {+price: qty}
        self.ts:    int   = 0    # Exchange timestamp ms
        self.cts:   int   = 0    # Client/receive timestamp ms
        self.seq:   int   = 0    # Sequence number
        self.ready: bool  = False

    # ------------------------------------------------------------------
    def apply(self, msg: dict) -> bool:
        """
        Apply one parsed JSON message to the book.
        Returns True if the book is valid and has at least one bid and ask.
        """
        self.ts  = msg.get('ts',  0)
        self.cts = msg.get('cts', 0)

        data = msg.get('data')
        if not data:
            return False

        self.seq = data.get('seq', self.seq)

        mtype = msg.get('type', '')
        if mtype == 'snapshot':
            self._apply_snapshot(data)
        elif mtype == 'delta':
            if not self.ready:
                return False   # Haven't seen snapshot yet — skip delta
            self._apply_delta(data)
        else:
            return False

        return self.ready and len(self.bids) > 0 and len(self.asks) > 0

    # ------------------------------------------------------------------
    def _apply_snapshot(self, data: dict):
        self.bids.clear()
        self.asks.clear()

        for price_str, qty_str in data.get('b', []):
            qty = float(qty_str)
            if qty > 0:
                self.bids[-float(price_str)] = qty

        for price_str, qty_str in data.get('a', []):
            qty = float(qty_str)
            if qty > 0:
                self.asks[float(price_str)] = qty

        self.ready = True

    # ------------------------------------------------------------------
    def _apply_delta(self, data: dict):
        for price_str, qty_str in data.get('b', []):
            key = -float(price_str)
            qty = float(qty_str)
            if qty == 0:
                self.bids.pop(key, None)
            else:
                self.bids[key] = qty

        for price_str, qty_str in data.get('a', []):
            key = float(price_str)
            qty = float(qty_str)
            if qty == 0:
                self.asks.pop(key, None)
            else:
                self.asks[key] = qty

    # ------------------------------------------------------------------
    # DERIVED QUANTITIES
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        return -self.bids.keys()[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks.keys()[0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return (ba - bb) if bb is not None and ba is not None else None

    @property
    def valid(self) -> bool:
        """True if book has valid crossed-free bids and asks."""
        bb, ba = self.best_bid, self.best_ask
        return (bb is not None and ba is not None and ba > bb)

    def microprice(self) -> Optional[float]:
        """
        Volume-weighted average of best bid and ask prices.
        Better estimate of fair value than simple mid.
        micro = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
        """
        if not self.bids or not self.asks:
            return None
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        bv = self.bids[-bb]   # qty at best bid
        av = self.asks[ba]    # qty at best ask
        total = bv + av
        return (bb * av + ba * bv) / total if total > 0 else (bb + ba) / 2.0

    def obi(self, n_levels: int = 5) -> float:
        """
        Order Book Imbalance over top n_levels.
        OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Range: [-1, +1]. Positive = more bid pressure.
        """
        bid_vol = sum(v for v in list(self.bids.values())[:n_levels])
        ask_vol = sum(v for v in list(self.asks.values())[:n_levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def depth_at(self, n_levels: int) -> Tuple[float, float]:
        """Total BTC volume within n_levels on each side."""
        bid_depth = sum(list(self.bids.values())[:n_levels])
        ask_depth = sum(list(self.asks.values())[:n_levels])
        return bid_depth, ask_depth
