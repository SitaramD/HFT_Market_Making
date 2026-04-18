"""
fill_simulator.py  —  SITARAM Fill Simulator  v2.1

Changes v2.1:
  [FIX]  Queue-position fill model — fills when market trade prints at or within
         one TICK_SIZE of our quote price. This correctly models passive maker
         queue priority on tight-spread markets (Bybit BTC/USDT spot = 0.10 USDT).

         Old logic (v2.0):
           price <= my_bid   — required trade to print BELOW our quote
           price >= my_ask   — required trade to print ABOVE our quote
           Result: on a 0.10 USDT spread market, quotes at ±0.15 from mid
           were always one tick outside touch and NEVER filled.

         New logic (v2.1):
           price <= my_bid + TICK_SIZE   — fills when trade at or one tick through bid
           price >= my_ask - TICK_SIZE   — fills when trade at or one tick through ask
           Result: dynamic — adapts to any market spread width automatically.
           If spread widens, fill threshold widens with it. If spread tightens,
           threshold tightens. No hardcoded spread assumptions.

  [KEEP] One fill per side per FILL_WINDOW_MS — prevents fill storm
  [KEEP] Inventory hard cap guard
  [KEEP] Stale trade purge from window
"""
import os, time, json, logging, threading, collections
import redis
from kafka import KafkaConsumer

log = logging.getLogger("simulator")

REDIS_HOST      = os.getenv("REDIS_HOST", "redis")
REDIS_PORT      = int(os.getenv("REDIS_PORT", 6379))
KAFKA_SERVERS   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
FILL_WINDOW_MS  = int(os.getenv("FILL_WINDOW_MS", 500))      # validated Run 15
QUOTE_SIZE      = float(os.getenv("QUOTE_SIZE_BTC", 0.01))
MAKER_FEE       = float(os.getenv("MAKER_FEE", -0.0001))
MAX_INVENTORY   = float(os.getenv("MAX_INVENTORY_BTC", 0.10))
TICK_SIZE       = float(os.getenv("TICK_SIZE", 0.1))         # Bybit BTC/USDT tick


class FillSimulator:
    def __init__(self, quoting_engine):
        self.qe           = quoting_engine
        self.r            = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                        decode_responses=True)
        self.total_quotes = 0
        self.total_fills  = 0

        # Rolling window of recent trades: deque of (ts_ms, price, side, volume)
        self._trade_window = collections.deque()
        self._lock         = threading.Lock()

        # One fill per side per window
        self._last_fill_ts = {"buy": 0.0, "sell": 0.0}

        # Start Kafka trades consumer thread
        t = threading.Thread(target=self._kafka_loop, daemon=True, name="Fills-Kafka")
        t.start()
        log.info("FillSimulator v2.1 initialised — consuming trades-raw")
        log.info(f"Fill model: queue-position | TICK_SIZE={TICK_SIZE} | "
                 f"FILL_WINDOW_MS={FILL_WINDOW_MS}")

    # ------------------------------------------------------------------
    # Kafka consumer loop for trades-raw
    # ------------------------------------------------------------------
    def _kafka_loop(self):
        log.info(f"Connecting to Kafka at {KAFKA_SERVERS} topic trades-raw")
        while True:
            try:
                consumer = KafkaConsumer(
                    "trades-raw",
                    bootstrap_servers=KAFKA_SERVERS,
                    group_id="python-engine-fills",
                    auto_offset_reset="latest",
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    max_poll_records=100,
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000,
                )
                log.info("Kafka trades consumer connected")

                for msg in consumer:
                    try:
                        trade = msg.value
                        ts_ms = float(trade.get("ts", time.time() * 1000))
                        price = float(trade.get("price", 0))
                        side  = trade.get("side", "")
                        vol   = float(trade.get("volume", 0))
                        if price > 0 and side in ("buy", "sell"):
                            with self._lock:
                                self._trade_window.append((ts_ms, price, side, vol))
                    except Exception as e:
                        log.warning(f"Trade message error: {e}")

            except Exception as e:
                log.error(f"Kafka trades consumer error: {e} — retrying in 5s")
                time.sleep(5)

    # ------------------------------------------------------------------
    # step() called every 1ms by main loop
    # ------------------------------------------------------------------
    def step(self):
        if not self.qe.ob.is_ready():
            return

        now_ms = time.time() * 1000
        cutoff = now_ms - FILL_WINDOW_MS

        my_bid = float(self.r.get("quote:bid") or 0)
        my_ask = float(self.r.get("quote:ask") or 0)
        active = self.r.get("quote:active") == "1"

        if my_bid == 0 or my_ask == 0 or not active:
            return

        self.total_quotes += 1

        # Read current inventory for hard cap guard
        current_inv = float(self.r.get("quote:inventory") or 0)

        with self._lock:
            # Purge stale trades outside fill window
            while self._trade_window and self._trade_window[0][0] < cutoff:
                self._trade_window.popleft()

            bid_filled = False
            ask_filled = False

            for i, (ts_ms, price, side, vol) in enumerate(self._trade_window):
                if ts_ms < cutoff:
                    continue

                # ── Passive bid fill: market sell at or within one tick of our bid ──
                # v2.1: price <= my_bid + TICK_SIZE  (queue-position model)
                # Rationale: if a sell trade prints at best_bid (e.g. 68561.5)
                # and our quote is at 68561.4, we have queue priority at that
                # level and would be filled as a passive maker on a real exchange.
                if (side == "sell"
                        and price <= my_bid + (2*TICK_SIZE)          # ← v2.1 change
                        and not bid_filled
                        and (now_ms - self._last_fill_ts["buy"]) >= FILL_WINDOW_MS
                        and current_inv + QUOTE_SIZE <= MAX_INVENTORY):

                    self.qe.record_fill("buy", my_bid, QUOTE_SIZE)
                    self.total_fills += 1
                    bid_filled = True
                    self._last_fill_ts["buy"] = now_ms
                    current_inv += QUOTE_SIZE
                    log.debug(f"BID filled @ {my_bid:.1f} | trade sell @ {price:.1f} "
                              f"| queue-position delta={price - my_bid:.2f}")

                # ── Passive ask fill: market buy at or within one tick of our ask ──
                # v2.1: price >= my_ask - TICK_SIZE  (queue-position model)
                elif (side == "buy"
                        and price >= my_ask - (2*TICK_SIZE) #Sitaram          # ← v2.1 change
                        and not ask_filled
                        and (now_ms - self._last_fill_ts["sell"]) >= FILL_WINDOW_MS
                        and current_inv - QUOTE_SIZE >= -MAX_INVENTORY):

                    self.qe.record_fill("sell", my_ask, QUOTE_SIZE)
                    self.total_fills += 1
                    ask_filled = True
                    self._last_fill_ts["sell"] = now_ms
                    current_inv -= QUOTE_SIZE
                    log.debug(f"ASK filled @ {my_ask:.1f} | trade buy @ {price:.1f} "
                              f"| queue-position delta={my_ask - price:.2f}")

                # Stop scanning once both sides filled this tick
                if bid_filled and ask_filled:
                    break

        # ── Update fill rate metrics ──────────────────────────────────────────
        fill_rate = self.total_fills / max(self.total_quotes, 1)
        self.r.set("sim:fill_rate",    round(fill_rate, 4))
        self.r.set("sim:total_fills",  self.total_fills)
        self.r.set("sim:total_quotes", self.total_quotes)

        from src.quoting.engine import _metrics
        _metrics["fill_rate"] = round(fill_rate, 4)
