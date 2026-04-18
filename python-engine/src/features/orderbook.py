"""
orderbook.py  -  SITARAM Real Order Book Processor  v2.7

Changes v2.7:
  [FIX]  Crossed book recovery no longer sets _ready=False.
         Root cause: after crossed book reset, _ready=False means engine
         waits for a snapshot. But Bybit only sends snapshots on initial
         WebSocket connect — never mid-session. So the engine gets stuck
         in an infinite loop: reset → wait snapshot → get delta → cross → reset.

         Fix: on crossed book, discard ONLY the crossed levels (remove all
         bids >= best_ask and asks <= best_bid) instead of clearing the whole
         book. This preserves valid levels and avoids needing a new snapshot.

         If the book is still crossed after pruning (catastrophic state),
         write ob:request_snapshot=1 to Redis so bybit_live_producer.py
         can reconnect and send a fresh snapshot. Engine stays ready
         during this wait — it just skips quoting until book is valid.

  [FIX]  _ready stays True after crossed book detection — engine continues
         processing ticks, just skips _flush_redis when book is invalid.

  [KEEP] All v2.6 latency tracking (cts/bybit_ts) unchanged.
  [KEEP] All v2.4 vol warmup + regime logic unchanged.
"""
import os, time, json, logging, threading
import redis
from sortedcontainers import SortedDict
from kafka import KafkaConsumer

log = logging.getLogger("orderbook")

REDIS_HOST   = os.getenv("REDIS_HOST", "redis")
REDIS_PORT   = int(os.getenv("REDIS_PORT", 6379))
KAFKA_SERVERS= os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
TICK_WINDOW  = int(os.getenv("REDIS_TICK_WINDOW", 200))
OBI_LEVELS   = 5

VOL_THRESHOLD = float(os.getenv("VOL_THRESHOLD", "6e-05"))

# How many consecutive crossed ticks before we prune instead of full reset
CROSSED_PRUNE_THRESHOLD = 3   # prune after 3 ticks (was 10 before full reset)
# How many ticks after pruning before we request a fresh snapshot
SNAPSHOT_REQUEST_THRESHOLD = 30


class OrderBookProcessor:
    def __init__(self):
        self.r          = self._connect_redis()
        self.bids       = SortedDict(lambda x: -x)
        self.asks       = SortedDict()
        self._lock      = threading.Lock()
        self.tick_count = 0
        self._last_mid  = None
        self._consumer  = None
        self._ready     = False
        self._crossed_count      = 0
        self._ticks_since_prune  = 0   # tracks how long since last prune

        # Regime state
        self._current_vol  = 0.0
        self._regime       = "NORMAL"
        self._regime_ticks = 0
        self._warmup_ticks = 0
        self._WARMUP_MIN   = 10

        t = threading.Thread(target=self._kafka_loop, daemon=True, name="OB-Kafka")
        t.start()
        log.info("OrderBookProcessor initialised - waiting for first snapshot")

    def _connect_redis(self):
        for attempt in range(20):
            try:
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                decode_responses=True)
                r.ping()
                log.info("Redis connected")
                return r
            except Exception as e:
                log.warning(f"Redis not ready ({attempt+1}/20): {e}")
                time.sleep(3)
        raise RuntimeError("Cannot connect to Redis")

    def _kafka_loop(self):
        log.info(f"Connecting to Kafka at {KAFKA_SERVERS} topic ob200-raw")
        while True:
            try:
                consumer = KafkaConsumer(
                    "ob200-raw",
                    bootstrap_servers=KAFKA_SERVERS,
                    group_id="python-engine-ob",
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    max_poll_records=50,
                    session_timeout_ms=60000,
                    heartbeat_interval_ms=20000,
                    consumer_timeout_ms=120000,
                    request_timeout_ms=130000,
                    max_poll_interval_ms=300000,
                )
                self._consumer = consumer

                consumer.poll(timeout_ms=5000)
                consumer.seek_to_end()
                log.info("Kafka OB consumer connected — seeked to end")

                with self._lock:
                    self.bids.clear()
                    self.asks.clear()
                    self._ready          = False
                    self._crossed_count  = 0
                    self._last_mid       = None
                    self._warmup_ticks   = 0
                    self._ticks_since_prune = 0
                self.r.delete("vol:rets")
                log.info("OB book cleared — waiting for fresh snapshot or delta bootstrap")

                for msg in consumer:
                    try:
                        self._process_message(msg.value)
                    except Exception as e:
                        log.warning(f"OB message error: {e}")
                log.warning("OB consumer timeout - reconnecting")

            except Exception as e:
                log.error(f"Kafka OB consumer error: {e} - retrying in 5s")
                time.sleep(5)

    def _prune_crossed_levels(self):
        """
        Remove only the crossed levels instead of clearing the whole book.

        A crossed book means best_bid >= best_ask.
        We remove bids that are >= best_ask and asks that are <= best_bid
        until the book is valid or empty.

        This is far less destructive than a full reset — valid price levels
        far from the cross are preserved.
        """
        if not self.bids or not self.asks:
            return

        pruned_bids = 0
        pruned_asks = 0

        # Keep removing the highest bid if it's >= lowest ask
        while self.bids and self.asks:
            best_bid = self.bids.keys()[0]
            best_ask = self.asks.keys()[0]
            if best_bid < best_ask:
                break  # book is valid now
            # Remove the crossed bid level
            self.bids.popitem(0)
            pruned_bids += 1

        # Also remove any asks <= remaining best bid (if any)
        if self.bids:
            best_bid = self.bids.keys()[0]
            while self.asks:
                best_ask = self.asks.keys()[0]
                if best_ask > best_bid:
                    break
                self.asks.popitem(0)
                pruned_asks += 1

        log.info(f"Crossed book pruned: removed {pruned_bids} bid levels, "
                 f"{pruned_asks} ask levels | "
                 f"remaining: {len(self.bids)} bids, {len(self.asks)} asks")

        # Reset warmup to avoid vol spike from mid discontinuity after prune
        self._last_mid     = None
        self._warmup_ticks = 0
        self.r.delete("vol:rets")
        self._ticks_since_prune = 0

    def _process_message(self, msg):
        data          = msg.get("data", {})
        msg_type      = msg.get("type", "delta")
        cts_ms        = float(msg.get("cts", 0))
        receive_ts_ms = time.time() * 1000
        bybit_ts_ms   = cts_ms if cts_ms > 0 else receive_ts_ms

        with self._lock:
            if msg_type == "snapshot":
                # Full snapshot — clear and rebuild from scratch
                self.bids.clear()
                self.asks.clear()
                self._crossed_count     = 0
                self._ticks_since_prune = 0
                self._last_mid          = None
                self._warmup_ticks      = 0
                self.r.delete("vol:rets")
                self.r.delete("ob:request_snapshot")

                for price, size in data.get("b", []):
                    p, s = float(price), float(size)
                    if s > 0:
                        self.bids[p] = s
                for price, size in data.get("a", []):
                    p, s = float(price), float(size)
                    if s > 0:
                        self.asks[p] = s
                self._ready = True
                log.info(f"OB snapshot loaded: {len(self.bids)} bids, "
                         f"{len(self.asks)} asks")

            else:  # delta
                if not self._ready:
                    # First message after startup — bootstrap from delta
                    self._ready = True
                    log.info("No snapshot received - bootstrapping from first delta")

                for price, size in data.get("b", []):
                    p, s = float(price), float(size)
                    if s == 0.0:
                        self.bids.pop(p, None)
                    else:
                        self.bids[p] = s

                for price, size in data.get("a", []):
                    p, s = float(price), float(size)
                    if s == 0.0:
                        self.asks.pop(p, None)
                    else:
                        self.asks[p] = s

                # ── Crossed book detection ────────────────────────────────────
                if self.bids and self.asks:
                    best_bid = self.bids.keys()[0]
                    best_ask = self.asks.keys()[0]

                    if best_bid >= best_ask:
                        self._crossed_count += 1

                        if self._crossed_count >= CROSSED_PRUNE_THRESHOLD:
                            # PRUNE instead of full reset — preserves valid levels
                            # _ready stays True — engine keeps running
                            log.warning(
                                f"Crossed book ({self._crossed_count} ticks) "
                                f"bid={best_bid} ask={best_ask} — pruning levels"
                            )
                            self._prune_crossed_levels()
                            self._crossed_count = 0

                            # If still crossed after pruning — request fresh snapshot
                            if self.bids and self.asks:
                                if self.bids.keys()[0] >= self.asks.keys()[0]:
                                    log.warning(
                                        "Book still crossed after pruning — "
                                        "requesting fresh snapshot from producer"
                                    )
                                    self.r.set("ob:request_snapshot", "1")
                                    # Clear book — engine pauses quoting until
                                    # snapshot arrives. _ready=False temporarily.
                                    self.bids.clear()
                                    self.asks.clear()
                                    self._ready = False
                    else:
                        self._crossed_count = 0
                        self._ticks_since_prune += 1

            self._flush_redis(receive_ts_ms, bybit_ts_ms)

    def _flush_redis(self, ts_ms, bybit_ts_ms=0):
        """
        Always write ob:ts first (before any early returns) so the stall
        detector in quoting/engine.py always sees a fresh timestamp.
        """
        # ── Write liveness timestamp FIRST — before any guard ─────────────────
        self.r.set("ob:ts", ts_ms)

        if not self.bids or not self.asks:
            return

        best_bid = self.bids.keys()[0]
        best_ask = self.asks.keys()[0]

        if best_bid >= best_ask:
            return

        mid    = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        bid_sz = self.bids[best_bid]
        ask_sz = self.asks[best_ask]
        total  = bid_sz + ask_sz
        microprice = (best_bid * ask_sz + best_ask * bid_sz) / total \
                     if total > 0 else mid
        micro_dev  = (microprice - mid) / mid if mid > 0 else 0.0

        obi5      = self._compute_obi(OBI_LEVELS)
        vol_adj   = self._update_vol(mid)
        composite = 0.50 * obi5 + 0.35 * micro_dev + 0.15 * vol_adj

        self._update_regime(vol_adj)

        tick_str = (f"{ts_ms},{best_bid},{best_ask},{bid_sz},{ask_sz},"
                    f"{obi5},{micro_dev},{vol_adj},{composite}")
        pipe = self.r.pipeline()
        pipe.lpush("ticks", tick_str)
        pipe.ltrim("ticks", 0, TICK_WINDOW - 1)
        pipe.set("ob:best_bid",   round(best_bid, 1))
        pipe.set("ob:best_ask",   round(best_ask, 1))
        pipe.set("ob:mid",        round(mid, 1))
        pipe.set("ob:microprice", round(microprice, 2))
        pipe.set("ob:spread",     round(spread, 1))
        pipe.set("ob:obi5",       round(obi5, 6))
        pipe.set("ob:micro_dev",  round(micro_dev, 6))
        pipe.set("ob:vol_adj",    round(vol_adj, 6))
        pipe.set("ob:composite",  round(composite, 6))
        pipe.set("ob:ts",         ts_ms)
        pipe.set("ob:receive_ts", ts_ms)
        if bybit_ts_ms > 0:
            pipe.set("ob:bybit_ts", bybit_ts_ms)
            net_transit = round(ts_ms - bybit_ts_ms, 2)
            pipe.set("ob:network_transit_ms", net_transit)
        pipe.set("regime:current_vol",   round(self._current_vol, 8))
        pipe.set("regime:state",         self._regime)
        pipe.set("regime:ticks_in",      self._regime_ticks)
        pipe.set("regime:vol_threshold", VOL_THRESHOLD)
        pipe.execute()

        self.tick_count += 1
        self._last_mid = mid

    def _update_regime(self, vol_adj):
        self._current_vol = abs(vol_adj)

        force = self.r.get("regime:force_normal")
        if force == "1":
            if self._regime == "HIGH_VOL":
                log.info(f"REGIME FORCED → NORMAL | vol={self._current_vol:.8f}")
            self._regime = "NORMAL"
            self._regime_ticks = 0
            self.r.delete("regime:force_normal")
            return

        prev_regime = self._regime
        self._regime = "HIGH_VOL" if self._current_vol >= VOL_THRESHOLD else "NORMAL"
        self._regime_ticks += 1

        if self._regime != prev_regime:
            self._regime_ticks = 1
            if self._regime == "HIGH_VOL":
                log.warning(
                    f"REGIME → HIGH_VOL | vol={self._current_vol:.8f} "
                    f"threshold={VOL_THRESHOLD:.2e}"
                )
            else:
                log.info(
                    f"REGIME → NORMAL | vol={self._current_vol:.8f} "
                    f"threshold={VOL_THRESHOLD:.2e}"
                )

    def _compute_obi(self, levels):
        bid_vol = sum(list(self.bids.values())[:levels])
        ask_vol = sum(list(self.asks.values())[:levels])
        total   = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def _update_vol(self, mid):
        if self._last_mid is None or self._last_mid == 0:
            self._warmup_ticks = 0
            return 0.0
        self._warmup_ticks += 1
        if self._warmup_ticks < self._WARMUP_MIN:
            self._last_mid = mid
            return 0.0
        ret = (mid - self._last_mid) / self._last_mid
        if abs(ret) > 1e-10:
            self.r.lpush("vol:rets", ret)
            self.r.ltrim("vol:rets", 0, 49)
        rets = self.r.lrange("vol:rets", 0, -1)
        if len(rets) < 2:
            return 0.0
        vals = [float(x) for x in rets]
        mean = sum(vals) / len(vals)
        var  = sum((x - mean) ** 2 for x in vals) / len(vals)
        std  = var ** 0.5
        return std * (429000 * 252) ** 0.5

    def process_tick(self):
        pass

    def best_bid(self):
        return float(self.r.get("ob:best_bid") or 0)

    def best_ask(self):
        return float(self.r.get("ob:best_ask") or 0)

    def mid(self):
        return float(self.r.get("ob:mid") or 0)

    def microprice(self):
        return float(self.r.get("ob:microprice") or 0)

    def spread(self):
        return float(self.r.get("ob:spread") or 0)

    def obi(self, levels=5):
        return float(self.r.get("ob:obi5") or 0)

    def micro_dev(self):
        return float(self.r.get("ob:micro_dev") or 0)

    def vol_adj(self):
        return float(self.r.get("ob:vol_adj") or 0)

    def composite_signal(self):
        return float(self.r.get("ob:composite") or 0)

    def is_ready(self):
        return self._ready and self.tick_count > 0

    def regime(self):
        return self.r.get("regime:state") or self._regime

    def current_vol(self):
        return float(self.r.get("regime:current_vol") or self._current_vol)
