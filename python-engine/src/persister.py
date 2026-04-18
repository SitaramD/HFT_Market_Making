"""
persister.py — SITARAM TimescaleDB Persister v1.0

Subscribes to:
  - Redis pub/sub channel "sitaram:fills"  → writes to trades table
  - Kafka topic "ob200-raw"               → writes to order_book_ticks table
  - Redis keys for feature metrics        → writes to feature_metrics table

TimescaleDB tables (confirmed schemas):
  trades:           trade_time, symbol, side, price, quantity, realized_pnl,
                    cumulative_pnl, fill_latency_ms, adverse_fill,
                    inventory_after, quote_id, session_id, quotes_sent
  order_book_ticks: time, symbol, best_bid, best_ask, mid_price, spread,
                    bid_depth_10, ask_depth_10, update_id, session_id
  feature_metrics:  computed_at, symbol, metric_name, ic_value,
                    feature_value, horizon_ms, rolling_window, session_id

Architecture:
  - Thread 1: Redis subscriber for fills
  - Thread 2: Kafka consumer for OB ticks (sampled every 10 ticks = ~2s)
  - Thread 3: Feature metrics poller (every 30s from Redis)
  - Main thread: health check loop
"""

import os
import json
import time
import logging
import threading
import uuid
from datetime import datetime, timezone

import redis
import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("persister")

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST    = os.getenv("REDIS_HOST",    "redis")
REDIS_PORT    = int(os.getenv("REDIS_PORT", 6379))
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
SYMBOL        = os.getenv("SYMBOL", "BTCUSDT")

DB_HOST = os.getenv("DB_HOST", "timescaledb")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "sitaram")
DB_USER = os.getenv("DB_USER", "sitaram_user")
DB_PASS = os.getenv("DB_PASS", "sitaram_secure_2026")

OB_SAMPLE_EVERY  = int(os.getenv("OB_SAMPLE_EVERY", 5))
METRICS_POLL_SEC = int(os.getenv("METRICS_POLL_SEC", 30))


# ── DB connection factory ─────────────────────────────────────────────────────
def make_db_conn():
    while True:
        try:
            conn = psycopg2.connect(
                host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                user=DB_USER, password=DB_PASS,
                connect_timeout=10
            )
            conn.autocommit = False
            log.info(f"TimescaleDB connected: {DB_HOST}:{DB_PORT}/{DB_NAME}")
            return conn
        except Exception as e:
            log.error(f"DB connection failed: {e} — retrying in 5s")
            time.sleep(5)


# ── Thread 1: Fill persister via Redis pub/sub ────────────────────────────────
class FillPersister(threading.Thread):
    """
    Listens on Redis channel "sitaram:fills" for fill events published
    by record_fill() in engine.py, then writes each fill to trades table.
    """

    # ── FIXED: session_id and quotes_sent now included ────────────────────────
    INSERT_SQL = """
        INSERT INTO trades
            (trade_time, symbol, side, price, quantity, realized_pnl,
             cumulative_pnl, fill_latency_ms, adverse_fill,
             inventory_after, quote_id, session_id, quotes_sent)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    def __init__(self):
        super().__init__(name="FillPersister", daemon=True)
        self.r    = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                decode_responses=True)
        self.conn = make_db_conn()
        self._last_trade_count = 0
        self._cumulative_pnl   = 0.0
        self._inventory        = 0.0
        self._quote_counter    = 0   # tracks quotes sent since last fill

    def _get_session_id(self):
        """Read current session_id from Redis (written by engine __init__)."""
        return self.r.get("session:id") or "UNKNOWN"

    def _get_quotes_sent(self):
        """
        Read cumulative quote counter from Redis.
        Engine increments quote:count on every update() tick.
        Falls back to 0 if key not present.
        """
        return int(self.r.get("quote:count") or 0)

    def run(self):
        log.info("FillPersister started — subscribing to sitaram:fills")
        try:
            pubsub = self.r.pubsub()
            pubsub.subscribe("sitaram:fills")
            log.info("Subscribed to Redis channel: sitaram:fills")
            self._pubsub_loop(pubsub)
        except Exception as e:
            log.warning(f"Pub/sub failed ({e}) — falling back to Redis polling")
            self._polling_loop()

    def _pubsub_loop(self, pubsub):
        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                fill = json.loads(message["data"])
                self._insert_fill(fill)
            except Exception as e:
                log.error(f"Fill insert error: {e}")

    def _polling_loop(self):
        log.info("FillPersister polling mode active (500ms interval)")
        while True:
            try:
                self._poll_once()
            except Exception as e:
                log.error(f"Poll error: {e}")
                try:
                    self.conn = make_db_conn()
                except Exception:
                    pass
            time.sleep(0.5)

    def _poll_once(self):
        current_count = int(self.r.get("sim:total_fills") or 0)
        if current_count <= self._last_trade_count:
            return

        new_fills = current_count - self._last_trade_count
        self._last_trade_count = current_count

        cum_pnl   = float(self.r.get("quote:cumulative_pnl") or
                          self._get_metrics_pnl())
        inventory = float(self.r.get("quote:inventory") or 0)
        bid       = float(self.r.get("quote:bid") or 0)
        ask       = float(self.r.get("quote:ask") or 0)

        inv_delta = inventory - self._inventory
        side      = "buy" if inv_delta > 0 else "sell"
        price     = bid if side == "buy" else ask
        qty       = 0.01
        pnl_delta = cum_pnl - self._cumulative_pnl

        self._cumulative_pnl = cum_pnl
        self._inventory      = inventory

        fill = {
            "time":            datetime.now(timezone.utc).isoformat(),
            "side":            side,
            "price":           price,
            "quantity":        qty,
            "realized_pnl":    round(pnl_delta, 6),
            "cumulative_pnl":  round(cum_pnl, 6),
            "fill_latency_ms": 10.0,
            "adverse_fill":    False,
            "inventory_after": round(inventory, 6),
            "quote_id":        str(uuid.uuid4()),
        }

        for _ in range(new_fills):
            self._insert_fill(fill)

    def _get_metrics_pnl(self):
        try:
            val = self.r.get("metrics:cumulative_pnl")
            return float(val) if val else 0.0
        except Exception:
            return 0.0

    def _insert_fill(self, fill):
        """Write one fill row to trades table — includes session_id + quotes_sent."""
        try:
            trade_time = fill.get("time", datetime.now(timezone.utc).isoformat())
            if isinstance(trade_time, str):
                trade_time = datetime.fromisoformat(
                    trade_time.replace("Z", "+00:00"))

            session_id  = fill.get("session_id") or self._get_session_id()
            quotes_sent = fill.get("quotes_sent") or self._get_quotes_sent()

            with self.conn.cursor() as cur:
                cur.execute(self.INSERT_SQL, (
                    trade_time,
                    SYMBOL,
                    fill.get("side", "buy"),
                    fill.get("price", 0.0),
                    fill.get("quantity", 0.01),
                    fill.get("realized_pnl", 0.0),
                    fill.get("cumulative_pnl", 0.0),
                    fill.get("fill_latency_ms", 10.0),
                    fill.get("adverse_fill", False),
                    fill.get("inventory_after", 0.0),
                    fill.get("quote_id", str(uuid.uuid4())),
                    session_id,    # ← NEW
                    quotes_sent,   # ← NEW
                ))
            self.conn.commit()
            log.info(f"Fill persisted: {fill.get('side')} @ {fill.get('price')} "
                     f"| cum_pnl={fill.get('cumulative_pnl', 0):.2f} "
                     f"| session={session_id} quotes={quotes_sent}")
        except psycopg2.Error as e:
            log.error(f"DB insert error: {e}")
            self.conn.rollback()
            try:
                self.conn = make_db_conn()
            except Exception:
                pass


# ── Thread 2: OB tick persister via Redis polling ─────────────────────────────
class OBPersister(threading.Thread):
    """
    Polls Redis every POLL_INTERVAL seconds for current OB state and
    writes one row to order_book_ticks — includes session_id.
    """

    INSERT_SQL = """
        INSERT INTO order_book_ticks
            (time, symbol, best_bid, best_ask, mid_price, spread,
             bid_depth_10, ask_depth_10, update_id, session_id)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    POLL_INTERVAL = 5.0

    def __init__(self):
        super().__init__(name="OBPersister", daemon=True)
        self.r    = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                decode_responses=True)
        self.conn = make_db_conn()
        self._last_ts = 0

    def run(self):
        log.info(f"OBPersister started — polling Redis every {self.POLL_INTERVAL}s")
        while True:
            try:
                self._poll_once()
            except psycopg2.Error as e:
                log.error(f"OB DB error: {e}")
                self.conn.rollback()
                try:
                    self.conn = make_db_conn()
                except Exception:
                    pass
            except Exception as e:
                log.warning(f"OB poll error: {e}")
            time.sleep(self.POLL_INTERVAL)

    def _poll_once(self):
        bid = self.r.get("quote:bid")
        ask = self.r.get("quote:ask")
        mid = self.r.get("ob:mid")
        ts  = self.r.get("ob:ts")

        if not bid or not ask:
            return

        best_bid  = float(bid)
        best_ask  = float(ask)
        mid_price = float(mid) if mid else round((best_bid + best_ask) / 2, 2)
        spread    = round(best_ask - best_bid, 2)
        ts_ms     = float(ts) if ts else time.time() * 1000
        tick_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        if ts_ms == self._last_ts:
            return
        self._last_ts = ts_ms

        session_id = self.r.get("session:id") or "UNKNOWN"

        with self.conn.cursor() as cur:
            cur.execute(self.INSERT_SQL, (
                tick_time, SYMBOL,
                best_bid, best_ask, mid_price, spread,
                0.0, 0.0,
                int(ts_ms),
                session_id,    # ← NEW
            ))
        self.conn.commit()
        log.debug(f"OB tick: bid={best_bid} ask={best_ask} spread={spread} "
                  f"session={session_id}")


# ── Thread 3: Feature metrics persister ──────────────────────────────────────
class MetricsPersister(threading.Thread):
    """
    Polls Redis every METRICS_POLL_SEC seconds for feature metrics
    and writes to feature_metrics table — includes session_id.
    """

    INSERT_SQL = """
        INSERT INTO feature_metrics
            (computed_at, symbol, metric_name, ic_value,
             feature_value, horizon_ms, rolling_window, session_id)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    METRICS_MAP = {
        "quote:obi":      ("obi_5",     200, 5),
        "quote:signal":   ("composite", 200, 50),
        "quote:inv_norm": ("inv_norm",  200, 1),
    }

    def __init__(self):
        super().__init__(name="MetricsPersister", daemon=True)
        self.r    = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                decode_responses=True)
        self.conn = make_db_conn()

    def run(self):
        log.info(f"MetricsPersister started — polling every {METRICS_POLL_SEC}s")
        while True:
            try:
                self._poll_once()
            except Exception as e:
                log.error(f"MetricsPersister error: {e}")
                try:
                    self.conn = make_db_conn()
                except Exception:
                    pass
            time.sleep(METRICS_POLL_SEC)

    def _poll_once(self):
        now        = datetime.now(timezone.utc)
        ic_value   = float(self.r.get("metrics:ic_200") or 0.0)
        session_id = self.r.get("session:id") or "UNKNOWN"

        rows = []
        for redis_key, (metric_name, horizon_ms, rolling_window) in \
                self.METRICS_MAP.items():
            val = self.r.get(redis_key)
            if val is None:
                continue
            rows.append((
                now, SYMBOL, metric_name,
                ic_value,
                float(val),
                horizon_ms,
                rolling_window,
                session_id,    # ← NEW
            ))

        if not rows:
            return

        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, self.INSERT_SQL, rows)
        self.conn.commit()
        log.debug(f"Feature metrics persisted: {len(rows)} rows | session={session_id}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("SITARAM Persister v1.1 starting")
    log.info(f"  DB:    {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    log.info(f"  Redis: {REDIS_HOST}:{REDIS_PORT}")
    log.info(f"  OB sample rate: every {OB_SAMPLE_EVERY} ticks")
    log.info(f"  Metrics poll: every {METRICS_POLL_SEC}s")
    log.info("=" * 60)

    threads = [
        FillPersister(),
        OBPersister(),
        MetricsPersister(),
    ]

    for t in threads:
        t.start()
        log.info(f"Thread started: {t.name}")

    while True:
        alive = [t.name for t in threads if t.is_alive()]
        dead  = [t.name for t in threads if not t.is_alive()]
        if dead:
            log.error(f"Dead threads: {dead} | Alive: {alive}")
        else:
            log.info(f"All threads alive: {alive}")
        time.sleep(60)


if __name__ == "__main__":
    main()
