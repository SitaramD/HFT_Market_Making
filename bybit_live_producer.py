#!/usr/bin/env python3
# =============================================================================
# HFT Market Making — Bybit Live WebSocket Producer  v2.0
#
# Changes v2.0:
#   [NEW]  Windows desktop notifications via plyer (pip install plyer)
#   [NEW]  Data feed stall watchdog — alerts if OB or Trades go silent > 30s
#   [NEW]  Disconnect notifications with exact timestamp
#   [NEW]  Session start/stop notifications
#   [NEW]  Polls Redis for halt events published by engine.py and notifies desktop
#   [NEW]  Reconnect loop with backoff (was: manual restart only)
#
# Usage (Windows PowerShell):
#   pip install websocket-client kafka-python plyer redis
#   python bybit_live_producer.py
#
# Environment variables (optional overrides):
#   KAFKA_BOOTSTRAP_SERVERS  default: localhost:9092
#   BYBIT_WS_URL             default: wss://stream.bybit.com/v5/public/spot
#   SYMBOL                   default: BTCUSDT
#   LOG_LEVEL                default: INFO
#   REDIS_HOST               default: localhost
#   STALL_THRESHOLD_SEC      default: 30   (seconds of silence before alert)
#   NOTIFY_COOLDOWN_SEC      default: 60   (min seconds between same notification)
# =============================================================================

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone

import websocket
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Desktop notifications — graceful fallback if plyer not installed
try:
    from plyer import notification as _plyer_notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

# Redis — for polling halt events published by engine.py
try:
    import redis as _redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
KAFKA_BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
BYBIT_WS_URL        = os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/spot")
SYMBOL              = os.getenv("SYMBOL", "BTCUSDT")
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
REDIS_HOST          = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT", 6379))
STALL_THRESHOLD_SEC = float(os.getenv("STALL_THRESHOLD_SEC", 30))
NOTIFY_COOLDOWN_SEC = float(os.getenv("NOTIFY_COOLDOWN_SEC", 60))
PING_INTERVAL_S     = 20
RECONNECT_DELAY_S   = 5    # base delay between reconnect attempts
MAX_RECONNECT_S     = 60   # cap backoff at 60s

TOPIC_OB        = "ob200-raw"
TOPIC_TRADES    = "trades-raw"
TOPIC_HEARTBEAT = "sitaram.heartbeat"   # Layer 3: producer liveness signal

APP_NAME     = "SITARAM HFT"

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bybit-live")

# =============================================================================
# COUNTERS & STATE
# =============================================================================
stats = {
    "ob_msgs":       0,
    "trade_msgs":    0,
    "ob_errors":     0,
    "trade_errors":  0,
    "ob_last_msg":   time.time(),       # timestamp of last OB message
    "trade_last_msg":time.time(),       # timestamp of last Trade message
    "started_at":    time.time(),
    "ob_connected":  False,
    "trade_connected": False,
}

# Notification cooldown tracker: key → last sent timestamp
_notify_cooldowns: dict = {}

# =============================================================================
# WINDOWS DESKTOP NOTIFICATIONS
# =============================================================================
def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _on_cooldown(key: str) -> bool:
    return (time.time() - _notify_cooldowns.get(key, 0)) < NOTIFY_COOLDOWN_SEC

def _mark_notified(key: str):
    _notify_cooldowns[key] = time.time()

def notify(title: str, message: str, urgency: str = "normal", cooldown_key: str = ""):
    """
    Send a Windows desktop notification via plyer.
    urgency: 'normal' | 'critical'  (visual distinction in title prefix)
    cooldown_key: if set, same alert won't fire again within NOTIFY_COOLDOWN_SEC
    """
    if cooldown_key and _on_cooldown(cooldown_key):
        return
    if cooldown_key:
        _mark_notified(cooldown_key)

    prefix = "🔴 " if urgency == "critical" else "🟡 "
    full_title = f"{prefix}{APP_NAME} — {title}"

    log.info(f"NOTIFY [{urgency.upper()}] {title}: {message}")

    if not PLYER_AVAILABLE:
        log.warning("plyer not installed — desktop notification suppressed. "
                    "Run: pip install plyer")
        return

    try:
        _plyer_notification.notify(
            title=full_title,
            message=message,
            app_name=APP_NAME,
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Desktop notification failed: {e}")

# =============================================================================
# REDIS CLIENT (optional — for halt event polling)
# =============================================================================
_redis_client = None

def _get_redis():
    global _redis_client
    if not REDIS_AVAILABLE:
        return None
    if _redis_client is None:
        try:
            _redis_client = _redis_lib.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                decode_responses=True, socket_connect_timeout=3
            )
            _redis_client.ping()
            log.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            log.warning(f"Redis unavailable — halt polling disabled: {e}")
            _redis_client = None
    return _redis_client

# =============================================================================
# TRADE TRANSFORM
# =============================================================================
def transform_trade(trade: dict) -> dict:
    return {
        "id":     int(str(trade.get("i", "0"))[-9:]),
        "ts":     float(trade.get("T", 0)),
        "symbol": trade.get("s", SYMBOL),
        "price":  float(trade.get("p", 0)),
        "volume": float(trade.get("v", 0)),
        "side":   trade.get("S", "Buy").lower(),
        "rpi":    0,
    }

# =============================================================================
# KAFKA
# =============================================================================
def connect_kafka(bootstrap: str) -> KafkaProducer:
    for attempt in range(30):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                compression_type="gzip",
                batch_size=65536,
                linger_ms=5,
                acks=1,
                retries=3,
            )
            log.info(f"Kafka connected: {bootstrap}")
            return producer
        except NoBrokersAvailable:
            log.warning(f"Kafka not ready (attempt {attempt+1}/30) — retrying in 3s...")
            time.sleep(3)
    raise RuntimeError(f"Cannot connect to Kafka at {bootstrap}")

# =============================================================================
# STATS LOGGER
# =============================================================================
def stats_logger():
    while True:
        time.sleep(60)
        elapsed = time.time() - stats["started_at"]
        log.info(
            f"Stats | OB: {stats['ob_msgs']:,}  "
            f"Trades: {stats['trade_msgs']:,}  "
            f"Errors: OB={stats['ob_errors']} Trade={stats['trade_errors']}  "
            f"Uptime: {elapsed/60:.1f}m"
        )

# =============================================================================
# DATA FEED STALL WATCHDOG
# Runs in its own thread. Fires desktop notification if either feed goes silent.
# =============================================================================
def stall_watchdog():
    """Watches ob_last_msg and trade_last_msg timestamps. Alerts on silence."""
    log.info(f"Stall watchdog started (threshold: {STALL_THRESHOLD_SEC}s)")
    while True:
        time.sleep(10)
        now = time.time()

        ob_silence    = now - stats["ob_last_msg"]
        trade_silence = now - stats["trade_last_msg"]

        if ob_silence > STALL_THRESHOLD_SEC and stats["ob_connected"]:
            stall_time = datetime.fromtimestamp(
                stats["ob_last_msg"], tz=timezone.utc).strftime("%H:%M:%S UTC")
            notify(
                title="OB Feed STALLED",
                message=(f"Order book feed silent for {ob_silence:.0f}s.\n"
                         f"Last tick received at {stall_time}.\n"
                         f"Data age exceeds {STALL_THRESHOLD_SEC}s threshold."),
                urgency="critical",
                cooldown_key="ob_stall",
            )

        if trade_silence > STALL_THRESHOLD_SEC and stats["trade_connected"]:
            stall_time = datetime.fromtimestamp(
                stats["trade_last_msg"], tz=timezone.utc).strftime("%H:%M:%S UTC")
            notify(
                title="Trades Feed STALLED",
                message=(f"Public trades feed silent for {trade_silence:.0f}s.\n"
                         f"Last tick received at {stall_time}."),
                urgency="critical",
                cooldown_key="trade_stall",
            )

# =============================================================================
# HALT EVENT POLLER
# Subscribes to Redis pub/sub channel 'sitaram:halts' published by engine.py.
# Converts each halt event into a Windows desktop notification.
# =============================================================================
def halt_event_poller():
    """Subscribe to Redis sitaram:halts channel and push desktop notifications."""
    r = _get_redis()
    if r is None:
        log.warning("Halt event poller disabled — Redis not available")
        return

    log.info("Halt event poller started — listening on sitaram:halts")
    while True:
        try:
            pubsub = r.pubsub()
            pubsub.subscribe("sitaram:halts")
            for raw_msg in pubsub.listen():
                if raw_msg["type"] != "message":
                    continue
                try:
                    event = json.loads(raw_msg["data"])
                    reason   = event.get("reason", "UNKNOWN")
                    inv      = event.get("inventory", 0)
                    pnl      = event.get("pnl", 0)
                    halt_time= event.get("time", _now_str())

                    notify(
                        title="TRADING HALTED",
                        message=(
                            f"Reason: {reason}\n"
                            f"Time: {halt_time}\n"
                            f"Inventory: {inv:.6f} BTC\n"
                            f"PnL: {pnl:+.2f} USDT"
                        ),
                        urgency="critical",
                        cooldown_key=f"halt_{reason[:20]}",
                    )
                except Exception as e:
                    log.warning(f"Halt event parse error: {e}")
        except Exception as e:
            log.warning(f"Halt poller Redis error: {e} — reconnecting in 10s")
            global _redis_client
            _redis_client = None   # force reconnect (must be global to actually clear it)
            time.sleep(10)

# =============================================================================
# LAYER 3: HEARTBEAT PUBLISHER
# Publishes a liveness signal to Kafka topic 'sitaram.heartbeat' every 10s.
# The claude-agent consumes this topic in monitor_heartbeat() to detect producer
# death independently of ob:ts (which only proves the engine is writing Redis,
# not that fresh Bybit data is actually flowing).
# =============================================================================
def heartbeat_publisher(kafka_producer: KafkaProducer):
    """Publish producer heartbeat to Kafka every 10 seconds."""
    log.info(f"Heartbeat publisher started → topic: {TOPIC_HEARTBEAT}")
    while True:
        try:
            payload = {
                "ts"         : int(time.time() * 1000),   # epoch ms
                "source"     : "bybit-producer",
                "symbol"     : SYMBOL,
                "ob_msgs"    : stats["ob_msgs"],
                "trade_msgs" : stats["trade_msgs"],
                "ob_connected"   : stats["ob_connected"],
                "trade_connected": stats["trade_connected"],
                "uptime_sec" : round(time.time() - stats["started_at"], 0),
            }
            kafka_producer.send(TOPIC_HEARTBEAT, value=payload)
        except Exception as e:
            log.warning(f"Heartbeat publish failed: {e}")
        time.sleep(10)

# =============================================================================
# ORDERBOOK WEBSOCKET PRODUCER (with reconnect loop)
# =============================================================================
class OBProducer:

    def __init__(self, kafka_producer: KafkaProducer):
        self.kafka        = kafka_producer
        self.ws           = None
        self._reconnect_n = 0

    def start(self):
        """Reconnect loop — never gives up."""
        while True:
            delay = min(RECONNECT_DELAY_S * (2 ** self._reconnect_n), MAX_RECONNECT_S)
            if self._reconnect_n > 0:
                log.info(f"OB reconnecting in {delay:.0f}s (attempt {self._reconnect_n})")
                time.sleep(delay)
            log.info(f"OB WebSocket connecting → {BYBIT_WS_URL}")
            self.ws = websocket.WebSocketApp(
                BYBIT_WS_URL,
                on_open    = self._on_open,
                on_message = self._on_message,
                on_error   = self._on_error,
                on_close   = self._on_close,
            )
            self.ws.run_forever(ping_interval=0)
            self._reconnect_n += 1

    def _on_open(self, ws):
        self._reconnect_n = 0
        stats["ob_connected"] = True
        log.info("OB WebSocket connected")
        ws.send(json.dumps({"op": "subscribe", "args": [f"orderbook.200.{SYMBOL}"]}))
        notify(
            title="OB Feed Connected",
            message=f"Order book feed live at {_now_str()}",
            urgency="normal",
            cooldown_key="ob_connect",
        )
        threading.Thread(target=self._keep_alive, daemon=True).start()

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
            if msg.get("op") in ("pong", "subscribe", "ping"):
                return
            if not msg.get("topic", "").startswith("orderbook"):
                return
            if "cts" not in msg:
                msg["cts"] = int(time.time() * 1000)
            self.kafka.send(TOPIC_OB, value=msg)
            stats["ob_msgs"]    += 1
            stats["ob_last_msg"] = time.time()  # update watchdog timestamp
            if stats["ob_msgs"] % 1000 == 0:
                log.info(f"OB  → Kafka: {stats['ob_msgs']:,} msgs")
        except Exception as e:
            stats["ob_errors"] += 1
            log.warning(f"OB msg error: {e}")

    def _on_error(self, ws, error):
        log.error(f"OB WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        stats["ob_connected"] = False
        disconnect_time = _now_str()
        log.warning(f"OB WebSocket closed (code={code}) at {disconnect_time}")
        notify(
            title="OB Feed DISCONNECTED",
            message=(f"Order book WebSocket closed at {disconnect_time}.\n"
                     f"Close code: {code}\n"
                     f"Auto-reconnecting..."),
            urgency="critical",
            cooldown_key="ob_disconnect",
        )

    def _keep_alive(self):
        while True:
            time.sleep(PING_INTERVAL_S)
            try:
                if self.ws:
                    self.ws.send(json.dumps({"op": "ping"}))
            except Exception:
                break

# =============================================================================
# TRADES WEBSOCKET PRODUCER (with reconnect loop)
# =============================================================================
class TradesProducer:

    def __init__(self, kafka_producer: KafkaProducer):
        self.kafka        = kafka_producer
        self.ws           = None
        self._reconnect_n = 0

    def start(self):
        """Reconnect loop — never gives up."""
        while True:
            delay = min(RECONNECT_DELAY_S * (2 ** self._reconnect_n), MAX_RECONNECT_S)
            if self._reconnect_n > 0:
                log.info(f"Trades reconnecting in {delay:.0f}s (attempt {self._reconnect_n})")
                time.sleep(delay)
            log.info(f"Trades WebSocket connecting → {BYBIT_WS_URL}")
            self.ws = websocket.WebSocketApp(
                BYBIT_WS_URL,
                on_open    = self._on_open,
                on_message = self._on_message,
                on_error   = self._on_error,
                on_close   = self._on_close,
            )
            self.ws.run_forever(ping_interval=0)
            self._reconnect_n += 1

    def _on_open(self, ws):
        self._reconnect_n = 0
        stats["trade_connected"] = True
        log.info("Trades WebSocket connected")
        ws.send(json.dumps({"op": "subscribe", "args": [f"publicTrade.{SYMBOL}"]}))
        notify(
            title="Trades Feed Connected",
            message=f"Public trades feed live at {_now_str()}",
            urgency="normal",
            cooldown_key="trades_connect",
        )
        threading.Thread(target=self._keep_alive, daemon=True).start()

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
            if msg.get("op") in ("pong", "subscribe", "ping"):
                return
            if not msg.get("topic", "").startswith("publicTrade"):
                return
            for trade in msg.get("data", []):
                try:
                    self.kafka.send(TOPIC_TRADES, value=transform_trade(trade))
                    stats["trade_msgs"]    += 1
                    stats["trade_last_msg"] = time.time()   # update watchdog timestamp
                except Exception as e:
                    stats["trade_errors"] += 1
                    log.warning(f"Trade transform error: {e}")
            if stats["trade_msgs"] % 5000 == 0 and stats["trade_msgs"] > 0:
                log.info(f"Trades → Kafka: {stats['trade_msgs']:,} msgs")
        except Exception as e:
            stats["trade_errors"] += 1
            log.warning(f"Trades msg error: {e}")

    def _on_error(self, ws, error):
        log.error(f"Trades WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        stats["trade_connected"] = False
        disconnect_time = _now_str()
        log.warning(f"Trades WebSocket closed (code={code}) at {disconnect_time}")
        notify(
            title="Trades Feed DISCONNECTED",
            message=(f"Public trades WebSocket closed at {disconnect_time}.\n"
                     f"Close code: {code}\n"
                     f"Auto-reconnecting..."),
            urgency="critical",
            cooldown_key="trades_disconnect",
        )

    def _keep_alive(self):
        while True:
            time.sleep(PING_INTERVAL_S)
            try:
                if self.ws:
                    self.ws.send(json.dumps({"op": "ping"}))
            except Exception:
                break

# =============================================================================
# MAIN
# =============================================================================
def main():
    session_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    log.info("=" * 60)
    log.info("  HFT Market Making — Bybit Live Producer  v2.0")
    log.info("=" * 60)
    log.info(f"  Kafka    : {KAFKA_BOOTSTRAP}")
    log.info(f"  WS URL   : {BYBIT_WS_URL}")
    log.info(f"  Symbol   : {SYMBOL}")
    log.info(f"  Redis    : {REDIS_HOST}:{REDIS_PORT}")
    log.info(f"  Stall thr: {STALL_THRESHOLD_SEC}s")
    log.info(f"  plyer    : {'available' if PLYER_AVAILABLE else 'NOT INSTALLED — pip install plyer'}")
    log.info(f"  Started  : {session_start}")
    log.info("=" * 60)

    # Session start notification
    notify(
        title="Producer Started",
        message=f"SITARAM data feed starting at {session_start}\nSymbol: {SYMBOL}",
        urgency="normal",
        cooldown_key="producer_start",
    )

    # Step 1 — Connect to Kafka
    kafka = connect_kafka(KAFKA_BOOTSTRAP)

    # Step 2 — Background threads
    threading.Thread(target=stats_logger,      daemon=True, name="stats").start()
    threading.Thread(target=stall_watchdog,    daemon=True, name="stall-watchdog").start()
    threading.Thread(target=halt_event_poller, daemon=True, name="halt-poller").start()
    threading.Thread(target=heartbeat_publisher, args=(kafka,), daemon=True, name="heartbeat").start()
    log.info("Heartbeat publisher thread started")

    # Step 3 — OB WebSocket thread
    ob = OBProducer(kafka)
    ob_thread = threading.Thread(target=ob.start, name="ob-ws", daemon=True)
    ob_thread.start()
    log.info("OB producer thread started")

    time.sleep(2)

    # Step 4 — Trades WebSocket thread
    tr = TradesProducer(kafka)
    tr_thread = threading.Thread(target=tr.start, name="trades-ws", daemon=True)
    tr_thread.start()
    log.info("Trades producer thread started")

    log.info("Both feeds active — streaming live Bybit data to Kafka")
    log.info("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        stop_time = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        log.info(f"Ctrl+C — shutting down at {stop_time}")
        notify(
            title="Producer STOPPED",
            message=(f"Data feed manually stopped at {stop_time}\n"
                     f"OB msgs: {stats['ob_msgs']:,} | "
                     f"Trade msgs: {stats['trade_msgs']:,}"),
            urgency="critical",
            cooldown_key="producer_stop",
        )
    finally:
        log.info(
            f"Shutdown | OB: {stats['ob_msgs']:,}  "
            f"Trades: {stats['trade_msgs']:,}  "
            f"Errors: {stats['ob_errors'] + stats['trade_errors']}"
        )
        try:
            kafka.flush(timeout=5)
            kafka.close()
        except Exception:
            pass
        log.info("Done.")


if __name__ == "__main__":
    main()
