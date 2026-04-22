#import eventlet
#eventlet.monkey_patch()
import gevent
from gevent import monkey; monkey.patch_all()
"""
=============================================================================
SITARAM — Claude AI Intelligence Agent  v2.0
=============================================================================
Changes v2.0:
  [FIX]  MAX_INVENTORY_BTC corrected to 0.10 (matches Run-15 validated engine)
  [NEW]  monitor_halts() — subscribes to Redis sitaram:halts pub/sub channel,
         forwards halt reason + timestamp to dashboard via WebSocket instantly
  [NEW]  monitor_data_feed() — watches session:data_stall Redis key,
         dispatches alert when feed goes silent
  [NEW]  /session/current endpoint — returns live session metadata + parameters
  [NEW]  /session/history endpoint — returns full sitaram_sessions.json
  [NEW]  halt_reason included in /metrics response
  [NEW]  data_stall flag included in /metrics response
=============================================================================
"""

import os, time, json, logging, threading
from datetime import datetime, timezone
from typing import Optional

import anthropic
import psycopg2
import redis
import psycopg2.extras
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Layer 3: Kafka heartbeat consumer
try:
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('claude-agent')

# =============================================================================
# CONFIGURATION
# =============================================================================
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
TSDB_HOST         = os.getenv('TSDB_HOST', 'timescaledb')
TSDB_USER         = os.getenv('TSDB_USER', 'sitaram_user')
TSDB_PASSWORD     = os.environ['TSDB_PASSWORD']
TSDB_DB           = os.getenv('TSDB_DB', 'sitaram')
REDIS_HOST        = os.getenv('REDIS_HOST', 'redis')

IC_MIN            = float(os.getenv('IC_MIN_THRESHOLD', '0.02'))
SHARPE_MIN        = float(os.getenv('SHARPE_MIN_THRESHOLD', '1.5'))
MAX_DRAWDOWN_PCT  = float(os.getenv('MAX_DRAWDOWN_PCT', '5.0'))
MAX_INVENTORY_BTC = float(os.getenv('MAX_INVENTORY_BTC', '0.10'))   # FIXED: was 0.5
LATENCY_SPIKE_MS  = float(os.getenv('LATENCY_SPIKE_MS', '100'))
FILL_RATE_MIN     = float(os.getenv('FILL_RATE_MIN', '0.05'))
ADVERSE_SEL_MAX   = float(os.getenv('ADVERSE_SELECTION_MAX', '0.30'))

HEALTH_INTERVAL   = int(os.getenv('HEALTH_CHECK_INTERVAL_SEC', '30'))
IC_INTERVAL       = int(os.getenv('IC_CHECK_INTERVAL_SEC', '60'))
PNL_INTERVAL      = int(os.getenv('PNL_CHECK_INTERVAL_SEC', '10'))
ALERT_COOLDOWN    = int(os.getenv('ALERT_COOLDOWN_SEC', '300'))
STALL_ALERT_SEC   = float(os.getenv('STALL_ALERT_SEC', '30'))   # seconds of silence → alert
STALL_COOLDOWN    = int(os.getenv('STALL_COOLDOWN_SEC', '120'))  # Layer 3: shorter cooldown for feed/heartbeat alerts

# Layer 3: Kafka heartbeat consumer config
KAFKA_BOOTSTRAP       = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
HEARTBEAT_TOPIC       = 'sitaram.heartbeat'
HEARTBEAT_TIMEOUT_SEC = float(os.getenv('HEARTBEAT_TIMEOUT_SEC', '30'))  # alert if no heartbeat for this long

# Path to session JSON (mounted from host E:\Binance\)
SESSION_JSON_PATH = os.getenv('SESSION_JSON_PATH', '/mnt/sessions/sitaram_sessions.json')

# =============================================================================
# FLASK + SOCKETIO
# =============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sitaram-claude-agent'
CORS(app, origins='*')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')

# =============================================================================
# CLIENTS
# =============================================================================
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
redis_client  = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def get_tsdb_conn():
    return psycopg2.connect(
        host=TSDB_HOST, port=5432,
        dbname=TSDB_DB,
        user=TSDB_USER, password=TSDB_PASSWORD,
        cursor_factory=psycopg2.extras.RealDictCursor
    )

# =============================================================================
# ALERT COOLDOWN
# =============================================================================
alert_cooldowns: dict = {}

def is_on_cooldown(key: str) -> bool:
    return (time.time() - alert_cooldowns.get(key, 0)) < ALERT_COOLDOWN

def mark_alert_sent(key: str):
    alert_cooldowns[key] = time.time()

# =============================================================================
# CLAUDE API
# =============================================================================
def ask_claude(context: dict, anomaly_type: str) -> str:
    try:
        prompt_path = '/app/prompts/monitor.txt'
        with open(prompt_path, 'r') as f:
            system_prompt = f.read()
        response = claude_client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=300,
            messages=[{'role': 'user', 'content': f"""
ANOMALY DETECTED: {anomaly_type}

CURRENT METRICS SNAPSHOT:
{json.dumps(context, indent=2)}

Provide a concise (3-5 line) analysis:
1. What is happening and why it matters for market making
2. The likely cause given the data
3. What the operator should watch for next
4. Severity: [CRITICAL / WARNING / INFO]
"""}],
            system=system_prompt
        )
        return response.content[0].text
    except Exception as e:
        log.error(f'Claude API call failed: {e}')
        return f'[Claude API unavailable] Anomaly detected: {anomaly_type}'

# =============================================================================
# ALERT DISPATCHER
# =============================================================================
def dispatch_alert(severity: str, category: str, message: str,
                   claude_analysis: Optional[str], metrics: dict):
    alert_payload = {
        'id'              : f"{category}_{int(time.time())}",
        'timestamp'       : datetime.now(timezone.utc).isoformat(),
        'severity'        : severity,
        'category'        : category,
        'message'         : message,
        'claude_analysis' : claude_analysis,
        'metrics_snapshot': metrics,
        'source'          : 'claude-agent'
    }
    socketio.emit('alert', alert_payload, namespace='/dashboard')
    log.info(f'Alert dispatched [{severity}][{category}]: {message}')
    redis_client.publish('sitaram:alerts', json.dumps({
        'severity': severity,
        'message' : f"[{category}] {message}",
        'time'    : datetime.now().strftime('%H:%M:%S')
    }))
    try:
        conn = get_tsdb_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO claude_alerts
                (alert_time, severity, category, message, claude_analysis, metrics_json)
            VALUES (NOW(), %s, %s, %s, %s, %s)
        """, (severity, category, message, claude_analysis, json.dumps(metrics)))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        log.error(f'Failed to persist alert to TimescaleDB: {e}')

# =============================================================================
# HELPER: Read live metrics from Redis
# =============================================================================
def get_live_metrics() -> dict:
    r = redis_client
    def rf(key, default=0.0):
        v = r.get(key)
        return float(v) if v is not None else default
    def ri(key, default=0):
        v = r.get(key)
        return int(float(v)) if v is not None else default
    def rs(key, default=''):
        return r.get(key) or default

    mid        = rf('ob:mid')
    best_bid   = rf('ob:best_bid')
    best_ask   = rf('ob:best_ask')
    spread     = rf('ob:spread')
    obi5       = rf('ob:obi5')
    composite  = rf('ob:composite')
    quote_bid  = rf('quote:bid')
    quote_ask  = rf('quote:ask')
    signal     = rf('quote:signal')
    regime     = rs('quote:regime', 'unknown')
    active     = rs('quote:active', '0') == '1'
    fill_rate  = rf('sim:fill_rate')
    tot_fills  = ri('sim:total_fills')
    tot_quotes = ri('sim:total_quotes')
    inventory      = rf('quote:inventory', 0.0)
    inv_norm       = rf('quote:inv_norm', 0.0)
    cumulative_pnl = rf('quote:cumulative_pnl', 0.0)
    daily_pnl      = rf('quote:daily_pnl', 0.0)
    tick_buf       = r.llen('ticks')
    ob_ts          = rf('ob:ts')

    # NEW: session + halt + stall fields
    halt_reason  = rs('session:halt_reason', '')
    halt_time    = rs('session:halt_time', '')
    data_stall   = rs('session:data_stall', '0') == '1'
    stall_since  = rs('session:stall_since', '')
    session_id   = rs('session:id', '')
    session_start= rs('session:start_time', '')

    return {
        'mid'          : round(mid, 1),
        'best_bid'     : round(best_bid, 1),
        'best_ask'     : round(best_ask, 1),
        'spread'       : round(spread, 2),
        'obi5'         : round(obi5, 4),
        'composite'    : round(composite, 4),
        'quote_bid'    : round(quote_bid, 1),
        'quote_ask'    : round(quote_ask, 1),
        'signal'       : round(signal, 4),
        'regime'       : regime,
        'active'       : active,
        'fill_rate'    : round(fill_rate, 4),
        'total_fills'  : tot_fills,
        'total_quotes' : tot_quotes,
        'inventory'      : round(inventory, 6),
        'inv_norm'       : round(inv_norm, 4),
        'cumulative_pnl' : round(cumulative_pnl, 4),
        'daily_pnl'      : round(daily_pnl, 4),
        'tick_buffer'    : tick_buf,
        'ob_ts'        : ob_ts,
        'data_age_sec' : round((time.time() * 1000 - ob_ts) / 1000, 1) if ob_ts > 0 else -1,
        # NEW fields
        'halt_reason'  : halt_reason,
        'halt_time'    : halt_time,
        'data_stall'   : data_stall,
        'stall_since'  : stall_since,
        'session_id'   : session_id,
        'session_start': session_start,
    }

# =============================================================================
# MONITOR 1: IC
# =============================================================================
def monitor_ic():
    log.info('IC monitor started')
    while True:
        try:
            conn = get_tsdb_conn()
            cur  = conn.cursor()
            cur.execute("""
                SELECT
                    time_bucket('1 hour', computed_at) AS hour,
                    AVG(ic_value) AS avg_ic,
                    STDDEV(ic_value) AS ic_stddev
                FROM feature_metrics
                WHERE computed_at > NOW() - INTERVAL '3 days'
                  AND metric_name = 'IC_OBI_3'
                GROUP BY hour ORDER BY hour DESC LIMIT 72
            """)
            rows = cur.fetchall()
            cur.close(); conn.close()

            if rows:
                latest_ic   = float(rows[0]['avg_ic'] or 0)
                ic_3day_avg = sum(float(r['avg_ic'] or 0) for r in rows) / len(rows)
                ic_trend    = latest_ic - ic_3day_avg
                metrics = {
                    'latest_ic'    : round(latest_ic, 4),
                    'ic_3day_avg'  : round(ic_3day_avg, 4),
                    'ic_trend'     : round(ic_trend, 4),
                    'threshold'    : IC_MIN,
                    'hours_sampled': len(rows)
                }
                if latest_ic < IC_MIN and not is_on_cooldown('ic_below_min'):
                    analysis = ask_claude(metrics, 'IC below minimum threshold')
                    dispatch_alert('CRITICAL', 'SIGNAL',
                        f'IC dropped to {latest_ic:.4f} (min: {IC_MIN})',
                        analysis, metrics)
                    mark_alert_sent('ic_below_min')
                socketio.emit('ic_update', metrics, namespace='/dashboard')

        except Exception as e:
            log.error(f'IC monitor error: {e}')
        time.sleep(IC_INTERVAL)

# =============================================================================
# MONITOR 2: INVENTORY — corrected MAX_INVENTORY_BTC = 0.10
# =============================================================================
def monitor_inventory():
    log.info('Inventory monitor started')
    while True:
        try:
            inventory = float(redis_client.get('quote:inventory') or 0)
            inv_norm  = float(redis_client.get('quote:inv_norm') or 0)
            abs_inv   = abs(inventory)
            pct_max   = abs_inv / MAX_INVENTORY_BTC * 100  # FIX: 0.10 not 0.5

            metrics = {
                'inventory_btc' : round(inventory, 6),
                'inv_norm'      : round(inv_norm, 4),
                'abs_inventory' : round(abs_inv, 6),
                'pct_of_max'    : round(pct_max, 1),
                'max_allowed'   : MAX_INVENTORY_BTC,
                'direction'     : 'LONG' if inventory > 0 else 'SHORT' if inventory < 0 else 'FLAT'
            }

            if pct_max > 80 and not is_on_cooldown('inventory_high'):
                analysis = ask_claude(metrics, 'Inventory approaching maximum limit')
                dispatch_alert('WARNING', 'RISK',
                    f'Inventory at {pct_max:.1f}% of max ({inventory:.6f} BTC {metrics["direction"]})',
                    analysis, metrics)
                mark_alert_sent('inventory_high')

            if pct_max > 100 and not is_on_cooldown('inventory_breach'):
                dispatch_alert('CRITICAL', 'RISK',
                    f'INVENTORY BREACH: {inventory:.6f} BTC exceeds max {MAX_INVENTORY_BTC} BTC',
                    ask_claude(metrics, 'Inventory breach — hard limit exceeded'), metrics)
                mark_alert_sent('inventory_breach')

            socketio.emit('inventory_update', metrics, namespace='/dashboard')

        except Exception as e:
            log.error(f'Inventory monitor error: {e}')
        time.sleep(5)

# =============================================================================
# MONITOR 3: P&L + DRAWDOWN
# =============================================================================
# Session peak P&L tracker — resets only when session_id changes
_pnl_peak_tracker = {'session_id': '', 'peak_pnl': 0.0}

def monitor_pnl():
    log.info('P&L monitor started')
    global _pnl_peak_tracker

    while True:
        try:
            # ── Read TRUE P&L from Redis (engine source of truth) ────────────
            # Engine writes self.pnl = cash + mark-to-market to Redis
            # This is always correct regardless of persister restarts
            current_pnl  = float(redis_client.get('quote:cumulative_pnl') or 0)
            session_id   = redis_client.get('session:id') or ''

            # Fallback: if quote:cumulative_pnl not set, try metrics endpoint
            if current_pnl == 0:
                try:
                    import urllib.request
                    with urllib.request.urlopen('http://localhost:8000/metrics', timeout=2) as r:
                        data = json.loads(r.read())
                        current_pnl = float(data.get('cumulative_pnl', 0))
                except Exception:
                    pass

            # ── Reset peak tracker when session changes ────────────────────
            if session_id != _pnl_peak_tracker['session_id']:
                _pnl_peak_tracker['session_id'] = session_id
                _pnl_peak_tracker['peak_pnl']   = current_pnl
                log.info(f'P&L peak tracker reset for session {session_id}')

            # ── Update peak (only goes up, never down) ─────────────────────
            if current_pnl > _pnl_peak_tracker['peak_pnl']:
                _pnl_peak_tracker['peak_pnl'] = current_pnl

            peak_pnl = _pnl_peak_tracker['peak_pnl']

            # ── True drawdown from session peak ────────────────────────────
            # Only calculate if peak > 0 to avoid division artifacts
            if peak_pnl > 0:
                drawdown = ((peak_pnl - current_pnl) / peak_pnl) * 100
            else:
                drawdown = 0.0

            # ── Latency and adverse selection from TimescaleDB ─────────────
            avg_latency  = float(redis_client.get('latency:avg_ms') or 0)
            trade_count  = int(redis_client.get('sim:total_fills') or 0)
            adverse_rate = 0.0

            try:
                conn = get_tsdb_conn()
                cur  = conn.cursor()
                cur.execute("""
                    SELECT
                        AVG(CASE WHEN adverse_fill THEN 1.0 ELSE 0.0 END) AS adverse_sel_rate
                    FROM trades
                    WHERE trade_time > NOW() - INTERVAL '1 day'
                """)
                row = cur.fetchone()
                cur.close(); conn.close()
                if row and row['adverse_sel_rate'] is not None:
                    adverse_rate = float(row['adverse_sel_rate'])
            except Exception as e:
                log.warning(f'TimescaleDB adverse selection query failed: {e}')

            metrics = {
                'total_pnl_usdt'  : round(current_pnl, 2),
                'peak_pnl_usdt'   : round(peak_pnl, 2),
                'drawdown_pct'    : round(drawdown, 2),
                'trade_count'     : trade_count,
                'avg_latency_ms'  : round(avg_latency, 1),
                'adverse_sel_rate': round(adverse_rate, 3),
                'pnl_source'      : 'redis',  # confirms using engine source
            }

            if drawdown > MAX_DRAWDOWN_PCT and not is_on_cooldown('drawdown_exceeded'):
                analysis = ask_claude(metrics, f'Drawdown {drawdown:.1f}% exceeds max {MAX_DRAWDOWN_PCT}%')
                dispatch_alert('CRITICAL', 'RISK',
                    f'DRAWDOWN ALERT: {drawdown:.1f}% from peak (peak=${peak_pnl:.2f}, now=${current_pnl:.2f})',
                    analysis, metrics)
                mark_alert_sent('drawdown_exceeded')

            socketio.emit('pnl_update', metrics, namespace='/dashboard')

        except Exception as e:
            log.error(f'P&L monitor error: {e}')
        time.sleep(PNL_INTERVAL)

# =============================================================================
# MONITOR 4: PIPELINE HEALTH
# =============================================================================
def monitor_pipeline():
    log.info('Pipeline health monitor started')
    while True:
        try:
            redis_info = redis_client.info()
            tick_buf   = redis_client.llen('ticks')
            lag_raw    = redis_client.get('SITARAM:kafka:consumer_lag')

            health = {
                'redis_memory_mb'   : round(redis_info['used_memory'] / 1048576, 1),
                'redis_connected'   : redis_info['connected_clients'],
                'tick_buffer_size'  : tick_buf,
                'kafka_consumer_lag': int(lag_raw or 0),
                'ob_mid'            : redis_client.get('ob:mid'),
                'quote_active'      : redis_client.get('quote:active'),
                'fill_rate'         : redis_client.get('sim:fill_rate'),
            }

            if health['kafka_consumer_lag'] > 1000 and not is_on_cooldown('kafka_lag'):
                dispatch_alert('WARNING', 'PIPELINE',
                    f"Kafka lag: {health['kafka_consumer_lag']} messages behind",
                    ask_claude(health, 'Kafka consumer lag'), health)
                mark_alert_sent('kafka_lag')

            if tick_buf < 10 and not redis_client.get('ob:mid') \
                    and not is_on_cooldown('tick_buffer_empty'):
                dispatch_alert('CRITICAL', 'PIPELINE',
                    f"Tick buffer empty: {tick_buf} ticks — data feed may be down",
                    ask_claude(health, 'Tick buffer empty'), health)
                mark_alert_sent('tick_buffer_empty')

            socketio.emit('pipeline_health', health, namespace='/dashboard')

        except Exception as e:
            log.error(f'Pipeline health monitor error: {e}')
        time.sleep(HEALTH_INTERVAL)

# =============================================================================
# MONITOR 5 (NEW): HALT EVENTS
# Subscribes to Redis pub/sub 'sitaram:halts' published by engine.py.
# Forwards each halt event to the dashboard WebSocket immediately.
# =============================================================================
def monitor_halts():
    log.info('Halt monitor started — subscribing to sitaram:halts')
    while True:
        try:
            pubsub = redis_client.pubsub()
            pubsub.subscribe('sitaram:halts')
            for raw_msg in pubsub.listen():
                if raw_msg['type'] != 'message':
                    continue
                try:
                    event = json.loads(raw_msg['data'])
                    reason    = event.get('reason', 'UNKNOWN')
                    inv       = event.get('inventory', 0)
                    pnl       = event.get('pnl', 0)
                    halt_time = event.get('time', datetime.now(timezone.utc).isoformat())
                    session   = event.get('session_id', '')

                    payload = {
                        'session_id' : session,
                        'halt_time'  : halt_time,
                        'reason'     : reason,
                        'inventory'  : round(inv, 6),
                        'pnl'        : round(pnl, 4),
                    }

                    # Push to dashboard immediately (no cooldown — every halt matters)
                    socketio.emit('trading_halted', payload, namespace='/dashboard')

                    # Also dispatch as a standard alert
                    analysis = ask_claude(payload, f'Trading halted: {reason}')
                    dispatch_alert('CRITICAL', 'HALT',
                        f'TRADING HALTED at {halt_time} | Reason: {reason}',
                        analysis, payload)

                    log.warning(f'Halt event received and forwarded: {reason}')

                except Exception as e:
                    log.error(f'Halt event parse error: {e}')

        except Exception as e:
            log.error(f'Halt monitor Redis error: {e} — reconnecting in 10s')
            time.sleep(10)

# =============================================================================
# MONITOR 6 (NEW): DATA FEED STALL
# Polls session:data_stall Redis key written by engine.py.
# Dispatches alert when feed goes silent beyond threshold.
# =============================================================================
def monitor_data_feed():
    log.info('Data feed monitor started')
    while True:
        try:
            stall      = redis_client.get('session:data_stall') == '1'
            stall_since= redis_client.get('session:stall_since') or ''
            ob_ts_raw  = redis_client.get('ob:ts')
            ob_ts      = float(ob_ts_raw) if ob_ts_raw else 0

            if ob_ts > 0:
                data_age_sec = (time.time() * 1000 - ob_ts) / 1000
            else:
                data_age_sec = -1

            feed_metrics = {
                'data_stall'   : stall,
                'stall_since'  : stall_since,
                'data_age_sec' : round(data_age_sec, 1),
                'ob_ts'        : ob_ts,
            }

            if stall and not is_on_cooldown('data_feed_stall'):
                analysis = ask_claude(feed_metrics, 'Data feed stalled — no ticks received')
                dispatch_alert('CRITICAL', 'FEED',
                    f'DATA FEED STALLED since {stall_since} '
                    f'(age: {data_age_sec:.0f}s)',
                    analysis, feed_metrics)
                alert_cooldowns['data_feed_stall'] = time.time() - (ALERT_COOLDOWN - STALL_COOLDOWN)

            # Also alert if data age exceeds threshold even without stall flag
            if data_age_sec > STALL_ALERT_SEC and not is_on_cooldown('data_age_high'):
                dispatch_alert('WARNING', 'FEED',
                    f'Data age high: {data_age_sec:.0f}s (threshold: {STALL_ALERT_SEC}s)',
                    None, feed_metrics)
                alert_cooldowns['data_age_high'] = time.time() - (ALERT_COOLDOWN - STALL_COOLDOWN)

            socketio.emit('feed_status', feed_metrics, namespace='/dashboard')

        except Exception as e:
            log.error(f'Data feed monitor error: {e}')
        time.sleep(15)

# =============================================================================
# MONITOR 7 (NEW): LAYER 3 HEARTBEAT CONSUMER
# Consumes 'sitaram.heartbeat' Kafka topic published by bybit_live_producer.py
# every 10s. If gap between heartbeats exceeds HEARTBEAT_TIMEOUT_SEC, the
# producer process has died — even if ob:ts looks fresh (engine still running).
# This is the only monitor that can distinguish "engine alive, producer dead".
# =============================================================================
def monitor_heartbeat():
    if not KAFKA_AVAILABLE:
        log.warning('kafka-python not installed — heartbeat monitor disabled')
        return

    log.info(f'Heartbeat monitor started — consuming {HEARTBEAT_TOPIC} from {KAFKA_BOOTSTRAP}')
    last_heartbeat_ts = time.time()   # assume alive at startup
    consumer = None

    while True:
        try:
            if consumer is None:
                consumer = KafkaConsumer(
                    HEARTBEAT_TOPIC,
                    bootstrap_servers=KAFKA_BOOTSTRAP,
                    group_id='claude-agent-heartbeat',
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    consumer_timeout_ms=5000,   # non-blocking poll
                    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                )
                log.info('Heartbeat Kafka consumer connected')

            # Poll for up to 5s
            for msg in consumer:
                hb = msg.value
                last_heartbeat_ts = time.time()
                ob_connected    = hb.get('ob_connected', False)
                trade_connected = hb.get('trade_connected', False)
                uptime          = hb.get('uptime_sec', 0)
                # Emit live producer status to dashboard
                socketio.emit('heartbeat', {
                    'ts'             : hb.get('ts'),
                    'ob_connected'   : ob_connected,
                    'trade_connected': trade_connected,
                    'ob_msgs'        : hb.get('ob_msgs', 0),
                    'trade_msgs'     : hb.get('trade_msgs', 0),
                    'uptime_sec'     : uptime,
                    'last_seen'      : datetime.now(timezone.utc).isoformat(),
                }, namespace='/dashboard')

            # Check gap since last heartbeat
            gap = time.time() - last_heartbeat_ts
            if gap > HEARTBEAT_TIMEOUT_SEC and not is_on_cooldown('heartbeat_missing'):
                metrics = {
                    'gap_sec'           : round(gap, 1),
                    'timeout_sec'       : HEARTBEAT_TIMEOUT_SEC,
                    'last_heartbeat_ago': round(gap, 1),
                    'topic'             : HEARTBEAT_TOPIC,
                }
                analysis = ask_claude(metrics,
                    f'Producer heartbeat missing for {gap:.0f}s — bybit_live_producer.py may have died')
                dispatch_alert('CRITICAL', 'FEED',
                    f'PRODUCER HEARTBEAT LOST: no signal for {gap:.0f}s '
                    f'(threshold: {HEARTBEAT_TIMEOUT_SEC}s). '
                    f'Restart bybit_live_producer.py.',
                    analysis, metrics)
                alert_cooldowns['heartbeat_missing'] = time.time() - (ALERT_COOLDOWN - STALL_COOLDOWN)

        except Exception as e:
            log.error(f'Heartbeat monitor error: {e} — reconnecting in 15s')
            try:
                if consumer:
                    consumer.close()
            except Exception:
                pass
            consumer = None
            time.sleep(15)

# =============================================================================
# HTTP ENDPOINTS
# =============================================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'claude-agent',
                    'time': datetime.utcnow().isoformat()})

@app.route('/metrics', methods=['GET'])
def metrics():
    """Live metrics from Redis — polled by React dashboard every 2s."""
    try:
        m = get_live_metrics()
        m['services'] = {
            'kafka'      : bool(redis_client.get('ob:ts')),
            'redis'      : True,
            'python_eng' : bool(redis_client.get('quote:active')),
            'timescaledb': True,
            'claude_agent': True,
            'ray_head'   : True,
        }
        return jsonify(m)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/session/current', methods=['GET'])
def session_current():
    """Return live session metadata from Redis."""
    try:
        return jsonify({
            'session_id'   : redis_client.get('session:id') or '',
            'start_time'   : redis_client.get('session:start_time') or '',
            'halt_reason'  : redis_client.get('session:halt_reason') or '',
            'halt_time'    : redis_client.get('session:halt_time') or '',
            'data_stall'   : redis_client.get('session:data_stall') == '1',
            'stall_since'  : redis_client.get('session:stall_since') or '',
            'inventory'    : float(redis_client.get('quote:inventory') or 0),
            'inv_norm'     : float(redis_client.get('quote:inv_norm') or 0),
            'regime'       : redis_client.get('quote:regime') or 'unknown',
            'active'       : redis_client.get('quote:active') == '1',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/session/history', methods=['GET'])
def session_history():
    """Return full session history from sitaram_sessions.json."""
    try:
        with open(SESSION_JSON_PATH, 'r') as f:
            sessions = json.load(f)
        return jsonify(sessions)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/jenkins-status', methods=['POST'])
def jenkins_status():
    data = request.json
    socketio.emit('jenkins_update', data, namespace='/dashboard')
    log.info(f'Jenkins status received: {data}')
    return jsonify({'received': True})

@app.route('/alerts/recent', methods=['GET'])
def recent_alerts():
    try:
        conn = get_tsdb_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT alert_time, severity, category, message, claude_analysis
            FROM claude_alerts
            ORDER BY alert_time DESC LIMIT 50
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e), 'alerts': []}), 200

# =============================================================================
# STARTUP
# =============================================================================
if __name__ == '__main__':
    log.info('🧠 SITARAM Claude Intelligence Agent v2.0 starting...')

    monitors = [
        threading.Thread(target=monitor_ic,        name='IC-Monitor',        daemon=True),
        threading.Thread(target=monitor_inventory, name='Inventory-Monitor',  daemon=True),
        threading.Thread(target=monitor_pnl,       name='PnL-Monitor',       daemon=True),
        threading.Thread(target=monitor_pipeline,  name='Pipeline-Monitor',   daemon=True),
        threading.Thread(target=monitor_halts,     name='Halt-Monitor',       daemon=True),
        threading.Thread(target=monitor_data_feed, name='Feed-Monitor',       daemon=True),
        threading.Thread(target=monitor_heartbeat, name='Heartbeat-Monitor',  daemon=True),  # NEW Layer 3
    ]
    for t in monitors:
        t.start()
        log.info(f'Started monitor thread: {t.name}')

    log.info('🟢 All monitors active — Claude Intelligence Layer LIVE')
    log.info(f'   IC threshold     : {IC_MIN}')
    log.info(f'   Max drawdown     : {MAX_DRAWDOWN_PCT}%')
    log.info(f'   Max inventory    : {MAX_INVENTORY_BTC} BTC')   # Now correctly 0.10
    log.info(f'   Latency spike    : {LATENCY_SPIKE_MS}ms')
    log.info(f'   Alert cooldown   : {ALERT_COOLDOWN}s')
    log.info(f'   Stall threshold  : {STALL_ALERT_SEC}s')
    log.info(f'   Stall cooldown   : {STALL_COOLDOWN}s (faster re-alert for feed issues)')
    log.info(f'   Heartbeat timeout: {HEARTBEAT_TIMEOUT_SEC}s')
    log.info(f'   Heartbeat topic  : {HEARTBEAT_TOPIC}')
    log.info(f'   Session history  : {SESSION_JSON_PATH}')

    socketio.run(app, host='0.0.0.0', port=8001)
