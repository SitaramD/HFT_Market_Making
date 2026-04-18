import os, json, time, logging, glob, csv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("trades-producer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC           = "trades-raw"
DATA_PATH       = os.getenv("DATA_PATH_TRADES", "/mnt/data")
PLAYBACK_SPEED  = float(os.getenv("PLAYBACK_SPEED", "1.0"))

def connect_kafka():
    for attempt in range(20):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                compression_type="gzip",
                batch_size=65536,
                linger_ms=5,
            )
            log.info(f"Kafka connected: {KAFKA_BOOTSTRAP}")
            return producer
        except NoBrokersAvailable:
            log.warning(f"Kafka not ready ({attempt+1}/20), retrying in 3s...")
            time.sleep(3)
    raise RuntimeError("Cannot connect to Kafka")

def stream_file(producer, filepath):
    log.info(f"Streaming: {filepath}")
    sent = 0
    prev_ts = None
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestamp"])
                msg = {
                    "id":     int(row["id"]),
                    "ts":     ts,
                    "symbol": "BTCUSDT",
                    "price":  float(row["price"]),
                    "volume": float(row["volume"]),
                    "side":   row["side"],
                    "rpi":    int(row["rpi"]),
                }
            except (KeyError, ValueError) as e:
                log.warning(f"Skipping bad row: {e}")
                continue
            if PLAYBACK_SPEED > 0 and prev_ts and ts > prev_ts:
                gap_ms = (ts - prev_ts) / PLAYBACK_SPEED
                if gap_ms > 0:
                    time.sleep(min(gap_ms / 1000.0, 1.0))
            prev_ts = ts
            producer.send(TOPIC, value=msg)
            sent += 1
            if sent % 10000 == 0:
                log.info(f"  Trades sent {sent} messages...")
    producer.flush()
    log.info(f"Done: {filepath} -- {sent} messages sent")
    return sent

def main():
    producer = connect_kafka()
    pattern = os.path.join(DATA_PATH, "**", "BTCUSDT_*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern2 = os.path.join(DATA_PATH, "BTCUSDT_*.csv")
        files = sorted(glob.glob(pattern2))
    if not files:
        log.error(f"No CSV files found in {DATA_PATH}")
        return
    log.info(f"Found {len(files)} trade files")
    total = 0
    for f in files:
        total += stream_file(producer, f)
    log.info(f"Trades done. Total: {total}")
    producer.close()

if __name__ == "__main__":
    main()
