import os, logging, threading
from src.ob_producer import main as ob_main
from src.trades_producer import main as trades_main

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("feeder")

def run():
    log.info("=" * 60)
    log.info("SITARAM Data Feeder Starting")
    log.info(f"Kafka: {os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')}")
    log.info(f"Speed: {os.getenv('PLAYBACK_SPEED', '1.0')}x")
    log.info("=" * 60)
    ob_thread     = threading.Thread(target=ob_main,     name="ob-producer",     daemon=True)
    trades_thread = threading.Thread(target=trades_main, name="trades-producer", daemon=True)
    ob_thread.start()
    trades_thread.start()
    ob_thread.join()
    trades_thread.join()
    log.info("All producers finished.")

if __name__ == "__main__":
    run()
