import os, time, logging, threading, signal, sys
from flask import Flask, jsonify
from src.features.orderbook import OrderBookProcessor
from src.quoting.engine import QuotingEngine
from src.simulator.fill_simulator import FillSimulator

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("main")

app = Flask(__name__)
_healthy = False
_engine: QuotingEngine = None
_engine_lock = threading.Lock()

@app.route("/health")
def health():
    return (jsonify({"status":"ok"}), 200) if _healthy else (jsonify({"status":"starting"}), 503)

@app.route("/metrics")
def metrics():
    return jsonify(QuotingEngine.last_metrics())

# =============================================================================
# SIGNAL HANDLERS — must be registered on main thread
# Calls engine.shutdown() which handles session close + gate calculation
# =============================================================================
def _handle_signal(signum, frame):
    log.info(f"Signal {signum} received on main thread — triggering engine shutdown")
    with _engine_lock:
        if _engine is not None:
            _engine.shutdown()
    log.info("Engine shutdown complete — exiting")
    sys.exit(0)

def run_pipeline():
    global _healthy, _engine
    log.info("Starting SITARAM Python Engine...")
    ob  = OrderBookProcessor()
    qe  = QuotingEngine(ob)
    sim = FillSimulator(qe)
    with _engine_lock:
        _engine = qe
    _healthy = True
    log.info("Pipeline ready — entering main loop")
    while True:
        try:
            ob.process_tick()
            qe.update()
            sim.step()
            time.sleep(0.001)
        except Exception as e:
            log.error(f"Pipeline error: {e}", exc_info=True)
            time.sleep(1)

if __name__ == "__main__":
    # Register on main thread BEFORE starting pipeline thread
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)
    log.info("Signal handlers registered on main thread (SIGTERM, SIGINT)")

    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()

    # use_reloader=False required — reloader spawns child process which
    # intercepts signals before they reach our handler
    app.run(host="0.0.0.0", port=8000, use_reloader=False, threaded=True)
