"""
engine.py  —  SITARAM Quoting Engine  v3.2
Avellaneda-Stoikov market making with:
  - Half-spread $0.15 (validated Run 15)
  - AS_GAMMA = 2.0
  - MAX_INVENTORY = 0.10 BTC
  - Composite signal: 0.50*OBI5 + 0.35*micro_dev + 0.15*vol_adj
  - Inventory skew + regime detection

Changes v3.3 (IC fix):
  - [FIX]  IC calculation now live — rolling 200-tick buffer of
           (signal, future_return) pairs, Spearman rank correlation.
           Result written to Redis "metrics:ic_200" every IC_WRITE_INTERVAL
           ticks so MetricsPersister picks up real non-zero IC values.
  - [KEEP] All v3.2 latency tracking unchanged
  - [KEEP] All Run-15 parameters unchanged
"""
import os, time, math, logging, json, signal, uuid, atexit
from datetime import datetime, timezone
import redis
import numpy as np

from src.gate_calculator import calculate_and_write_gates
from src.session_logger import SessionLogger

log = logging.getLogger("quoting")

REDIS_HOST    = os.getenv("REDIS_HOST", "redis")
REDIS_PORT    = int(os.getenv("REDIS_PORT", 6379))

# ── Validated Run-15 parameters ───────────────────────────────────────────────
HALF_SPREAD   = float(os.getenv("HALF_SPREAD_USDT",       0.15))
AS_GAMMA      = float(os.getenv("AS_GAMMA",               2.0))
MAX_INVENTORY = float(os.getenv("MAX_INVENTORY_BTC",      0.10))
QUOTE_SIZE    = float(os.getenv("QUOTE_SIZE_BTC",         0.01))
TICK_SIZE     = float(os.getenv("TICK_SIZE",              0.1))
MAKER_FEE     = float(os.getenv("MAKER_FEE",             -0.0001))
TAKER_FEE     = float(os.getenv("TAKER_FEE",              0.0005))
INITIAL_CAP   = float(os.getenv("INITIAL_CAPITAL",        100000.0))
DAILY_LOSS_LIM= float(os.getenv("DAILY_LOSS_LIMIT",      -2000.0))
SIMULATED_LAT = float(os.getenv("SIMULATED_LATENCY_MS",  10))

REGIME_VOL_THRESHOLD = float(os.getenv("REGIME_VOL_THRESHOLD", 6e-5))
REGIME_VOL_WINDOW    = int(os.getenv("REGIME_VOL_WINDOW",       50))
DATA_STALL_SEC       = float(os.getenv("DATA_STALL_SEC",        30.0))

W_OBI   = 0.50
W_MICRO = 0.35
W_VOL   = 0.15

# ── IC calculation parameters ─────────────────────────────────────────────────
IC_WINDOW        = 200   # rolling window: number of ticks for IC calculation
IC_WRITE_INTERVAL = 30   # write IC to Redis every N ticks (not every tick)

# ── Snapshot interval ─────────────────────────────────────────────────────────
LIVE_SNAPSHOT_INTERVAL = 60   # seconds between live metric snapshots to master file

_metrics = {
    "inventory":       0.0,
    "cumulative_pnl":  0.0,
    "fill_rate":       0.0,
    "sharpe":          0.0,
    "ic_200":          0.0,
    "spread_ticks":    0.0,
    "total_trades":    0,
    "uptime_sec":      0,
    "regime":          "normal",
    "halt_reason":     "",
    "daily_pnl":       0.0,
    "signal":          0.0,
    "quote_bid":       0.0,
    "quote_ask":       0.0,
    "latency_min_ms":  0.0,
    "latency_avg_ms":  0.0,
    "latency_max_ms":  0.0,
}

_start      = time.time()
_day_start  = time.strftime("%Y%m%d")
_day_pnl    = 0.0


def _spearman_ic(predictions, actuals):
    """
    Spearman rank correlation between signal predictions and actual returns.
    More robust than Pearson for HFT signals — not distorted by outliers.
    Returns float in [-1, +1]. Returns 0.0 if insufficient data.
    """
    n = len(predictions)
    if n < 10:
        return 0.0
    p = np.array(predictions, dtype=float)
    a = np.array(actuals, dtype=float)
    # Check for zero variance — avoid divide-by-zero
    if p.std() == 0 or a.std() == 0:
        return 0.0
    # Rank both arrays
    rank_p = np.argsort(np.argsort(p)).astype(float)
    rank_a = np.argsort(np.argsort(a)).astype(float)
    # Pearson on ranks = Spearman
    corr = np.corrcoef(rank_p, rank_a)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


class QuotingEngine:

    def __init__(self, ob):
        self.ob               = ob
        self.inventory        = 0.0
        self.cash             = 0.0
        self.pnl              = 0.0
        self._prev_pnl        = 0.0
        self.trades           = 0
        self.fills            = []
        self._pnl_window      = []
        self._halted          = False
        self._halt_reason     = ""
        self._last_stall_alert= 0.0
        self._last_ob_ts      = 0.0
        self._last_snapshot   = 0.0
        self._last_day_logged = time.strftime("%Y-%m-%d")

        # ── IC rolling buffers ────────────────────────────────────────────────
        # Strategy: at each tick, record the current composite signal.
        # One tick later (future return = mid_t+1 - mid_t), pair them up.
        # IC = Spearman correlation over the last IC_WINDOW paired observations.
        self._ic_signal_buf   = []   # list of signal values (predictions)
        self._ic_return_buf   = []   # list of actual 1-tick returns (actuals)
        self._ic_prev_mid     = 0.0  # mid price at previous tick
        self._ic_prev_signal  = None # signal at previous tick (paired with current return)
        self._ic_tick_counter = 0    # counts ticks since last IC write to Redis
        self._ic_last_value   = 0.0  # last computed IC (for _metrics dict)

        # ── Latency tracking ──────────────────────────────────────────────────
        self._lat_min   = float("inf")
        self._lat_max   = 0.0
        self._lat_sum   = 0.0
        self._lat_count = 0

        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                             decode_responses=True)

        # ── Inventory recovery ────────────────────────────────────────────────
        saved_inv  = self.r.get("persist:inventory")
        saved_cash = self.r.get("persist:cash")
        saved_pnl  = self.r.get("persist:pnl")
        if saved_inv is not None:
            self.inventory  = float(saved_inv)
            self.cash       = float(saved_cash or 0.0)
            self.pnl        = float(saved_pnl  or 0.0)
            self._prev_pnl  = self.pnl
            log.info(f"Inventory restored: inv={self.inventory:.6f} BTC "
                     f"cash={self.cash:.4f} pnl={self.pnl:.4f}")
        else:
            log.info("No saved inventory — starting fresh at 0 BTC")

        # ── Session Logger ────────────────────────────────────────────────────
        self.logger = SessionLogger(params={
            "HALF_SPREAD":           HALF_SPREAD,
            "AS_GAMMA":              AS_GAMMA,
            "MAX_INVENTORY":         MAX_INVENTORY,
            "QUOTE_SIZE":            QUOTE_SIZE,
            "TICK_SIZE":             TICK_SIZE,
            "MAKER_FEE":             MAKER_FEE,
            "TAKER_FEE":             TAKER_FEE,
            "INITIAL_CAP":           INITIAL_CAP,
            "DAILY_LOSS_LIM":        DAILY_LOSS_LIM,
            "REGIME_VOL_THRESHOLD":  REGIME_VOL_THRESHOLD,
            "REGIME_VOL_WINDOW":     REGIME_VOL_WINDOW,
            "DATA_STALL_SEC":        DATA_STALL_SEC,
            "SIMULATED_LATENCY_MS":  SIMULATED_LAT,
        })
        self.session_id = self.logger.open_session()

        # ── Write session state to Redis ──────────────────────────────────────
        self.r.set("session:id",          self.session_id)
        self.r.set("session:start_time",  datetime.now(timezone.utc).isoformat())
        self.r.set("session:halt_reason", "")
        self.r.set("session:data_stall",  "0")
        self.r.set("quote:count",         "0")
        self.r.set("metrics:ic_200",      "0.0")  # initialise so persister never reads stale

        self._start_time      = datetime.now(timezone.utc)
        self._shutdown_called = False

        # Signal handlers registered on main thread in main.py
        # signal.signal() silently fails from non-main thread
        atexit.register(self._atexit_shutdown)

        log.info(f"QuotingEngine v3.3 | session={self.session_id} "
                 f"| MAX_INV={MAX_INVENTORY} BTC | DAILY_LOSS={DAILY_LOSS_LIM}")

    # ──────────────────────────────────────────────────────────────────────────
    # IC update — called every tick BEFORE writing quotes
    # ──────────────────────────────────────────────────────────────────────────
    def _update_ic(self, current_signal: float, current_mid: float):
        """
        Pairs last tick's signal with this tick's actual return,
        then updates the rolling IC buffer and conditionally writes
        the IC value to Redis key "metrics:ic_200".

        Flow:
          tick T:   record signal_T, record mid_T
          tick T+1: actual_return = mid_(T+1) - mid_T
                    pair (signal_T, actual_return) → append to buffers
                    compute IC over last IC_WINDOW pairs
                    every IC_WRITE_INTERVAL ticks → write to Redis
        """
        if self._ic_prev_mid > 0 and self._ic_prev_signal is not None:
            # Actual 1-tick mid-price return (in USDT)
            actual_return = current_mid - self._ic_prev_mid

            # Append to rolling buffers
            self._ic_signal_buf.append(self._ic_prev_signal)
            self._ic_return_buf.append(actual_return)

            # Trim to IC_WINDOW
            if len(self._ic_signal_buf) > IC_WINDOW:
                self._ic_signal_buf.pop(0)
                self._ic_return_buf.pop(0)

            # Compute IC every IC_WRITE_INTERVAL ticks
            self._ic_tick_counter += 1
            if self._ic_tick_counter >= IC_WRITE_INTERVAL:
                self._ic_tick_counter = 0
                ic = _spearman_ic(self._ic_signal_buf, self._ic_return_buf)
                self._ic_last_value = ic
                # Write to Redis — MetricsPersister reads this every 30s
                self.r.set("metrics:ic_200", str(round(ic, 6)))
                log.debug(f"IC updated: {ic:.6f} (n={len(self._ic_signal_buf)})")

        # Store current tick values for next tick pairing
        self._ic_prev_signal = current_signal
        self._ic_prev_mid    = current_mid

    # ──────────────────────────────────────────────────────────────────────────
    # Main update — called every tick by pipeline loop
    # ──────────────────────────────────────────────────────────────────────────
    def update(self):
        global _day_pnl, _day_start

        if not self.ob.is_ready():
            return

        mid = self.ob.mid()
        if mid == 0:
            return

        # ── Data stall detection ──────────────────────────────────────────────
        current_ob_ts = float(self.r.get("ob:ts") or 0)
        if current_ob_ts > self._last_ob_ts:
            self._last_ob_ts = current_ob_ts
            self.r.set("session:data_stall", "0")
        else:
            if self._last_ob_ts > 0:
                silence_sec = time.time() - (self._last_ob_ts / 1000)
                if silence_sec > DATA_STALL_SEC:
                    self.r.set("session:data_stall", "1")
                    self.r.set("session:stall_since", str(self._last_ob_ts))
                    if time.time() - self._last_stall_alert > 60:
                        self._last_stall_alert = time.time()
                        self.logger.append_stall_event(silence_sec, self._last_ob_ts)
                        log.warning(f"Feed stall: {silence_sec:.1f}s silent")

        # ── Daily reset ───────────────────────────────────────────────────────
        today     = time.strftime("%Y%m%d")
        today_iso = time.strftime("%Y-%m-%d")
        if today != _day_start:
            yesterday_iso = __import__("datetime").datetime(
                int(_day_start[:4]), int(_day_start[4:6]), int(_day_start[6:8])
            ).strftime("%Y-%m-%d")
            self.logger.record_daily_pnl(yesterday_iso, _day_pnl)
            _day_start        = today
            _day_pnl          = 0.0
            self._halted      = False
            self._halt_reason = ""
            self.r.set("session:halt_reason", "")
            log.info("New trading day — daily loss limit reset")

        # ── Daily loss limit halt ─────────────────────────────────────────────
        if _day_pnl <= DAILY_LOSS_LIM and self.trades > 10:
            if not self._halted:
                reason = (f"DAILY_LOSS_LIMIT | pnl={_day_pnl:.2f} "
                          f"<= limit={DAILY_LOSS_LIM}")
                self._trigger_halt(reason, mid)
            self._write_redis_halt(mid, self._halt_reason)
            return

        # ── Regime ────────────────────────────────────────────────────────────
        vol    = self.ob.vol_adj()
        regime = "high_vol" if vol > REGIME_VOL_THRESHOLD else "normal"

        # ── Composite signal ──────────────────────────────────────────────────
        obi5      = self.ob.obi(5)
        micro_dev = self.ob.micro_dev()
        signal_val = W_OBI * obi5 + W_MICRO * micro_dev + W_VOL * vol

        # ── IC update — pair previous signal with current actual return ───────
        # Must be called AFTER computing signal_val and mid, BEFORE writing Redis
        self._update_ic(signal_val, mid)

        # ── AS reservation price ──────────────────────────────────────────────
        inv_norm = max(-1.0, min(1.0, self.inventory / MAX_INVENTORY))
        sigma    = vol if vol > 0 else 1e-4
        r_price  = mid - AS_GAMMA * (sigma ** 2) * inv_norm * 1.0

        # ── Spread ────────────────────────────────────────────────────────────
        half        = HALF_SPREAD
        signal_skew = signal_val * TICK_SIZE

        my_bid = round(r_price - half - signal_skew, 1)
        my_ask = round(r_price + half - signal_skew, 1)
        if my_ask - my_bid < TICK_SIZE:
            my_ask = my_bid + TICK_SIZE

        # ── Write quotes + IC to Redis ─────────────────────────────────────────
        pipe = self.r.pipeline()
        pipe.set("quote:bid",            my_bid)
        pipe.set("quote:ask",            my_ask)
        pipe.set("quote:obi",            round(obi5, 4))
        pipe.set("quote:signal",         round(signal_val, 4))
        pipe.set("quote:regime",         regime)
        pipe.set("quote:active",         "1" if not self._halted else "0")
        pipe.set("quote:inventory",      round(self.inventory, 6))
        pipe.set("quote:inv_norm",       round(inv_norm, 4))
        pipe.set("session:halt_reason",  self._halt_reason)
        pipe.set("quote:cumulative_pnl", round(self.pnl, 4))
        pipe.set("quote:cash_balance",   round(self.cash, 4))
        pipe.set("quote:daily_pnl",      round(_day_pnl, 4))
        pipe.incr("quote:count")
        pipe.execute()

        # ── Sharpe ────────────────────────────────────────────────────────────
        self._pnl_window.append(self.pnl)
        if len(self._pnl_window) > 1000:
            self._pnl_window.pop(0)
        sharpe = self._rolling_sharpe()

        # ── Update metrics dict ───────────────────────────────────────────────
        lat_avg = round(self._lat_sum / self._lat_count, 2) if self._lat_count > 0 else 0.0
        lat_min = round(self._lat_min, 2) if self._lat_min != float("inf") else 0.0
        _metrics.update({
            "inventory":       round(self.inventory, 6),
            "cumulative_pnl":  round(self.pnl, 4),
            "spread_ticks":    round(my_ask - my_bid, 1),
            "uptime_sec":      int(time.time() - _start),
            "total_trades":    self.trades,
            "regime":          regime,
            "halt_reason":     self._halt_reason,
            "daily_pnl":       round(_day_pnl, 4),
            "signal":          round(signal_val, 4),
            "sharpe":          round(sharpe, 4),
            "ic_200":          round(self._ic_last_value, 6),
            "quote_bid":       my_bid,
            "quote_ask":       my_ask,
            "latency_min_ms":  lat_min,
            "latency_avg_ms":  lat_avg,
            "latency_max_ms":  round(self._lat_max, 2),
        })

        # ── Live snapshot to master file every 60s ────────────────────────────
        if time.time() - self._last_snapshot > LIVE_SNAPSHOT_INTERVAL:
            self._last_snapshot = time.time()
            lat_avg_snap = round(self._lat_sum / self._lat_count, 2) if self._lat_count > 0 else 0.0
            lat_min_snap = round(self._lat_min, 2) if self._lat_min != float("inf") else 0.0
            self.logger.update_live_metrics(
                cumulative_pnl   = self.pnl,
                total_trades     = self.trades,
                inventory_btc    = self.inventory,
                regime           = regime,
                data_feed_active = self.r.get("session:data_stall") != "1",
                latency_min_ms   = lat_min_snap,
                latency_avg_ms   = lat_avg_snap,
                latency_max_ms   = round(self._lat_max, 2),
            )
            self.logger.record_daily_pnl(today_iso, _day_pnl)

    # ──────────────────────────────────────────────────────────────────────────
    # Record a fill
    # ──────────────────────────────────────────────────────────────────────────
    def record_fill(self, side, price, size):
        global _day_pnl

        # ── Latency ───────────────────────────────────────────────────────────
        fill_time_ms   = time.time() * 1000
        bybit_ts_ms    = float(self.r.get("ob:bybit_ts") or 0)
        receive_ts_ms  = float(self.r.get("ob:receive_ts") or 0)

        if bybit_ts_ms > 0:
            latency_ms = round(fill_time_ms - bybit_ts_ms, 2)
        elif receive_ts_ms > 0:
            latency_ms = round(fill_time_ms - receive_ts_ms, 2)
        else:
            latency_ms = 0.0

        if latency_ms < 0 or latency_ms > 30000:
            latency_ms = 0.0

        if latency_ms > 0:
            self._lat_sum   += latency_ms
            self._lat_count += 1
            self._lat_min    = min(self._lat_min, latency_ms)
            self._lat_max    = max(self._lat_max, latency_ms)

        lat_avg = round(self._lat_sum / self._lat_count, 2) if self._lat_count > 0 else 0.0
        lat_min = round(self._lat_min, 2) if self._lat_min != float("inf") else 0.0
        pipe_lat = self.r.pipeline()
        pipe_lat.set("latency:last_ms",  latency_ms)
        pipe_lat.set("latency:avg_ms",   lat_avg)
        pipe_lat.set("latency:min_ms",   lat_min)
        pipe_lat.set("latency:max_ms",   round(self._lat_max, 2))
        pipe_lat.set("latency:count",    self._lat_count)
        pipe_lat.execute()

        # ── Cash + inventory ──────────────────────────────────────────────────
        if side == "buy":
            self.inventory += size
            self.cash -= price * size * (1 + MAKER_FEE)
        else:
            self.inventory -= size
            self.cash += price * size * (1 - MAKER_FEE)

        mid       = float(self.r.get("ob:mid") or price)
        self.pnl  = self.cash + (self.inventory * mid)
        _day_pnl  = self.pnl

        fill_pnl       = self.pnl - self._prev_pnl
        self._prev_pnl = self.pnl

        self.trades += 1
        self.fills.append(1)

        log.debug(f"Fill: {side} {size:.4f} @ {price:.1f} | "
                  f"inv={self.inventory:.4f} pnl={self.pnl:.2f} "
                  f"fill_pnl={fill_pnl:.4f} latency={latency_ms:.1f}ms")

        if abs(self.inventory) > MAX_INVENTORY:
            reason = (f"INVENTORY_BREACH | inv={self.inventory:.6f} BTC "
                      f"| max={MAX_INVENTORY} BTC")
            self._trigger_halt(reason, price)
            self.r.set("quote:active", "0")

        self.r.publish("sitaram:fills", json.dumps({
            "time":            datetime.now(timezone.utc).isoformat(),
            "side":            side,
            "price":           price,
            "quantity":        size,
            "realized_pnl":    round(fill_pnl, 6),
            "cumulative_pnl":  round(self.pnl, 6),
            "cash_balance":    round(self.cash, 6),
            "fill_latency_ms": latency_ms,
            "adverse_fill":    False,
            "inventory_after": round(self.inventory, 6),
            "quote_id":        str(uuid.uuid4()),
        }))

    # ──────────────────────────────────────────────────────────────────────────
    # Centralised halt
    # ──────────────────────────────────────────────────────────────────────────
    def _trigger_halt(self, reason: str, mid: float):
        self._halted      = True
        self._halt_reason = reason
        halt_time         = datetime.now(timezone.utc).isoformat()

        self.r.set("session:halt_reason", reason)
        self.r.set("session:halt_time",   halt_time)
        self.r.set("quote:active",        "0")
        self.r.set("quote:regime",        "halted")

        self.r.publish("sitaram:halts", json.dumps({
            "session_id": self.session_id,
            "time":       halt_time,
            "reason":     reason,
            "inventory":  round(self.inventory, 6),
            "pnl":        round(self.pnl, 4),
            "mid":        round(mid, 1),
        }))

        self.logger.append_halt_event(reason, self.inventory, self.pnl, mid)
        log.warning(f"HALT | {reason}")

    # ──────────────────────────────────────────────────────────────────────────
    # Graceful shutdown
    # ──────────────────────────────────────────────────────────────────────────
    def shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True

        global _day_pnl
        today_iso = time.strftime("%Y-%m-%d")
        self.logger.record_daily_pnl(today_iso, _day_pnl)

        lat_avg_final = round(self._lat_sum / self._lat_count, 2) if self._lat_count > 0 else 0.0
        lat_min_final = round(self._lat_min, 2) if self._lat_min != float("inf") else 0.0

        final_metrics = {
            "cumulative_pnl_usdt":          round(self.pnl, 4),
            "daily_pnl_usdt":               round(_day_pnl, 4),
            "cash_balance_usdt":            round(self.cash, 4),
            "total_trades":                 self.trades,
            "final_inventory_btc":          round(self.inventory, 6),
            "max_inventory_btc":            round(max(
                abs(f) for f in self._pnl_window) if self._pnl_window else 0, 6),
            "uptime_sec":                   int(time.time() - _start),
            "sharpe":                       round(self._rolling_sharpe(), 4),
            "fill_rate":                    _metrics.get("fill_rate", 0),
            "ic_value":                     round(self._ic_last_value, 6),
            "max_drawdown_pct":             0.0,
            "adverse_selection_rate":       0.0,
            "avg_spread_usdt":              _metrics.get("spread_ticks", 0),
            "avg_obi":                      0.0,
            "avg_composite_signal":         _metrics.get("signal", 0),
            "inventory_mean_reversion_pct": 0.0,
            "halt_reason":                  self._halt_reason or "MANUAL_STOP",
            "latency_min_ms":               lat_min_final,
            "latency_avg_ms":               lat_avg_final,
            "latency_max_ms":               round(self._lat_max, 2),
            "latency_fill_count":           self._lat_count,
        }

        # Persist inventory for next restart
        self.r.set("persist:inventory", str(round(self.inventory, 6)))
        self.r.set("persist:cash",      str(round(self.cash, 4)))
        self.r.set("persist:pnl",       str(round(self.pnl, 4)))
        self.r.set("persist:session_id", self.session_id)
        self.r.set("persist:saved_at",  datetime.now(timezone.utc).isoformat())
        log.info(f"Inventory saved: inv={self.inventory:.6f} BTC "
                 f"cash={self.cash:.4f} pnl={self.pnl:.4f}")

        self.logger.close_session(final_metrics)
        log.info(f"Session {self.session_id} written to master file")

        # ── Gate evaluation on shutdown ───────────────────────────────────────
        try:
            log.info("Running gate calculator — evaluating all 11 Jenkins gates...")
            calculate_and_write_gates(
                session_id = self.session_id,
                start_time = self._start_time,
                end_time   = datetime.now(timezone.utc),
            )
            log.info("Gate evaluation complete — session marked COMPLETED in JSON")
        except Exception as e:
            log.error(f"Gate calculator failed (non-fatal): {e}", exc_info=True)

    def _handle_sigterm(self, signum, frame):
        log.info("SIGTERM received — shutting down cleanly")
        self.shutdown()

    def _handle_sigint(self, signum, frame):
        log.info("SIGINT received — shutting down cleanly")
        self.shutdown()

    def _atexit_shutdown(self):
        if not self._shutdown_called:
            log.info("atexit triggered — writing session as safety net")
            self.shutdown()

    # ──────────────────────────────────────────────────────────────────────────
    # Rolling Sharpe
    # ──────────────────────────────────────────────────────────────────────────
    def _rolling_sharpe(self):
        if len(self._pnl_window) < 10:
            return 0.0
        rets = [self._pnl_window[i] - self._pnl_window[i-1]
                for i in range(1, len(self._pnl_window))]
        mean = sum(rets) / len(rets)
        var  = sum((r - mean)**2 for r in rets) / len(rets)
        std  = var ** 0.5
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(429000)

    def _write_redis_halt(self, mid, reason=""):
        self.r.set("quote:bid",    round(mid - HALF_SPREAD, 1))
        self.r.set("quote:ask",    round(mid + HALF_SPREAD, 1))
        self.r.set("quote:active", "0")
        self.r.set("quote:regime", "halted")
        if reason:
            self.r.set("session:halt_reason", reason)

    @staticmethod
    def last_metrics():
        return dict(_metrics)
