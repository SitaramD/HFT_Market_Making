"""
engine.py  —  SITARAM Quoting Engine  v3.4
Avellaneda-Stoikov market making with:
  - Half-spread $0.15 (validated Run 15)
  - AS_GAMMA = 2.0
  - MAX_INVENTORY = 0.01 BTC  [BUG FIX v3.4: was 0.10]
  - Composite signal: 0.50*OBI5 + 0.35*micro_dev + 0.15*vol_adj
  - Inventory skew + regime detection

Changes v3.4:
  [FIX]  MAX_INVENTORY default corrected: 0.10 → 0.01 BTC
  [FIX]  P&L NOT restored across sessions — only inventory + cash restored
  [FIX]  _day_pnl now tracks delta from session-start cash, not cumulative pnl
  [FIX]  gate_calculator spread fallback removed (caused cascade transaction abort)
  [NEW]  US market session gate: 09:30–16:00 US/Eastern, DST-aware (zoneinfo)
  [NEW]  Pre-market boot window: engine boots before 09:30, holds quotes until open
  [NEW]  Auto-stop at 16:00 EST — triggers graceful shutdown
  [NEW]  Inventory close window: 15:15–15:45 aggressive quoting to flatten position
  [NEW]  Hard flatten at 15:45 — synthetic fill at mid if still open
  [NEW]  On shutdown: inventory zeroed, cash balance persisted for next session
  [NEW]  Daily P&L = realised P&L from fills this session only (no cumulative carryover)
  [KEEP] All v3.3 IC calculation unchanged
  [KEEP] All v3.3 latency tracking unchanged
"""
import os, time, math, logging, json, signal, uuid, atexit, sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
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
MAX_INVENTORY = float(os.getenv("MAX_INVENTORY_BTC",      0.01))   # [FIX v3.4] was 0.10
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
IC_WINDOW         = 200   # rolling window: number of ticks for IC calculation
IC_WRITE_INTERVAL = 30    # write IC to Redis every N ticks (not every tick)

# ── Snapshot interval ─────────────────────────────────────────────────────────
LIVE_SNAPSHOT_INTERVAL = 60   # seconds between live metric snapshots to master file

# ── US Market session parameters ──────────────────────────────────────────────
US_TZ                   = ZoneInfo("America/New_York")   # handles EST/EDT automatically
MARKET_OPEN_H           = 9
MARKET_OPEN_M           = 30
MARKET_CLOSE_H          = 16
MARKET_CLOSE_M          = 0
CLOSE_WINDOW_START_H    = 15   # begin aggressive closing quotes at 15:15
CLOSE_WINDOW_START_M    = 15
HARD_FLATTEN_H          = 15   # force-flatten any residual inventory at 15:45
HARD_FLATTEN_M          = 45

# Aggressive closing spread — tight enough to attract fills, wide enough to
# not cross. Uses half of normal half-spread.
CLOSING_HALF_SPREAD = HALF_SPREAD * 0.3   # 0.045 USDT — very aggressive

_metrics = {
    "inventory":       0.0,
    "daily_pnl":       0.0,
    "fill_rate":       0.0,
    "sharpe":          0.0,
    "ic_200":          0.0,
    "spread_ticks":    0.0,
    "total_trades":    0,
    "uptime_sec":      0,
    "regime":          "normal",
    "halt_reason":     "",
    "signal":          0.0,
    "quote_bid":       0.0,
    "quote_ask":       0.0,
    "latency_min_ms":  0.0,
    "latency_avg_ms":  0.0,
    "latency_max_ms":  0.0,
    "session_phase":   "pre_market",   # pre_market | trading | closing | closed
}

_start      = time.time()
_day_start  = time.strftime("%Y%m%d")
_day_pnl    = 0.0   # [FIX v3.4] delta from session-start cash, not cumulative pnl


# ── DST-aware US market time helpers ──────────────────────────────────────────

def _now_us_eastern() -> datetime:
    """Return current time in US/Eastern (handles EST/EDT automatically)."""
    return datetime.now(US_TZ)


def _market_open_today() -> datetime:
    """Return today's market open as a timezone-aware datetime in US/Eastern."""
    now_et = _now_us_eastern()
    return now_et.replace(
        hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
        second=0, microsecond=0
    )


def _market_close_today() -> datetime:
    """Return today's market close as a timezone-aware datetime in US/Eastern."""
    now_et = _now_us_eastern()
    return now_et.replace(
        hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M,
        second=0, microsecond=0
    )


def _close_window_start_today() -> datetime:
    """Return today's closing-window start (15:15 ET)."""
    now_et = _now_us_eastern()
    return now_et.replace(
        hour=CLOSE_WINDOW_START_H, minute=CLOSE_WINDOW_START_M,
        second=0, microsecond=0
    )


def _hard_flatten_today() -> datetime:
    """Return today's hard-flatten deadline (15:45 ET)."""
    now_et = _now_us_eastern()
    return now_et.replace(
        hour=HARD_FLATTEN_H, minute=HARD_FLATTEN_M,
        second=0, microsecond=0
    )


def _is_before_market_open() -> bool:
    return _now_us_eastern() < _market_open_today()


def _is_after_market_close() -> bool:
    return _now_us_eastern() >= _market_close_today()


def _is_in_closing_window() -> bool:
    now = _now_us_eastern()
    return _close_window_start_today() <= now < _hard_flatten_today()


def _is_past_hard_flatten() -> bool:
    return _now_us_eastern() >= _hard_flatten_today()


def _seconds_to_market_open() -> float:
    delta = _market_open_today() - _now_us_eastern()
    return max(0.0, delta.total_seconds())


def _seconds_to_market_close() -> float:
    delta = _market_close_today() - _now_us_eastern()
    return max(0.0, delta.total_seconds())


# ── Spearman IC ───────────────────────────────────────────────────────────────

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
    if p.std() == 0 or a.std() == 0:
        return 0.0
    rank_p = np.argsort(np.argsort(p)).astype(float)
    rank_a = np.argsort(np.argsort(a)).astype(float)
    corr = np.corrcoef(rank_p, rank_a)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
class QuotingEngine:

    def __init__(self, ob):
        self.ob               = ob
        self.inventory        = 0.0
        self.cash             = 0.0
        self.pnl              = 0.0   # mark-to-market total: cash + inventory*mid
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

        # ── Session phase flags ───────────────────────────────────────────────
        self._phase               = "pre_market"   # pre_market|trading|closing|closed
        self._hard_flatten_done   = False
        self._auto_close_logged   = False
        self._closing_window_logged = False

        # ── IC rolling buffers ────────────────────────────────────────────────
        self._ic_signal_buf   = []
        self._ic_return_buf   = []
        self._ic_prev_mid     = 0.0
        self._ic_prev_signal  = None
        self._ic_tick_counter = 0
        self._ic_last_value   = 0.0

        # ── Latency tracking ──────────────────────────────────────────────────
        self._lat_min   = float("inf")
        self._lat_max   = 0.0
        self._lat_sum   = 0.0
        self._lat_count = 0

        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                             decode_responses=True)

        # ── Check market hours on startup ─────────────────────────────────────
        # Reject if started after market close
        if _is_after_market_close():
            log.error(
                f"Engine started after market close "
                f"({_now_us_eastern().strftime('%H:%M:%S %Z')}). "
                f"Next session starts at 09:30 ET tomorrow. Exiting."
            )
            sys.exit(1)

        # Log pre-market boot if applicable
        if _is_before_market_open():
            secs = _seconds_to_market_open()
            log.info(
                f"Pre-market boot — current time: "
                f"{_now_us_eastern().strftime('%H:%M:%S %Z')}. "
                f"Quoting will begin at 09:30 ET "
                f"(in {secs/60:.1f} minutes). "
                f"Order book warming up..."
            )
            self._phase = "pre_market"
        else:
            self._phase = "trading"
            log.info(
                f"Engine started within market hours: "
                f"{_now_us_eastern().strftime('%H:%M:%S %Z')} ET"
            )

        # ── Inventory recovery (cash only — P&L always resets) ────────────────
        # [FIX v3.4] Restore inventory + cash from prior session close.
        #            Do NOT restore pnl — daily P&L always starts at 0.
        #            Cash carries over because session close flattened inventory
        #            at mid price and updated cash accordingly.
        saved_inv  = self.r.get("persist:inventory")
        saved_cash = self.r.get("persist:cash")
        if saved_inv is not None:
            self.inventory = float(saved_inv)
            self.cash      = float(saved_cash or INITIAL_CAP)
            # P&L starts at 0 — will be calculated from first tick
            self.pnl       = 0.0
            self._prev_pnl = 0.0
            log.info(
                f"Session restored: inv={self.inventory:.6f} BTC "
                f"cash={self.cash:.4f} USDT | daily P&L starting at 0"
            )
        else:
            # First ever session — start fresh
            self.cash = INITIAL_CAP
            log.info(
                f"First session — starting fresh: "
                f"cash={self.cash:.2f} USDT, inventory=0 BTC"
            )

        # [FIX v3.4] Baseline cash at session start — used to compute daily P&L.
        # daily_pnl = current_cash + inventory*mid - self._session_start_cash
        # This isolates THIS session's realised P&L from prior sessions.
        self._session_start_cash = self.cash

        # ── Session Logger ────────────────────────────────────────────────────
        self.logger = SessionLogger(params={
            "half_spread_usdt":      HALF_SPREAD,
            "as_gamma":              AS_GAMMA,
            "max_inventory_btc":     MAX_INVENTORY,
            "quote_size_btc":        QUOTE_SIZE,
            "tick_size":             TICK_SIZE,
            "maker_fee":             MAKER_FEE,
            "taker_fee":             TAKER_FEE,
            "initial_capital_usdt":  INITIAL_CAP,
            "daily_loss_limit_usdt": DAILY_LOSS_LIM,
            "regime_vol_threshold":  REGIME_VOL_THRESHOLD,
            "regime_vol_window":     REGIME_VOL_WINDOW,
            "data_stall_threshold_sec": DATA_STALL_SEC,
            "simulated_latency_ms":  SIMULATED_LAT,
        })
        self.session_id = self.logger.open_session()

        # ── Write session state to Redis ──────────────────────────────────────
        self.r.set("session:id",           self.session_id)
        self.r.set("session:start_time",   datetime.now(timezone.utc).isoformat())
        self.r.set("session:halt_reason",  "")
        self.r.set("session:data_stall",   "0")
        self.r.set("session:phase",        self._phase)
        self.r.set("quote:count",          "0")
        self.r.set("quote:daily_pnl",      "0.0")
        self.r.set("metrics:ic_200",       "0.0")

        self._start_time      = datetime.now(timezone.utc)
        self._shutdown_called = False

        atexit.register(self._atexit_shutdown)

        log.info(
            f"QuotingEngine v3.4 | session={self.session_id} "
            f"| MAX_INV={MAX_INVENTORY} BTC "
            f"| DAILY_LOSS={DAILY_LOSS_LIM} "
            f"| session_start_cash={self._session_start_cash:.2f}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # IC update — called every tick BEFORE writing quotes
    # ──────────────────────────────────────────────────────────────────────────
    def _update_ic(self, current_signal: float, current_mid: float):
        """
        Pairs last tick's signal with this tick's actual return,
        then updates the rolling IC buffer and conditionally writes
        the IC value to Redis key "metrics:ic_200".
        """
        if self._ic_prev_mid > 0 and self._ic_prev_signal is not None:
            actual_return = current_mid - self._ic_prev_mid
            self._ic_signal_buf.append(self._ic_prev_signal)
            self._ic_return_buf.append(actual_return)
            if len(self._ic_signal_buf) > IC_WINDOW:
                self._ic_signal_buf.pop(0)
                self._ic_return_buf.pop(0)
            self._ic_tick_counter += 1
            if self._ic_tick_counter >= IC_WRITE_INTERVAL:
                self._ic_tick_counter = 0
                ic = _spearman_ic(self._ic_signal_buf, self._ic_return_buf)
                self._ic_last_value = ic
                self.r.set("metrics:ic_200", str(round(ic, 6)))
                log.debug(f"IC updated: {ic:.6f} (n={len(self._ic_signal_buf)})")
        self._ic_prev_signal = current_signal
        self._ic_prev_mid    = current_mid

    # ──────────────────────────────────────────────────────────────────────────
    # Session phase management
    # ──────────────────────────────────────────────────────────────────────────
    def _update_phase(self) -> str:
        """
        Returns the current session phase and handles transitions.
        Phases: pre_market → trading → closing → closed

        Transitions:
          pre_market → trading   at 09:30 ET
          trading    → closing   at 15:15 ET
          closing    → closed    at 16:00 ET (triggers shutdown)
        """
        now_et = _now_us_eastern()

        # Already in closed phase — handled by caller
        if self._phase == "closed":
            return self._phase

        # ── Transition: pre_market → trading ──────────────────────────────────
        if self._phase == "pre_market" and not _is_before_market_open():
            self._phase = "trading"
            self.r.set("session:phase", "trading")
            log.info(
                f"MARKET OPEN — quoting begins. "
                f"Time: {now_et.strftime('%H:%M:%S %Z')}"
            )

        # ── Transition: trading → closing ─────────────────────────────────────
        if self._phase == "trading" and _is_in_closing_window():
            self._phase = "closing"
            self.r.set("session:phase", "closing")
            if not self._closing_window_logged:
                self._closing_window_logged = True
                log.info(
                    f"CLOSING WINDOW — aggressive inventory reduction begins. "
                    f"Target: flat by 15:45 ET. "
                    f"Current inventory: {self.inventory:.6f} BTC. "
                    f"Time: {now_et.strftime('%H:%M:%S %Z')}"
                )

        # ── Hard flatten at 15:45 ─────────────────────────────────────────────
        if self._phase == "closing" and _is_past_hard_flatten() \
                and not self._hard_flatten_done:
            self._hard_flatten()

        # ── Auto-shutdown at 16:00 ────────────────────────────────────────────
        if _is_after_market_close():
            if not self._auto_close_logged:
                self._auto_close_logged = True
                log.info(
                    f"MARKET CLOSE — 16:00 ET reached. "
                    f"Triggering auto-shutdown. "
                    f"Time: {now_et.strftime('%H:%M:%S %Z')}"
                )
            self._phase = "closed"
            self.r.set("session:phase", "closed")

        return self._phase

    # ──────────────────────────────────────────────────────────────────────────
    # Hard flatten — synthetic fill at mid to zero inventory
    # ──────────────────────────────────────────────────────────────────────────
    def _hard_flatten(self):
        """
        Called at 15:45 ET if inventory is still non-zero.
        Books a synthetic fill at current mid price to zero the position.
        Records as a normal fill in TimescaleDB with quote_id="SESSION_FLATTEN".
        """
        if abs(self.inventory) < 1e-8:
            self._hard_flatten_done = True
            log.info("Hard flatten called — inventory already flat. No action needed.")
            return

        mid = float(self.r.get("ob:mid") or 0)
        if mid == 0:
            log.warning("Hard flatten: ob:mid not available — using last known price")
            mid = float(self.r.get("ob:last_price") or 77000.0)

        flatten_qty  = abs(self.inventory)
        flatten_side = "sell" if self.inventory > 0 else "buy"

        log.warning(
            f"HARD FLATTEN at 15:45 ET | "
            f"side={flatten_side} qty={flatten_qty:.6f} BTC @ {mid:.1f} "
            f"(synthetic fill at mid)"
        )

        # Book the closing fill using taker fee (market-style execution)
        if flatten_side == "sell":
            self.cash      += mid * flatten_qty * (1 - TAKER_FEE)
            self.inventory -= flatten_qty
        else:
            self.cash      -= mid * flatten_qty * (1 + TAKER_FEE)
            self.inventory += flatten_qty

        # Snap inventory to exactly 0.0 (avoid floating point residuals)
        self.inventory = 0.0

        # Recalculate P&L after flatten
        self.pnl = self.cash   # inventory is 0, so pnl = cash only

        # [FIX v3.4] daily_pnl = change in cash from session start (realised only)
        global _day_pnl
        _day_pnl = self.cash - self._session_start_cash
        self._prev_pnl = self.pnl

        # Publish as a fill so persister writes it to TimescaleDB
        self.trades += 1
        self.r.publish("sitaram:fills", json.dumps({
            "time":            datetime.now(timezone.utc).isoformat(),
            "side":            flatten_side,
            "price":           mid,
            "quantity":        flatten_qty,
            "realized_pnl":    round(_day_pnl, 6),
            "cumulative_pnl":  round(self.pnl, 6),
            "cash_balance":    round(self.cash, 6),
            "fill_latency_ms": 0.0,
            "adverse_fill":    False,
            "inventory_after": 0.0,
            "quote_id":        "SESSION_FLATTEN_1545",
            "session_id":      self.session_id,
        }))

        # Update Redis
        pipe = self.r.pipeline()
        pipe.set("quote:inventory",      "0.0")
        pipe.set("quote:cash_balance",   round(self.cash, 4))
        pipe.set("quote:daily_pnl",      round(_day_pnl, 4))
        pipe.set("session:halt_reason",  "SESSION_FLATTEN_1545")
        pipe.execute()

        self._hard_flatten_done = True
        log.info(
            f"Hard flatten complete | "
            f"cash={self.cash:.4f} USDT | "
            f"daily_pnl={_day_pnl:.4f} USDT | "
            f"inventory=0.0 BTC"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Closing spread — aggressive quotes to reduce inventory
    # ──────────────────────────────────────────────────────────────────────────
    def _closing_quotes(self, mid: float) -> tuple:
        """
        Returns (bid, ask) during the 15:15–15:45 closing window.
        Uses tighter spread to attract fills. Skews heavily toward
        the side that reduces inventory:
          - Long inventory → skew ask down aggressively (want to sell)
          - Short inventory → skew bid up aggressively (want to buy)
        """
        inv_sign   = 1.0 if self.inventory > 0 else -1.0
        half       = CLOSING_HALF_SPREAD

        # Skew: push the reducing side toward mid (or even cross it slightly)
        # so fills happen at near-mid prices
        if self.inventory > 0:
            # Long — want to sell: push ask toward mid
            my_bid = round(mid - half * 2, 1)   # wide bid (don't want more longs)
            my_ask = round(mid - half, 1)        # tight ask (aggressive sell)
        elif self.inventory < 0:
            # Short — want to buy: push bid toward mid
            my_bid = round(mid + half, 1)        # tight bid (aggressive buy)
            my_ask = round(mid + half * 2, 1)   # wide ask (don't want more shorts)
        else:
            # Already flat — use normal spread to capture any late fills
            my_bid = round(mid - half, 1)
            my_ask = round(mid + half, 1)

        # Ensure minimum tick separation
        if my_ask - my_bid < TICK_SIZE:
            my_ask = my_bid + TICK_SIZE

        return my_bid, my_ask

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

        # ── Update session phase ──────────────────────────────────────────────
        phase = self._update_phase()

        # ── If closed — trigger shutdown and return ───────────────────────────
        if phase == "closed":
            if not self._shutdown_called:
                log.info("Phase=closed — calling shutdown()")
                self.shutdown()
            return

        # ── If pre-market — warm up OB but do not quote ───────────────────────
        if phase == "pre_market":
            # Still update IC and write a minimal Redis heartbeat
            # so the dashboard shows the engine is alive
            secs_left = _seconds_to_market_open()
            if int(secs_left) % 60 == 0 and secs_left > 0:
                log.info(
                    f"Pre-market: {secs_left/60:.1f} min to open. "
                    f"OB mid={mid:.1f}"
                )
            self.r.set("quote:active",       "0")
            self.r.set("session:phase",      "pre_market")
            self.r.set("quote:cumulative_pnl", "0.0")
            self.r.set("quote:daily_pnl",    "0.0")
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
        obi5       = self.ob.obi(5)
        micro_dev  = self.ob.micro_dev()
        signal_val = W_OBI * obi5 + W_MICRO * micro_dev + W_VOL * vol

        # ── IC update ─────────────────────────────────────────────────────────
        self._update_ic(signal_val, mid)

        # ── Quote generation — phase-aware ────────────────────────────────────
        if phase == "closing":
            # Aggressive closing quotes
            my_bid, my_ask = self._closing_quotes(mid)
        else:
            # Normal AS quoting
            inv_norm = max(-1.0, min(1.0, self.inventory / MAX_INVENTORY))
            sigma    = vol if vol > 0 else 1e-4
            r_price  = mid - AS_GAMMA * (sigma ** 2) * inv_norm * 1.0
            half        = HALF_SPREAD
            signal_skew = signal_val * TICK_SIZE
            my_bid = round(r_price - half - signal_skew, 1)
            my_ask = round(r_price + half - signal_skew, 1)
            if my_ask - my_bid < TICK_SIZE:
                my_ask = my_bid + TICK_SIZE

        # ── Write quotes to Redis ──────────────────────────────────────────────
        pipe = self.r.pipeline()
        pipe.set("quote:bid",            my_bid)
        pipe.set("quote:ask",            my_ask)
        pipe.set("quote:obi",            round(obi5, 4))
        pipe.set("quote:signal",         round(signal_val, 4))
        pipe.set("quote:regime",         regime)
        pipe.set("quote:active",         "1" if not self._halted else "0")
        pipe.set("quote:inventory",      round(self.inventory, 6))
        pipe.set("quote:inv_norm",       round(
            max(-1.0, min(1.0, self.inventory / MAX_INVENTORY)), 4))
        pipe.set("session:halt_reason",  self._halt_reason)
        pipe.set("quote:daily_pnl",      round(_day_pnl, 4))
        pipe.set("quote:cash_balance",   round(self.cash, 4))
        pipe.set("session:phase",        phase)
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
            "daily_pnl":       round(_day_pnl, 4),
            "spread_ticks":    round(my_ask - my_bid, 1),
            "uptime_sec":      int(time.time() - _start),
            "total_trades":    self.trades,
            "regime":          regime,
            "halt_reason":     self._halt_reason,
            "signal":          round(signal_val, 4),
            "sharpe":          round(sharpe, 4),
            "ic_200":          round(self._ic_last_value, 6),
            "quote_bid":       my_bid,
            "quote_ask":       my_ask,
            "latency_min_ms":  lat_min,
            "latency_avg_ms":  lat_avg,
            "latency_max_ms":  round(self._lat_max, 2),
            "session_phase":   phase,
        })

        # ── Live snapshot to master file every 60s ────────────────────────────
        if time.time() - self._last_snapshot > LIVE_SNAPSHOT_INTERVAL:
            self._last_snapshot = time.time()
            lat_avg_snap = round(self._lat_sum / self._lat_count, 2) if self._lat_count > 0 else 0.0
            lat_min_snap = round(self._lat_min, 2) if self._lat_min != float("inf") else 0.0
            today_iso    = time.strftime("%Y-%m-%d")
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
    # Record a fill — called by FillSimulator on each simulated execution
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

        # [FIX v3.4] daily_pnl = cash delta from session start + MTM on open inventory
        # This correctly measures only THIS session's realised + unrealised P&L.
        _day_pnl  = (self.cash - self._session_start_cash) + (self.inventory * mid)

        fill_pnl       = self.pnl - self._prev_pnl
        self._prev_pnl = self.pnl

        self.trades += 1
        self.fills.append(1)

        log.debug(
            f"Fill: {side} {size:.4f} @ {price:.1f} | "
            f"inv={self.inventory:.4f} daily_pnl={_day_pnl:.2f} "
            f"fill_pnl={fill_pnl:.4f} latency={latency_ms:.1f}ms"
        )

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
            "session_id":      self.session_id,
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
            "pnl":        round(_day_pnl, 4),
            "mid":        round(mid, 1),
        }))

        self.logger.append_halt_event(reason, self.inventory, _day_pnl, mid)
        log.warning(f"HALT | {reason}")

    # ──────────────────────────────────────────────────────────────────────────
    # Graceful shutdown
    # ──────────────────────────────────────────────────────────────────────────
    def shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True

        global _day_pnl

        # ── Ensure inventory is flat before closing ───────────────────────────
        # If hard flatten wasn't triggered (e.g. manual stop mid-session),
        # do it now at current mid price.
        if abs(self.inventory) > 1e-8 and not self._hard_flatten_done:
            log.info(
                f"Shutdown: inventory not flat ({self.inventory:.6f} BTC). "
                f"Flattening at mid price before closing session."
            )
            self._hard_flatten()

        today_iso = time.strftime("%Y-%m-%d")
        self.logger.record_daily_pnl(today_iso, _day_pnl)

        lat_avg_final = round(self._lat_sum / self._lat_count, 2) if self._lat_count > 0 else 0.0
        lat_min_final = round(self._lat_min, 2) if self._lat_min != float("inf") else 0.0

        final_metrics = {
            "daily_pnl_usdt":               round(_day_pnl, 4),
            "cash_balance_usdt":            round(self.cash, 4),
            "session_start_cash_usdt":      round(self._session_start_cash, 4),
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
            "halt_reason":                  self._halt_reason or "SESSION_CLOSE_1600",
            "latency_min_ms":               lat_min_final,
            "latency_avg_ms":               lat_avg_final,
            "latency_max_ms":               round(self._lat_max, 2),
            "latency_fill_count":           self._lat_count,
            "session_phase_at_close":       self._phase,
        }

        # ── Persist for next session ──────────────────────────────────────────
        # [FIX v3.4] Inventory is guaranteed flat here (zeroed by _hard_flatten).
        #            Cash balance carries forward — next session starts with this cash.
        #            P&L is NOT persisted — each session computes daily P&L fresh.
        self.r.set("persist:inventory", "0.0")                          # always flat
        self.r.set("persist:cash",      str(round(self.cash, 4)))       # carry forward
        self.r.set("persist:session_id", self.session_id)
        self.r.set("persist:saved_at",  datetime.now(timezone.utc).isoformat())
        # Explicitly delete stale pnl key so next session never reads it
        self.r.delete("persist:pnl")

        log.info(
            f"Session close persisted: "
            f"inventory=0.0 BTC (flat) | "
            f"cash={self.cash:.4f} USDT (carries to next session) | "
            f"daily_pnl={_day_pnl:.4f} USDT"
        )

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
