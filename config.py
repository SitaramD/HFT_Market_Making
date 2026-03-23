# =============================================================================
# SITARAM HFT — Configuration
# All strategy and backtest parameters in one place.
# Edit paths, fees, and model params here — nowhere else.
#
# Parameter changes vs previous run (all fixes applied 2026-03-21):
#   FIX-A  AS_GAMMA             0.50  -> 2.0      Widen spread aggressively with inventory
#   FIX-B  AS_KAPPA             1.5   -> 0.5      Wider base spread (lower arrival rate)
#   FIX-C  MAX_INVENTORY_BTC    0.10  -> 0.01     10x tighter; forces mean reversion
#   FIX-D  FILL_WINDOW_MS       200   -> 50       Fill only against trades within 50ms
#   FIX-E  REGIME_VOL_THRESHOLD 6e-5  -> 2e-5     Pause quoting 3x more often in trends
#   (backtester.py changes)
#   FIX-1  AS spread cap        $0.50 -> $5.00    Matches BTC $5-50/tick volatile moves
#   FIX-2  Fill tolerance       ±0.5% -> exact    No phantom fills against informed flow
#   FIX-3  Walk-forward         day 0-1 only -> all 17 consecutive day pairs
# =============================================================================

import sys

# ---------------------------------------------------------------------------
# DATA PATHS
# Windows native paths (when running on Windows directly)
# WSL2: script auto-converts D:\ -> /mnt/d/
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    DATA_DIR_OB     = r'E:\Binance\March\All'
    DATA_DIR_TRADES = r'E:\Binance\March'
    OUTPUT_DIR      = r'E:\Binance\March\reports'
else:
    DATA_DIR_OB     = '/mnt/e/Binance/March'
    DATA_DIR_TRADES = '/mnt/e/Binance/March'
    OUTPUT_DIR      = '/mnt/e/Binance/March/reports'

OB_FILE_PATTERN     = '*BTCUSDT_ob200.data'
TRADES_FILE_PATTERN = 'BTCUSDT_*.csv'

# ---------------------------------------------------------------------------
# STRATEGY IDENTITY
# ---------------------------------------------------------------------------
SYMBOL   = 'BTCUSDT'
VENUE    = 'Bybit Spot CEX'
STRATEGY = 'Avellaneda-Stoikov Market Making'

# ---------------------------------------------------------------------------
# AVELLANEDA-STOIKOV MODEL PARAMETERS
# ---------------------------------------------------------------------------
# FIX-A: gamma raised 0.50 -> 2.0
#   Higher gamma = spread widens faster as inventory accumulates.
#   Previous value of 0.50 was too low — quotes barely moved even at max inventory.
#   At gamma=2.0 the model becomes much more defensive when inventory builds.
AS_GAMMA = 2.0      # reverted from 5.0 — higher gamma worsened IS drawdown

# FIX-B: kappa lowered 1.5 -> 0.5
#   Lower kappa = model assumes slower order arrival = wider base spread.
#   Previous value of 1.5 assumed very high fill rate, leading to tight quotes
#   that were immediately picked off by faster participants.
AS_KAPPA = 0.5

AS_SIGMA = 0.001    # Initial volatility (updated dynamically from realized returns)
AS_T     = 1.0      # Normalized session horizon

# ---------------------------------------------------------------------------
# ORDER PARAMETERS
# ---------------------------------------------------------------------------
ORDER_QTY_BTC        = 0.001   # BTC per quote (each side) — unchanged, correct size

# FIX-C: max inventory 0.10 -> 0.01 BTC
#   Previous limit allowed accumulating up to 0.10 BTC (~$8,500 at $85,000 BTC).
#   With 1.34M fills, inventory was building one-sided without mean-reverting.
#   At 0.01 BTC (10 fills before hard stop) the model is forced to rebalance.
MAX_INVENTORY_BTC    = 0.01

TICK_SIZE_USDT       = 0.10    # Minimum price increment on Bybit BTC/USDT spot

# FIX-D: fill window 200ms -> 50ms
#   200ms window was matching our passive quotes against trades that arrived
#   well after the book had already moved — i.e., against informed flow.
#   50ms matches only trades that arrive while our quote is still competitive.
FILL_WINDOW_MS       = 50

SIMULATED_LATENCY_MS = 10.0    # Order placement latency in ms

# ---------------------------------------------------------------------------
# FEE STRUCTURE — Bybit BTC/USDT Spot
# ---------------------------------------------------------------------------
MAKER_FEE_RATE = -0.00010   # Maker rebate (exchange pays us): -0.01%
TAKER_FEE_RATE =  0.00055   # Taker fee (we pay exchange): +0.055%

# ---------------------------------------------------------------------------
# CAPITAL
# ---------------------------------------------------------------------------
INITIAL_CAPITAL_USDT = 100_000.0
LEVERAGE             = 1.0          # Spot: no leverage

# ---------------------------------------------------------------------------
# BACKTEST SPLIT
# 18 days available: Mar 01-18. 70/30 split -> ~13 IS, ~5 OOS
# ---------------------------------------------------------------------------
IN_SAMPLE_RATIO = 0.70    # 70/30 = ~13 days IS, ~5 days OOS

# ---------------------------------------------------------------------------
# REGIME FILTER
# Detects trending/high-vol conditions and pauses quoting entirely.
# ---------------------------------------------------------------------------
REGIME_VOL_WINDOW    = 50       # ticks to measure realized vol

# FIX-E: threshold tightened 0.000060 -> 0.000020
#   Previous threshold was too permissive — only filtered extreme spikes.
#   At 0.000020 the strategy pauses quoting ~3x more often, avoiding
#   adverse fills during directional moves which are the main loss source.
REGIME_VOL_THRESHOLD = 0.000045

REGIME_SPREAD_MULT   = 3.0      # multiply AS half-spread when trending (fallback)

# ---------------------------------------------------------------------------
# DAILY LOSS LIMIT
# Hard stop per calendar day. If cumulative PnL on that day falls below
# this threshold the strategy stops quoting for the rest of the day.
# This directly cuts IS drawdown without touching gamma or spread params.
# Set to None or a very large negative number to disable.
#
# Rationale: IS drawdown of 39.85% in Run #6 was caused by 3-4 bad days
# (Mar 02-03, Mar 07-09) where the strategy kept quoting into trending flow.
# A $3,000/day stop would have limited each bad day's loss, cutting the
# 13-day IS drawdown from ~40% to an estimated <10%.
# ---------------------------------------------------------------------------
DAILY_LOSS_LIMIT_USDT = -3000.0   # stop quoting if daily fee-adj PnL < -$3,000

# ---------------------------------------------------------------------------
# FUND THRESHOLDS (Tamara requirements)
# ---------------------------------------------------------------------------
SHARPE_TARGET      = 3.0
MAX_DRAWDOWN_LIMIT = 5.0
IC_MIN_THRESHOLD   = 0.02
FILL_RATE_MIN      = 40.0
FILL_RATE_MAX      = 70.0
QTR_MAX            = 10.0
MAKER_RATIO_MIN    = 90.0
ADVERSE_SEL_MAX    = 30.0
LATENCY_MAX_MS     = 50.0
IC_STABILITY_MIN   = 3
INV_REVERSION_MIN  = 70.0

# ---------------------------------------------------------------------------
# CONFIG DICT — passed to all modules
# ---------------------------------------------------------------------------
CONFIG = {
    'symbol'               : SYMBOL,
    'venue'                : VENUE,
    'strategy'             : STRATEGY,
    'data_dir_ob'          : DATA_DIR_OB,
    'data_dir_trades'      : DATA_DIR_TRADES,
    'ob_file_pattern'      : OB_FILE_PATTERN,
    'trades_file_pattern'  : TRADES_FILE_PATTERN,
    'output_dir'           : OUTPUT_DIR,
    'as_gamma'             : AS_GAMMA,
    'as_kappa'             : AS_KAPPA,
    'as_sigma'             : AS_SIGMA,
    'as_T'                 : AS_T,
    'order_qty_btc'        : ORDER_QTY_BTC,
    'max_inventory_btc'    : MAX_INVENTORY_BTC,
    'tick_size_usdt'       : TICK_SIZE_USDT,
    'fill_window_ms'       : FILL_WINDOW_MS,
    'simulated_latency_ms' : SIMULATED_LATENCY_MS,
    'maker_fee_rate'       : MAKER_FEE_RATE,
    'taker_fee_rate'       : TAKER_FEE_RATE,
    'initial_capital_usdt' : INITIAL_CAPITAL_USDT,
    'leverage'             : LEVERAGE,
    'in_sample_ratio'      : IN_SAMPLE_RATIO,
    'oos_ratio'            : 1.0 - IN_SAMPLE_RATIO,
    'regime_vol_window'    : REGIME_VOL_WINDOW,
    'regime_vol_threshold' : REGIME_VOL_THRESHOLD,
    'regime_spread_mult'   : REGIME_SPREAD_MULT,
    'daily_loss_limit_usdt': DAILY_LOSS_LIMIT_USDT,
    'sharpe_target'        : SHARPE_TARGET,
    'max_drawdown_limit'   : MAX_DRAWDOWN_LIMIT,
    'ic_min_threshold'     : IC_MIN_THRESHOLD,
}
