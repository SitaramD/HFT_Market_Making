#!/usr/bin/env python3
# =============================================================================
# SITARAM HFT — Backtest Runner
#
# Usage (Windows PowerShell from D:\1\Project\Trading_Strategy\github\):
#   python run_backtest.py
#
# Usage (WSL2):
#   cd /mnt/d/1/Project/Trading_Strategy/github/
#   python run_backtest.py
#
# Output:
#   - Console: live progress + final metrics table
#   - JSON:    D:\1\Project\Trading_Strategy\github\data\reports\SITARAM_report_<ts>.json
#   - PDF:     D:\1\Project\Trading_Strategy\github\data\reports\SITARAM_report_<ts>.pdf
# =============================================================================

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config   import CONFIG
from src.backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)-22s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('sitaram.main')


def print_metrics_table(results: dict):
    """Print a clean summary table to console after the run."""
    is_r  = results['in_sample']
    oos_r = results['out_of_sample']
    full  = results['full']

    def row(label, is_v, oos_v, fmt=''):
        try:
            is_s  = (f'{is_v:{fmt}}')  if fmt else str(is_v)
            oos_s = (f'{oos_v:{fmt}}') if fmt else str(oos_v)
        except Exception:
            is_s, oos_s = str(is_v), str(oos_v)
        print(f"  {label:<38} {is_s:>12}  {oos_s:>12}")

    print()
    print('=' * 68)
    print('  SITARAM HFT — BACKTEST RESULTS')
    print('=' * 68)
    print(f"  {'Metric':<38} {'In-Sample':>12}  {'Out-of-Sample':>12}")
    print(f"  {'(Mar 01)':<38} {'':>12}  {'(Mar 02)':>12}")
    print('  ' + '─' * 65)

    print('  RETURN METRICS')
    row('1  Sharpe Ratio',              is_r.sharpe_ratio,          oos_r.sharpe_ratio,         '.4f')
    row('2  Sortino Ratio',             is_r.sortino_ratio,         oos_r.sortino_ratio,         '.4f')
    row('3  Calmar Ratio',              is_r.calmar_ratio,          oos_r.calmar_ratio,          '.4f')
    row('4  APY Post-Leverage (%)',     is_r.apy_post_leverage,     oos_r.apy_post_leverage,     '.2f')

    print('  RISK METRICS')
    row('5  Max Drawdown (%)',          is_r.max_drawdown_pct,      oos_r.max_drawdown_pct,      '.4f')
    row('6  Max Drawdown (USDT)',       is_r.max_drawdown_usdt,     oos_r.max_drawdown_usdt,     '.2f')
    row('7  VaR 95% Daily (USDT)',      is_r.var_95_daily,          oos_r.var_95_daily,          '.2f')
    row('8  Drawdown Duration (periods)',is_r.drawdown_duration_days,oos_r.drawdown_duration_days,'.1f')
    row('9  Inventory Mean Reversion (%)',is_r.inventory_mean_reversion,oos_r.inventory_mean_reversion,'.2f')

    print('  EDGE & SIGNAL METRICS')
    row('10 IC Rolling',               is_r.ic_rolling,            oos_r.ic_rolling,            '.6f')
    row('11 IC Stability (periods)',   is_r.ic_stability_days,     oos_r.ic_stability_days,     'd')
    row('12 IC Decay (500ms proxy)',   is_r.ic_decay_500ms,        oos_r.ic_decay_500ms,        '.6f')
    row('13 Adverse Selection (%)',    is_r.adverse_selection_rate,oos_r.adverse_selection_rate,'.2f')

    print('  EXECUTION METRICS')
    row('14 Fill Rate (%)',            is_r.fill_rate,             oos_r.fill_rate,             '.2f')
    row('15 Quote-to-Trade Ratio',     is_r.quote_to_trade_ratio,  oos_r.quote_to_trade_ratio,  '.1f')
    row('16 Maker Ratio (%)',          is_r.maker_ratio,           oos_r.maker_ratio,           '.1f')
    row('17 Fee-Adjusted PnL (USDT)',  is_r.fee_adjusted_pnl,      oos_r.fee_adjusted_pnl,      '.2f')
    row('18 Avg Latency (ms)',         is_r.avg_latency_ms,        oos_r.avg_latency_ms,        '.1f')

    print('  CAPACITY METRICS')
    row('19 Max Capacity (USDT)',      is_r.max_capacity_usdt,     oos_r.max_capacity_usdt,     ',.0f')
    row('20 Daily Turnover (USDT)',    is_r.daily_turnover_usdt,   oos_r.daily_turnover_usdt,   ',.0f')
    row('21 Market Impact (bps)',      is_r.market_impact_bps,     oos_r.market_impact_bps,     '.4f')
    row('22 Regime Sensitivity',       is_r.regime_sensitivity,    oos_r.regime_sensitivity,    '.4f')

    print('  OOS INTEGRITY')
    row('23 Sharpe Degradation (%)',   '—', oos_r.sharpe_degradation_pct, '.2f')

    print('  ' + '─' * 65)
    print('  SUPPORTING')
    row('   Gross PnL (USDT)',         is_r.gross_pnl_usdt,        oos_r.gross_pnl_usdt,        '.2f')
    row('   Total Fees (USDT)',        is_r.total_fees_usdt,       oos_r.total_fees_usdt,       '.2f')
    row('   Total Fills',             is_r.total_trades,          oos_r.total_trades,          ',d')
    row('   Quotes Placed',           is_r.quotes_placed,         oos_r.quotes_placed,         ',d')
    row('   Avg Spread (USDT)',        is_r.avg_spread_usdt,       oos_r.avg_spread_usdt,       '.4f')
    row('   Daily Turnover (BTC)',     is_r.daily_turnover_btc,    oos_r.daily_turnover_btc,    '.6f')

    print('=' * 68)

    # Fund target check
    target = CONFIG['sharpe_target']
    passed = oos_r.sharpe_ratio >= target
    status = '✅ PASS' if passed else f'❌ BELOW TARGET ({target:.1f})'
    print(f"\n  Fund target (Sharpe ≥ {target:.1f}):  OOS Sharpe = {oos_r.sharpe_ratio:.4f}  →  {status}")

    # Walk-forward summary
    wf = results.get('walk_forward_windows', [])
    if wf:
        print(f"\n  Walk-Forward Windows ({len(wf)} windows):")
        for w in wf:
            print(f"    WF-{w['window']}  {w['label']:<36} "
                  f"Train={w['train'].sharpe_ratio:.3f}  "
                  f"Test={w['test'].sharpe_ratio:.3f}  "
                  f"Δ={w['test'].sharpe_degradation_pct:+.1f}%")

    print()


def main():
    log.info('SITARAM HFT Backtester — starting')
    log.info(f"OB   data: {CONFIG['data_dir_ob']}")
    log.info(f"Trade data: {CONFIG['data_dir_trades']}")
    log.info(f"Output dir: {CONFIG['output_dir']}")

    bt      = Backtester(CONFIG)
    results = bt.run()

    print_metrics_table(results)

    # ── JSON report ───────────────────────────────────────────────────────
    try:
        from src.json_report import generate_json
        json_path = generate_json(results, CONFIG['output_dir'])
        log.info(f"JSON report: {json_path}")
    except ImportError:
        log.warning("json_report.py not found — skipping JSON output.")

    # ── PDF report ────────────────────────────────────────────────────────
    try:
        from src.pdf_report import generate_pdf
        pdf_path = generate_pdf(results, CONFIG['output_dir'])
        log.info(f"PDF report:  {pdf_path}")
    except ImportError:
        log.warning("pdf_report.py not found — skipping PDF output.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
