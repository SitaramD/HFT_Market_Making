#!/usr/bin/env python3
# =============================================================================
# SITARAM HFT — RL Backtest + Validation Entry Point
# Step 5 of 5
#
# Loads the trained RL policy and runs it on BOTH Mar 01 (IS) and Mar 02 (OOS).
# Computes all 23 metrics. Prints validation table to console.
# Generates JSON and PDF reports for Tamara submission.
#
# Usage (from D:\1\Project\Trading_Strategy\crypto_BTCUSDT\):
#   python run_rl_backtest.py
#
# Run AFTER: python run_rl_train.py
# =============================================================================

import sys
import os
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config        import CONFIG
from src.rl_backtester import RLBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)-26s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('sitaram.rl_backtest')

POLICY_PATH = os.path.join(CONFIG['output_dir'], 'rl_policy.json')


# =============================================================================
# METRICS TABLE PRINTER
# =============================================================================

def print_metrics_table(results: dict):
    is_r  = results['in_sample']
    oos_r = results['out_of_sample']

    def row(label, is_v, oos_v, fmt=''):
        try:
            is_s  = f'{is_v:{fmt}}'  if fmt else str(is_v)
            oos_s = f'{oos_v:{fmt}}' if fmt else str(oos_v)
        except Exception:
            is_s, oos_s = str(is_v), str(oos_v)

        # % improvement IS → OOS
        try:
            if isinstance(is_v, float) and is_v != 0:
                delta = (oos_v - is_v) / abs(is_v) * 100
                delta_s = f'{delta:+.1f}%'
            else:
                delta_s = '—'
        except Exception:
            delta_s = '—'

        print(f"  {label:<38} {is_s:>12}  {oos_s:>12}  {delta_s:>8}")

    print()
    print('=' * 78)
    print('  SITARAM HFT — RL POLICY BACKTEST RESULTS')
    print('=' * 78)
    print(f"  {'Metric':<38} {'IS (Mar 01)':>12}  {'OOS (Mar 02)':>12}  {'Δ IS→OOS':>8}")
    print('  ' + '─' * 74)

    print('  ── RETURN METRICS ──')
    row('1  Sharpe Ratio',                is_r.sharpe_ratio,            oos_r.sharpe_ratio,            '.4f')
    row('2  Sortino Ratio',               is_r.sortino_ratio,           oos_r.sortino_ratio,            '.4f')
    row('3  Calmar Ratio',                is_r.calmar_ratio,            oos_r.calmar_ratio,             '.4f')
    row('4  APY Post-Leverage (%)',        is_r.apy_post_leverage,       oos_r.apy_post_leverage,        '.2f')

    print('  ── RISK METRICS ──')
    row('5  Max Drawdown (%)',             is_r.max_drawdown_pct,        oos_r.max_drawdown_pct,         '.4f')
    row('6  Max Drawdown (USDT)',          is_r.max_drawdown_usdt,       oos_r.max_drawdown_usdt,        '.2f')
    row('7  VaR 95% Daily (USDT)',         is_r.var_95_daily,            oos_r.var_95_daily,             '.2f')
    row('8  Drawdown Duration (periods)',  is_r.drawdown_duration_days,  oos_r.drawdown_duration_days,   '.1f')
    row('9  Inventory Mean Reversion (%)', is_r.inventory_mean_reversion,oos_r.inventory_mean_reversion, '.2f')

    print('  ── EDGE & SIGNAL METRICS ──')
    row('10 IC Rolling',                  is_r.ic_rolling,              oos_r.ic_rolling,               '.6f')
    row('11 IC Stability (periods)',       is_r.ic_stability_days,       oos_r.ic_stability_days,        'd')
    row('12 IC Decay (500ms proxy)',       is_r.ic_decay_500ms,          oos_r.ic_decay_500ms,           '.6f')
    row('13 Adverse Selection (%)',        is_r.adverse_selection_rate,  oos_r.adverse_selection_rate,   '.2f')

    print('  ── EXECUTION METRICS ──')
    row('14 Fill Rate (%)',                is_r.fill_rate,               oos_r.fill_rate,                '.2f')
    row('15 Quote-to-Trade Ratio',         is_r.quote_to_trade_ratio,    oos_r.quote_to_trade_ratio,     '.1f')
    row('16 Maker Ratio (%)',              is_r.maker_ratio,             oos_r.maker_ratio,              '.1f')
    row('17 Fee-Adjusted PnL (USDT)',      is_r.fee_adjusted_pnl,        oos_r.fee_adjusted_pnl,         '.2f')
    row('18 Avg Latency (ms)',             is_r.avg_latency_ms,          oos_r.avg_latency_ms,           '.1f')

    print('  ── CAPACITY METRICS ──')
    row('19 Max Capacity (USDT)',          is_r.max_capacity_usdt,       oos_r.max_capacity_usdt,        ',.0f')
    row('20 Daily Turnover (USDT)',        is_r.daily_turnover_usdt,     oos_r.daily_turnover_usdt,      ',.0f')
    row('21 Market Impact (bps)',          is_r.market_impact_bps,       oos_r.market_impact_bps,        '.4f')
    row('22 Regime Sensitivity',           is_r.regime_sensitivity,      oos_r.regime_sensitivity,       '.4f')

    print('  ── OOS INTEGRITY ──')
    row('23 Sharpe Degradation (%)',       '—',  oos_r.sharpe_degradation_pct, '.2f')

    print('  ' + '─' * 74)
    print('  SUPPORTING')
    row('   Gross PnL (USDT)',             is_r.gross_pnl_usdt,          oos_r.gross_pnl_usdt,           '.2f')
    row('   Total Fees (USDT)',            is_r.total_fees_usdt,         oos_r.total_fees_usdt,          '.2f')
    row('   Total Fills',                  is_r.total_trades,            oos_r.total_trades,             ',d')
    row('   Quotes Placed',                is_r.quotes_placed,           oos_r.quotes_placed,            ',d')
    row('   Avg Spread (USDT)',            is_r.avg_spread_usdt,         oos_r.avg_spread_usdt,          '.4f')

    print('=' * 78)

    # ── Fund target check ─────────────────────────────────────────────────
    target = CONFIG['sharpe_target']
    dd_lim = CONFIG['max_drawdown_limit']

    checks = [
        ('Sharpe ≥ 3.0',         oos_r.sharpe_ratio       >= target,    f'{oos_r.sharpe_ratio:.4f}'),
        ('Max DD < 5%',          oos_r.max_drawdown_pct   <  dd_lim,   f'{oos_r.max_drawdown_pct:.4f}%'),
        ('Fill Rate 40-70%',     40 <= oos_r.fill_rate    <= 70,        f'{oos_r.fill_rate:.2f}%'),
        ('Maker Ratio > 90%',    oos_r.maker_ratio        >= 90,        f'{oos_r.maker_ratio:.1f}%'),
        ('Adverse Sel < 30%',    oos_r.adverse_selection_rate < 30,     f'{oos_r.adverse_selection_rate:.2f}%'),
        ('Fee-Adj PnL > 0',      oos_r.fee_adjusted_pnl   >  0,        f'${oos_r.fee_adjusted_pnl:.2f}'),
        ('IC > 0.02',            oos_r.ic_rolling         >  0.02,     f'{oos_r.ic_rolling:.6f}'),
        ('Sharpe Deg < 30%',     abs(oos_r.sharpe_degradation_pct) < 30,f'{oos_r.sharpe_degradation_pct:+.2f}%'),
    ]

    print()
    print('  FUND VALIDATION GATES (OOS)')
    print('  ' + '─' * 55)
    all_pass = True
    for name, passed, val in checks:
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"  {status}  {name:<30}  {val}")
        if not passed:
            all_pass = False
    print('  ' + '─' * 55)
    final = '✅ ALL GATES PASSED — Ready for Tamara submission' \
            if all_pass else '❌ Some gates failed — review before submission'
    print(f"  {final}")
    print()

    # ── Walk-forward summary ──────────────────────────────────────────────
    wf = results.get('walk_forward_windows', [])
    if wf:
        print(f"  Walk-Forward Validation ({len(wf)} windows):")
        for w in wf:
            print(f"    WF-{w['window']}  {w['label']:<36}  "
                  f"Train Sharpe={w['train'].sharpe_ratio:.3f}  "
                  f"Test Sharpe={w['test'].sharpe_ratio:.3f}  "
                  f"Δ={w['test'].sharpe_degradation_pct:+.1f}%")
        print()

    # ── RL parameter stats ────────────────────────────────────────────────
    gk = results.get('gamma_kappa_log', [])
    if gk:
        gammas = [g for g, k in gk]
        kappas = [k for g, k in gk]
        print(f"  RL Parameter Decisions (across all ticks):")
        print(f"    gamma  mean={np.mean(gammas):.4f}  std={np.std(gammas):.4f}  "
              f"min={np.min(gammas):.4f}  max={np.max(gammas):.4f}")
        print(f"    kappa  mean={np.mean(kappas):.4f}  std={np.std(kappas):.4f}  "
              f"min={np.min(kappas):.4f}  max={np.max(kappas):.4f}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info('=' * 60)
    log.info('  SITARAM HFT — RL Policy Backtest & Validation')
    log.info('=' * 60)

    if not os.path.exists(POLICY_PATH):
        log.error(
            f"\n❌ Policy file not found: {POLICY_PATH}\n"
            f"   Run training first:  python run_rl_train.py\n"
        )
        return 1

    log.info(f"Policy file : {POLICY_PATH}")
    log.info(f"OB data     : {CONFIG['data_dir_ob']}")
    log.info(f"Trade data  : {CONFIG['data_dir_trades']}")
    log.info(f"Output dir  : {CONFIG['output_dir']}")

    bt      = RLBacktester(CONFIG, POLICY_PATH)
    results = bt.run()

    print_metrics_table(results)

    # ── JSON report ───────────────────────────────────────────────────
    try:
        from src.json_report import generate_json
        path = generate_json(results, CONFIG['output_dir'])
        log.info(f"JSON report : {path}")
    except ImportError:
        log.warning("json_report.py not found — skipping JSON output.")

    # ── PDF report ────────────────────────────────────────────────────
    try:
        from src.pdf_report import generate_pdf
        path = generate_pdf(results, CONFIG['output_dir'])
        log.info(f"PDF report  : {path}")
    except ImportError:
        log.warning("pdf_report.py not found — skipping PDF output.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
