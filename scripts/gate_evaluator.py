"""
SITARAM HFT — Gate Evaluator Script
Called by Jenkins Gate Evaluation stage.
Reads live metrics from Redis (source of truth) and computes gate results.
Writes a JSON report for Jenkins to parse and display.

Usage:
    python scripts/gate_evaluator.py \
        --sharpe-target 3.0 \
        --max-dd 5.0 \
        --fill-rate-min 5.0 \
        --output reports/gate_report.json
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

# ── Optional Redis import ─────────────────────────────────────
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


def read_redis_metrics(host: str, port: int) -> dict:
    """
    Read live session metrics from Redis.
    Redis is the source of truth for live state.
    """
    if not HAS_REDIS:
        print("[GATE] ⚠️  redis-py not installed — using placeholder metrics")
        return {}

    try:
        r = redis.Redis(host=host, port=port, socket_timeout=3)
        r.ping()

        def _float(key, default=None):
            v = r.get(key)
            if v is None:
                return default
            try:
                return float(v.decode())
            except (ValueError, AttributeError):
                return default

        def _str(key, default=None):
            v = r.get(key)
            return v.decode() if v else default

        return {
            'cumulative_pnl':  _float('quote:cumulative_pnl',  0.0),
            'session_pnl':     _float('quote:session_pnl',     0.0),
            'inventory':       _float('quote:inventory',        0.0),
            'avg_latency_ms':  _float('latency:avg_ms',        None),
            'regime':          _str('regime:state',            'UNKNOWN'),
            'sharpe':          _float('metrics:sharpe',        None),
            'max_drawdown_pct':_float('metrics:max_drawdown',  None),
            'fill_rate_pct':   _float('metrics:fill_rate',     None),
            'total_trades':    _float('metrics:total_trades',  0),
            'quotes_sent':     _float('metrics:quotes_sent',   0),
            'redis_ok':        True,
        }
    except Exception as e:
        print(f"[GATE] ⚠️  Redis read failed: {e}")
        return {'redis_ok': False}


def evaluate_gates(
    sharpe: float,
    max_dd_pct: float,
    fill_rate_pct: float,
    sharpe_target: float,
    max_dd_limit: float,
    fill_rate_min: float,
) -> dict:
    """Evaluate all three Go-Live gates."""
    sharpe_ok    = sharpe       >= sharpe_target  if sharpe       is not None else False
    dd_ok        = max_dd_pct   <= max_dd_limit   if max_dd_pct   is not None else False
    fill_ok      = fill_rate_pct >= fill_rate_min  if fill_rate_pct is not None else False

    return {
        'sharpe_pass':    sharpe_ok,
        'drawdown_pass':  dd_ok,
        'fill_rate_pass': fill_ok,
        'overall_pass':   sharpe_ok and dd_ok and fill_ok,
    }


def main():
    parser = argparse.ArgumentParser(description='SITARAM HFT Gate Evaluator')
    parser.add_argument('--sharpe-target',  type=float, default=3.0)
    parser.add_argument('--max-dd',         type=float, default=5.0)
    parser.add_argument('--fill-rate-min',  type=float, default=5.0)
    parser.add_argument('--output',         type=str,   default='reports/gate_report.json')
    parser.add_argument('--redis-host',     type=str,   default=os.getenv('REDIS_HOST', 'localhost'))
    parser.add_argument('--redis-port',     type=int,   default=int(os.getenv('REDIS_PORT', '6379')))
    args = parser.parse_args()

    print(f"[GATE] Reading metrics from Redis {args.redis_host}:{args.redis_port}")
    metrics = read_redis_metrics(args.redis_host, args.redis_port)

    sharpe        = metrics.get('sharpe')
    max_dd_pct    = metrics.get('max_drawdown_pct')
    fill_rate_pct = metrics.get('fill_rate_pct')

    # If metrics unavailable from Redis (no active session), report advisory
    no_session = (sharpe is None and max_dd_pct is None and fill_rate_pct is None)

    if no_session:
        print("[GATE] ⚠️  No active session metrics in Redis — gate is advisory only")
        gate_results = {
            'sharpe_pass':    None,
            'drawdown_pass':  None,
            'fill_rate_pass': None,
            'overall_pass':   None,
        }
    else:
        gate_results = evaluate_gates(
            sharpe        = sharpe        or 0.0,
            max_dd_pct    = max_dd_pct    or 0.0,
            fill_rate_pct = fill_rate_pct or 0.0,
            sharpe_target = args.sharpe_target,
            max_dd_limit  = args.max_dd,
            fill_rate_min = args.fill_rate_min,
        )

    report = {
        'timestamp':        datetime.now(timezone.utc).isoformat(),
        'thresholds': {
            'sharpe_target':  args.sharpe_target,
            'max_dd_pct':     args.max_dd,
            'fill_rate_min':  args.fill_rate_min,
        },
        'metrics': {
            'sharpe':           sharpe,
            'max_drawdown_pct': max_dd_pct,
            'fill_rate_pct':    fill_rate_pct,
            'avg_latency_ms':   metrics.get('avg_latency_ms'),
            'regime':           metrics.get('regime'),
            'inventory':        metrics.get('inventory'),
            'cumulative_pnl':   metrics.get('cumulative_pnl'),
            'total_trades':     metrics.get('total_trades'),
        },
        **gate_results,
        'redis_ok':         metrics.get('redis_ok', False),
        'no_session':       no_session,
        'advisory_only':    True,   # gates never block — always advisory
    }

    # Print summary to Jenkins console
    print("\n" + "="*60)
    print("  SITARAM HFT — GATE EVALUATION REPORT")
    print("="*60)
    if no_session:
        print("  ⚠️  NO ACTIVE SESSION — metrics unavailable")
    else:
        def fmt(val, fmt_str='.4f'):
            return f"{val:{fmt_str}}" if val is not None else "N/A"

        def tick(passed):
            if passed is None: return "❓"
            return "✅" if passed else "⚠️ "

        print(f"  Sharpe:    {fmt(sharpe)}  (target ≥ {args.sharpe_target})  {tick(gate_results.get('sharpe_pass'))}")
        print(f"  Drawdown:  {fmt(max_dd_pct)}%  (limit ≤ {args.max_dd}%)     {tick(gate_results.get('drawdown_pass'))}")
        print(f"  Fill Rate: {fmt(fill_rate_pct)}%  (min ≥ {args.fill_rate_min}%)    {tick(gate_results.get('fill_rate_pass'))}")
        overall = gate_results.get('overall_pass')
        print(f"\n  Overall: {'✅ PASS' if overall else '⚠️  ADVISORY FAIL (non-blocking)'}")
        print(f"  Regime:  {metrics.get('regime', 'UNKNOWN')}")
        print(f"  Avg Latency: {fmt(metrics.get('avg_latency_ms'), '.1f')} ms")
    print("="*60 + "\n")

    # Write report JSON
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[GATE] Report written to: {args.output}")


if __name__ == '__main__':
    main()
