"""
SITARAM HFT — Gate Persistence Script
Writes gate evaluation results to TimescaleDB jenkins_approvals table.
Called by Jenkins after gate_evaluator.py produces the report JSON.

Usage:
    python scripts/persist_gate.py \
        --gate-json reports/gate_report.json \
        --build-number 42 \
        --git-commit abc1234
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

try:
    import psycopg2
    HAS_PG = True
except ImportError:
    HAS_PG = False


def persist(gate_json_path: str, build_number: int, git_commit: str,
            pg_host: str, pg_port: int, pg_db: str, pg_user: str, pg_pass: str):

    if not os.path.exists(gate_json_path):
        print(f"[PERSIST] ⚠️  Gate report not found: {gate_json_path}")
        return

    with open(gate_json_path) as f:
        report = json.load(f)

    if not HAS_PG:
        print("[PERSIST] ⚠️  psycopg2 not installed — skipping DB write")
        print(f"[PERSIST] Gate data: {json.dumps(report, indent=2)}")
        return

    try:
        conn = psycopg2.connect(
            host=pg_host, port=pg_port,
            dbname=pg_db, user=pg_user, password=pg_pass,
            connect_timeout=5,
        )
        cur = conn.cursor()

        metrics = report.get('metrics', {})
        cur.execute("""
            INSERT INTO jenkins_approvals (
                timestamp, build_number, git_commit,
                sharpe, max_drawdown_pct, fill_rate_pct,
                sharpe_pass, drawdown_pass, fill_rate_pass, overall_pass,
                avg_latency_ms, regime, inventory, cumulative_pnl,
                redis_ok, no_session, advisory_only
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (build_number) DO UPDATE SET
                timestamp        = EXCLUDED.timestamp,
                sharpe           = EXCLUDED.sharpe,
                max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                fill_rate_pct    = EXCLUDED.fill_rate_pct,
                overall_pass     = EXCLUDED.overall_pass
        """, (
            report.get('timestamp', datetime.now(timezone.utc).isoformat()),
            build_number,
            git_commit,
            metrics.get('sharpe'),
            metrics.get('max_drawdown_pct'),
            metrics.get('fill_rate_pct'),
            report.get('sharpe_pass'),
            report.get('drawdown_pass'),
            report.get('fill_rate_pass'),
            report.get('overall_pass'),
            metrics.get('avg_latency_ms'),
            metrics.get('regime'),
            metrics.get('inventory'),
            metrics.get('cumulative_pnl'),
            report.get('redis_ok', False),
            report.get('no_session', True),
            report.get('advisory_only', True),
        ))
        conn.commit()
        print(f"[PERSIST] ✅  Gate results written to jenkins_approvals "
              f"(build #{build_number}, commit {git_commit[:8]})")
    except psycopg2.errors.UndefinedTable:
        print("[PERSIST] ⚠️  jenkins_approvals table does not exist — "
              "run schema migration first")
    except Exception as e:
        print(f"[PERSIST] ⚠️  DB write failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate-json',    type=str, required=True)
    parser.add_argument('--build-number', type=int, default=0)
    parser.add_argument('--git-commit',   type=str, default='local')
    parser.add_argument('--pg-host',  default=os.getenv('TIMESCALE_HOST', 'localhost'))
    parser.add_argument('--pg-port',  type=int, default=int(os.getenv('TIMESCALE_PORT', '5432')))
    parser.add_argument('--pg-db',    default=os.getenv('TIMESCALE_DB',   'sitaram'))
    parser.add_argument('--pg-user',  default=os.getenv('TIMESCALE_USER', 'sitaram_user'))
    parser.add_argument('--pg-pass',  default=os.getenv('TIMESCALE_PASS', 'sitaram_secure_2026'))
    args = parser.parse_args()

    persist(
        gate_json_path=args.gate_json,
        build_number=args.build_number,
        git_commit=args.git_commit,
        pg_host=args.pg_host,
        pg_port=args.pg_port,
        pg_db=args.pg_db,
        pg_user=args.pg_user,
        pg_pass=args.pg_pass,
    )


if __name__ == '__main__':
    main()
