// ============================================================
// SITARAM HFT — Jenkins Pipeline Orchestrator
// Triggered: every git push (via post-commit hook + 1-min poll)
// Stages: Lint → Unit → Integration → Backtest → Gate Report
// Gate failures are ADVISORY only — pipeline never hard-blocks
// ============================================================

pipeline {

    agent {
        docker {
            image 'python:3.11-slim'
            args  '--network host -v /var/run/docker.sock:/var/run/docker.sock'
        }
    }

    options {
        timestamps()
        ansiColor('xterm')
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '30'))
        disableConcurrentBuilds()          // one pipeline at a time
    }

    environment {
        // ── project paths ──────────────────────────────────────
        PROJECT_ROOT   = "D:\\1\\Project\\Trading_Strategy\\sitaram"
        TESTS_DIR      = "${WORKSPACE}\\tests"
        REPORTS_DIR    = "${WORKSPACE}\\reports"

        // ── strategy constants (must match validated Run-15) ───
        AS_GAMMA       = "2.0"
        AS_KAPPA       = "0.5"
        MAX_INV_BTC    = "0.01"
        MAKER_FEE      = "-0.0001"
        TAKER_FEE      = "0.00055"
        FILL_WINDOW_MS = "500"

        // ── gate thresholds (advisory — never hard-block) ──────
        SHARPE_TARGET  = "3.0"
        MAX_DD_PCT     = "5.0"
        FILL_RATE_MIN  = "5.0"

        // ── infrastructure (host Docker network) ───────────────
        REDIS_HOST     = "localhost"
        REDIS_PORT     = "6379"
        TIMESCALE_HOST = "localhost"
        TIMESCALE_PORT = "5432"
        TIMESCALE_DB   = "sitaram"
        TIMESCALE_USER = "sitaram_user"
        TIMESCALE_PASS = "sitaram_secure_2026"

        // ── report artefact names ──────────────────────────────
        JUNIT_UNIT    = "reports/junit_unit.xml"
        JUNIT_INTG    = "reports/junit_integration.xml"
        GATE_REPORT   = "reports/gate_report.json"
        COV_HTML      = "reports/coverage_html"
    }

    stages {

        // ── 0. SETUP ──────────────────────────────────────────
        stage('Setup') {
            steps {
                echo '⚙️  Installing Python dependencies...'
                sh '''
                    pip install --quiet --upgrade pip
                    pip install --quiet \
                        pytest pytest-cov pytest-html pytest-xdist \
                        hypothesis \
                        redis \
                        psycopg2-binary \
                        numpy pandas \
                        docker \
                        colorama
                    mkdir -p reports
                '''
            }
        }

        // ── 1. LINT ───────────────────────────────────────────
        stage('Lint') {
            steps {
                echo '🔍  Running linters...'
                sh '''
                    pip install --quiet flake8 black isort
                    echo "--- flake8 ---"
                    flake8 tests/ \
                        --max-line-length=120 \
                        --ignore=E501,W503,E203 \
                        --count --statistics || true
                    echo "--- black (check only) ---"
                    black --check --diff tests/ || true
                    echo "--- isort (check only) ---"
                    isort --check-only tests/ || true
                    echo "✅ Lint complete (advisory)"
                '''
            }
        }

        // ── 2. UNIT TESTS (mocked infra) ──────────────────────
        stage('Unit Tests') {
            steps {
                echo '🧪  Running unit tests (mocked Redis/Kafka/DB)...'
                sh '''
                    pytest tests/unit/ \
                        -v \
                        --tb=short \
                        --junitxml=${JUNIT_UNIT} \
                        --cov=tests/unit \
                        --cov-report=html:${COV_HTML}/unit \
                        --cov-report=term-missing \
                        -n auto \
                        --hypothesis-seed=42 \
                        -q
                '''
            }
            post {
                always {
                    junit "${JUNIT_UNIT}"
                }
                failure {
                    echo '❌ Unit tests FAILED — pipeline halted'
                }
            }
        }

        // ── 3. INTEGRATION TESTS (real Docker services) ───────
        stage('Integration Tests') {
            steps {
                echo '🔗  Running integration tests against live Docker stack...'
                sh '''
                    # Verify Docker services are reachable before running
                    python -c "
import redis, sys
try:
    r = redis.Redis(host='${REDIS_HOST}', port=${REDIS_PORT}, socket_timeout=3)
    r.ping()
    print('✅  Redis reachable')
except Exception as e:
    print(f'⚠️  Redis not reachable: {e}')
    sys.exit(1)
"
                    pytest tests/integration/ \
                        -v \
                        --tb=short \
                        --junitxml=${JUNIT_INTG} \
                        --cov=tests/integration \
                        --cov-report=html:${COV_HTML}/integration \
                        -q
                '''
            }
            post {
                always {
                    junit "${JUNIT_INTG}"
                }
            }
        }

        // ── 4. BACKTEST VALIDATION ────────────────────────────
        stage('Backtest Validation') {
            steps {
                echo '📊  Running strategy backtest validation...'
                sh '''
                    pytest tests/unit/test_backtest_validation.py \
                        -v \
                        --tb=long \
                        -q
                '''
            }
        }

        // ── 5. GATE EVALUATION (advisory — never fails build) ──
        stage('Gate Evaluation') {
            steps {
                echo '🚦  Evaluating Go-Live gates (ADVISORY — no block)...'
                sh '''
                    python scripts/gate_evaluator.py \
                        --sharpe-target ${SHARPE_TARGET} \
                        --max-dd ${MAX_DD_PCT} \
                        --fill-rate-min ${FILL_RATE_MIN} \
                        --output ${GATE_REPORT} || true
                '''
            }
            post {
                always {
                    script {
                        if (fileExists("${GATE_REPORT}")) {
                            def gate = readJSON file: "${GATE_REPORT}"
                            def status = gate.overall_pass ? '✅ PASS' : '⚠️  ADVISORY FAIL'
                            echo "Gate result: ${status}"
                            echo "  Sharpe:    ${gate.sharpe}  (target ≥ ${SHARPE_TARGET})"
                            echo "  Drawdown:  ${gate.max_drawdown_pct}%  (limit ≤ ${MAX_DD_PCT}%)"
                            echo "  Fill Rate: ${gate.fill_rate_pct}%  (min ≥ ${FILL_RATE_MIN}%)"
                            // Write gate to TimescaleDB jenkins_approvals table
                            sh """
                                python scripts/persist_gate.py \
                                    --gate-json ${GATE_REPORT} \
                                    --build-number ${BUILD_NUMBER} \
                                    --git-commit ${GIT_COMMIT ?: 'local'} || true
                            """
                        }
                    }
                }
            }
        }

        // ── 6. DEPLOY TO PAPER TRADING ────────────────────────
        stage('Deploy — Paper Trading') {
            when {
                expression {
                    // Only deploy when unit + integration tests pass
                    currentBuild.currentResult == 'SUCCESS'
                }
            }
            steps {
                echo '🚀  Deploying to paper trading stack...'
                sh '''
                    # Hot-reload Python engine (no full rebuild)
                    docker cp . sitaram-python-engine:/app/
                    docker restart sitaram-python-engine
                    echo "✅  Paper trading engine restarted with new code"
                '''
            }
        }
    }

    // ── POST-PIPELINE ACTIONS ─────────────────────────────────
    post {
        always {
            echo '📦  Archiving reports...'
            archiveArtifacts artifacts: 'reports/**/*', allowEmptyArchive: true
            publishHTML([
                allowMissing: true,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: "${COV_HTML}/unit",
                reportFiles: 'index.html',
                reportName: 'Unit Test Coverage'
            ])
        }
        success {
            echo '🎉  Pipeline PASSED — all tests green'
        }
        failure {
            echo '🔴  Pipeline FAILED — check unit/integration test output'
        }
        unstable {
            echo '🟡  Pipeline UNSTABLE — some tests had warnings'
        }
    }
}
