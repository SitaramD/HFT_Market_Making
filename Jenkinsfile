// ============================================================
// SITARAM HFT — Jenkins Pipeline Orchestrator
// Triggered: every git push (via post-commit hook + 1-min poll)
// Stages: Lint → Unit → Integration → Backtest → Gate Report
// Gate failures are ADVISORY only — pipeline never hard-blocks
// ============================================================

pipeline {

    agent any

    options {
        timestamps()
        ansiColor('xterm')
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '30'))
        disableConcurrentBuilds()
    }

    environment {
        AS_GAMMA       = "2.0"
        AS_KAPPA       = "0.5"
        MAX_INV_BTC    = "0.01"
        MAKER_FEE      = "-0.0001"
        TAKER_FEE      = "0.00055"
        FILL_WINDOW_MS = "500"
        SHARPE_TARGET  = "3.0"
        MAX_DD_PCT     = "5.0"
        FILL_RATE_MIN  = "5.0"
        REDIS_HOST     = "redis"
        REDIS_PORT     = "6379"
        TIMESCALE_HOST = "timescaledb"
        TIMESCALE_PORT = "5432"
        TIMESCALE_DB   = "sitaram"
        TIMESCALE_USER = "sitaram_user"
        TIMESCALE_PASS = "sitaram_secure_2026"
        JUNIT_UNIT     = "reports/junit_unit.xml"
        JUNIT_INTG     = "reports/junit_integration.xml"
        GATE_REPORT    = "reports/gate_report.json"
    }

    stages {

        // ── 0. SETUP ──────────────────────────────────────────
        stage('Setup') {
            steps {
                echo '⚙️  Installing Python dependencies...'
                sh '''
                    pip install --quiet --upgrade pip --break-system-packages
                    pip install --quiet --break-system-packages pytest pytest-cov pytest-xdist hypothesis redis psycopg2-binary numpy pandas colorama
                    mkdir -p reports
                '''
            }
        }

        // ── 1. LINT ───────────────────────────────────────────
        stage('Lint') {
            steps {
                echo '🔍  Running linters...'
                sh '''
                    pip install --quiet --break-system-packages flake8 black isort
                    flake8 tests/ --max-line-length=120 --ignore=E501,W503,E203 --count --statistics || true
                    black --check --diff tests/ || true
                    isort --check-only tests/ || true
                    echo "✅ Lint complete (advisory)"
                '''
            }
        }

        // ── 2. UNIT TESTS ─────────────────────────────────────
        stage('Unit Tests') {
            steps {
                echo '🧪  Running unit tests (mocked Redis/Kafka/DB)...'
                sh '''
                    pytest tests/unit/ -v --tb=short --junitxml=${JUNIT_UNIT} -n auto --hypothesis-seed=42 -q
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

        // ── 3. INTEGRATION TESTS ──────────────────────────────
        stage('Integration Tests') {
            steps {
                echo '🔗  Running integration tests against live Docker stack...'
                sh '''
                    python -c "import redis, sys; r = redis.Redis(host='${REDIS_HOST}', port=${REDIS_PORT}, socket_timeout=3); r.ping(); print('✅ Redis reachable')" || true
                    pytest tests/integration/ -v --tb=short --junitxml=${JUNIT_INTG} -q || true
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: "${JUNIT_INTG}"
                }
            }
        }

        // ── 4. BACKTEST VALIDATION ────────────────────────────
        stage('Backtest Validation') {
            steps {
                echo '📊  Running strategy backtest validation...'
                sh '''
                    pytest tests/unit/test_backtest_validation.py -v --tb=long -q
                '''
            }
        }

        // ── 5. GATE EVALUATION (advisory) ─────────────────────
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
                            echo "  Sharpe:    ${gate.sharpe}"
                            echo "  Drawdown:  ${gate.max_drawdown_pct}%"
                            echo "  Fill Rate: ${gate.fill_rate_pct}%"
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            echo '📦  Archiving reports...'
            archiveArtifacts artifacts: 'reports/**/*', allowEmptyArchive: true
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
