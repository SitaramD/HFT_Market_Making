# SITARAM HFT — CI/CD Setup Guide
# Run these steps ONCE to wire Jenkins + git hook to your project.
# All commands are PowerShell (Windows 11).

# ════════════════════════════════════════════════════════════════
# STEP 1: Copy CI files into your project 
# ════════════════════════════════════════════════════════════════

# From this directory, copy to your project root:
# D:\1\Project\Trading_Strategy\sitaram\

# Copy test suite
Copy-Item -Recurse tests\  D:\1\Project\Trading_Strategy\sitaram\tests\
Copy-Item -Recurse scripts\ D:\1\Project\Trading_Strategy\sitaram\scripts\
Copy-Item pytest.ini        D:\1\Project\Trading_Strategy\sitaram\pytest.ini
Copy-Item Jenkinsfile       D:\1\Project\Trading_Strategy\sitaram\Jenkinsfile


# ════════════════════════════════════════════════════════════════
# STEP 2: Install git post-commit hook
# ════════════════════════════════════════════════════════════════

Copy-Item jenkins\post-commit D:\1\Project\Trading_Strategy\sitaram\.git\hooks\post-commit

# Git hooks on Windows don't need chmod, but if using Git Bash:
# chmod +x .git/hooks/post-commit


# ════════════════════════════════════════════════════════════════
# STEP 3: Create Jenkins API token
# ════════════════════════════════════════════════════════════════

# 1. Open browser: http://localhost:8080
# 2. Login as admin
# 3. Go to: admin → Configure → API Token → Add New Token
# 4. Name it: sitaram-ci
# 5. Copy the token, then:

Set-Content D:\1\Project\Trading_Strategy\sitaram\.jenkins_token "YOUR_TOKEN_HERE"

# Add to .gitignore so it's never committed:
Add-Content D:\1\Project\Trading_Strategy\sitaram\.gitignore ".jenkins_token"


# ════════════════════════════════════════════════════════════════
# STEP 4: Create Jenkins pipeline job
# ════════════════════════════════════════════════════════════════

# In Jenkins UI (http://localhost:8080):
# 1. New Item → Name: "sitaram-hft-pipeline" → Type: Pipeline → OK
# 2. General:
#    ✅ Discard old builds → Keep max 30
# 3. Build Triggers:
#    ✅ Poll SCM → Schedule: H/1 * * * *   (every 1 minute)
# 4. Pipeline:
#    Definition: Pipeline script from SCM
#    SCM: Git
#    Repository URL: D:/1/Project/Trading_Strategy/sitaram
#    Branch: */master  (or your branch name)
#    Script Path: Jenkinsfile
# 5. Save


# ════════════════════════════════════════════════════════════════
# STEP 5: Verify the pipeline
# ════════════════════════════════════════════════════════════════

# Make a test commit:
cd D:\1\Project\Trading_Strategy\sitaram
git add .
git commit -m "ci: add Jenkins pipeline and test suite"

# You should see:
# [SITARAM CI] Triggering Jenkins pipeline: sitaram-hft-pipeline
# [SITARAM CI] Jenkins build triggered successfully (HTTP 201)

# Then open: http://localhost:8080/job/sitaram-hft-pipeline/
# and watch the build run.


# ════════════════════════════════════════════════════════════════
# STEP 6: Run tests locally (before committing)
# ════════════════════════════════════════════════════════════════

cd D:\1\Project\Trading_Strategy\sitaram

# Install test deps:
pip install pytest pytest-cov pytest-xdist hypothesis redis psycopg2-binary numpy

# Run unit tests only (fast, no Docker needed):
pytest tests/unit/ -v --tb=short

# Run integration tests (requires Docker stack running):
pytest tests/integration/ -v --tb=short

# Run everything:
pytest tests/ -v --tb=short


# ════════════════════════════════════════════════════════════════
# PIPELINE FLOW (every git commit)
# ════════════════════════════════════════════════════════════════
#
#  git commit
#      │
#      ▼
#  post-commit hook ──────────► Jenkins triggered immediately
#      │                              │
#      │                              ▼
#      │                    [1] Setup (pip install)
#      │                              │
#      │                    [2] Lint (flake8, black) ← advisory
#      │                              │
#      │                    [3] Unit Tests ← BLOCKING if fail
#      │                         ├─ test_as_spread.py
#      │                         ├─ test_gates.py
#      │                         ├─ test_pnl_fees.py
#      │                         └─ test_orderbook_fills.py
#      │                              │
#      │                    [4] Integration Tests (real Docker)
#      │                         └─ test_live_stack.py
#      │                              │
#      │                    [5] Backtest Validation
#      │                         └─ test_backtest_validation.py
#      │                              │
#      │                    [6] Gate Evaluation ← ADVISORY only
#      │                         └─ gate_evaluator.py → gate_report.json
#      │                              │
#      │                    [7] Deploy to Paper Trading
#      │                         └─ docker cp + docker restart
#      │
#  Jenkins polls every 1 min as fallback if hook failed
