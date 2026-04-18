#!/bin/bash
# =============================================================================
# SITARAM — Full Platform Startup Script
# Run this from WSL2: bash scripts/start.sh
# =============================================================================

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          SITARAM HFT Market Making Platform              ║"
echo "║          Starting full pipeline...                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Check .env exists ─────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found"
    echo "   Copy .env.example to .env and fill in your API keys"
    echo "   cp .env.example .env"
    exit 1
fi

# ── Check Docker is running ───────────────────────────────────────────────────
if ! docker info > /dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running. Start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"
echo "✅ .env file found"
echo ""

# ── Pull latest base images ───────────────────────────────────────────────────
echo "📦 Pulling base images..."
docker compose pull --ignore-pull-failures 2>/dev/null || true

# ── Build all custom images ───────────────────────────────────────────────────
echo ""
echo "🔨 Building custom images..."
docker compose build --parallel

# ── Start infrastructure first (Kafka, Redis, TimescaleDB) ───────────────────
echo ""
echo "🚀 Stage 1: Starting infrastructure..."
docker compose up -d zookeeper kafka redis timescaledb

echo "   Waiting for infrastructure health checks..."
sleep 15

# ── Initialize Kafka topics ───────────────────────────────────────────────────
echo "📋 Initializing Kafka topics..."
docker compose up kafka-init

# ── Start compute engines ─────────────────────────────────────────────────────
echo ""
echo "🚀 Stage 2: Starting compute engines..."
docker compose up -d python-engine ray-head

echo "   Waiting for compute engines..."
sleep 10

# ── Start Claude Intelligence Agent ──────────────────────────────────────────
echo ""
echo "🧠 Stage 3: Starting Claude Intelligence Agent..."
docker compose up -d claude-agent
sleep 5

# ── Start Frontend ────────────────────────────────────────────────────────────
echo ""
echo "🎨 Stage 4: Starting React Dashboard & Nginx..."
docker compose up -d react-dashboard nginx

# ── Start Jenkins LAST (it will orchestrate from here) ───────────────────────
echo ""
echo "🔧 Starting Jenkins conductor..."
docker compose up -d jenkins

# ── Wait for Jenkins to boot ──────────────────────────────────────────────────
echo "   Waiting for Jenkins to initialize (60s)..."
sleep 60

# ── Show status ───────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  SITARAM IS LIVE                         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  React Dashboard  →  http://localhost:3000               ║"
echo "║  Jenkins UI       →  http://localhost:8090               ║"
echo "║  Ray Dashboard    →  http://localhost:8265               ║"
echo "║  TimescaleDB      →  localhost:5432                      ║"
echo "║  Kafka            →  localhost:9092                      ║"
echo "║  Redis            →  localhost:6379                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Jenkins will now run validation pipeline...             ║"
echo "║  Trading approved only after ALL 11 gates pass           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Show running containers ───────────────────────────────────────────────────
docker compose ps
