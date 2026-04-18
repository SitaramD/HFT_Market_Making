# SITARAM — HFT Market Making Platform
## Complete Docker + Jenkins + Claude AI Setup Guide

---

## Architecture

```
docker compose up
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    SITARAM PLATFORM                          │
│                                                             │
│  ┌──────────┐    ┌───────┐    ┌─────────────┐              │
│  │  Kafka   │───▶│ Redis │───▶│Python Engine│              │
│  │(ob200-raw│    │(200   │    │(Features,   │              │
│  │trades-raw│    │ ticks)│    │ Quoting)    │              │
│  └──────────┘    └───────┘    └──────┬──────┘              │
│        ▲                             │                      │
│        │                             ▼                      │
│  Global Datafeed               ┌──────────┐                 │
│  WebSocket Feed                │   Ray    │                 │
│                                │(Parallel │                 │
│                                │ Compute) │                 │
│                                └──────┬───┘                 │
│                                       │                     │
│                                       ▼                     │
│  ┌─────────────┐              ┌─────────────┐              │
│  │  Jenkins    │◀────────────▶│TimescaleDB  │              │
│  │ (Conductor) │  validates   │(Metrics,    │              │
│  │ 11 gates    │              │ P&L, Alerts)│              │
│  └─────────────┘              └──────┬──────┘              │
│                                      │                      │
│                                      ▼                      │
│                              ┌─────────────┐               │
│                              │Claude Agent │               │
│                              │(Intelligence│               │
│                              │ Monitoring) │               │
│                              └──────┬──────┘               │
│                                     │                      │
│                                     ▼                      │
│                              ┌─────────────┐               │
│                              │   React     │               │
│                              │  Dashboard  │               │
│                              │(6 panels)   │               │
│                              └─────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **Docker Desktop** installed and running on Windows 11
2. **WSL2** enabled (Ubuntu recommended)
3. **Anthropic API key** from https://console.anthropic.com
4. Your data files in `E:\Binance\March\`

---

## Quick Start

### Step 1: Open WSL2 terminal

```bash
# In Windows: press Win+R → type wsl → Enter
```

### Step 2: Clone / copy this project

```bash
cd ~
# Copy the sitaram folder here
```

### Step 3: Create your .env file

```bash
cd ~/sitaram
cp .env.example .env
nano .env   # Fill in ANTHROPIC_API_KEY and other values
```

### Step 4: Launch everything

```bash
bash scripts/start.sh
```

That's it. One command starts the entire platform.

---

## What Happens After `docker compose up`

```
T+0s    Zookeeper starts
T+5s    Kafka starts (depends on Zookeeper healthy)
T+10s   Redis starts
T+15s   TimescaleDB starts, runs SQL schema init
T+20s   Kafka topics created (ob200-raw, trades-raw, etc.)
T+30s   Python Engine starts consuming from Kafka
T+40s   Ray cluster starts
T+50s   Claude Agent starts monitoring
T+60s   React Dashboard goes live
T+90s   Jenkins boots, runs first validation pipeline
T+120s  IF all 11 gates pass → Redis flag set → trading APPROVED
T+150s  Jenkins re-runs every 30 minutes
```

---

## Jenkins Validation Gates (ALL must pass)

| # | Gate | Threshold | Blocks Trading? |
|---|------|-----------|-----------------|
| 1 | Information Coefficient | IC > 0.02 | YES |
| 2 | Sharpe Ratio | > 1.5 | YES |
| 3 | Max Drawdown | < 5% | YES |
| 4 | Fill Rate | > 60% | YES |
| 5 | Adverse Selection | < 30% | YES |
| 6 | Quote-to-Trade Ratio | < 10 | YES |
| 7 | Minimum Spread | > 1 tick | YES |
| 8 | IC Stability | >= 3 days | YES |
| 9 | Tick-to-Fill Latency | < 50ms | YES |
| 10 | Fee-Adjusted P&L | > 0 | YES |
| 11 | Inventory Mean Reversion | > 70% | YES |
| + | All Unit Tests | 100% pass | YES |

---

## Claude AI Alert Types

| Category | Trigger | Severity |
|----------|---------|----------|
| SIGNAL | IC < 0.02 | CRITICAL |
| SIGNAL | IC degrading rapidly | WARNING |
| RISK | Inventory > 80% of max | WARNING |
| RISK | Inventory at 100% max | CRITICAL |
| RISK | Drawdown > 5% | CRITICAL |
| EXECUTION | Latency > 100ms | WARNING |
| EXECUTION | Adverse selection > 30% | WARNING |
| PIPELINE | Kafka consumer lag > 1000 | WARNING |
| PIPELINE | Tick buffer empty | CRITICAL |
| PIPELINE | Any service down | CRITICAL |

---

## Access URLs

| Service | URL | Notes |
|---------|-----|-------|
| React Dashboard | http://localhost:3000 | Main operator view |
| Jenkins | http://localhost:8090 | Pipeline management |
| Ray Dashboard | http://localhost:8265 | Compute monitoring |
| TimescaleDB | localhost:5432 | PostgreSQL client |
| Kafka | localhost:9092 | Kafka tools |
| Redis | localhost:6379 | redis-cli |

---

## Dashboard Panels (React)

1. **Live Order Book** — Real-time bid/ask depth visualization
2. **P&L Curve** — Cumulative P&L with drawdown overlay
3. **Inventory Gauge** — Current BTC position vs max
4. **IC Decay Chart** — Rolling IC across 5 horizons (200ms–1000ms)
5. **Jenkins Pipeline Status** — Current gate results, last build
6. **Claude AI Alerts Feed** — Real-time intelligence alerts with analysis

---

## File Structure

```
sitaram/
├── docker-compose.yml          ← Single entry point
├── .env.example                ← Copy to .env, fill in keys
├── jenkins/
│   ├── Dockerfile
│   ├── plugins.txt
│   └── pipelines/
│       └── Jenkinsfile         ← 11-gate validation pipeline
├── kafka/config/
├── timescaledb/init/
│   └── 01_schema.sql           ← All tables + hypertables
├── python-engine/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/                    ← Feature engine, quoting engine
├── ray-engine/
│   └── Dockerfile
├── claude-agent/
│   ├── Dockerfile
│   ├── src/agent.py            ← Intelligence monitoring layer
│   └── prompts/monitor.txt     ← Claude system prompt (editable)
├── react-dashboard/
│   ├── Dockerfile
│   └── src/
├── nginx/nginx.conf
└── scripts/
    ├── start.sh                ← Full platform startup
    └── stop.sh                 ← Graceful shutdown
```

---

## Troubleshooting

**Kafka not starting:**
```bash
docker logs sitaram-zookeeper
docker logs sitaram-kafka
```

**Jenkins can't connect to services:**
```bash
# Check all services are healthy
docker compose ps
# Restart Jenkins after others are ready
docker compose restart jenkins
```

**TimescaleDB schema not created:**
```bash
docker exec -it sitaram-timescaledb psql -U sitaram_user -d sitaram -c "\dt"
```

**Claude Agent not alerting:**
```bash
docker logs sitaram-claude-agent
# Check ANTHROPIC_API_KEY is set in .env
```
