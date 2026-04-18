#!/bin/bash
# =============================================================================
# SITARAM — Graceful Shutdown
# =============================================================================
echo "🛑 Stopping SITARAM platform..."
docker compose down
echo "✅ All services stopped. Data volumes preserved."
echo "   To also remove data volumes: docker compose down -v"
