-- =============================================================================
-- SITARAM — TimescaleDB Schema Initialization
-- Runs automatically on first container startup
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- TABLE: order_book_ticks — Raw order book state per tick
-- =============================================================================
CREATE TABLE IF NOT EXISTS order_book_ticks (
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL DEFAULT 'BTCUSDT',
    best_bid        NUMERIC(12,1),
    best_ask        NUMERIC(12,1),
    mid_price       NUMERIC(12,2),
    spread          NUMERIC(8,2),
    bid_depth_10    NUMERIC(16,6),  -- Total BTC volume at top 10 bid levels
    ask_depth_10    NUMERIC(16,6),  -- Total BTC volume at top 10 ask levels
    update_id       BIGINT
);
SELECT create_hypertable('order_book_ticks', 'time', if_not_exists => TRUE);
CREATE INDEX ON order_book_ticks (symbol, time DESC);

-- =============================================================================
-- TABLE: feature_metrics — All 10 computed features per tick
-- =============================================================================
CREATE TABLE IF NOT EXISTS feature_metrics (
    computed_at         TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL DEFAULT 'BTCUSDT',
    metric_name         TEXT NOT NULL,   -- 'OBI_3', 'IC_OBI_3', 'SPREAD', etc.
    ic_value            NUMERIC(8,6),    -- IC at this horizon
    feature_value       NUMERIC(16,8),   -- Raw feature value
    horizon_ms          INTEGER,         -- IC horizon in milliseconds
    rolling_window      INTEGER          -- Window size used
);
SELECT create_hypertable('feature_metrics', 'computed_at', if_not_exists => TRUE);
CREATE INDEX ON feature_metrics (metric_name, computed_at DESC);

-- =============================================================================
-- TABLE: trades — All simulated fills
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
    trade_time          TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL DEFAULT 'BTCUSDT',
    side                TEXT NOT NULL,   -- 'buy' or 'sell'
    price               NUMERIC(12,1),
    quantity            NUMERIC(16,6),
    realized_pnl        NUMERIC(16,6),   -- P&L after fees
    cumulative_pnl      NUMERIC(16,6),   -- Running total P&L
    fill_latency_ms     NUMERIC(8,2),    -- Time from quote to fill
    adverse_fill        BOOLEAN,         -- True if fill was against our position
    inventory_after     NUMERIC(16,6),   -- Inventory after this trade
    quote_id            TEXT             -- Which quote was filled
);
SELECT create_hypertable('trades', 'trade_time', if_not_exists => TRUE);
CREATE INDEX ON trades (symbol, trade_time DESC);

-- =============================================================================
-- TABLE: claude_alerts — All AI-generated alerts
-- =============================================================================
CREATE TABLE IF NOT EXISTS claude_alerts (
    alert_time          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    severity            TEXT NOT NULL,   -- CRITICAL / WARNING / INFO
    category            TEXT NOT NULL,   -- SIGNAL / RISK / EXECUTION / PIPELINE
    message             TEXT NOT NULL,
    claude_analysis     TEXT,            -- Full Claude API response
    metrics_json        JSONB            -- Snapshot of metrics at alert time
);
SELECT create_hypertable('claude_alerts', 'alert_time', if_not_exists => TRUE);

-- =============================================================================
-- TABLE: jenkins_approvals — Jenkins validation audit trail
-- =============================================================================
CREATE TABLE IF NOT EXISTS jenkins_approvals (
    approved_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    build_number        INTEGER,
    all_gates_passed    BOOLEAN NOT NULL,
    gate_results        JSONB            -- Individual gate pass/fail details
);

-- =============================================================================
-- VIEW: v_latest_strategy_metrics — Used by Jenkins validation stage
-- Returns all metrics Jenkins needs in ONE query
-- =============================================================================
CREATE OR REPLACE VIEW v_latest_strategy_metrics AS
SELECT
    -- IC: 3-day rolling average
    (SELECT AVG(ic_value) FROM feature_metrics
     WHERE metric_name = 'IC_OBI_3'
       AND computed_at > NOW() - INTERVAL '3 days') AS ic_value,

    -- Sharpe ratio from last 24h of trades
    (SELECT CASE WHEN STDDEV(realized_pnl) > 0
            THEN AVG(realized_pnl) / STDDEV(realized_pnl) * SQRT(86400)
            ELSE 0 END
     FROM trades WHERE trade_time > NOW() - INTERVAL '1 day') AS sharpe_ratio,

    -- Max drawdown %
    (SELECT CASE WHEN MAX(cumulative_pnl) > 0
            THEN (MAX(cumulative_pnl) - MIN(cumulative_pnl)) / MAX(cumulative_pnl) * 100
            ELSE 0 END
     FROM trades WHERE trade_time > NOW() - INTERVAL '1 day') AS drawdown_pct,

    -- Fill rate (filled quotes / total quotes placed)
    (SELECT COUNT(*)::float / NULLIF((SELECT COUNT(*) FROM trades
     WHERE trade_time > NOW() - INTERVAL '1 day'), 0)
     FROM trades WHERE trade_time > NOW() - INTERVAL '1 day') AS fill_rate,

    -- Adverse selection rate
    (SELECT AVG(CASE WHEN adverse_fill THEN 1.0 ELSE 0.0 END)
     FROM trades WHERE trade_time > NOW() - INTERVAL '1 day') AS adverse_sel_rate,

    -- Quote to trade ratio (stored by Python engine)
    (SELECT feature_value FROM feature_metrics
     WHERE metric_name = 'QTR'
     ORDER BY computed_at DESC LIMIT 1) AS qtr,

    -- Average spread in ticks
    (SELECT AVG(spread) / 0.1 FROM order_book_ticks
     WHERE time > NOW() - INTERVAL '1 hour') AS spread_ticks,

    -- IC stability: days with IC > threshold
    (SELECT COUNT(DISTINCT DATE(computed_at))
     FROM feature_metrics
     WHERE metric_name = 'IC_OBI_3'
       AND ic_value > 0.02
       AND computed_at > NOW() - INTERVAL '3 days') AS ic_stable_days,

    -- Average fill latency
    (SELECT AVG(fill_latency_ms) FROM trades
     WHERE trade_time > NOW() - INTERVAL '1 hour') AS avg_latency_ms,

    -- Fee-adjusted P&L
    (SELECT SUM(realized_pnl) FROM trades
     WHERE trade_time > NOW() - INTERVAL '1 day') AS fee_adj_pnl,

    -- Inventory mean reversion rate
    (SELECT AVG(CASE WHEN ABS(inventory_after) < 0.1 THEN 1.0 ELSE 0.0 END)
     FROM trades WHERE trade_time > NOW() - INTERVAL '1 day') AS inv_reversion_rate;

-- =============================================================================
-- CONTINUOUS AGGREGATES — Pre-compute hourly stats for dashboard charts
-- =============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_pnl
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', trade_time) AS hour,
    SUM(realized_pnl)     AS hourly_pnl,
    COUNT(*)              AS trade_count,
    AVG(fill_latency_ms)  AS avg_latency
FROM trades
GROUP BY hour;

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_ic
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', computed_at) AS hour,
    metric_name,
    AVG(ic_value)    AS avg_ic,
    STDDEV(ic_value) AS ic_stddev
FROM feature_metrics
WHERE metric_name LIKE 'IC_%'
GROUP BY hour, metric_name;

COMMIT;
