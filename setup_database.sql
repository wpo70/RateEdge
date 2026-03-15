-- RateEdge Swaption Database Setup
-- Run this in Azure PostgreSQL to create the swaption database

-- First, connect to the default 'postgres' database and create the new database:
-- CREATE DATABASE swaption;

-- Then connect to 'swaption' database and run:

-- User configs table - stores curves, vols, SABR params per user
CREATE TABLE IF NOT EXISTS user_configs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    config_type VARCHAR(50) NOT NULL,  -- 'curve', 'atm_vols', 'sabr_alpha', 'sabr_beta', 'sabr_rho', 'sabr_nu', 'basis_6v3', 'basis_3v1', 'basis_ois'
    currency VARCHAR(3) NOT NULL,       -- 'AUD', 'NZD', 'USD'
    data JSONB NOT NULL,                -- Actual data as JSON
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, config_type, currency)
);

-- Index for fast lookups by user
CREATE INDEX IF NOT EXISTS idx_user_configs_user ON user_configs(user_id);

-- Index for fast lookups by config type
CREATE INDEX IF NOT EXISTS idx_user_configs_type ON user_configs(config_type);

-- Optional: Priced trades history
CREATE TABLE IF NOT EXISTS trade_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    trade_date TIMESTAMP DEFAULT NOW(),
    instrument_type VARCHAR(20) NOT NULL,  -- 'Swaption', 'Cap', 'Floor'
    currency VARCHAR(3) NOT NULL,
    side VARCHAR(20) NOT NULL,
    expiry VARCHAR(10),
    tenor VARCHAR(10),
    notional DECIMAL(18,2),
    strike DECIMAL(10,6),
    forward_rate DECIMAL(10,6),
    vol_used DECIMAL(10,4),
    pv DECIMAL(18,2),
    pv_bp DECIMAL(10,4),
    delta DECIMAL(18,2),
    gamma DECIMAL(18,2),
    vega DECIMAL(18,2),
    theta DECIMAL(18,2),
    label VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_trade_history_user ON trade_history(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_history_date ON trade_history(trade_date);

-- Grant permissions (adjust username as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- View all configs for a user:
-- SELECT config_type, currency, updated_at FROM user_configs WHERE user_id = 'wpo' ORDER BY updated_at DESC;

-- Delete old configs (older than 30 days):
-- DELETE FROM user_configs WHERE updated_at < NOW() - INTERVAL '30 days';
