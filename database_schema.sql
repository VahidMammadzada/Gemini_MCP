CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: stock_transactions
-- Stores all buy/sell transactions for the portfolio
CREATE TABLE IF NOT EXISTS stock_transactions (
    id SERIAL PRIMARY KEY,
    transaction_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
    quantity DECIMAL(15, 4) NOT NULL CHECK (quantity > 0),
    price DECIMAL(15, 4) NOT NULL CHECK (price >= 0),
    transaction_date DATE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: portfolio_positions
-- Stores current aggregated positions (calculated from transactions)
CREATE TABLE IF NOT EXISTS portfolio_positions (
    symbol VARCHAR(10) PRIMARY KEY,
    total_quantity DECIMAL(15, 4) NOT NULL CHECK (total_quantity >= 0),
    avg_cost_basis DECIMAL(15, 4) NOT NULL CHECK (avg_cost_basis >= 0),
    first_purchase_date DATE,
    last_transaction_date DATE,
    total_invested DECIMAL(15, 2) NOT NULL DEFAULT 0,
    realized_gains DECIMAL(15, 2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: portfolio_snapshots
-- Stores historical portfolio value snapshots for performance tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(15, 2) NOT NULL,
    total_cost_basis DECIMAL(15, 2) NOT NULL,
    total_gain_loss DECIMAL(15, 2) NOT NULL,
    total_gain_loss_pct DECIMAL(10, 4) NOT NULL,
    num_positions INTEGER NOT NULL,
    snapshot_data JSONB,  -- Store detailed position data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: stock_metadata
-- Cache stock information to reduce API calls
CREATE TABLE IF NOT EXISTS stock_metadata (
    symbol VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON stock_transactions(symbol);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON stock_transactions(transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON stock_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_metadata_sector ON stock_metadata(sector);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers to automatically update updated_at
CREATE TRIGGER update_stock_transactions_updated_at
    BEFORE UPDATE ON stock_transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolio_positions_updated_at
    BEFORE UPDATE ON portfolio_positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_stock_metadata_updated_at
    BEFORE UPDATE ON stock_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to recalculate portfolio positions
-- This ensures positions are always in sync with transactions
CREATE OR REPLACE FUNCTION recalculate_position(p_symbol VARCHAR)
RETURNS VOID AS $$
DECLARE
    v_total_quantity DECIMAL(15, 4);
    v_avg_cost_basis DECIMAL(15, 4);
    v_total_invested DECIMAL(15, 2);
    v_realized_gains DECIMAL(15, 2);
    v_first_purchase DATE;
    v_last_transaction DATE;
BEGIN
    -- Calculate position from transactions
    WITH position_calc AS (
        SELECT
            SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as net_quantity,
            SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price ELSE 0 END) as total_cost,
            SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE 0 END) as total_bought,
            MIN(CASE WHEN transaction_type = 'BUY' THEN transaction_date END) as first_buy,
            MAX(transaction_date) as last_trans,
            SUM(CASE WHEN transaction_type = 'SELL' THEN quantity * price ELSE 0 END) -
            SUM(CASE WHEN transaction_type = 'SELL' THEN quantity *
                (SELECT AVG(price) FROM stock_transactions WHERE symbol = p_symbol AND transaction_type = 'BUY')
                ELSE 0 END) as realized_gain
        FROM stock_transactions
        WHERE symbol = p_symbol
    )
    SELECT
        net_quantity,
        CASE WHEN total_bought > 0 THEN total_cost / total_bought ELSE 0 END,
        total_cost,
        COALESCE(realized_gain, 0),
        first_buy,
        last_trans
    INTO v_total_quantity, v_avg_cost_basis, v_total_invested, v_realized_gains, v_first_purchase, v_last_transaction
    FROM position_calc;

    -- Update or delete position
    IF v_total_quantity > 0 THEN
        INSERT INTO portfolio_positions (
            symbol, total_quantity, avg_cost_basis, total_invested,
            realized_gains, first_purchase_date, last_transaction_date
        )
        VALUES (
            p_symbol, v_total_quantity, v_avg_cost_basis, v_total_invested,
            v_realized_gains, v_first_purchase, v_last_transaction
        )
        ON CONFLICT (symbol) DO UPDATE SET
            total_quantity = EXCLUDED.total_quantity,
            avg_cost_basis = EXCLUDED.avg_cost_basis,
            total_invested = EXCLUDED.total_invested,
            realized_gains = EXCLUDED.realized_gains,
            first_purchase_date = EXCLUDED.first_purchase_date,
            last_transaction_date = EXCLUDED.last_transaction_date;
    ELSE
        -- If position is fully closed, delete it
        DELETE FROM portfolio_positions WHERE symbol = p_symbol;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically recalculate positions when transactions change
CREATE OR REPLACE FUNCTION trigger_recalculate_position()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM recalculate_position(OLD.symbol);
        RETURN OLD;
    ELSE
        PERFORM recalculate_position(NEW.symbol);
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_recalculate_position
    AFTER INSERT OR UPDATE OR DELETE ON stock_transactions
    FOR EACH ROW
    EXECUTE FUNCTION trigger_recalculate_position();

-- View: current_portfolio_summary
-- Provides a quick overview of the portfolio
CREATE OR REPLACE VIEW current_portfolio_summary AS
SELECT
    COUNT(*) as total_positions,
    SUM(total_quantity) as total_shares,
    SUM(total_invested) as total_invested,
    SUM(realized_gains) as total_realized_gains
FROM portfolio_positions;
