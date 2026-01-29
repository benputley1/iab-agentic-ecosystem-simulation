-- ledger.sql
-- Sui proxy ledger schema for Scenario C (Alkimi ledger-backed)
-- Simulates immutable blockchain records with Walrus blob storage

-- Ledger entries (simulates Sui objects)
CREATE TABLE ledger_entries (
    id VARCHAR(50) PRIMARY KEY,
    blob_id VARCHAR(100) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hash VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64),
    estimated_sui_gas DECIMAL(18, 8) NOT NULL,
    estimated_walrus_cost DECIMAL(18, 8) NOT NULL,
    payload_bytes INT NOT NULL
);

-- Gas estimation log (for cost comparison)
CREATE TABLE gas_estimates (
    id SERIAL PRIMARY KEY,
    ledger_entry_id VARCHAR(50) REFERENCES ledger_entries(id),
    payload_bytes INT NOT NULL,
    sui_gas_estimate DECIMAL(18, 8) NOT NULL,
    walrus_cost_estimate DECIMAL(18, 8) NOT NULL,
    total_sui DECIMAL(18, 8) NOT NULL,
    total_usd DECIMAL(15, 6) NOT NULL,
    sui_price_usd DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cumulative gas costs per campaign (for comparison with exchange fees)
CREATE TABLE campaign_blockchain_costs (
    campaign_id VARCHAR(50) NOT NULL REFERENCES campaigns(id),
    total_transactions INT DEFAULT 0,
    total_sui_gas DECIMAL(18, 8) DEFAULT 0,
    total_walrus_cost DECIMAL(18, 8) DEFAULT 0,
    total_cost_usd DECIMAL(15, 6) DEFAULT 0,
    equivalent_exchange_fees DECIMAL(15, 2) DEFAULT 0,
    savings_vs_exchange DECIMAL(15, 2) DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id)
);

-- State recovery log (when agents recover from ledger)
CREATE TABLE ledger_recovery_events (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    recovery_type VARCHAR(50) NOT NULL,
    entries_recovered INT NOT NULL,
    recovery_time_ms INT NOT NULL,
    state_completeness DECIMAL(5, 4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_ledger_type ON ledger_entries(transaction_type);
CREATE INDEX idx_ledger_created_by ON ledger_entries(created_by);
CREATE INDEX idx_ledger_hash ON ledger_entries(hash);
CREATE INDEX idx_gas_estimates_entry ON gas_estimates(ledger_entry_id);

-- View for total blockchain costs
CREATE VIEW blockchain_cost_summary AS
SELECT
    SUM(total_transactions) as total_transactions,
    SUM(total_sui_gas) as total_sui_gas,
    SUM(total_walrus_cost) as total_walrus_cost,
    SUM(total_cost_usd) as total_cost_usd,
    SUM(equivalent_exchange_fees) as equivalent_exchange_fees,
    SUM(savings_vs_exchange) as total_savings
FROM campaign_blockchain_costs;

-- View for cost per 1000 impressions
CREATE VIEW cost_per_1k_impressions AS
SELECT
    c.id as campaign_id,
    c.impressions_delivered,
    cbc.total_cost_usd,
    CASE
        WHEN c.impressions_delivered > 0
        THEN (cbc.total_cost_usd / c.impressions_delivered) * 1000
        ELSE 0
    END as cost_per_1k
FROM campaigns c
JOIN campaign_blockchain_costs cbc ON c.id = cbc.campaign_id
WHERE c.scenario = 'C';

-- Function to verify ledger chain integrity
CREATE OR REPLACE FUNCTION verify_ledger_integrity()
RETURNS TABLE (
    entry_id VARCHAR(50),
    is_valid BOOLEAN,
    error_message TEXT
) AS $$
DECLARE
    prev_hash VARCHAR(64) := NULL;
    entry RECORD;
BEGIN
    FOR entry IN
        SELECT id, hash, previous_hash
        FROM ledger_entries
        ORDER BY created_at
    LOOP
        IF entry.previous_hash IS DISTINCT FROM prev_hash THEN
            RETURN QUERY SELECT
                entry.id,
                FALSE,
                'Hash chain broken: expected ' || COALESCE(prev_hash, 'NULL') ||
                ' but got ' || COALESCE(entry.previous_hash, 'NULL');
        ELSE
            RETURN QUERY SELECT entry.id, TRUE, NULL::TEXT;
        END IF;
        prev_hash := entry.hash;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
