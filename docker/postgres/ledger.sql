-- ledger.sql
-- Sui/Walrus proxy ledger for Scenario C
-- Simulates immutable blockchain records with gas cost estimation

-- Ledger entries (simulates Sui objects + Walrus blobs)
CREATE TABLE ledger_entries (
    id VARCHAR(50) PRIMARY KEY,  -- Simulates Sui Object ID
    blob_id VARCHAR(100) NOT NULL,  -- Simulates Walrus blob reference

    -- Transaction data
    transaction_type VARCHAR(50) NOT NULL,  -- "bid_request", "bid_response", "deal", "delivery"
    payload JSONB NOT NULL,
    payload_bytes INT NOT NULL,

    -- Provenance
    created_by VARCHAR(50) NOT NULL,  -- Agent ID

    -- Immutability simulation (hash chain)
    content_hash VARCHAR(64) NOT NULL,  -- SHA256 of payload
    previous_hash VARCHAR(64),  -- Link to previous entry (chain)

    -- Gas estimation
    estimated_sui_gas DECIMAL(20, 10) NOT NULL,  -- In SUI
    estimated_walrus_cost DECIMAL(20, 10) NOT NULL,  -- In SUI
    total_cost_sui DECIMAL(20, 10) NOT NULL,
    total_cost_usd DECIMAL(15, 6) NOT NULL,  -- At current SUI price

    -- Metadata
    sui_price_usd DECIMAL(10, 4) DEFAULT 1.50,  -- SUI/USD rate used

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gas cost aggregates (for reporting)
CREATE TABLE gas_cost_summary (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,

    -- Counts
    total_transactions INT NOT NULL,
    bid_requests INT DEFAULT 0,
    bid_responses INT DEFAULT 0,
    deals INT DEFAULT 0,
    deliveries INT DEFAULT 0,

    -- Costs
    total_sui_gas DECIMAL(20, 10) NOT NULL,
    total_walrus_cost DECIMAL(20, 10) NOT NULL,
    total_sui DECIMAL(20, 10) NOT NULL,
    total_usd DECIMAL(15, 6) NOT NULL,

    -- Averages
    avg_cost_per_tx_sui DECIMAL(20, 10),
    avg_cost_per_tx_usd DECIMAL(15, 6),

    -- Comparison metrics
    impressions_recorded BIGINT,
    cost_per_1000_impressions_usd DECIMAL(15, 6),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(date)
);

-- Verification log (for dispute resolution)
CREATE TABLE ledger_verification (
    id SERIAL PRIMARY KEY,
    entry_id VARCHAR(50) REFERENCES ledger_entries(id),

    verification_type VARCHAR(50) NOT NULL,  -- "hash_check", "chain_integrity", "payload_match"
    verified BOOLEAN NOT NULL,
    discrepancy TEXT,

    verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- View: Cost comparison (blockchain vs exchange fees)
CREATE VIEW cost_comparison AS
SELECT
    l.date,
    l.total_usd as blockchain_cost,
    COALESCE(t.exchange_fees, 0) as exchange_fees,
    COALESCE(t.total_spend, 0) as total_spend,
    CASE
        WHEN COALESCE(t.total_spend, 0) > 0
        THEN ROUND((l.total_usd / t.total_spend) * 100, 4)
        ELSE 0
    END as blockchain_pct_of_spend,
    CASE
        WHEN COALESCE(t.total_spend, 0) > 0
        THEN ROUND((COALESCE(t.exchange_fees, 0) / t.total_spend) * 100, 2)
        ELSE 0
    END as exchange_pct_of_spend
FROM gas_cost_summary l
LEFT JOIN (
    SELECT
        DATE(created_at) as date,
        SUM(exchange_fee) as exchange_fees,
        SUM(buyer_spend) as total_spend
    FROM transactions
    WHERE scenario = 'A'
    GROUP BY DATE(created_at)
) t ON l.date = t.date;

-- View: Ledger chain integrity
CREATE VIEW chain_integrity AS
SELECT
    e1.id,
    e1.content_hash,
    e1.previous_hash,
    e2.content_hash as expected_previous,
    CASE
        WHEN e1.previous_hash IS NULL THEN TRUE  -- Genesis entry
        WHEN e1.previous_hash = e2.content_hash THEN TRUE
        ELSE FALSE
    END as chain_valid
FROM ledger_entries e1
LEFT JOIN ledger_entries e2 ON e1.previous_hash = e2.content_hash
ORDER BY e1.created_at;

-- Gas estimation constants (based on ADS Explorer data)
-- These can be updated based on actual Sui/Walrus costs
CREATE TABLE gas_constants (
    id SERIAL PRIMARY KEY,
    constant_name VARCHAR(50) UNIQUE NOT NULL,
    value DECIMAL(20, 10) NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO gas_constants (constant_name, value, description) VALUES
    ('SUI_BASE_GAS', 0.001, 'Base transaction cost in SUI'),
    ('SUI_PER_BYTE', 0.0000001, 'Cost per byte of data in SUI'),
    ('WALRUS_BASE_COST', 0.0005, 'Minimum blob storage cost in SUI'),
    ('WALRUS_PER_KB', 0.00001, 'Cost per KB stored in SUI'),
    ('SUI_PRICE_USD', 1.50, 'Current SUI/USD exchange rate');

-- Indexes
CREATE INDEX idx_ledger_created_by ON ledger_entries(created_by);
CREATE INDEX idx_ledger_type ON ledger_entries(transaction_type);
CREATE INDEX idx_ledger_created_at ON ledger_entries(created_at);
CREATE INDEX idx_gas_summary_date ON gas_cost_summary(date);
