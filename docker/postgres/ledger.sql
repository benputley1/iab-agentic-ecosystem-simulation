-- ledger.sql
-- Sui/Walrus proxy ledger for Scenario C (Alkimi)
-- Simulates immutable blockchain storage with cost estimation
-- Executed after init.sql and ground_truth.sql

-- =============================================================================
-- LEDGER ENTRIES
-- Each entry represents an immutable record (like a Sui object)
-- =============================================================================

CREATE TABLE ledger_entries (
    -- Identifiers (simulating Sui Object ID)
    entry_id VARCHAR(50) PRIMARY KEY DEFAULT 'ledger-' || uuid_generate_v4()::text,
    blob_id VARCHAR(100) NOT NULL,        -- Simulates Walrus blob reference

    -- Transaction type
    transaction_type VARCHAR(50) NOT NULL, -- 'bid_request', 'bid_response', 'deal', 'delivery', 'settlement'

    -- Full transaction data (immutable)
    payload JSONB NOT NULL,
    payload_size_bytes INTEGER NOT NULL,

    -- Provenance
    created_by VARCHAR(50) NOT NULL,      -- Agent ID
    created_by_type VARCHAR(50) NOT NULL, -- 'buyer', 'seller', 'exchange', 'system'

    -- Chain linking (simulating blockchain)
    entry_hash VARCHAR(64) NOT NULL,      -- SHA256 of payload
    previous_hash VARCHAR(64),            -- Link to previous entry (chain)
    block_number BIGINT NOT NULL,         -- Simulated block number

    -- Gas estimation (Sui)
    estimated_sui_gas DECIMAL(12,8) NOT NULL,
    sui_gas_price DECIMAL(12,8) DEFAULT 0.001,  -- Price per gas unit

    -- Storage cost (Walrus)
    estimated_walrus_cost DECIMAL(12,8) NOT NULL,
    storage_duration_epochs INTEGER DEFAULT 100,

    -- Combined costs
    total_cost_sui DECIMAL(12,8) NOT NULL,
    total_cost_usd DECIMAL(12,6) NOT NULL,

    -- Metadata
    simulation_day INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_hash CHECK (length(entry_hash) = 64)
);

CREATE INDEX idx_ledger_entries_type ON ledger_entries(transaction_type);
CREATE INDEX idx_ledger_entries_created_by ON ledger_entries(created_by);
CREATE INDEX idx_ledger_entries_block ON ledger_entries(block_number);
CREATE INDEX idx_ledger_entries_day ON ledger_entries(simulation_day);
CREATE INDEX idx_ledger_entries_hash ON ledger_entries(entry_hash);
CREATE INDEX idx_ledger_entries_prev_hash ON ledger_entries(previous_hash);

-- =============================================================================
-- DEAL RECORDS (Immutable deal history)
-- =============================================================================

CREATE TABLE ledger_deals (
    id SERIAL PRIMARY KEY,
    entry_id VARCHAR(50) NOT NULL REFERENCES ledger_entries(entry_id),
    deal_id VARCHAR(50) NOT NULL,

    -- Deal parties
    buyer_id VARCHAR(50) NOT NULL,
    seller_id VARCHAR(50) NOT NULL,

    -- Deal terms (immutable once recorded)
    impressions BIGINT NOT NULL,
    cpm DECIMAL(10,2) NOT NULL,
    total_cost DECIMAL(12,2) NOT NULL,

    -- Status at time of recording
    deal_status VARCHAR(50) NOT NULL,

    -- Verification
    buyer_signature VARCHAR(128),         -- Simulated signature
    seller_signature VARCHAR(128),

    -- Timing
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ledger_deals_entry ON ledger_deals(entry_id);
CREATE INDEX idx_ledger_deals_deal ON ledger_deals(deal_id);
CREATE INDEX idx_ledger_deals_buyer ON ledger_deals(buyer_id);
CREATE INDEX idx_ledger_deals_seller ON ledger_deals(seller_id);

-- =============================================================================
-- DELIVERY RECORDS (Impression delivery proof)
-- =============================================================================

CREATE TABLE ledger_deliveries (
    id SERIAL PRIMARY KEY,
    entry_id VARCHAR(50) NOT NULL REFERENCES ledger_entries(entry_id),
    deal_id VARCHAR(50) NOT NULL,

    -- Delivery batch
    batch_number INTEGER NOT NULL,
    impressions_in_batch BIGINT NOT NULL,
    cumulative_impressions BIGINT NOT NULL,

    -- Verification
    delivery_proof_hash VARCHAR(64),      -- Hash of delivery evidence
    timestamp_proof TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Timing
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ledger_deliveries_entry ON ledger_deliveries(entry_id);
CREATE INDEX idx_ledger_deliveries_deal ON ledger_deliveries(deal_id);
CREATE INDEX idx_ledger_deliveries_batch ON ledger_deliveries(deal_id, batch_number);

-- =============================================================================
-- SETTLEMENT RECORDS (Financial settlements)
-- =============================================================================

CREATE TABLE ledger_settlements (
    id SERIAL PRIMARY KEY,
    entry_id VARCHAR(50) NOT NULL REFERENCES ledger_entries(entry_id),
    deal_id VARCHAR(50) NOT NULL,

    -- Settlement details
    buyer_spend DECIMAL(12,2) NOT NULL,
    seller_revenue DECIMAL(12,2) NOT NULL,
    platform_fee DECIMAL(12,2) DEFAULT 0.0,  -- Minimal for Alkimi

    -- Blockchain costs (deducted from spread)
    blockchain_costs DECIMAL(12,6) NOT NULL,

    -- Net amounts
    net_to_seller DECIMAL(12,2) NOT NULL,

    -- Verification
    settlement_hash VARCHAR(64) NOT NULL,

    -- Timing
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ledger_settlements_entry ON ledger_settlements(entry_id);
CREATE INDEX idx_ledger_settlements_deal ON ledger_settlements(deal_id);

-- =============================================================================
-- STATE RECOVERY LOG
-- Track when agents recover state from ledger
-- =============================================================================

CREATE TABLE ledger_recovery_log (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,

    -- Recovery details
    recovery_reason VARCHAR(100) NOT NULL,  -- 'restart', 'context_rot', 'checkpoint_miss'
    entries_recovered INTEGER NOT NULL,
    bytes_recovered BIGINT NOT NULL,

    -- Recovery success
    recovery_complete BOOLEAN NOT NULL,
    state_accuracy DECIMAL(5,4),          -- How accurate compared to pre-loss state
    missing_entries TEXT[],               -- Any entries that couldn't be recovered

    -- Performance
    recovery_time_ms INTEGER NOT NULL,

    -- Cost (reading from ledger has minimal cost)
    read_cost_sui DECIMAL(12,8) DEFAULT 0.0,

    -- Timing
    simulation_day INTEGER NOT NULL,
    recovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ledger_recovery_agent ON ledger_recovery_log(agent_id);
CREATE INDEX idx_ledger_recovery_day ON ledger_recovery_log(simulation_day);
CREATE INDEX idx_ledger_recovery_reason ON ledger_recovery_log(recovery_reason);

-- =============================================================================
-- GAS ESTIMATION PARAMETERS
-- Configurable parameters for cost estimation
-- =============================================================================

CREATE TABLE gas_parameters (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(50) NOT NULL UNIQUE,
    parameter_value DECIMAL(20,10) NOT NULL,
    unit VARCHAR(20),
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default parameters based on Sui/Walrus pricing
INSERT INTO gas_parameters (parameter_name, parameter_value, unit, description) VALUES
    ('sui_base_gas', 0.001, 'SUI', 'Base transaction gas cost'),
    ('sui_per_byte', 0.0000001, 'SUI/byte', 'Gas cost per byte of data'),
    ('sui_price_usd', 1.50, 'USD/SUI', 'SUI to USD conversion rate'),
    ('walrus_base_cost', 0.0005, 'SUI', 'Minimum blob storage cost'),
    ('walrus_per_kb', 0.00001, 'SUI/KB', 'Storage cost per KB'),
    ('walrus_epoch_duration', 86400, 'seconds', 'Duration of one storage epoch'),
    ('batch_size_impressions', 1000, 'impressions', 'Impressions per ledger entry (batching)');

-- =============================================================================
-- BLOCKCHAIN SIMULATION STATE
-- =============================================================================

CREATE TABLE blockchain_state (
    id SERIAL PRIMARY KEY,
    current_block_number BIGINT NOT NULL DEFAULT 0,
    total_entries BIGINT NOT NULL DEFAULT 0,
    total_gas_used DECIMAL(20,8) NOT NULL DEFAULT 0.0,
    total_storage_used_bytes BIGINT NOT NULL DEFAULT 0,
    total_cost_sui DECIMAL(20,8) NOT NULL DEFAULT 0.0,
    total_cost_usd DECIMAL(20,6) NOT NULL DEFAULT 0.0,
    last_entry_hash VARCHAR(64),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Initialize blockchain state
INSERT INTO blockchain_state (current_block_number) VALUES (0);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to calculate gas cost for a payload
CREATE OR REPLACE FUNCTION calculate_gas_cost(payload_bytes INTEGER)
RETURNS DECIMAL AS $$
DECLARE
    base_gas DECIMAL;
    per_byte DECIMAL;
BEGIN
    SELECT parameter_value INTO base_gas
    FROM gas_parameters WHERE parameter_name = 'sui_base_gas';

    SELECT parameter_value INTO per_byte
    FROM gas_parameters WHERE parameter_name = 'sui_per_byte';

    RETURN base_gas + (payload_bytes * per_byte);
END;
$$ LANGUAGE plpgsql;

-- Function to calculate Walrus storage cost
CREATE OR REPLACE FUNCTION calculate_walrus_cost(payload_bytes INTEGER)
RETURNS DECIMAL AS $$
DECLARE
    base_cost DECIMAL;
    per_kb DECIMAL;
    payload_kb DECIMAL;
BEGIN
    SELECT parameter_value INTO base_cost
    FROM gas_parameters WHERE parameter_name = 'walrus_base_cost';

    SELECT parameter_value INTO per_kb
    FROM gas_parameters WHERE parameter_name = 'walrus_per_kb';

    payload_kb := payload_bytes::DECIMAL / 1024;

    RETURN base_cost + (payload_kb * per_kb);
END;
$$ LANGUAGE plpgsql;

-- Function to convert SUI to USD
CREATE OR REPLACE FUNCTION sui_to_usd(sui_amount DECIMAL)
RETURNS DECIMAL AS $$
DECLARE
    price DECIMAL;
BEGIN
    SELECT parameter_value INTO price
    FROM gas_parameters WHERE parameter_name = 'sui_price_usd';

    RETURN sui_amount * price;
END;
$$ LANGUAGE plpgsql;

-- Function to create a new ledger entry with automatic cost calculation
CREATE OR REPLACE FUNCTION create_ledger_entry(
    p_transaction_type VARCHAR,
    p_payload JSONB,
    p_created_by VARCHAR,
    p_created_by_type VARCHAR,
    p_simulation_day INTEGER
) RETURNS VARCHAR AS $$
DECLARE
    v_entry_id VARCHAR;
    v_blob_id VARCHAR;
    v_payload_size INTEGER;
    v_entry_hash VARCHAR;
    v_previous_hash VARCHAR;
    v_block_number BIGINT;
    v_sui_gas DECIMAL;
    v_walrus_cost DECIMAL;
    v_total_sui DECIMAL;
    v_total_usd DECIMAL;
BEGIN
    -- Generate IDs
    v_entry_id := 'ledger-' || uuid_generate_v4()::text;
    v_blob_id := 'walrus-' || encode(gen_random_bytes(16), 'hex');

    -- Calculate payload size
    v_payload_size := length(p_payload::text);

    -- Calculate hash
    v_entry_hash := encode(digest(p_payload::text, 'sha256'), 'hex');

    -- Get blockchain state
    SELECT current_block_number + 1, last_entry_hash
    INTO v_block_number, v_previous_hash
    FROM blockchain_state
    LIMIT 1;

    -- Calculate costs
    v_sui_gas := calculate_gas_cost(v_payload_size);
    v_walrus_cost := calculate_walrus_cost(v_payload_size);
    v_total_sui := v_sui_gas + v_walrus_cost;
    v_total_usd := sui_to_usd(v_total_sui);

    -- Insert entry
    INSERT INTO ledger_entries (
        entry_id, blob_id, transaction_type, payload, payload_size_bytes,
        created_by, created_by_type, entry_hash, previous_hash, block_number,
        estimated_sui_gas, estimated_walrus_cost, total_cost_sui, total_cost_usd,
        simulation_day
    ) VALUES (
        v_entry_id, v_blob_id, p_transaction_type, p_payload, v_payload_size,
        p_created_by, p_created_by_type, v_entry_hash, v_previous_hash, v_block_number,
        v_sui_gas, v_walrus_cost, v_total_sui, v_total_usd, p_simulation_day
    );

    -- Update blockchain state
    UPDATE blockchain_state SET
        current_block_number = v_block_number,
        total_entries = total_entries + 1,
        total_gas_used = total_gas_used + v_sui_gas,
        total_storage_used_bytes = total_storage_used_bytes + v_payload_size,
        total_cost_sui = total_cost_sui + v_total_sui,
        total_cost_usd = total_cost_usd + v_total_usd,
        last_entry_hash = v_entry_hash,
        updated_at = NOW();

    RETURN v_entry_id;
END;
$$ LANGUAGE plpgsql;

-- Function to recover agent state from ledger
CREATE OR REPLACE FUNCTION recover_agent_state(
    p_agent_id VARCHAR,
    p_from_day INTEGER DEFAULT 0
) RETURNS JSONB AS $$
DECLARE
    v_state JSONB;
BEGIN
    -- Aggregate all entries for this agent
    SELECT jsonb_agg(
        jsonb_build_object(
            'entry_id', entry_id,
            'type', transaction_type,
            'payload', payload,
            'day', simulation_day,
            'timestamp', created_at
        ) ORDER BY block_number
    )
    INTO v_state
    FROM ledger_entries
    WHERE created_by = p_agent_id
      AND simulation_day >= p_from_day;

    RETURN COALESCE(v_state, '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS FOR COST ANALYSIS
-- =============================================================================

-- Total blockchain costs by day
CREATE VIEW ledger_costs_by_day AS
SELECT
    simulation_day,
    COUNT(*) as entries,
    SUM(payload_size_bytes) as total_bytes,
    SUM(estimated_sui_gas) as total_gas_sui,
    SUM(estimated_walrus_cost) as total_walrus_sui,
    SUM(total_cost_sui) as total_cost_sui,
    SUM(total_cost_usd) as total_cost_usd
FROM ledger_entries
GROUP BY simulation_day
ORDER BY simulation_day;

-- Costs by transaction type
CREATE VIEW ledger_costs_by_type AS
SELECT
    transaction_type,
    COUNT(*) as entries,
    AVG(payload_size_bytes) as avg_bytes,
    SUM(total_cost_sui) as total_cost_sui,
    SUM(total_cost_usd) as total_cost_usd,
    AVG(total_cost_usd) as avg_cost_per_entry
FROM ledger_entries
GROUP BY transaction_type;

-- Cost per 1000 impressions (key metric)
CREATE VIEW ledger_cost_per_1k_impressions AS
SELECT
    ld.deal_id,
    d.impressions,
    SUM(le.total_cost_usd) as total_ledger_cost_usd,
    CASE
        WHEN d.impressions > 0
        THEN (SUM(le.total_cost_usd) / d.impressions) * 1000
        ELSE 0
    END as cost_per_1k_impressions
FROM ledger_deals ld
JOIN deals d ON ld.deal_id = d.deal_id
JOIN ledger_entries le ON ld.entry_id = le.entry_id
GROUP BY ld.deal_id, d.impressions;

-- Comparison: Blockchain costs vs Exchange fees
CREATE VIEW blockchain_vs_exchange_costs AS
SELECT
    'Scenario A (Exchange)' as scenario,
    SUM(exchange_fee) as total_intermediary_cost,
    SUM(buyer_spend) as total_spend,
    (SUM(exchange_fee) / NULLIF(SUM(buyer_spend), 0)) * 100 as cost_pct
FROM transactions
WHERE scenario = 'A'
UNION ALL
SELECT
    'Scenario C (Blockchain)' as scenario,
    SUM(total_cost_usd) as total_intermediary_cost,
    (SELECT SUM(buyer_spend) FROM transactions WHERE scenario = 'C') as total_spend,
    CASE
        WHEN (SELECT SUM(buyer_spend) FROM transactions WHERE scenario = 'C') > 0
        THEN (SUM(total_cost_usd) / (SELECT SUM(buyer_spend) FROM transactions WHERE scenario = 'C')) * 100
        ELSE 0
    END as cost_pct
FROM ledger_entries;

-- Recovery success rate
CREATE VIEW ledger_recovery_stats AS
SELECT
    recovery_reason,
    COUNT(*) as total_recoveries,
    SUM(CASE WHEN recovery_complete THEN 1 ELSE 0 END) as successful,
    AVG(state_accuracy) as avg_accuracy,
    AVG(recovery_time_ms) as avg_recovery_ms,
    SUM(bytes_recovered) as total_bytes_recovered
FROM ledger_recovery_log
GROUP BY recovery_reason;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE ledger_entries IS 'Immutable ledger entries simulating Sui blockchain with Walrus storage';
COMMENT ON TABLE ledger_deals IS 'Deal records stored on ledger for audit trail';
COMMENT ON TABLE ledger_deliveries IS 'Impression delivery proofs on ledger';
COMMENT ON TABLE ledger_settlements IS 'Financial settlement records on ledger';
COMMENT ON TABLE ledger_recovery_log IS 'Log of agent state recoveries from ledger';
COMMENT ON TABLE gas_parameters IS 'Configurable gas and storage cost parameters';
COMMENT ON TABLE blockchain_state IS 'Current state of simulated blockchain';

COMMENT ON FUNCTION create_ledger_entry IS 'Create immutable ledger entry with automatic cost calculation';
COMMENT ON FUNCTION recover_agent_state IS 'Recover agent state from ledger entries';

COMMENT ON VIEW blockchain_vs_exchange_costs IS 'Compare blockchain costs (Scenario C) vs exchange fees (Scenario A)';
COMMENT ON VIEW ledger_cost_per_1k_impressions IS 'Calculate blockchain cost per 1000 impressions';
