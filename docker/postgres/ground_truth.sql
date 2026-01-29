-- ground_truth.sql
-- Reality database that agents CANNOT read
-- Used for post-hoc verification of agent claims and hallucination detection
-- Executed after init.sql

-- =============================================================================
-- INVENTORY REALITY
-- What inventory actually exists (agents see estimates, not truth)
-- =============================================================================

CREATE TABLE inventory_reality (
    id SERIAL PRIMARY KEY,
    publisher_id VARCHAR(50) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    date DATE NOT NULL,

    -- True inventory levels
    actual_avails BIGINT NOT NULL,        -- Real available impressions
    actual_fill_rate DECIMAL(5,4),        -- Real historical fill rate
    actual_viewability DECIMAL(5,4),      -- Real viewability rate
    actual_ctr DECIMAL(7,6),              -- Real click-through rate

    -- Price floors (what sellers actually accept)
    true_floor_cpm DECIMAL(10,2) NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(publisher_id, channel, date)
);

CREATE INDEX idx_inventory_reality_publisher ON inventory_reality(publisher_id);
CREATE INDEX idx_inventory_reality_date ON inventory_reality(date);
CREATE INDEX idx_inventory_reality_channel ON inventory_reality(channel);

-- =============================================================================
-- CAMPAIGN DELIVERY REALITY
-- What campaigns actually delivered (vs what agents report)
-- =============================================================================

CREATE TABLE campaign_delivery_reality (
    id SERIAL PRIMARY KEY,
    campaign_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,

    -- True delivery metrics
    actual_impressions BIGINT NOT NULL,
    actual_clicks BIGINT NOT NULL,
    actual_conversions BIGINT NOT NULL,
    actual_spend DECIMAL(12,2) NOT NULL,

    -- True performance metrics
    actual_viewability DECIMAL(5,4),
    actual_ctr DECIMAL(7,6),
    actual_conversion_rate DECIMAL(7,6),

    -- Derived metrics
    actual_cpm DECIMAL(10,2) GENERATED ALWAYS AS (
        CASE WHEN actual_impressions > 0
        THEN (actual_spend / actual_impressions) * 1000
        ELSE 0 END
    ) STORED,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(campaign_id, date)
);

CREATE INDEX idx_delivery_reality_campaign ON campaign_delivery_reality(campaign_id);
CREATE INDEX idx_delivery_reality_date ON campaign_delivery_reality(date);

-- =============================================================================
-- AGENT CLAIMS
-- Record what agents claim vs reality for hallucination detection
-- =============================================================================

CREATE TABLE agent_claims (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,      -- 'buyer', 'seller', 'exchange'
    scenario scenario_type NOT NULL,

    -- Claim details
    claim_type VARCHAR(100) NOT NULL,     -- e.g., 'inventory_level', 'delivery_count', 'price_quote'
    claim_context JSONB,                  -- Additional context about the claim
    claimed_value TEXT NOT NULL,          -- What the agent claimed
    actual_value TEXT NOT NULL,           -- Ground truth value

    -- Hallucination determination
    is_hallucination BOOLEAN NOT NULL,
    discrepancy_pct DECIMAL(7,4),         -- How far off (if numeric)
    severity VARCHAR(20),                 -- 'minor', 'moderate', 'severe'

    -- Impact tracking
    decision_id VARCHAR(50),              -- Link to the decision this affected
    impact_description TEXT,              -- What happened as a result

    -- Timing
    simulation_day INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_claims_agent ON agent_claims(agent_id);
CREATE INDEX idx_agent_claims_scenario ON agent_claims(scenario);
CREATE INDEX idx_agent_claims_type ON agent_claims(claim_type);
CREATE INDEX idx_agent_claims_hallucination ON agent_claims(is_hallucination);
CREATE INDEX idx_agent_claims_day ON agent_claims(simulation_day);
CREATE INDEX idx_agent_claims_severity ON agent_claims(severity);

-- =============================================================================
-- AGENT DECISIONS
-- Track all agent decisions for hallucination analysis
-- =============================================================================

CREATE TABLE agent_decisions (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'dec-' || uuid_generate_v4()::text,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    scenario scenario_type NOT NULL,

    -- Decision details
    decision_type VARCHAR(100) NOT NULL,  -- 'bid', 'accept', 'reject', 'counter', 'allocate'
    decision_input JSONB NOT NULL,        -- What data the agent used
    decision_output JSONB NOT NULL,       -- What the agent decided
    decision_reasoning TEXT,              -- Agent's stated reasoning (if available)

    -- Verification
    claimed_fact_id VARCHAR(50),          -- FK to a specific claim
    decision_basis_verified BOOLEAN,      -- Was the basis factually correct?

    -- Timing
    simulation_day INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_decisions_agent ON agent_decisions(agent_id);
CREATE INDEX idx_agent_decisions_scenario ON agent_decisions(scenario);
CREATE INDEX idx_agent_decisions_type ON agent_decisions(decision_type);
CREATE INDEX idx_agent_decisions_verified ON agent_decisions(decision_basis_verified);
CREATE INDEX idx_agent_decisions_day ON agent_decisions(simulation_day);

-- =============================================================================
-- CONTEXT ROT EVENTS
-- Track when agents lose context and what happens
-- =============================================================================

CREATE TABLE context_rot_events (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    scenario scenario_type NOT NULL,

    -- Event details
    event_type VARCHAR(50) NOT NULL,      -- 'decay', 'restart', 'memory_pressure'
    keys_lost TEXT[],                     -- Which memory keys were lost
    keys_corrupted TEXT[],                -- Which keys had corrupted data

    -- Recovery attempt
    recovery_attempted BOOLEAN DEFAULT FALSE,
    recovery_successful BOOLEAN,
    recovery_accuracy DECIMAL(5,4),       -- How accurate was the recovered state
    recovery_source VARCHAR(50),          -- 'ledger', 'checkpoint', 'peer', 'none'

    -- Impact
    impact_description TEXT,
    decisions_affected INTEGER DEFAULT 0,

    -- Timing
    simulation_day INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_context_rot_agent ON context_rot_events(agent_id);
CREATE INDEX idx_context_rot_scenario ON context_rot_events(scenario);
CREATE INDEX idx_context_rot_type ON context_rot_events(event_type);
CREATE INDEX idx_context_rot_day ON context_rot_events(simulation_day);

-- =============================================================================
-- HALLUCINATION INJECTION LOG
-- Track injected hallucinations (Scenario B) for validation
-- =============================================================================

CREATE TABLE hallucination_injections (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    scenario scenario_type NOT NULL,

    -- Injection details
    injection_type VARCHAR(50) NOT NULL,  -- 'inventory', 'price', 'history'
    original_value TEXT NOT NULL,
    injected_value TEXT NOT NULL,
    injection_factor DECIMAL(5,4),        -- e.g., 1.3 for 30% inflation

    -- Detection tracking
    was_detected BOOLEAN DEFAULT FALSE,
    detection_delay_seconds INTEGER,      -- How long until detected
    detected_by VARCHAR(50),              -- Which mechanism detected it

    -- Timing
    simulation_day INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_hallucination_injections_agent ON hallucination_injections(agent_id);
CREATE INDEX idx_hallucination_injections_scenario ON hallucination_injections(scenario);
CREATE INDEX idx_hallucination_injections_type ON hallucination_injections(injection_type);
CREATE INDEX idx_hallucination_injections_detected ON hallucination_injections(was_detected);

-- =============================================================================
-- FACT REGISTRY
-- Source of truth for all verifiable facts in the simulation
-- =============================================================================

CREATE TABLE fact_registry (
    fact_id VARCHAR(50) PRIMARY KEY DEFAULT 'fact-' || uuid_generate_v4()::text,
    fact_type VARCHAR(100) NOT NULL,      -- 'inventory', 'price', 'delivery', 'transaction'
    entity_id VARCHAR(50) NOT NULL,       -- ID of related entity
    entity_type VARCHAR(50) NOT NULL,     -- 'publisher', 'campaign', 'deal'

    -- Fact details
    fact_key VARCHAR(100) NOT NULL,       -- e.g., 'avail_impressions', 'floor_cpm'
    fact_value TEXT NOT NULL,
    fact_value_numeric DECIMAL(20,6),     -- For numeric comparisons
    fact_unit VARCHAR(20),                -- e.g., 'impressions', 'USD'

    -- Validity window
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE,

    -- Metadata
    simulation_day INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_fact_registry_type ON fact_registry(fact_type);
CREATE INDEX idx_fact_registry_entity ON fact_registry(entity_id, entity_type);
CREATE INDEX idx_fact_registry_key ON fact_registry(fact_key);
CREATE INDEX idx_fact_registry_validity ON fact_registry(valid_from, valid_until);

-- =============================================================================
-- VIEWS FOR HALLUCINATION ANALYSIS
-- =============================================================================

-- Hallucination rate by scenario and agent type
CREATE VIEW hallucination_rates AS
SELECT
    scenario,
    agent_type,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN decision_basis_verified = FALSE THEN 1 ELSE 0 END) as hallucinated_decisions,
    CASE
        WHEN COUNT(*) > 0
        THEN (SUM(CASE WHEN decision_basis_verified = FALSE THEN 1 ELSE 0 END)::float / COUNT(*)) * 100
        ELSE 0
    END as hallucination_rate
FROM agent_decisions
GROUP BY scenario, agent_type;

-- Hallucination severity distribution
CREATE VIEW hallucination_severity_dist AS
SELECT
    scenario,
    severity,
    COUNT(*) as count,
    AVG(discrepancy_pct) as avg_discrepancy
FROM agent_claims
WHERE is_hallucination = TRUE
GROUP BY scenario, severity;

-- Context rot impact by day
CREATE VIEW context_rot_impact AS
SELECT
    scenario,
    simulation_day,
    COUNT(*) as rot_events,
    SUM(array_length(keys_lost, 1)) as total_keys_lost,
    AVG(recovery_accuracy) as avg_recovery_accuracy,
    SUM(decisions_affected) as total_decisions_affected
FROM context_rot_events
GROUP BY scenario, simulation_day
ORDER BY scenario, simulation_day;

-- Injection detection rate
CREATE VIEW injection_detection_rates AS
SELECT
    scenario,
    injection_type,
    COUNT(*) as total_injections,
    SUM(CASE WHEN was_detected THEN 1 ELSE 0 END) as detected,
    CASE
        WHEN COUNT(*) > 0
        THEN (SUM(CASE WHEN was_detected THEN 1 ELSE 0 END)::float / COUNT(*)) * 100
        ELSE 0
    END as detection_rate_pct,
    AVG(detection_delay_seconds) as avg_detection_delay
FROM hallucination_injections
GROUP BY scenario, injection_type;

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to verify a claim against ground truth
CREATE OR REPLACE FUNCTION verify_claim(
    p_claim_type VARCHAR,
    p_entity_id VARCHAR,
    p_claimed_value TEXT,
    p_simulation_day INTEGER
) RETURNS TABLE (
    is_valid BOOLEAN,
    actual_value TEXT,
    discrepancy_pct DECIMAL
) AS $$
DECLARE
    v_actual TEXT;
    v_claimed_numeric DECIMAL;
    v_actual_numeric DECIMAL;
BEGIN
    -- Look up the fact
    SELECT fact_value, fact_value_numeric
    INTO v_actual, v_actual_numeric
    FROM fact_registry
    WHERE entity_id = p_entity_id
      AND fact_key = p_claim_type
      AND simulation_day = p_simulation_day
    LIMIT 1;

    IF v_actual IS NULL THEN
        RETURN QUERY SELECT NULL::BOOLEAN, NULL::TEXT, NULL::DECIMAL;
        RETURN;
    END IF;

    -- Try numeric comparison
    BEGIN
        v_claimed_numeric := p_claimed_value::DECIMAL;
        IF v_actual_numeric IS NOT NULL AND v_actual_numeric != 0 THEN
            RETURN QUERY SELECT
                ABS(v_claimed_numeric - v_actual_numeric) / v_actual_numeric < 0.01,
                v_actual,
                ABS(v_claimed_numeric - v_actual_numeric) / v_actual_numeric * 100;
            RETURN;
        END IF;
    EXCEPTION WHEN OTHERS THEN
        -- Not numeric, do string comparison
        NULL;
    END;

    -- String comparison
    RETURN QUERY SELECT
        p_claimed_value = v_actual,
        v_actual,
        NULL::DECIMAL;
END;
$$ LANGUAGE plpgsql;

-- Function to record an agent claim with automatic verification
CREATE OR REPLACE FUNCTION record_agent_claim(
    p_agent_id VARCHAR,
    p_agent_type VARCHAR,
    p_scenario scenario_type,
    p_claim_type VARCHAR,
    p_entity_id VARCHAR,
    p_claimed_value TEXT,
    p_simulation_day INTEGER
) RETURNS INTEGER AS $$
DECLARE
    v_is_valid BOOLEAN;
    v_actual TEXT;
    v_discrepancy DECIMAL;
    v_claim_id INTEGER;
    v_severity VARCHAR;
BEGIN
    -- Verify the claim
    SELECT * INTO v_is_valid, v_actual, v_discrepancy
    FROM verify_claim(p_claim_type, p_entity_id, p_claimed_value, p_simulation_day);

    -- Determine severity
    IF v_discrepancy IS NOT NULL THEN
        IF v_discrepancy < 5 THEN
            v_severity := 'minor';
        ELSIF v_discrepancy < 20 THEN
            v_severity := 'moderate';
        ELSE
            v_severity := 'severe';
        END IF;
    END IF;

    -- Insert the claim
    INSERT INTO agent_claims (
        agent_id, agent_type, scenario, claim_type, claimed_value,
        actual_value, is_hallucination, discrepancy_pct, severity, simulation_day
    ) VALUES (
        p_agent_id, p_agent_type, p_scenario, p_claim_type, p_claimed_value,
        COALESCE(v_actual, 'UNKNOWN'), COALESCE(NOT v_is_valid, TRUE),
        v_discrepancy, v_severity, p_simulation_day
    ) RETURNING id INTO v_claim_id;

    RETURN v_claim_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE inventory_reality IS 'Ground truth inventory levels - agents cannot read this';
COMMENT ON TABLE campaign_delivery_reality IS 'Actual campaign delivery metrics for verification';
COMMENT ON TABLE agent_claims IS 'Record of agent claims vs reality for hallucination detection';
COMMENT ON TABLE agent_decisions IS 'All agent decisions for post-hoc verification';
COMMENT ON TABLE context_rot_events IS 'Context loss events for Scenario B analysis';
COMMENT ON TABLE hallucination_injections IS 'Log of intentionally injected hallucinations';
COMMENT ON TABLE fact_registry IS 'Authoritative source of truth for all facts';

COMMENT ON VIEW hallucination_rates IS 'Hallucination detection rate by scenario and agent type';
COMMENT ON VIEW context_rot_impact IS 'Impact of context rot over simulation days';
