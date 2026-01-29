-- ground_truth.sql
-- This database maintains "reality" that agents cannot read
-- Used for post-hoc verification of agent claims

-- Actual inventory levels (agents see estimated/cached versions)
CREATE TABLE inventory_reality (
    publisher_id VARCHAR(50) NOT NULL REFERENCES publishers(id),
    channel VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    actual_avails BIGINT NOT NULL,
    actual_fill_rate DECIMAL(5, 4) NOT NULL,
    actual_floor_cpm DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (publisher_id, channel, date)
);

-- Actual campaign delivery (vs what agents report)
CREATE TABLE campaign_delivery_reality (
    campaign_id VARCHAR(50) NOT NULL REFERENCES campaigns(id),
    date DATE NOT NULL,
    actual_impressions BIGINT NOT NULL,
    actual_clicks BIGINT NOT NULL,
    actual_conversions BIGINT NOT NULL,
    actual_spend DECIMAL(15, 2) NOT NULL,
    actual_cpm DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (campaign_id, date)
);

-- Agent claims vs reality (for hallucination detection)
CREATE TABLE agent_claims (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    claim_type VARCHAR(100) NOT NULL,
    claimed_value TEXT NOT NULL,
    actual_value TEXT,
    is_hallucination BOOLEAN NOT NULL DEFAULT FALSE,
    hallucination_source VARCHAR(100),
    scenario scenario_type NOT NULL,
    simulation_day INT NOT NULL,
    impact_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Context rot events
CREATE TABLE context_rot_events (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    scenario scenario_type NOT NULL,
    simulation_day INT NOT NULL,
    keys_lost TEXT[] NOT NULL,
    recovery_attempted BOOLEAN DEFAULT FALSE,
    recovery_successful BOOLEAN,
    recovery_accuracy DECIMAL(5, 4),
    impact_on_campaign VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent restart events (full context wipe)
CREATE TABLE agent_restarts (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    scenario scenario_type NOT NULL,
    simulation_day INT NOT NULL,
    reason VARCHAR(100),
    state_before_restart JSONB,
    state_after_restart JSONB,
    recovery_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fact registry (ground truth facts that agents may claim knowledge of)
CREATE TABLE facts (
    id VARCHAR(50) PRIMARY KEY,
    fact_type VARCHAR(100) NOT NULL,
    fact_value JSONB NOT NULL,
    valid_from TIMESTAMP NOT NULL,
    valid_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for analysis queries
CREATE INDEX idx_agent_claims_hallucination ON agent_claims(is_hallucination);
CREATE INDEX idx_agent_claims_scenario ON agent_claims(scenario);
CREATE INDEX idx_agent_claims_day ON agent_claims(simulation_day);
CREATE INDEX idx_context_rot_scenario ON context_rot_events(scenario);
CREATE INDEX idx_context_rot_day ON context_rot_events(simulation_day);
CREATE INDEX idx_inventory_date ON inventory_reality(date);
CREATE INDEX idx_delivery_date ON campaign_delivery_reality(date);

-- View for hallucination rates by scenario
CREATE VIEW hallucination_rates AS
SELECT
    scenario,
    agent_type,
    COUNT(*) as total_claims,
    SUM(CASE WHEN is_hallucination THEN 1 ELSE 0 END) as hallucinated_claims,
    (SUM(CASE WHEN is_hallucination THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100 as hallucination_rate
FROM agent_claims
GROUP BY scenario, agent_type;

-- View for context rot impact over time
CREATE VIEW context_rot_impact AS
SELECT
    scenario,
    simulation_day,
    COUNT(*) as rot_events,
    AVG(recovery_accuracy) as avg_recovery_accuracy,
    SUM(CASE WHEN recovery_successful THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) * 100 as recovery_success_rate
FROM context_rot_events
GROUP BY scenario, simulation_day
ORDER BY scenario, simulation_day;
