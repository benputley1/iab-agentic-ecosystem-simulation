-- ground_truth.sql
-- This database maintains "reality" that agents cannot read directly
-- Used for post-hoc verification of agent claims

-- Actual inventory levels (what publishers really have)
CREATE TABLE inventory_reality (
    id SERIAL PRIMARY KEY,
    publisher_id VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    date DATE NOT NULL,

    actual_avails BIGINT NOT NULL,  -- Real available impressions
    actual_fill_rate DECIMAL(5, 4),
    actual_floor_cpm DECIMAL(10, 2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(publisher_id, channel, date)
);

-- Actual campaign delivery (what really happened)
CREATE TABLE campaign_delivery_reality (
    id SERIAL PRIMARY KEY,
    campaign_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,

    actual_impressions BIGINT NOT NULL,
    actual_clicks BIGINT DEFAULT 0,
    actual_conversions BIGINT DEFAULT 0,
    actual_spend DECIMAL(15, 2) NOT NULL,
    actual_cpm DECIMAL(10, 2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(campaign_id, date)
);

-- Agent claims vs reality (for hallucination detection)
CREATE TABLE agent_claims (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(20) NOT NULL,

    claim_type VARCHAR(50) NOT NULL,  -- "inventory_level", "delivery_count", "price_quote", etc.
    claimed_value TEXT NOT NULL,
    actual_value TEXT,  -- From ground truth

    is_hallucination BOOLEAN,
    discrepancy_pct DECIMAL(10, 2),  -- How far off was the claim

    scenario CHAR(1) NOT NULL,
    simulation_day INT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Price history (actual market rates)
CREATE TABLE price_history_reality (
    id SERIAL PRIMARY KEY,
    seller_id VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    date DATE NOT NULL,

    actual_avg_cpm DECIMAL(10, 2) NOT NULL,
    actual_min_cpm DECIMAL(10, 2),
    actual_max_cpm DECIMAL(10, 2),
    transactions_count INT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(seller_id, channel, date)
);

-- View: Hallucination summary by agent type and scenario
CREATE VIEW hallucination_summary AS
SELECT
    scenario,
    agent_type,
    COUNT(*) as total_claims,
    SUM(CASE WHEN is_hallucination THEN 1 ELSE 0 END) as hallucinated_claims,
    ROUND(
        (SUM(CASE WHEN is_hallucination THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100,
        2
    ) as hallucination_rate_pct,
    AVG(CASE WHEN is_hallucination THEN discrepancy_pct ELSE NULL END) as avg_discrepancy
FROM agent_claims
GROUP BY scenario, agent_type;

-- View: Daily ground truth comparison
CREATE VIEW daily_truth_comparison AS
SELECT
    c.scenario,
    DATE(c.created_at) as date,
    SUM(c.impressions_delivered) as claimed_impressions,
    SUM(COALESCE(r.actual_impressions, 0)) as actual_impressions,
    SUM(c.spent) as claimed_spend,
    SUM(COALESCE(r.actual_spend, 0)) as actual_spend
FROM campaigns c
LEFT JOIN campaign_delivery_reality r
    ON c.id = r.campaign_id
    AND DATE(c.created_at) = r.date
GROUP BY c.scenario, DATE(c.created_at);

-- Indexes
CREATE INDEX idx_claims_scenario_day ON agent_claims(scenario, simulation_day);
CREATE INDEX idx_claims_hallucination ON agent_claims(is_hallucination);
CREATE INDEX idx_inventory_reality_date ON inventory_reality(date);
CREATE INDEX idx_delivery_reality_campaign ON campaign_delivery_reality(campaign_id);
