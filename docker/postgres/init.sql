-- init.sql
-- Base schema for RTB Simulation

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Buyers (advertisers)
CREATE TABLE buyers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    total_budget DECIMAL(15, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sellers (publishers)
CREATE TABLE sellers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    channels TEXT[] NOT NULL,
    floor_cpm DECIMAL(10, 2) NOT NULL,
    daily_avails BIGINT NOT NULL,
    audience_segments TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Campaigns
CREATE TABLE campaigns (
    id VARCHAR(50) PRIMARY KEY,
    buyer_id VARCHAR(50) REFERENCES buyers(id),
    name VARCHAR(255) NOT NULL,

    -- Budget
    total_budget DECIMAL(15, 2) NOT NULL,
    daily_budget DECIMAL(15, 2) NOT NULL,

    -- Goals
    primary_kpi VARCHAR(50) NOT NULL,
    target_impressions BIGINT NOT NULL,
    target_cpm DECIMAL(10, 2) NOT NULL,
    target_reach BIGINT,

    -- Targeting
    channels TEXT[] NOT NULL,
    publishers TEXT[],
    audience_segments TEXT[],
    geo_targets TEXT[],

    -- Timing
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,

    -- State
    spent DECIMAL(15, 2) DEFAULT 0.0,
    impressions_delivered BIGINT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',

    -- Scenario tracking
    scenario CHAR(1) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Deals (negotiated agreements)
CREATE TABLE deals (
    id VARCHAR(50) PRIMARY KEY,
    campaign_id VARCHAR(50) REFERENCES campaigns(id),
    buyer_id VARCHAR(50) REFERENCES buyers(id),
    seller_id VARCHAR(50) REFERENCES sellers(id),

    -- Deal terms
    deal_type VARCHAR(10) NOT NULL, -- PG, PD, PA
    impressions BIGINT NOT NULL,
    cpm DECIMAL(10, 2) NOT NULL,
    total_cost DECIMAL(15, 2) NOT NULL,

    -- Fees
    exchange_fee DECIMAL(15, 2) DEFAULT 0,
    seller_revenue DECIMAL(15, 2) NOT NULL,

    -- Status
    status VARCHAR(20) DEFAULT 'pending',
    negotiation_rounds INT DEFAULT 0,

    -- Scenario
    scenario CHAR(1) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions (actual deliveries)
CREATE TABLE transactions (
    id VARCHAR(50) PRIMARY KEY,
    deal_id VARCHAR(50) REFERENCES deals(id),

    -- Amounts
    buyer_spend DECIMAL(15, 2) NOT NULL,
    seller_revenue DECIMAL(15, 2) NOT NULL,
    exchange_fee DECIMAL(15, 2) DEFAULT 0,

    -- Delivery
    impressions_delivered BIGINT NOT NULL,

    -- Scenario
    scenario CHAR(1) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bid requests
CREATE TABLE bid_requests (
    id VARCHAR(50) PRIMARY KEY,
    buyer_id VARCHAR(50) REFERENCES buyers(id),
    campaign_id VARCHAR(50) REFERENCES campaigns(id),

    channel VARCHAR(20) NOT NULL,
    impressions_requested BIGINT NOT NULL,
    max_cpm DECIMAL(10, 2) NOT NULL,
    targeting JSONB,

    scenario CHAR(1) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bid responses
CREATE TABLE bid_responses (
    id VARCHAR(50) PRIMARY KEY,
    request_id VARCHAR(50) REFERENCES bid_requests(id),
    seller_id VARCHAR(50) REFERENCES sellers(id),

    offered_cpm DECIMAL(10, 2) NOT NULL,
    available_impressions BIGINT NOT NULL,
    deal_type VARCHAR(10) NOT NULL,

    scenario CHAR(1) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent decisions (for hallucination tracking)
CREATE TABLE agent_decisions (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(20) NOT NULL,

    decision_type VARCHAR(50) NOT NULL,
    decision_basis TEXT NOT NULL,
    claimed_fact_id VARCHAR(50),
    decision_basis_verified BOOLEAN,

    scenario CHAR(1) NOT NULL,
    simulation_day INT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily metrics
CREATE TABLE daily_metrics (
    id SERIAL PRIMARY KEY,
    scenario CHAR(1) NOT NULL,
    simulation_day INT NOT NULL,

    -- Performance
    avg_goal_attainment DECIMAL(5, 2),
    total_deals INT,
    total_spend DECIMAL(15, 2),
    total_impressions BIGINT,

    -- Health
    context_loss_events INT DEFAULT 0,
    state_recovery_accuracy DECIMAL(5, 2),
    agent_restart_events INT DEFAULT 0,
    hallucination_count INT DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(scenario, simulation_day)
);

-- Indexes for performance
CREATE INDEX idx_campaigns_buyer ON campaigns(buyer_id);
CREATE INDEX idx_campaigns_scenario ON campaigns(scenario);
CREATE INDEX idx_deals_campaign ON deals(campaign_id);
CREATE INDEX idx_deals_scenario ON deals(scenario);
CREATE INDEX idx_transactions_scenario ON transactions(scenario);
CREATE INDEX idx_agent_decisions_scenario_day ON agent_decisions(scenario, simulation_day);
