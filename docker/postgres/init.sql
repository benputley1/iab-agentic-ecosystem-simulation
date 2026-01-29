-- init.sql
-- Base schema for RTB simulation

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enums
CREATE TYPE scenario_type AS ENUM ('A', 'B', 'C');
CREATE TYPE kpi_type AS ENUM ('impressions', 'reach', 'clicks', 'conversions', 'viewability');
CREATE TYPE deal_type AS ENUM ('PG', 'PD', 'PA');
CREATE TYPE campaign_status AS ENUM ('active', 'paused', 'completed', 'cancelled');

-- Publishers (sellers)
CREATE TABLE publishers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    channels TEXT[] NOT NULL,
    floor_cpm DECIMAL(10, 2) NOT NULL,
    daily_avails BIGINT NOT NULL,
    audience_segments TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Buyers
CREATE TABLE buyers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    total_budget DECIMAL(15, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Campaigns
CREATE TABLE campaigns (
    id VARCHAR(50) PRIMARY KEY,
    buyer_id VARCHAR(50) NOT NULL REFERENCES buyers(id),
    name VARCHAR(255) NOT NULL,
    total_budget DECIMAL(15, 2) NOT NULL,
    daily_budget DECIMAL(15, 2) NOT NULL,
    primary_kpi kpi_type NOT NULL,
    target_impressions BIGINT NOT NULL,
    target_cpm DECIMAL(10, 2) NOT NULL,
    target_reach BIGINT,
    channels TEXT[] NOT NULL,
    publishers TEXT[],
    audience_segments TEXT[],
    geo_targets TEXT[],
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    spent DECIMAL(15, 2) DEFAULT 0,
    impressions_delivered BIGINT DEFAULT 0,
    status campaign_status DEFAULT 'active',
    scenario scenario_type NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Deals
CREATE TABLE deals (
    id VARCHAR(50) PRIMARY KEY,
    campaign_id VARCHAR(50) NOT NULL REFERENCES campaigns(id),
    seller_id VARCHAR(50) NOT NULL REFERENCES publishers(id),
    deal_type deal_type NOT NULL,
    impressions BIGINT NOT NULL,
    cpm DECIMAL(10, 2) NOT NULL,
    total_cost DECIMAL(15, 2) NOT NULL,
    exchange_fee DECIMAL(15, 2) DEFAULT 0,
    seller_revenue DECIMAL(15, 2) NOT NULL,
    scenario scenario_type NOT NULL,
    negotiation_rounds INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bid requests
CREATE TABLE bid_requests (
    id VARCHAR(50) PRIMARY KEY,
    buyer_id VARCHAR(50) NOT NULL REFERENCES buyers(id),
    campaign_id VARCHAR(50) NOT NULL REFERENCES campaigns(id),
    channel VARCHAR(50) NOT NULL,
    impressions_requested BIGINT NOT NULL,
    max_cpm DECIMAL(10, 2) NOT NULL,
    targeting JSONB,
    scenario scenario_type NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bid responses
CREATE TABLE bid_responses (
    id VARCHAR(50) PRIMARY KEY,
    request_id VARCHAR(50) NOT NULL REFERENCES bid_requests(id),
    seller_id VARCHAR(50) NOT NULL REFERENCES publishers(id),
    offered_cpm DECIMAL(10, 2) NOT NULL,
    available_impressions BIGINT NOT NULL,
    deal_type deal_type NOT NULL,
    accepted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions (completed exchanges)
CREATE TABLE transactions (
    id VARCHAR(50) PRIMARY KEY,
    deal_id VARCHAR(50) NOT NULL REFERENCES deals(id),
    buyer_id VARCHAR(50) NOT NULL REFERENCES buyers(id),
    seller_id VARCHAR(50) NOT NULL REFERENCES publishers(id),
    buyer_spend DECIMAL(15, 2) NOT NULL,
    seller_revenue DECIMAL(15, 2) NOT NULL,
    exchange_fee DECIMAL(15, 2) DEFAULT 0,
    impressions BIGINT NOT NULL,
    scenario scenario_type NOT NULL,
    simulation_day INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent decisions (for hallucination tracking)
CREATE TABLE agent_decisions (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    decision_type VARCHAR(100) NOT NULL,
    claimed_fact_id VARCHAR(50),
    decision_basis JSONB,
    decision_basis_verified BOOLEAN,
    scenario scenario_type NOT NULL,
    simulation_day INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily metrics
CREATE TABLE daily_metrics (
    id SERIAL PRIMARY KEY,
    scenario scenario_type NOT NULL,
    simulation_day INT NOT NULL,
    total_spend DECIMAL(15, 2) DEFAULT 0,
    total_impressions BIGINT DEFAULT 0,
    total_deals INT DEFAULT 0,
    avg_cpm DECIMAL(10, 2),
    avg_goal_attainment DECIMAL(5, 2),
    context_losses INT DEFAULT 0,
    state_recovery_accuracy DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scenario, simulation_day)
);

-- Indexes for common queries
CREATE INDEX idx_campaigns_buyer ON campaigns(buyer_id);
CREATE INDEX idx_campaigns_scenario ON campaigns(scenario);
CREATE INDEX idx_deals_campaign ON deals(campaign_id);
CREATE INDEX idx_deals_seller ON deals(seller_id);
CREATE INDEX idx_deals_scenario ON deals(scenario);
CREATE INDEX idx_transactions_scenario ON transactions(scenario);
CREATE INDEX idx_transactions_day ON transactions(simulation_day);
CREATE INDEX idx_agent_decisions_agent ON agent_decisions(agent_id);
CREATE INDEX idx_agent_decisions_verified ON agent_decisions(decision_basis_verified);

-- Updated at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_publishers_updated_at BEFORE UPDATE ON publishers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_buyers_updated_at BEFORE UPDATE ON buyers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
