-- init.sql
-- Base schema for IAB Agentic RTB Simulation
-- This file is executed first when PostgreSQL container starts

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- ENUMS
-- =============================================================================

CREATE TYPE kpi_type AS ENUM (
    'impressions',
    'reach',
    'clicks',
    'conversions',
    'viewability'
);

CREATE TYPE deal_type AS ENUM (
    'PG',   -- Programmatic Guaranteed
    'PD',   -- Private Deals
    'PA'    -- Programmatic Auction
);

CREATE TYPE scenario_type AS ENUM (
    'A',    -- Current State (rent-seeking exchanges)
    'B',    -- IAB Pure A2A (direct buyer-seller)
    'C'     -- Alkimi Ledger-backed
);

CREATE TYPE campaign_status AS ENUM (
    'draft',
    'active',
    'paused',
    'completed',
    'cancelled'
);

CREATE TYPE bid_status AS ENUM (
    'pending',
    'accepted',
    'rejected',
    'countered',
    'expired'
);

CREATE TYPE deal_status AS ENUM (
    'proposed',
    'negotiating',
    'confirmed',
    'delivering',
    'completed',
    'cancelled'
);

-- =============================================================================
-- BUYERS
-- =============================================================================

CREATE TABLE buyers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(20) DEFAULT 'standard',  -- For identity-based pricing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_buyers_tier ON buyers(tier);

-- =============================================================================
-- SELLERS (Publishers)
-- =============================================================================

CREATE TABLE sellers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    channels TEXT[] NOT NULL,             -- e.g., ['display', 'video', 'ctv']
    floor_cpm DECIMAL(10,2) NOT NULL,
    daily_avails BIGINT NOT NULL,
    audience_segments TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sellers_channels ON sellers USING GIN(channels);
CREATE INDEX idx_sellers_floor_cpm ON sellers(floor_cpm);

-- =============================================================================
-- CAMPAIGNS
-- =============================================================================

CREATE TABLE campaigns (
    id VARCHAR(50) PRIMARY KEY,
    buyer_id VARCHAR(50) NOT NULL REFERENCES buyers(id),
    name VARCHAR(255) NOT NULL,

    -- Budget
    total_budget DECIMAL(12,2) NOT NULL,
    daily_budget DECIMAL(12,2) NOT NULL,

    -- Goals
    primary_kpi kpi_type NOT NULL,
    target_impressions BIGINT NOT NULL,
    target_cpm DECIMAL(10,2) NOT NULL,
    target_reach BIGINT,

    -- Targeting
    channels TEXT[] NOT NULL,
    publishers TEXT[],                    -- Preferred publisher IDs
    audience_segments TEXT[],
    geo_targets TEXT[],

    -- Timing
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date TIMESTAMP WITH TIME ZONE NOT NULL,

    -- State
    spent DECIMAL(12,2) DEFAULT 0.0,
    impressions_delivered BIGINT DEFAULT 0,
    clicks BIGINT DEFAULT 0,
    conversions BIGINT DEFAULT 0,
    status campaign_status DEFAULT 'draft',

    -- Scenario tracking
    scenario scenario_type NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_campaigns_buyer ON campaigns(buyer_id);
CREATE INDEX idx_campaigns_scenario ON campaigns(scenario);
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaigns_dates ON campaigns(start_date, end_date);
CREATE INDEX idx_campaigns_channels ON campaigns USING GIN(channels);

-- =============================================================================
-- BID REQUESTS
-- =============================================================================

CREATE TABLE bid_requests (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'req-' || uuid_generate_v4()::text,
    buyer_id VARCHAR(50) NOT NULL REFERENCES buyers(id),
    campaign_id VARCHAR(50) NOT NULL REFERENCES campaigns(id),
    channel VARCHAR(50) NOT NULL,
    impressions_requested BIGINT NOT NULL,
    max_cpm DECIMAL(10,2) NOT NULL,
    targeting JSONB,
    scenario scenario_type NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_bid_requests_campaign ON bid_requests(campaign_id);
CREATE INDEX idx_bid_requests_buyer ON bid_requests(buyer_id);
CREATE INDEX idx_bid_requests_scenario ON bid_requests(scenario);
CREATE INDEX idx_bid_requests_created ON bid_requests(created_at);

-- =============================================================================
-- BID RESPONSES
-- =============================================================================

CREATE TABLE bid_responses (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'resp-' || uuid_generate_v4()::text,
    request_id VARCHAR(50) NOT NULL REFERENCES bid_requests(id),
    seller_id VARCHAR(50) NOT NULL REFERENCES sellers(id),
    offered_cpm DECIMAL(10,2) NOT NULL,
    available_impressions BIGINT NOT NULL,
    deal_type deal_type NOT NULL,
    status bid_status DEFAULT 'pending',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_bid_responses_request ON bid_responses(request_id);
CREATE INDEX idx_bid_responses_seller ON bid_responses(seller_id);
CREATE INDEX idx_bid_responses_status ON bid_responses(status);

-- =============================================================================
-- DEALS
-- =============================================================================

CREATE TABLE deals (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'deal-' || uuid_generate_v4()::text,
    request_id VARCHAR(50) REFERENCES bid_requests(id),
    campaign_id VARCHAR(50) NOT NULL REFERENCES campaigns(id),
    buyer_id VARCHAR(50) NOT NULL REFERENCES buyers(id),
    seller_id VARCHAR(50) NOT NULL REFERENCES sellers(id),

    -- Deal terms
    impressions BIGINT NOT NULL,
    cpm DECIMAL(10,2) NOT NULL,
    total_cost DECIMAL(12,2) NOT NULL,
    deal_type deal_type NOT NULL,

    -- Status
    status deal_status DEFAULT 'proposed',
    negotiation_rounds INTEGER DEFAULT 0,

    -- Delivery tracking
    impressions_delivered BIGINT DEFAULT 0,

    -- Scenario tracking
    scenario scenario_type NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confirmed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_deals_campaign ON deals(campaign_id);
CREATE INDEX idx_deals_buyer ON deals(buyer_id);
CREATE INDEX idx_deals_seller ON deals(seller_id);
CREATE INDEX idx_deals_scenario ON deals(scenario);
CREATE INDEX idx_deals_status ON deals(status);
CREATE INDEX idx_deals_created ON deals(created_at);

-- =============================================================================
-- TRANSACTIONS (Financial Records)
-- =============================================================================

CREATE TABLE transactions (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'txn-' || uuid_generate_v4()::text,
    deal_id VARCHAR(50) NOT NULL REFERENCES deals(id),

    -- Financial breakdown
    buyer_spend DECIMAL(12,2) NOT NULL,
    seller_revenue DECIMAL(12,2) NOT NULL,
    exchange_fee DECIMAL(12,2) DEFAULT 0.0,  -- 0 for scenarios B, C

    -- Fee details (Scenario A)
    exchange_fee_pct DECIMAL(5,4) DEFAULT 0.0,

    -- Blockchain costs (Scenario C)
    sui_gas_cost DECIMAL(12,8) DEFAULT 0.0,
    walrus_storage_cost DECIMAL(12,8) DEFAULT 0.0,
    blockchain_cost_usd DECIMAL(12,6) DEFAULT 0.0,

    -- Scenario tracking
    scenario scenario_type NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_transactions_deal ON transactions(deal_id);
CREATE INDEX idx_transactions_scenario ON transactions(scenario);
CREATE INDEX idx_transactions_created ON transactions(created_at);

-- =============================================================================
-- AUCTION RESULTS (Scenario A - Second Price Auction)
-- =============================================================================

CREATE TABLE auction_results (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'auction-' || uuid_generate_v4()::text,
    request_id VARCHAR(50) NOT NULL REFERENCES bid_requests(id),

    -- Winner details
    winner_id VARCHAR(50) REFERENCES sellers(id),
    winning_bid DECIMAL(10,2),
    clearing_price DECIMAL(10,2),

    -- Auction stats
    bid_count INTEGER NOT NULL,
    valid_bid_count INTEGER NOT NULL,
    floor_price DECIMAL(10,2) NOT NULL,

    -- Exchange take
    exchange_fee DECIMAL(10,2) DEFAULT 0.0,
    seller_receives DECIMAL(10,2),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_auction_results_request ON auction_results(request_id);
CREATE INDEX idx_auction_results_winner ON auction_results(winner_id);

-- =============================================================================
-- DAILY METRICS (Aggregated per day per scenario)
-- =============================================================================

CREATE TABLE daily_metrics (
    id SERIAL PRIMARY KEY,
    scenario scenario_type NOT NULL,
    simulation_day INTEGER NOT NULL,

    -- Campaign metrics
    total_campaigns INTEGER DEFAULT 0,
    active_campaigns INTEGER DEFAULT 0,
    campaigns_hit_goal INTEGER DEFAULT 0,
    avg_goal_attainment DECIMAL(5,2) DEFAULT 0.0,

    -- Financial metrics
    total_spend DECIMAL(14,2) DEFAULT 0.0,
    total_seller_revenue DECIMAL(14,2) DEFAULT 0.0,
    total_exchange_fees DECIMAL(14,2) DEFAULT 0.0,
    total_blockchain_costs DECIMAL(14,6) DEFAULT 0.0,

    -- Volume metrics
    total_impressions BIGINT DEFAULT 0,
    total_deals INTEGER DEFAULT 0,
    total_bids INTEGER DEFAULT 0,

    -- Context rot metrics (Scenario B)
    agent_restart_events INTEGER DEFAULT 0,
    context_loss_events INTEGER DEFAULT 0,
    state_recovery_accuracy DECIMAL(5,4) DEFAULT 1.0,

    -- Hallucination metrics
    hallucination_events INTEGER DEFAULT 0,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(scenario, simulation_day)
);

CREATE INDEX idx_daily_metrics_scenario_day ON daily_metrics(scenario, simulation_day);

-- =============================================================================
-- AGENT STATE (For context rot simulation)
-- =============================================================================

CREATE TABLE agent_states (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,  -- 'buyer', 'seller', 'exchange'
    scenario scenario_type NOT NULL,

    -- State data
    state_data JSONB NOT NULL,
    state_hash VARCHAR(64) NOT NULL,  -- SHA256 for verification

    -- Context tracking
    context_size_bytes INTEGER,
    memory_keys_count INTEGER,

    -- Metadata
    simulation_day INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(agent_id, scenario, simulation_day)
);

CREATE INDEX idx_agent_states_agent ON agent_states(agent_id);
CREATE INDEX idx_agent_states_scenario_day ON agent_states(scenario, simulation_day);

-- =============================================================================
-- SIMULATION RUNS
-- =============================================================================

CREATE TABLE simulation_runs (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'sim-' || uuid_generate_v4()::text,

    -- Configuration
    scenarios scenario_type[] NOT NULL,
    duration_days INTEGER NOT NULL,
    buyer_count INTEGER NOT NULL,
    seller_count INTEGER NOT NULL,
    campaigns_per_buyer INTEGER NOT NULL,
    time_acceleration DECIMAL(10,2) NOT NULL,

    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed
    current_day INTEGER DEFAULT 0,

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    config JSONB
);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_buyers_updated_at
    BEFORE UPDATE ON buyers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sellers_updated_at
    BEFORE UPDATE ON sellers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_campaigns_updated_at
    BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_deals_updated_at
    BEFORE UPDATE ON deals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- VIEWS FOR ANALYSIS
-- =============================================================================

-- Fee extraction comparison view
CREATE VIEW fee_extraction_by_scenario AS
SELECT
    scenario,
    SUM(buyer_spend) as gross_spend,
    SUM(seller_revenue) as net_to_publisher,
    SUM(buyer_spend - seller_revenue) as intermediary_take,
    CASE
        WHEN SUM(buyer_spend) > 0
        THEN (SUM(buyer_spend - seller_revenue) / SUM(buyer_spend)) * 100
        ELSE 0
    END as take_rate_pct
FROM transactions
GROUP BY scenario;

-- Campaign goal achievement view
CREATE VIEW campaign_goal_achievement AS
SELECT
    scenario,
    COUNT(*) as total_campaigns,
    SUM(CASE WHEN impressions_delivered >= target_impressions THEN 1 ELSE 0 END) as hit_impression_goal,
    SUM(CASE WHEN spent / NULLIF(impressions_delivered, 0) * 1000 <= target_cpm THEN 1 ELSE 0 END) as hit_cpm_goal,
    AVG(LEAST(impressions_delivered::float / NULLIF(target_impressions, 0), 1.0) * 100) as avg_goal_attainment
FROM campaigns
WHERE status = 'completed'
GROUP BY scenario;

-- Daily performance trend view
CREATE VIEW daily_performance_trend AS
SELECT
    scenario,
    simulation_day,
    avg_goal_attainment as daily_goal_attainment,
    agent_restart_events + context_loss_events as context_losses,
    state_recovery_accuracy as recovery_fidelity,
    hallucination_events
FROM daily_metrics
ORDER BY scenario, simulation_day;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE campaigns IS 'Buyer advertising campaigns with targeting and budget';
COMMENT ON TABLE deals IS 'Confirmed deals between buyers and sellers';
COMMENT ON TABLE transactions IS 'Financial record of each transaction with fee breakdown';
COMMENT ON TABLE auction_results IS 'Scenario A second-price auction results';
COMMENT ON TABLE daily_metrics IS 'Aggregated daily metrics for trend analysis';
COMMENT ON TABLE agent_states IS 'Agent state snapshots for context rot simulation';

COMMENT ON VIEW fee_extraction_by_scenario IS 'Compare intermediary take rates across scenarios';
COMMENT ON VIEW campaign_goal_achievement IS 'Campaign KPI achievement rates by scenario';
COMMENT ON VIEW daily_performance_trend IS 'Daily performance metrics for context rot analysis';
