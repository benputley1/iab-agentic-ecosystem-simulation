-- Parallel Schema Setup for IAB Agentic Simulation
-- Creates isolated schemas for each scenario to enable parallel execution

-- Create schemas for each scenario
CREATE SCHEMA IF NOT EXISTS scenario_a;
CREATE SCHEMA IF NOT EXISTS scenario_b;
CREATE SCHEMA IF NOT EXISTS scenario_c;

-- Grant permissions
GRANT ALL ON SCHEMA scenario_a TO rtb_sim;
GRANT ALL ON SCHEMA scenario_b TO rtb_sim;
GRANT ALL ON SCHEMA scenario_c TO rtb_sim;

-- Create tables in each schema
-- (Duplicates the main tables into each scenario's namespace)

-- Scenario A tables
CREATE TABLE IF NOT EXISTS scenario_a.campaigns (LIKE public.campaigns INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_a.deals (LIKE public.deals INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_a.transactions (LIKE public.transactions INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_a.agent_decisions (LIKE public.agent_decisions INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_a.agent_claims (LIKE public.agent_claims INCLUDING ALL);

-- Scenario B tables
CREATE TABLE IF NOT EXISTS scenario_b.campaigns (LIKE public.campaigns INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_b.deals (LIKE public.deals INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_b.transactions (LIKE public.transactions INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_b.agent_decisions (LIKE public.agent_decisions INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_b.agent_claims (LIKE public.agent_claims INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_b.context_rot_events (LIKE public.context_rot_events INCLUDING ALL);

-- Scenario C tables
CREATE TABLE IF NOT EXISTS scenario_c.campaigns (LIKE public.campaigns INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_c.deals (LIKE public.deals INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_c.transactions (LIKE public.transactions INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_c.agent_decisions (LIKE public.agent_decisions INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_c.agent_claims (LIKE public.agent_claims INCLUDING ALL);
CREATE TABLE IF NOT EXISTS scenario_c.ledger_entries (LIKE public.ledger_entries INCLUDING ALL);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_scenario_a_deals_campaign ON scenario_a.deals(campaign_id);
CREATE INDEX IF NOT EXISTS idx_scenario_a_deals_buyer ON scenario_a.deals(buyer_id);
CREATE INDEX IF NOT EXISTS idx_scenario_a_deals_seller ON scenario_a.deals(seller_id);

CREATE INDEX IF NOT EXISTS idx_scenario_b_deals_campaign ON scenario_b.deals(campaign_id);
CREATE INDEX IF NOT EXISTS idx_scenario_b_deals_buyer ON scenario_b.deals(buyer_id);
CREATE INDEX IF NOT EXISTS idx_scenario_b_deals_seller ON scenario_b.deals(seller_id);
CREATE INDEX IF NOT EXISTS idx_scenario_b_rot_agent ON scenario_b.context_rot_events(agent_id);

CREATE INDEX IF NOT EXISTS idx_scenario_c_deals_campaign ON scenario_c.deals(campaign_id);
CREATE INDEX IF NOT EXISTS idx_scenario_c_deals_buyer ON scenario_c.deals(buyer_id);
CREATE INDEX IF NOT EXISTS idx_scenario_c_deals_seller ON scenario_c.deals(seller_id);
CREATE INDEX IF NOT EXISTS idx_scenario_c_ledger_agent ON scenario_c.ledger_entries(agent_id);

-- Comment for documentation
COMMENT ON SCHEMA scenario_a IS 'Isolated schema for Scenario A (Exchange) simulation';
COMMENT ON SCHEMA scenario_b IS 'Isolated schema for Scenario B (IAB A2A) simulation';
COMMENT ON SCHEMA scenario_c IS 'Isolated schema for Scenario C (Alkimi Ledger) simulation';

-- Verification query
SELECT 
    schema_name, 
    COUNT(*) as table_count
FROM information_schema.tables 
WHERE schema_name IN ('scenario_a', 'scenario_b', 'scenario_c')
GROUP BY schema_name;
