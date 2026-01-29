# Optimized Execution Plan: IAB Agentic Ecosystem Simulation

> **Version:** 2.0 | **Date:** 2026-01-30 | **Status:** Ready to Execute

---

## Executive Summary

This plan optimizes the 30-day IAB Agentic RTB Simulation for **minimal time** and **minimal cost** while maintaining result validity. Key optimizations:

| Metric | Baseline | Optimized | Savings |
|--------|----------|-----------|---------|
| Real-time duration | 7.2 hours | **43 minutes** | 90% |
| LLM cost per run | $277-300 | **$8-15** | 95% |
| Infrastructure cost | $600-900/mo | **$50-100/mo** | 90% |
| Total runs possible | 1 per day | **10+ per day** | 10x |

---

## Part 1: Time Optimization

### 1.1 Acceleration Analysis

**Current Baseline:** 100x acceleration = 7.2 real hours for 30 simulated days

| Acceleration | Real Duration | Limiting Factor | Viable? |
|--------------|---------------|-----------------|---------|
| 100x | 7.2 hours | None | ✅ Safe |
| 500x | 86 minutes | LLM latency | ✅ With batching |
| 1000x | 43 minutes | DB write throughput | ✅ With async writes |
| 2000x | 22 minutes | Agent decision frequency | ⚠️ Marginal |
| 5000x | 9 minutes | Simulation validity | ❌ Not recommended |

**What breaks at higher speeds:**
- **500x:** LLM API calls become bottleneck (300-500ms each)
- **1000x:** PostgreSQL write throughput may lag
- **2000x+:** Agent decisions skip meaningful time intervals, reducing realism

**Recommendation:** Use **1000x acceleration** with async database writes

```python
# src/config/.env
RTB_TIME_ACCELERATION=1000  # 43 minutes for 30 days
RTB_DB_ASYNC_WRITES=true
RTB_DECISION_BATCHING=true
```

### 1.2 Parallel Scenario Execution

Run all three scenarios **simultaneously** instead of sequentially:

```bash
# Sequential (current): 3 × 43 min = 129 minutes
# Parallel (optimized): 1 × 43 min = 43 minutes

# Implementation: Use process isolation
python -m src.scenarios.scenario_a --days 30 --mock-llm &
python -m src.scenarios.scenario_b --days 30 --mock-llm &
python -m src.scenarios.scenario_c --days 30 --mock-llm &
wait
python -m src.metrics.compare_scenarios
```

**Isolation Requirements:**
- Separate Redis namespaces per scenario
- Separate PostgreSQL schemas per scenario
- Shared InfluxDB (different measurement names)

```yaml
# docker/docker-compose.parallel.yml
services:
  postgres:
    # Single DB with 3 schemas: scenario_a, scenario_b, scenario_c
  redis:
    # Namespace by prefix: rtb:a:*, rtb:b:*, rtb:c:*
```

### 1.3 Time Optimization Summary

| Phase | Baseline | Optimized |
|-------|----------|-----------|
| Infrastructure startup | 5 min | 2 min (prebuilt images) |
| Scenario A | 2.4 hours | 14 min (parallel + 1000x) |
| Scenario B | 2.4 hours | 14 min (parallel + 1000x) |
| Scenario C | 2.4 hours | 14 min (parallel + 1000x) |
| Analysis | 30 min | 15 min (auto-generated) |
| **Total** | **7.7 hours** | **~45 minutes** |

---

## Part 2: Cost Optimization

### 2.1 LLM Cost Breakdown

**Baseline costs (from FINAL_IMPLEMENTATION_PLAN.md):**

| Scenario | Calls | Input Tokens | Output Tokens | Cost (Haiku) |
|----------|-------|--------------|---------------|--------------|
| A | 15,000 | 30M | 3M | $75 |
| B | 22,500 | 45M | 4.5M | $112 |
| C | 18,000 | 36M | 3.6M | $90 |
| **Total** | **55,500** | **111M** | **11.1M** | **$277** |

### 2.2 LLM Cost Reduction Strategies

#### Strategy A: Use Mock LLM for Infrastructure Testing (FREE)

```bash
# 100% cost reduction for infrastructure validation
rtb-sim run --days 30 --mock-llm --scenario a,b,c
```

Mock LLM returns deterministic responses based on input patterns:
- Bid acceptance: `floor_price <= bid <= floor_price * 1.5`
- Deal negotiation: Always accept within 15% of target CPM
- Campaign allocation: Round-robin across available inventory

**Use for:** Infrastructure testing, debugging, dashboard validation

#### Strategy B: Cached LLM Responses (90% reduction)

Cache deterministic decision patterns:

```python
# src/agents/buyer/config.py
LLM_CACHE_CONFIG = {
    "enabled": True,
    "cache_ttl_hours": 24,
    "cacheable_decisions": [
        "bid_acceptance",      # Same inputs → same output
        "inventory_selection", # Deterministic ranking
        "price_negotiation",   # Rule-based
    ],
    "non_cacheable_decisions": [
        "campaign_strategy",   # Needs LLM creativity
        "anomaly_detection",   # Needs pattern recognition
    ],
}
```

**Cost with caching:** ~$27 (90% reduction from $277)

#### Strategy C: Haiku-Only Mode (Additional 50% reduction)

Use Claude Haiku instead of Sonnet for ALL agent decisions:

| Model | Input/1K | Output/1K | 30-day Cost |
|-------|----------|-----------|-------------|
| Claude 3 Opus | $15 | $75 | ~$4,155 |
| Claude 3.5 Sonnet | $3 | $15 | ~$831 |
| Claude 3 Haiku | $0.25 | $1.25 | **$69** |
| Haiku + Caching | $0.25 | $1.25 | **$8-15** |

```bash
# .env configuration
RTB_MANAGER_MODEL=anthropic/claude-3-5-haiku-20241022
RTB_AGENT_MODEL=anthropic/claude-3-5-haiku-20241022
```

#### Strategy D: Batched Decision Making

Instead of 1 LLM call per bid request, batch decisions:

```python
# Before: 15,000 calls for Scenario A
# After: 1,500 calls (batch size = 10)

async def batch_bid_decisions(bids: list[BidRequest]) -> list[BidResponse]:
    """Process 10 bids in a single LLM call."""
    prompt = f"""
    Evaluate these {len(bids)} bid requests and return decisions:
    {json.dumps([b.dict() for b in bids])}
    
    Return JSON array of decisions.
    """
    response = await llm.complete(prompt)
    return [BidResponse(**d) for d in json.loads(response)]
```

**Cost with batching:** ~$30 (89% reduction)

#### Strategy E: Pre-computed Decision Trees

For 80% of common scenarios, use rule-based decisions:

```python
# src/agents/buyer/strategies/decision_tree.py
def should_accept_bid(bid: BidRequest, campaign: Campaign) -> bool:
    """
    Rule-based bid acceptance (no LLM needed).
    Only escalate to LLM for edge cases.
    """
    # Common case: Accept if within budget and CPM target
    if bid.offered_cpm <= campaign.target_cpm * 1.1:
        if campaign.spent + bid.total_cost <= campaign.total_budget:
            return True  # Rule-based accept
    
    # Edge case: Near budget boundary, complex targeting
    if campaign.remaining_budget / campaign.total_budget < 0.1:
        return None  # Escalate to LLM
    
    return False  # Rule-based reject
```

**LLM calls reduced by 80%**, cost: ~$55

### 2.3 Cost Optimization Summary

| Strategy | LLM Cost | Validity |
|----------|----------|----------|
| Baseline (Haiku) | $277 | Full |
| + Caching | $27 | Full |
| + Batching | $15 | Full |
| + Decision Trees | $8-15 | Full (LLM for edge cases) |
| Mock Mode | $0 | Infrastructure only |

**Recommended for production runs:** Haiku + Caching + Decision Trees = **$8-15 per full run**

### 2.4 Infrastructure Cost Optimization

| Component | Railway | Ubuntu VPS | Local (Mac) |
|-----------|---------|------------|-------------|
| PostgreSQL | $20/mo | Included | Free |
| Redis | $10/mo | Included | Free |
| InfluxDB | $15/mo | Included | Free |
| Grafana | $10/mo | Included | Free |
| Compute | $50/mo | $48/mo | Free |
| **Total** | **$105/mo** | **$48/mo** | **$0** |

**Recommendation:** 
- **Development:** Run on Ben's Mac (Gas Town)
- **Production runs:** $48/mo DigitalOcean droplet (2 vCPU, 2GB, already available)

---

## Part 3: Gas Town Deployment

### 3.1 Prerequisites

```bash
# Verify Gas Town installation
cd /Users/ben/Desktop/Gas\ Town/gastown
./gt --version

# Verify Git access
gh auth status
```

### 3.2 Add Repository as Gas Town Rig

```bash
# Navigate to Gas Town workspace
cd ~/gt

# Add the repo as a rig
gt rig add rtb-sim https://github.com/benputley1/iab-agentic-ecosystem-simulation

# Verify
gt rig list
```

### 3.3 Create Convoys for Parallel Execution

**Phase 1 Convoy: Infrastructure**
```bash
cd ~/gt/rtb-sim

# Create beads for infrastructure tasks
bd create "Set up Docker services (Postgres, Redis, InfluxDB, Grafana)" --tag infra-docker
bd create "Initialize database schemas (campaigns, deals, ground_truth, ledger)" --tag infra-db
bd create "Configure Redis message bus with scenario namespaces" --tag infra-redis
bd create "Set up Grafana dashboards from templates" --tag infra-grafana

# Create convoy
gt convoy create "Phase 1: Infrastructure" <bead-ids...>
```

**Phase 2 Convoy: Agent Adapters**
```bash
bd create "Implement buyer agent wrapper with decision batching" --tag agent-buyer
bd create "Implement seller agent wrapper with inventory management" --tag agent-seller
bd create "Build exchange agent for Scenario A with fee extraction" --tag agent-exchange
bd create "Add hallucination injection for Scenario B" --tag agent-ucp

gt convoy create "Phase 2: Agent Adapters" <bead-ids...>
```

**Phase 3 Convoy: Scenario Engines**
```bash
bd create "Complete Scenario A implementation with auction logic" --tag scenario-a
bd create "Complete Scenario B with context rot simulation" --tag scenario-b
bd create "Complete Scenario C with ledger persistence" --tag scenario-c

gt convoy create "Phase 3: Scenarios" <bead-ids...>
```

**Phase 4 Convoy: Simulation & Analysis**
```bash
bd create "Implement 1000x time acceleration with async writes" --tag sim-accel
bd create "Add parallel scenario execution support" --tag sim-parallel
bd create "Build automated comparison reports" --tag sim-reports

gt convoy create "Phase 4: Simulation" <bead-ids...>
```

### 3.4 Spawn Polecats for Parallel Work

```bash
# Sling beads to polecats (spawns parallel workers)
gt sling <infra-docker-id> rtb-sim
gt sling <infra-db-id> rtb-sim
gt sling <infra-redis-id> rtb-sim
gt sling <infra-grafana-id> rtb-sim

# Monitor progress
gt convoy status "Phase 1: Infrastructure"
```

### 3.5 tmux Monitoring Setup

```bash
# Start Gas Town with full stack
gt start --all

# Attach to Mayor for overview
gt mayor attach

# Split tmux for multiple views (within Mayor session)
# Ctrl+B then % = vertical split
# Ctrl+B then " = horizontal split

# Watch convoy progress
watch -n 5 'gt convoy list'

# Tail simulation logs
tail -f ~/gt/rtb-sim/logs/simulation.log
```

### 3.6 Quick Start Script

Create `~/gt/rtb-sim/scripts/gastown-start.sh`:

```bash
#!/bin/bash
# Gas Town Quick Start for RTB Simulation

set -e

echo "🚀 Starting Gas Town for RTB Simulation..."

# 1. Start infrastructure
cd ~/gt/rtb-sim
docker-compose -f docker/docker-compose.yml up -d

# 2. Wait for services
echo "⏳ Waiting for services..."
sleep 10

# 3. Initialize database
docker exec rtb_postgres psql -U rtb_sim -d rtb_simulation -f /docker-entrypoint-initdb.d/01-init.sql

# 4. Start Gas Town
gt start --all

echo "✅ Gas Town ready!"
echo ""
echo "Commands:"
echo "  gt mayor attach     - Main coordination view"
echo "  gt convoy list      - See all convoys"
echo "  gt polecat list     - See all workers"
echo ""
echo "Run simulation:"
echo "  rtb-sim run --days 30 --mock-llm"
```

---

## Part 4: Optimized Execution Plan

### 4.1 Phase Overview

| Phase | Tasks | Duration | Parallel? |
|-------|-------|----------|-----------|
| 0. Setup | Clone, dependencies, Docker | 15 min | No |
| 1. Infrastructure | DB, Redis, InfluxDB, Grafana | 30 min | Yes (4 polecats) |
| 2. Validation | Test each component | 20 min | Yes |
| 3. Mock Run | Full sim with mock LLM | 45 min | Yes |
| 4. Production Run | Full sim with Haiku | 45 min | Yes |
| 5. Analysis | Compare, report, content | 30 min | No |
| **Total** | | **~3 hours** | |

### 4.2 Detailed Steps

#### Phase 0: Initial Setup (15 minutes)

```bash
# 1. Clone repo (if not already)
cd /tmp
git clone https://github.com/benputley1/iab-agentic-ecosystem-simulation.git
cd iab-agentic-ecosystem-simulation

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[full]"

# 4. Copy environment file
cp .env.example .env

# 5. Configure .env
cat >> .env << 'EOF'
# Optimized settings
RTB_TIME_ACCELERATION=1000
RTB_MOCK_LLM=true
RTB_MANAGER_MODEL=anthropic/claude-3-5-haiku-20241022
RTB_AGENT_MODEL=anthropic/claude-3-5-haiku-20241022
ANTHROPIC_API_KEY=your_key_here
EOF
```

#### Phase 1: Infrastructure (30 minutes, parallel)

```bash
# Start Docker services
cd docker
docker-compose up -d

# Verify services
docker-compose ps  # All should be "Up"
docker exec rtb_postgres pg_isready -U rtb_sim  # Should say "accepting connections"
docker exec rtb_redis redis-cli ping  # Should say "PONG"
curl -s http://localhost:8086/health | jq  # InfluxDB health
curl -s http://localhost:3000/api/health | jq  # Grafana health
```

#### Phase 2: Validation (20 minutes)

```bash
# Test database connection
python -c "
from src.infrastructure.database import get_engine
import asyncio
async def test():
    engine = await get_engine()
    print('✅ Database connected')
asyncio.run(test())
"

# Test Redis connection
python -c "
from src.infrastructure.redis_bus import create_redis_bus
import asyncio
async def test():
    bus = await create_redis_bus('test')
    print('✅ Redis connected')
    await bus.disconnect()
asyncio.run(test())
"

# Test scenario imports
python -c "
from src.scenarios.scenario_a import ScenarioA
from src.scenarios.scenario_b import ScenarioB
from src.scenarios.scenario_c import ScenarioC
print('✅ All scenarios importable')
"

# Run unit tests
pytest tests/ -v --tb=short
```

#### Phase 3: Mock Run (45 minutes)

```bash
# Full simulation with mock LLM (validates everything except LLM integration)
rtb-sim run \
    --scenario a,b,c \
    --days 30 \
    --buyers 5 \
    --sellers 5 \
    --mock-llm \
    --output results/mock_run.json

# Verify results
cat results/mock_run.json | jq '.scenario_results[] | {scenario_id, total_deals, total_buyer_spend}'
```

#### Phase 4: Production Run (45 minutes)

```bash
# Full simulation with real LLM (Haiku + caching)
rtb-sim run \
    --scenario a,b,c \
    --days 30 \
    --buyers 5 \
    --sellers 5 \
    --real-llm \
    --output results/production_run.json

# Monitor in another terminal
watch -n 10 'rtb-sim status'
```

#### Phase 5: Analysis (30 minutes)

```bash
# Generate comparison report
rtb-sim compare results/production_run.json --format markdown > ANALYSIS.md

# Generate content outputs
python -m src.logging.content.article_generator results/production_run.json

# View Grafana dashboards
open http://localhost:3000  # Login: admin / admin
```

---

## Part 5: Time/Cost Comparison Table

### Baseline vs Optimized

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Time** | | | |
| Real duration per run | 7.2 hours | 45 minutes | **90%** |
| Runs per day | 1-2 | 10+ | **5-10x** |
| Setup time | 2 hours | 15 minutes | **88%** |
| **LLM Costs** | | | |
| Per full run | $277 | $8-15 | **95-97%** |
| Development runs | $277 | $0 (mock) | **100%** |
| Monthly (10 runs) | $2,770 | $80-150 | **95%** |
| **Infrastructure** | | | |
| Monthly hosting | $600-900 | $48-100 | **85%** |
| Development | $600-900/mo | $0 (local) | **100%** |

### Cost Breakdown by Mode

| Mode | LLM | Infra | Total/Run | Use Case |
|------|-----|-------|-----------|----------|
| **Mock (Free)** | $0 | $0 | $0 | Development, CI/CD |
| **Haiku + Cache** | $8-15 | $0 | $8-15 | Production runs |
| **Full (Railway)** | $8-15 | $3.50* | $12-20 | Cloud production |
| **Full (Ubuntu)** | $8-15 | $1.60* | $10-17 | Dedicated server |

*Infrastructure amortized per run (assuming 30 runs/month)

---

## Part 6: Validation Checklist

### Pre-Run Validation

- [ ] Docker services running (`docker-compose ps`)
- [ ] PostgreSQL accepts connections
- [ ] Redis responds to PING
- [ ] InfluxDB health check passes
- [ ] Grafana accessible at :3000
- [ ] Python environment activated
- [ ] Dependencies installed (`pip install -e ".[full]"`)
- [ ] `.env` configured with API keys
- [ ] Unit tests pass (`pytest tests/`)

### During-Run Monitoring

- [ ] CPU usage < 80%
- [ ] Memory usage < 70%
- [ ] PostgreSQL query time < 100ms
- [ ] Redis latency < 10ms
- [ ] LLM response time < 2s
- [ ] No error logs in `logs/simulation.log`

### Post-Run Validation

- [ ] Results JSON created
- [ ] All 3 scenarios completed
- [ ] 30 simulation days recorded
- [ ] Metrics populated in InfluxDB
- [ ] Grafana dashboards show data
- [ ] Comparison report generated

### Result Validity Checks

| Metric | Expected Range | Invalid If |
|--------|---------------|------------|
| Total deals (A) | 4,000-6,000 | < 1,000 or > 10,000 |
| Exchange fees (A) | 10-20% | < 5% or > 30% |
| Context rot events (B) | 20-50 | 0 or > 100 |
| Hallucination rate (B) | 5-15% | < 1% or > 30% |
| Recovery rate (C) | 100% | < 99% |
| Ledger entries (C) | 10,000+ | < 5,000 |

---

## Part 7: Quick Reference Commands

### Simulation

```bash
# Quick test (1 day, mock LLM)
rtb-sim run --days 1 --mock-llm

# Full mock run (validates infrastructure)
rtb-sim run --days 30 --mock-llm --scenario a,b,c

# Production run (minimal cost)
rtb-sim run --days 30 --real-llm --scenario a,b,c

# Single scenario test
rtb-sim test-scenario --scenario c --mock-llm

# Test ledger recovery (Scenario C)
rtb-sim test-recovery --agent test-buyer-001
```

### Infrastructure

```bash
# Start services
cd docker && docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f postgres redis

# Reset database
docker-compose down -v && docker-compose up -d
```

### Gas Town

```bash
# Add rig
gt rig add rtb-sim https://github.com/benputley1/iab-agentic-ecosystem-simulation

# Create convoy
gt convoy create "Phase 1" <bead-ids>

# Sling work
gt sling <bead-id> rtb-sim

# Monitor
gt convoy status "Phase 1"
gt mayor attach
```

### Analysis

```bash
# Compare scenarios
rtb-sim compare results/run.json --format markdown

# Generate article content
python -m src.logging.content.article_generator results/run.json

# Export metrics
python -m src.metrics.export --output metrics.csv
```

---

## Appendix A: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Connection refused" on PostgreSQL | Service not started | `docker-compose up -d postgres` |
| "ANTHROPIC_API_KEY not set" | Missing env var | Add to `.env` or export |
| "ModuleNotFoundError" | Dependencies missing | `pip install -e ".[full]"` |
| LLM timeout | Rate limiting | Reduce `RTB_MAX_ITERATIONS` |
| Memory errors | Too many agents | Reduce buyers/sellers count |

### Performance Tuning

```bash
# If simulations are slow:
export RTB_TIME_ACCELERATION=500  # Lower for stability
export RTB_DB_ASYNC_WRITES=true   # Don't wait for DB

# If LLM costs are high:
export RTB_MOCK_LLM=true          # Use mocks
export RTB_DECISION_BATCHING=true # Batch calls
```

---

## Appendix B: Files to Commit

After implementing optimizations, commit these changes:

```bash
git add -A
git commit -m "Add optimized execution plan with 90% time/cost reduction

- Time: 7.2 hours → 45 minutes (1000x acceleration + parallel)
- LLM costs: $277 → $8-15 (Haiku + caching + batching)
- Infrastructure: $600-900/mo → $48-100/mo

New files:
- OPTIMIZED_EXECUTION_PLAN.md - This document
- scripts/gastown-start.sh - Quick start for Gas Town
- docker/docker-compose.parallel.yml - Parallel scenario support"

git push origin main
```

---

*Prepared by NJ | 2026-01-30*
*Based on analysis of iab-agentic-ecosystem-simulation repo structure and FINAL_IMPLEMENTATION_PLAN.md*
