# IAB Agentic Ecosystem Simulation - Comprehensive Plan

**Repo:** `github.com/benputley1/iab-agentic-ecosystem-simulation`
**Orchestration:** Gastown (20-30 polecats organized by component)
**Duration:** 30-day simulated campaigns

---

## Executive Summary

Build a simulation environment comparing THREE scenarios for programmatic advertising:

| Scenario | Description | Exchange Role | State Persistence |
|----------|-------------|---------------|-------------------|
| **A: Current State** | Rent-seeking exchanges | Exchange agent extracts 10-20% fees | Centralized DB |
| **B: IAB Pure A2A** | Direct buyer↔seller per IAB spec | No exchange (passive infrastructure) | In-memory (context rot) |
| **C: Alkimi Ledger** | Beads → Walrus, internal ledger → Sui | Decentralized audit trail | Immutable records |

**Key Research Questions:**
1. How much do intermediaries extract in each scenario?
2. What % of campaigns achieve their stated KPIs?
3. How does context rot degrade performance over 30 days?
4. How often do agents hallucinate (decide on fabricated data)?

---

## Architecture

```
                         SIMULATION CONTROL
                    ┌─────────────────────────────┐
                    │     Scenario Engine         │
                    │  • Event injection          │
                    │  • Time acceleration        │
                    │  • Ground truth DB          │
                    └──────────────┬──────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────────┐
│                           GAS TOWN LAYER                                 │
│                                  │                                       │
│  ┌────────────┐         ┌────────▼────────┐        ┌────────────┐       │
│  │   MAYOR    │◄───────►│     DEACON      │◄──────►│  WITNESS   │       │
│  │  (Coord)   │         │  (Supervisor)   │        │ (Monitor)  │       │
│  └────────────┘         └────────┬────────┘        └────────────┘       │
│                                  │                                       │
│     ┌────────────────────────────┼────────────────────────────┐         │
│     │                  CONVOY: Campaign-001                    │         │
│     │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │         │
│     │  │buyer-01 │ │buyer-02 │ │seller-01│ │exchange │        │         │
│     │  │(Polecat)│ │(Polecat)│ │(Polecat)│ │(Polecat)│        │         │
│     │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │         │
│     └───────┼───────────┼───────────┼───────────┼─────────────┘         │
│             │           │           │           │                        │
│  ┌──────────▼───────────▼───────────▼───────────▼──────────────────┐    │
│  │                    BEADS (State Ledger)                          │    │
│  │  campaign-goals │ deal-states │ bid-history │ agent-checkpoints  │    │
│  │  ─────────────────────────────────────────────────────────────── │    │
│  │  Scenario C: Beads = Walrus proxy (immutable blob references)    │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
  ┌──────▼──────┐          ┌───────▼───────┐         ┌──────▼──────┐
  │   REDIS     │          │  POSTGRESQL   │         │  INFLUXDB   │
  │ (Msg Bus)   │          │ (Ground Truth │         │  (Metrics)  │
  │             │          │  + Ledger)    │         │             │
  └─────────────┘          └───────────────┘         └─────────────┘
```

---

## Scenario Details

### Scenario A: Current State (Rent-Seeking Exchanges)

Models today's programmatic ecosystem with exchange intermediaries.

```
Buyer Agent                  Exchange Agent               Seller Agent
    │                             │                            │
    │──── bid request ───────────►│                            │
    │                             │──── bid request ──────────►│
    │                             │◄─── bid response ──────────│
    │                             │   (applies 15% fee)        │
    │◄─── bid response ───────────│                            │
    │   (price inflated)          │                            │
```

**Exchange Agent Behavior:**
- Applies platform fee (configurable 10-20%)
- May favor certain buyers/sellers
- Controls auction mechanics
- Extracts "tech tax" on every transaction

### Scenario B: IAB Pure A2A (No Exchange)

Direct agent-to-agent per IAB agentic-rtb-framework spec.

```
Buyer Agent                                            Seller Agent
    │                                                       │
    │──────────── A2A discovery query ─────────────────────►│
    │◄─────────── product listings (MCP) ──────────────────│
    │──────────── pricing request (identity + volume) ─────►│
    │◄─────────── tiered price response ───────────────────│
    │──────────── proposal submission (MCP create_order) ──►│
    │◄─────────── accept/counter/reject ───────────────────│
    │──────────── deal confirmation ───────────────────────►│
    │◄─────────── deal ID for DSP ─────────────────────────│
```

**Critical Gap (Intentional):**
- No persistent state across agent restarts
- Context rot accumulates over 30 days
- No single source of truth for disputes
- Hallucination risk from stale embeddings

### Scenario C: Alkimi Ledger-Backed

Same A2A flow as Scenario B, but with immutable audit trail.

```
Buyer Agent                    Beads/Ledger                Seller Agent
    │                              │                            │
    │──── proposal ────────────────┼───────────────────────────►│
    │                              │                            │
    │                         ┌────▼────┐                       │
    │                         │ Record  │                       │
    │                         │ to Bead │                       │
    │                         └────┬────┘                       │
    │                              │                            │
    │◄─────────────────────────────┼──── accept + deal ID ──────│
    │                              │                            │
    │                         ┌────▼────┐                       │
    │                         │ Record  │                       │
    │                         │ to Bead │                       │
    │                         └─────────┘                       │
```

**Alkimi Advantage:**
- Beads = Walrus blob proxy (immutable transaction record)
- Internal ledger = Sui object proxy (tamper-proof reference)
- Agents can always recover state from ledger
- Ground truth for dispute resolution
- No context rot (state reconstructable)

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| **Buyers** | 5 agents |
| **Sellers** | 5 agents (each = publisher with inventory) |
| **Campaigns** | 10 per buyer = 50 total |
| **Duration** | 30 simulated days |
| **Distribution** | Random campaigns across publishers |
| **Time acceleration** | Configurable (100x default = 7.2 hours real time) |

---

## Key Metrics

### 1. Fee Extraction Comparison

```sql
-- Per scenario, measure total intermediary take
SELECT
    scenario,
    SUM(buyer_spend) as gross_spend,
    SUM(seller_revenue) as net_to_publisher,
    SUM(buyer_spend - seller_revenue) as intermediary_take,
    (SUM(buyer_spend - seller_revenue) / SUM(buyer_spend)) * 100 as take_rate_pct
FROM transactions
GROUP BY scenario;
```

**Expected Results:**
- Scenario A: 15-25% intermediary take
- Scenario B: ~5% (residual fees in agent pricing)
- Scenario C: ~2% (Sui gas estimation + minimal fees)

### 2. Campaign Goal Achievement

```sql
-- What % of campaigns hit their KPIs?
SELECT
    scenario,
    COUNT(*) as total_campaigns,
    SUM(CASE WHEN actual_impressions >= target_impressions THEN 1 ELSE 0 END) as hit_impression_goal,
    SUM(CASE WHEN actual_cpm <= target_cpm THEN 1 ELSE 0 END) as hit_cpm_goal,
    AVG(goal_attainment_score) as avg_goal_attainment
FROM campaigns
GROUP BY scenario;
```

### 3. Context Rot Impact

```sql
-- Performance degradation over time
SELECT
    scenario,
    simulation_day,
    AVG(goal_attainment_score) as daily_goal_attainment,
    COUNT(DISTINCT agent_restart_events) as context_losses,
    AVG(state_recovery_accuracy) as recovery_fidelity
FROM daily_metrics
GROUP BY scenario, simulation_day
ORDER BY scenario, simulation_day;
```

**Hypothesis:**
- Scenario B: Degradation accelerates after day 15
- Scenario C: Stable performance (ledger recovery)

### 4. Hallucination Rate

```sql
-- Decisions based on fabricated data
SELECT
    scenario,
    agent_type,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN decision_basis_verified = FALSE THEN 1 ELSE 0 END) as hallucinated_decisions,
    (SUM(CASE WHEN decision_basis_verified = FALSE THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 as hallucination_rate
FROM agent_decisions
JOIN ground_truth ON agent_decisions.claimed_fact_id = ground_truth.fact_id
GROUP BY scenario, agent_type;
```

**Ground Truth Database:**
- Maintains actual inventory levels
- Records real bid history
- Tracks true campaign delivery
- Agents cannot read this directly
- Used for post-hoc verification of agent claims

---

## Development Phases (Gastown Parallelization)

### Phase 1: Infrastructure (5 polecats in parallel)

| Polecat | Component | Deliverables |
|---------|-----------|--------------|
| `infra-repo` | Repository scaffold | Clone repo, pyproject.toml, docker-compose.yml |
| `infra-db` | PostgreSQL | Schema for campaigns, deals, ground truth, ledger |
| `infra-redis` | Message bus | Redis Streams wrapper, A2A message routing |
| `infra-metrics` | InfluxDB + Grafana | Metric collection, dashboard templates |
| `infra-beads` | Beads types | Custom types: campaign, deal, bid, transaction |

### Phase 2: Agent Adapters (4 polecats in parallel)

| Polecat | Component | Deliverables |
|---------|-----------|--------------|
| `agent-buyer` | Buyer adapter | Wrap IAB buyer-agent CrewAI flows |
| `agent-seller` | Seller adapter | Wrap IAB seller-agent flows |
| `agent-exchange` | Exchange agent | Scenario A: rent-seeking auction logic |
| `agent-ucp` | Audience/UCP | Embedding exchange, hallucination injection points |

### Phase 3: Scenario Engines (3 polecats in parallel)

| Polecat | Component | Deliverables |
|---------|-----------|--------------|
| `scenario-a` | Current state | Exchange agent integration, fee extraction logic |
| `scenario-b` | IAB Pure A2A | Direct buyer↔seller, context rot simulation |
| `scenario-c` | Alkimi ledger | Beads persistence, ledger recovery, cost estimation |

### Phase 4: Orchestration (3 polecats in parallel)

| Polecat | Component | Deliverables |
|---------|-----------|--------------|
| `orch-convoy` | Campaign↔Convoy | Map campaigns to Gastown convoys |
| `orch-ground-truth` | Ground truth DB | Maintain reality database, verify agent claims |
| `orch-logging` | Event + narrative logs | Comprehensive events, narrative-ready formatting |

### Phase 5: Simulation & Analysis (3 polecats in parallel)

| Polecat | Component | Deliverables |
|---------|-----------|--------------|
| `sim-runner` | Simulation engine | Time control, event injection, chaos testing |
| `sim-metrics` | Metric analysis | KPI calculations, comparative reports |
| `sim-content` | Content generation | Extract insights for article series |

---

## File Structure

```
iab-agentic-ecosystem-simulation/
├── .beads/
│   ├── types.jsonl                  # campaign, deal, bid, transaction types
│   ├── formulas/
│   │   └── rtb-simulation.formula.toml
│   └── PRIME.md                     # Polecat context
├── docker/
│   ├── docker-compose.yml           # PostgreSQL, Redis, InfluxDB, Grafana
│   └── postgres/
│       ├── init.sql                 # Base schema
│       ├── ground_truth.sql         # Reality database
│       └── ledger.sql               # Sui proxy ledger
├── src/
│   ├── infrastructure/
│   │   ├── database.py              # SQLAlchemy models
│   │   ├── redis_bus.py             # Message routing
│   │   ├── ledger.py                # Sui proxy (internal ledger)
│   │   └── beads_client.py          # Walrus proxy (Beads wrapper)
│   ├── agents/
│   │   ├── buyer/
│   │   │   ├── adapter.py           # IAB buyer-agent wrapper
│   │   │   └── strategies/          # Bidding strategies
│   │   ├── seller/
│   │   │   ├── adapter.py           # IAB seller-agent wrapper
│   │   │   └── inventory.py         # Publisher inventory
│   │   ├── exchange/
│   │   │   ├── auction.py           # Scenario A: rent-seeking
│   │   │   └── fees.py              # Fee extraction logic
│   │   └── ucp/
│   │       ├── embeddings.py        # User context protocol
│   │       └── hallucination.py     # Inject/detect hallucinations
│   ├── scenarios/
│   │   ├── base.py                  # Common scenario interface
│   │   ├── scenario_a.py            # Current state (with exchange)
│   │   ├── scenario_b.py            # IAB Pure A2A
│   │   └── scenario_c.py            # Alkimi ledger-backed
│   ├── orchestration/
│   │   ├── convoy_sync.py           # Campaign ↔ Convoy mapping
│   │   ├── ground_truth.py          # Reality database
│   │   ├── context_rot.py           # Measure/inject context loss
│   │   └── time_controller.py       # Simulation time acceleration
│   ├── metrics/
│   │   ├── collector.py             # InfluxDB writer
│   │   ├── kpis.py                  # KPI calculations
│   │   └── dashboards/              # Grafana JSON
│   └── logging/
│       ├── event_logger.py          # Comprehensive events
│       ├── narrative_logger.py      # Content-ready formatting
│       └── analysis/                # Post-hoc analysis scripts
├── tests/
│   ├── integration/
│   │   ├── test_scenario_a.py
│   │   ├── test_scenario_b.py
│   │   └── test_scenario_c.py
│   └── hallucination/
│       └── test_ground_truth.py     # Verify hallucination detection
├── AGENTS.md                        # Polecat instructions
├── README.md
└── pyproject.toml
```

---

## Verification Plan

### Phase 1 Complete
```bash
docker-compose up -d
psql -h localhost -U postgres -c "SELECT * FROM campaigns LIMIT 1"
redis-cli PING
curl http://localhost:3000/api/health  # Grafana
```

### Phase 2 Complete
```bash
# Test buyer adapter
python -m src.agents.buyer.adapter --test

# Test seller adapter
python -m src.agents.seller.adapter --test

# Test exchange agent (Scenario A)
python -m src.agents.exchange.auction --test
```

### Phase 3 Complete
```bash
# Run mini-simulation (1 day, 1 buyer, 1 seller)
python -m src.scenarios.scenario_a --days 1 --buyers 1 --sellers 1
python -m src.scenarios.scenario_b --days 1 --buyers 1 --sellers 1
python -m src.scenarios.scenario_c --days 1 --buyers 1 --sellers 1
```

### Full Simulation
```bash
# Run all scenarios with full parameters
gt convoy create "Full Simulation" --rig rtb-sim

# Start simulation
python -m src.orchestration.run_simulation \
    --scenarios a,b,c \
    --days 30 \
    --buyers 5 \
    --sellers 5 \
    --campaigns-per-buyer 10

# Generate comparative report
python -m src.metrics.compare_scenarios
```

---

## Content Series Outputs

The simulation will generate:

1. **Comprehensive Event Logs** (`logs/events/`)
   - Every bid request, response, decision
   - Agent reasoning traces
   - Fee extraction events
   - Context rot incidents

2. **Narrative-Ready Logs** (`logs/narrative/`)
   - Human-readable summaries
   - Key moments highlighted
   - Quote-worthy agent "reasoning"
   - Comparative stats formatted for articles

3. **Comparative Metrics** (`reports/`)
   - Fee extraction by scenario (charts)
   - Goal achievement trends (30-day graphs)
   - Context rot impact visualization
   - Hallucination rate heatmaps

4. **Article Drafts** (`content/`)
   - Auto-generated findings summaries
   - Key statistics extracted
   - Scenario comparison tables

---

## Gas Fee Estimation (Scenario C)

Based on ADS Explorer data (262M+ impressions):
- Estimate Walrus blob storage cost per transaction
- Estimate Sui object creation cost
- Calculate per-impression blockchain cost
- Compare to Scenario A exchange fees

This enables the key comparison: **"Blockchain fees vs Exchange fees"**

---

## Critical Dependencies

1. **IAB Repos** (will be cloned during build)
   - https://github.com/IABTechLab/buyer-agent
   - https://github.com/IABTechLab/seller-agent
   - https://github.com/IABTechLab/agentic-direct
   - https://github.com/IABTechLab/agentic-rtb-framework
   - https://github.com/IABTechLab/agentic-audiences

2. **Gastown** (already installed)
   - `/Users/ben/Desktop/Gas Town/gastown/`

3. **Docker Services**
   - PostgreSQL 15+
   - Redis 7+
   - InfluxDB 2+
   - Grafana 10+

---

## Deployment Instructions (Full Stack Mode with tmux)

### Step 0: Write Plan to GitHub Repo

```bash
# Clone the empty repo
cd ~
gh repo clone benputley1/iab-agentic-ecosystem-simulation
cd iab-agentic-ecosystem-simulation

# Initialize if needed
git checkout -b main 2>/dev/null || git checkout main

# Copy the plan from Claude's plan file to the repo
cp ~/.claude/plans/compiled-conjuring-valley.md IMPLEMENTATION_PLAN.md

# Commit the plan
git add IMPLEMENTATION_PLAN.md
git commit -m "Add comprehensive implementation plan

This plan defines the IAB Agentic RTB Simulation:
- 3 scenarios (Current Exchange, IAB A2A, Alkimi Ledger)
- 5 development phases with parallel Gastown execution
- 30 gap-filling specifications
- Complete code templates for all components"

git push origin main
```

### Step 1: Clone Your GitHub Repo (Already Done)

```bash
# Repo is now cloned with IMPLEMENTATION_PLAN.md
cd ~/iab-agentic-ecosystem-simulation
```

### Step 2: Add as Gastown Rig

```bash
# Navigate to Gas Town workspace
cd ~/gt

# Add the repo as a new rig
gt rig add rtb-sim https://github.com/benputley1/iab-agentic-ecosystem-simulation

# Verify rig was added
gt rig list
```

### Step 3: Start Gas Town (Full Stack Mode with tmux)

```bash
# Start the Deacon + Mayor (tmux sessions)
gt start --all

# OR start with daemon for background operation
gt daemon start

# Verify everything is running
gt status
```

### Step 4: Create Phase 1 Infrastructure Beads

```bash
# Navigate to rig
cd ~/gt/rtb-sim

# Create beads for Phase 1 tasks
bd create "Set up repository scaffold with pyproject.toml and docker-compose.yml" --tag infra-repo
bd create "Create PostgreSQL schema for campaigns, deals, ground truth, ledger" --tag infra-db
bd create "Implement Redis Streams wrapper and A2A message routing" --tag infra-redis
bd create "Set up InfluxDB + Grafana with metric collection templates" --tag infra-metrics
bd create "Define custom Beads types: campaign, deal, bid, transaction" --tag infra-beads

# List created beads
bd list
```

### Step 5: Create Convoy and Sling Work to Polecats

```bash
# Get bead IDs from 'bd list' output
# Create convoy with all Phase 1 beads
gt convoy create "Phase 1: Infrastructure" <bead-id-1> <bead-id-2> <bead-id-3> <bead-id-4> <bead-id-5>

# Sling each bead to a polecat (spawns parallel workers)
gt sling <bead-id-1> rtb-sim  # infra-repo
gt sling <bead-id-2> rtb-sim  # infra-db
gt sling <bead-id-3> rtb-sim  # infra-redis
gt sling <bead-id-4> rtb-sim  # infra-metrics
gt sling <bead-id-5> rtb-sim  # infra-beads

# View convoy progress
gt convoy list
gt convoy status "Phase 1: Infrastructure"
```

### Step 6: Monitor Polecats (tmux Sessions)

```bash
# Attach to Mayor (coordinator view)
gt mayor attach

# Attach to specific polecat's tmux session
gt polecat attach <polecat-name>

# View all running sessions
tmux list-sessions

# Switch between tmux sessions
# Ctrl+B then ) = next session
# Ctrl+B then ( = previous session
# Ctrl+B then d = detach (back to terminal)
```

### Step 7: Continue with Subsequent Phases

```bash
# After Phase 1 completes, create Phase 2 beads
bd create "Wrap IAB buyer-agent CrewAI flows for simulation" --tag agent-buyer
bd create "Wrap IAB seller-agent flows for simulation" --tag agent-seller
bd create "Build rent-seeking exchange agent for Scenario A" --tag agent-exchange
bd create "Implement UCP embedding exchange and hallucination detection" --tag agent-ucp

# Create Phase 2 convoy
gt convoy create "Phase 2: Agent Adapters" <buyer-bead> <seller-bead> <exchange-bead> <ucp-bead>

# Sling to polecats
gt sling <buyer-bead> rtb-sim
gt sling <seller-bead> rtb-sim
gt sling <exchange-bead> rtb-sim
gt sling <ucp-bead> rtb-sim
```

### Step 8: Shutdown (When Done for the Day)

```bash
# Graceful shutdown (saves state, cleans up polecats)
gt shutdown --graceful

# Or if you want to pause but keep worktrees
gt down

# Resume next day
gt start --all
```

---

## Quick Reference: Gastown Commands

| Command | Purpose |
|---------|---------|
| `gt start --all` | Start Deacon, Mayor, Witnesses, Refineries (tmux) |
| `gt daemon start` | Start daemon in background |
| `gt status` | Show workspace status |
| `gt rig add <name> <url>` | Add new project rig |
| `gt rig list` | List all rigs |
| `bd create "<task>"` | Create a bead (issue) |
| `bd list` | List all beads |
| `gt convoy create "<name>" <beads...>` | Create convoy from beads |
| `gt convoy list` | List all convoys |
| `gt sling <bead-id> <rig>` | Assign bead to rig (spawns polecat) |
| `gt polecat list` | List all polecats |
| `gt polecat attach <name>` | Attach to polecat tmux session |
| `gt mayor attach` | Attach to Mayor tmux session |
| `gt shutdown --graceful` | Stop everything, cleanup polecats |
| `gt doctor` | Run health checks |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| All 3 scenarios run to completion | 30 simulated days each |
| Fee extraction data captured | Per-transaction granularity |
| Context rot measured | Daily degradation scores |
| Hallucinations detected | >90% of fabricated claims caught |
| Comparative report generated | Charts + narrative ready for publication |
| Content series inputs | 5+ article-worthy findings documented |

---

## Gap Analysis & Additional Specifications

### Gap 1: PRIME.md (Polecat Instructions)

```markdown
# RTB Simulation - Polecat Context

You are a worker agent in the IAB Agentic RTB Simulation project.

## Your Mission
Build components that simulate programmatic advertising across 3 scenarios:
- Scenario A: Current state with rent-seeking exchanges
- Scenario B: IAB Pure A2A (direct buyer↔seller)
- Scenario C: Alkimi ledger-backed (Beads = immutable records)

## Critical Rules
1. **All state must be persisted** - Never hold important data only in memory
2. **Use Beads for state** - Create/update beads for every significant event
3. **Log comprehensively** - Both event logs AND narrative logs
4. **Test as you build** - Each component needs integration tests

## Your Hook
Check your hook bead for your current assignment. It contains:
- Component to build
- Dependencies (wait for these first)
- Acceptance criteria

## Key Repos to Reference
- IAB buyer-agent: https://github.com/IABTechLab/buyer-agent
- IAB seller-agent: https://github.com/IABTechLab/seller-agent
- IAB agentic-rtb-framework: https://github.com/IABTechLab/agentic-rtb-framework

## When Done
1. Run tests: `pytest tests/`
2. Update your bead: `bd update <id> --status done`
3. Push to branch: `git push origin <your-branch>`
```

---

### Gap 2: Campaign Model Definition

```python
# src/models/campaign.py
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional

class KPIType(Enum):
    IMPRESSIONS = "impressions"
    REACH = "reach"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"
    VIEWABILITY = "viewability"

class Campaign(BaseModel):
    """A buyer's advertising campaign."""
    id: str
    buyer_id: str
    name: str

    # Budget
    total_budget: float  # USD
    daily_budget: float

    # Goals
    primary_kpi: KPIType
    target_impressions: int
    target_cpm: float  # Max willing to pay
    target_reach: Optional[int] = None

    # Targeting
    channels: list[str]  # ["display", "video", "ctv"]
    publishers: list[str]  # Preferred publisher IDs
    audience_segments: list[str]
    geo_targets: list[str]

    # Timing
    start_date: datetime
    end_date: datetime

    # State
    spent: float = 0.0
    impressions_delivered: int = 0
    status: str = "active"

    # Scenario tracking
    scenario: str  # "A", "B", or "C"
```

---

### Gap 3: Agent Behavior Specifications

#### Buyer Agent Decision Tree
```
1. RECEIVE campaign brief
2. QUERY sellers for available inventory
   - Filter by: channels, geo, audience overlap
3. FOR each seller response:
   - CALCULATE expected value = (predicted_CTR * conversion_value) - cost
   - IF expected_value > threshold: ADD to consideration set
4. RANK consideration set by expected_value / cost ratio
5. SUBMIT proposals to top N sellers
6. NEGOTIATE on counter-offers:
   - ACCEPT if within 10% of target CPM
   - COUNTER if within 20%
   - REJECT if >20% above target
7. EXECUTE deals via MCP create_order
8. TRACK delivery against goals
```

#### Seller Agent Decision Tree
```
1. RECEIVE bid request from buyer
2. CHECK inventory availability
3. CALCULATE floor price based on:
   - Historical fill rate
   - Time of day
   - Buyer tier (identity-based pricing)
4. IF bid >= floor: ACCEPT
   ELIF bid >= floor * 0.9: COUNTER at floor
   ELSE: REJECT
5. ON acceptance: GENERATE deal ID
6. RECORD to ledger (Scenario C only)
```

#### Exchange Agent Decision Tree (Scenario A only)
```
1. RECEIVE bid request from buyer
2. APPLY platform fee (15% default)
3. FORWARD to sellers with buyer identity masked (optional)
4. COLLECT responses
5. RUN second-price auction
6. DEDUCT margin from winning bid
7. FORWARD reduced bid to seller
8. RECORD transaction (fee extracted)
```

---

### Gap 4: Ground Truth Database Schema

```sql
-- ground_truth.sql
-- This database maintains "reality" that agents cannot read

CREATE TABLE inventory_reality (
    publisher_id VARCHAR(50),
    channel VARCHAR(20),
    date DATE,
    actual_avails BIGINT,  -- Real available impressions
    actual_fill_rate FLOAT,
    PRIMARY KEY (publisher_id, channel, date)
);

CREATE TABLE campaign_delivery_reality (
    campaign_id VARCHAR(50),
    date DATE,
    actual_impressions BIGINT,
    actual_clicks BIGINT,
    actual_conversions BIGINT,
    actual_spend FLOAT,
    PRIMARY KEY (campaign_id, date)
);

CREATE TABLE agent_claims (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50),
    claim_type VARCHAR(50),  -- "inventory_level", "delivery_count", etc.
    claimed_value TEXT,
    actual_value TEXT,  -- From ground truth
    is_hallucination BOOLEAN,
    timestamp TIMESTAMP
);
```

---

### Gap 5: Hallucination Injection (Scenario B)

```python
# src/agents/ucp/hallucination.py

class HallucinationInjector:
    """Injects hallucinations for Scenario B testing."""

    def __init__(self, injection_rate: float = 0.05):
        self.injection_rate = injection_rate  # 5% of queries

    def maybe_corrupt_inventory(self, real_inventory: dict) -> dict:
        """Randomly corrupt inventory data."""
        if random.random() < self.injection_rate:
            # Inflate by 10-50%
            multiplier = random.uniform(1.1, 1.5)
            return {k: int(v * multiplier) for k, v in real_inventory.items()}
        return real_inventory

    def maybe_corrupt_price(self, real_price: float) -> float:
        """Randomly corrupt pricing data."""
        if random.random() < self.injection_rate:
            # Deflate by 10-30% (makes deals look better than they are)
            return real_price * random.uniform(0.7, 0.9)
        return real_price

    def maybe_fabricate_history(self, agent_id: str) -> dict:
        """Create false memory of past transactions."""
        if random.random() < self.injection_rate:
            return {
                "fabricated": True,
                "fake_deal_id": f"FAKE-{uuid.uuid4().hex[:8]}",
                "fake_success_rate": random.uniform(0.8, 0.95)
            }
        return None
```

---

### Gap 6: Context Rot Simulation

```python
# src/orchestration/context_rot.py

class ContextRotSimulator:
    """Simulates context loss over time for Scenario B."""

    def __init__(self, decay_rate: float = 0.02):
        self.decay_rate = decay_rate  # 2% per day

    def apply_decay(self, agent_state: dict, day: int) -> dict:
        """
        Decay agent's memory over time.

        By day 30: ~55% of original context remains (0.98^30 ≈ 0.545)
        """
        survival_rate = (1 - self.decay_rate) ** day

        # Randomly drop items from agent's memory
        decayed_state = {}
        for key, value in agent_state.items():
            if random.random() < survival_rate:
                decayed_state[key] = value
            else:
                # Log the loss
                logger.info(f"Context rot: Agent lost memory of {key}")

        return decayed_state

    def trigger_restart(self, agent_id: str) -> bool:
        """
        Simulate random agent restarts (context wipe).

        Probability increases over time (simulating memory pressure).
        """
        # 0.5% chance per day, increasing
        base_probability = 0.005
        return random.random() < base_probability
```

---

### Gap 7: Message Bus Protocol

```python
# src/infrastructure/message_schemas.py

class BidRequest(BaseModel):
    """Buyer → Seller/Exchange"""
    request_id: str
    buyer_id: str
    campaign_id: str
    channel: str
    impressions_requested: int
    max_cpm: float
    targeting: dict
    timestamp: datetime

class BidResponse(BaseModel):
    """Seller → Buyer/Exchange"""
    request_id: str
    seller_id: str
    offered_cpm: float
    available_impressions: int
    deal_type: str  # "PG", "PD", "PA"
    timestamp: datetime

class DealConfirmation(BaseModel):
    """Final deal record"""
    deal_id: str
    request_id: str
    buyer_id: str
    seller_id: str
    impressions: int
    cpm: float
    total_cost: float
    exchange_fee: float  # 0 for Scenarios B, C
    scenario: str
    timestamp: datetime

# Redis Stream names
STREAMS = {
    "bid_requests": "rtb:requests",
    "bid_responses": "rtb:responses",
    "deals": "rtb:deals",
    "events": "rtb:events"
}
```

---

### Gap 8: IAB Adapter Integration

```python
# src/agents/buyer/adapter.py

from buyer_agent.flows import DealBookingFlow
from buyer_agent.models import BookingState

class SimulatedBuyerAgent:
    """Wraps IAB buyer-agent for simulation."""

    def __init__(self, agent_id: str, scenario: str):
        self.agent_id = agent_id
        self.scenario = scenario

        # Initialize IAB buyer flow
        self.booking_flow = DealBookingFlow()

        # Scenario-specific modifications
        if scenario == "B":
            # Inject hallucination possibility
            self.hallucinator = HallucinationInjector()
        elif scenario == "C":
            # Connect to Beads for persistence
            self.beads = BeadsClient()

    async def execute_campaign(self, campaign: Campaign) -> dict:
        """Execute a campaign using IAB buyer-agent flows."""

        # Convert our Campaign to IAB BookingState
        booking_state = BookingState(
            campaign_brief={
                "budget": campaign.total_budget,
                "objectives": {campaign.primary_kpi.value: True},
                "targeting": {
                    "channels": campaign.channels,
                    "audience": campaign.audience_segments
                }
            }
        )

        # Run the CrewAI flow
        result = await self.booking_flow.kickoff(booking_state)

        # Extract results
        return {
            "deals": result.booked_lines,
            "spend": result.total_cost,
            "errors": result.errors
        }
```

---

### Gap 9: Ledger Schema (Scenario C)

```python
# src/infrastructure/ledger.py

class LedgerEntry(BaseModel):
    """Immutable record for Sui-proxy ledger."""

    # Identifiers (like Sui Object ID)
    entry_id: str  # UUID
    blob_id: str  # Simulates Walrus blob reference

    # Transaction data
    transaction_type: str  # "bid_request", "bid_response", "deal", "delivery"
    payload: dict  # Full transaction data

    # Provenance
    created_by: str  # Agent ID
    created_at: datetime

    # Immutability simulation
    hash: str  # SHA256 of payload
    previous_hash: str  # Chain link

    # Gas estimation
    estimated_sui_gas: float  # Based on payload size
    estimated_walrus_cost: float  # Based on blob size

class Ledger:
    """Simulated Sui blockchain with Walrus blob storage."""

    def __init__(self, db_path: str):
        self.db = PostgresClient(db_path)
        self.chain = []  # In-memory chain

    async def write(self, entry: LedgerEntry) -> str:
        """Write entry to ledger (atomic, immutable)."""
        # Calculate hash
        entry.hash = hashlib.sha256(
            json.dumps(entry.payload).encode()
        ).hexdigest()

        # Link to previous
        if self.chain:
            entry.previous_hash = self.chain[-1].hash

        # Persist
        await self.db.insert("ledger", entry.dict())
        self.chain.append(entry)

        # Estimate gas costs
        payload_size = len(json.dumps(entry.payload))
        entry.estimated_sui_gas = self._estimate_gas(payload_size)
        entry.estimated_walrus_cost = self._estimate_walrus(payload_size)

        return entry.entry_id
```

---

### Gap 10: Phase Dependencies

```
Phase 1: Infrastructure (NO DEPENDENCIES - can start immediately)
├── infra-repo: None
├── infra-db: None
├── infra-redis: None
├── infra-metrics: None
└── infra-beads: None

Phase 2: Agent Adapters (DEPENDS ON Phase 1)
├── agent-buyer: Requires infra-db, infra-redis
├── agent-seller: Requires infra-db, infra-redis
├── agent-exchange: Requires infra-db, infra-redis
└── agent-ucp: Requires infra-db

Phase 3: Scenario Engines (DEPENDS ON Phase 2)
├── scenario-a: Requires agent-buyer, agent-seller, agent-exchange
├── scenario-b: Requires agent-buyer, agent-seller, agent-ucp
└── scenario-c: Requires agent-buyer, agent-seller, infra-beads

Phase 4: Orchestration (DEPENDS ON Phase 3)
├── orch-convoy: Requires all scenario engines
├── orch-ground-truth: Requires infra-db
└── orch-logging: Requires infra-metrics

Phase 5: Simulation & Analysis (DEPENDS ON Phase 4)
├── sim-runner: Requires all Phase 4
├── sim-metrics: Requires orch-logging
└── sim-content: Requires sim-metrics
```

---

### Beads Type Definitions

```jsonl
{"type": "campaign", "fields": {"id": "string", "buyer_id": "string", "budget": "number", "kpi": "string", "status": "string"}}
{"type": "deal", "fields": {"id": "string", "campaign_id": "string", "seller_id": "string", "impressions": "number", "cpm": "number", "scenario": "string"}}
{"type": "bid", "fields": {"id": "string", "campaign_id": "string", "seller_id": "string", "offered_cpm": "number", "status": "string"}}
{"type": "transaction", "fields": {"id": "string", "deal_id": "string", "buyer_spend": "number", "seller_revenue": "number", "exchange_fee": "number"}}
{"type": "hallucination", "fields": {"id": "string", "agent_id": "string", "claim_type": "string", "claimed_value": "string", "actual_value": "string"}}
{"type": "context-rot", "fields": {"id": "string", "agent_id": "string", "day": "number", "keys_lost": "array", "recovery_accuracy": "number"}}
```

---

### Gap 11: Docker Compose Configuration

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: rtb_sim
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-rtb_sim_dev}
      POSTGRES_DB: rtb_simulation
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/01-init.sql
      - ./postgres/ground_truth.sql:/docker-entrypoint-initdb.d/02-ground_truth.sql
      - ./postgres/ledger.sql:/docker-entrypoint-initdb.d/03-ledger.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rtb_sim"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  influxdb:
    image: influxdb:2.7
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: ${INFLUX_PASSWORD:-rtb_sim_dev}
      DOCKER_INFLUXDB_INIT_ORG: alkimi
      DOCKER_INFLUXDB_INIT_BUCKET: rtb_metrics
    ports:
      - "8086:8086"
    volumes:
      - influx_data:/var/lib/influxdb2

  grafana:
    image: grafana/grafana:10.2.0
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - influxdb
      - postgres

volumes:
  postgres_data:
  redis_data:
  influx_data:
  grafana_data:
```

---

### Gap 12: Python Dependencies (pyproject.toml)

```toml
[project]
name = "iab-agentic-ecosystem-simulation"
version = "0.1.0"
description = "Simulation of IAB Agentic RTB Framework"
requires-python = ">=3.11"

dependencies = [
    # Core
    "pydantic>=2.0",
    "asyncio>=3.4",

    # Database
    "sqlalchemy>=2.0",
    "asyncpg>=0.28",
    "alembic>=1.12",

    # Redis
    "redis>=5.0",

    # Metrics
    "influxdb-client>=1.38",

    # IAB Agent Dependencies
    "crewai>=0.86.0",
    "anthropic>=0.18",
    "litellm>=1.0",
    "httpx>=0.25",

    # Utilities
    "python-dotenv>=1.0",
    "structlog>=23.0",
    "typer>=0.9",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.1",
    "black>=23.0",
    "ruff>=0.1",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

### Gap 13: LLM Configuration & Cost Management

```python
# src/config/llm_config.py

import os
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """LLM configuration for simulation agents."""

    # Model selection (use cheaper models for simulation)
    orchestrator_model: str = "claude-3-haiku-20240307"  # Cheaper than Opus
    specialist_model: str = "claude-3-haiku-20240307"

    # Rate limiting
    max_requests_per_minute: int = 50
    max_tokens_per_request: int = 4000

    # Cost tracking
    cost_per_1k_input_tokens: float = 0.00025  # Haiku pricing
    cost_per_1k_output_tokens: float = 0.00125

    # Simulation mode (mock LLM for testing)
    mock_mode: bool = os.getenv("RTB_MOCK_LLM", "false").lower() == "true"

class LLMCostTracker:
    """Track LLM costs during simulation."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls_by_agent = {}

    def record(self, agent_id: str, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if agent_id not in self.calls_by_agent:
            self.calls_by_agent[agent_id] = {"input": 0, "output": 0, "calls": 0}
        self.calls_by_agent[agent_id]["input"] += input_tokens
        self.calls_by_agent[agent_id]["output"] += output_tokens
        self.calls_by_agent[agent_id]["calls"] += 1

    def total_cost(self, config: LLMConfig) -> float:
        input_cost = (self.total_input_tokens / 1000) * config.cost_per_1k_input_tokens
        output_cost = (self.total_output_tokens / 1000) * config.cost_per_1k_output_tokens
        return input_cost + output_cost

# IMPORTANT: For full simulation (30 days, 5 buyers, 5 sellers)
# Estimated LLM cost: $50-200 depending on decision frequency
# Use mock_mode=true for testing infrastructure without LLM costs
```

---

### Gap 14: Time Acceleration Implementation

```python
# src/orchestration/time_controller.py

import asyncio
from datetime import datetime, timedelta
from typing import Callable

class TimeController:
    """
    Controls simulated time acceleration.

    100x acceleration = 1 real second = 100 simulated seconds
    30 simulated days at 100x = 7.2 real hours
    """

    def __init__(self, acceleration: float = 100.0):
        self.acceleration = acceleration
        self.sim_start_real = datetime.now()
        self.sim_start_virtual = datetime(2025, 1, 1)  # Simulation epoch
        self.paused = False

    @property
    def current_sim_time(self) -> datetime:
        """Get current simulated time."""
        if self.paused:
            return self._paused_at
        real_elapsed = (datetime.now() - self.sim_start_real).total_seconds()
        sim_elapsed = real_elapsed * self.acceleration
        return self.sim_start_virtual + timedelta(seconds=sim_elapsed)

    @property
    def sim_day(self) -> int:
        """Get current simulation day (1-30)."""
        elapsed = self.current_sim_time - self.sim_start_virtual
        return elapsed.days + 1

    async def wait_sim_seconds(self, sim_seconds: float):
        """Wait for simulated time to pass."""
        real_seconds = sim_seconds / self.acceleration
        await asyncio.sleep(real_seconds)

    async def wait_sim_hours(self, sim_hours: float):
        await self.wait_sim_seconds(sim_hours * 3600)

    async def wait_until_day(self, day: int):
        """Wait until simulation reaches specified day."""
        target = self.sim_start_virtual + timedelta(days=day - 1)
        while self.current_sim_time < target:
            await asyncio.sleep(0.1)

    def schedule_at_day(self, day: int, callback: Callable):
        """Schedule callback to run at start of specified day."""
        # Implementation uses asyncio.create_task with wait_until_day
        pass
```

---

### Gap 15: Seed Data Generation

```python
# src/data/seed_data.py

import random
from typing import List

def generate_publishers(count: int = 5) -> List[dict]:
    """Generate realistic publisher data."""
    publishers = []
    templates = [
        {"name": "Premium News Network", "channels": ["display", "video"], "floor_cpm": 15.0},
        {"name": "Sports Media Group", "channels": ["display", "video", "ctv"], "floor_cpm": 20.0},
        {"name": "Entertainment Hub", "channels": ["display", "native"], "floor_cpm": 12.0},
        {"name": "Tech Review Sites", "channels": ["display", "video"], "floor_cpm": 18.0},
        {"name": "Lifestyle Network", "channels": ["display", "native", "video"], "floor_cpm": 14.0},
    ]

    for i, template in enumerate(templates[:count]):
        publishers.append({
            "id": f"pub-{i+1:03d}",
            "name": template["name"],
            "channels": template["channels"],
            "floor_cpm": template["floor_cpm"],
            "daily_avails": random.randint(1_000_000, 10_000_000),
            "audience_segments": random.sample([
                "sports_enthusiasts", "tech_early_adopters", "luxury_shoppers",
                "auto_intenders", "travel_planners", "health_conscious"
            ], k=random.randint(2, 4))
        })

    return publishers

def generate_campaigns(buyer_id: str, count: int = 10) -> List[dict]:
    """Generate realistic campaign data for a buyer."""
    campaigns = []
    kpis = ["impressions", "clicks", "conversions", "viewability"]

    for i in range(count):
        budget = random.choice([10000, 25000, 50000, 100000])
        campaigns.append({
            "id": f"camp-{buyer_id}-{i+1:03d}",
            "buyer_id": buyer_id,
            "name": f"Q1 Campaign {i+1}",
            "total_budget": budget,
            "daily_budget": budget / 30,
            "primary_kpi": random.choice(kpis),
            "target_impressions": int(budget / 0.015 * 1000),  # ~$15 CPM
            "target_cpm": random.uniform(10, 25),
            "channels": random.sample(["display", "video", "ctv", "native"], k=random.randint(1, 3)),
            "audience_segments": random.sample([
                "sports_enthusiasts", "tech_early_adopters", "luxury_shoppers"
            ], k=random.randint(1, 2)),
        })

    return campaigns
```

---

### Gap 16: Second-Price Auction Implementation

```python
# src/agents/exchange/auction.py

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Bid:
    bidder_id: str
    amount: float
    deal_type: str

class SecondPriceAuction:
    """
    Implements second-price (Vickrey) auction.

    Winner pays the second-highest bid + $0.01
    """

    def __init__(self, floor_price: float = 0.0, exchange_fee_pct: float = 0.15):
        self.floor_price = floor_price
        self.exchange_fee_pct = exchange_fee_pct

    def run(self, bids: List[Bid]) -> Optional[dict]:
        """Run auction and return winning bid details."""
        # Filter bids above floor
        valid_bids = [b for b in bids if b.amount >= self.floor_price]

        if not valid_bids:
            return None

        # Sort by amount descending
        sorted_bids = sorted(valid_bids, key=lambda b: b.amount, reverse=True)

        winner = sorted_bids[0]

        # Second price: pay second-highest bid or floor
        if len(sorted_bids) > 1:
            clearing_price = sorted_bids[1].amount + 0.01
        else:
            clearing_price = self.floor_price

        # Calculate exchange fee
        exchange_fee = clearing_price * self.exchange_fee_pct
        seller_receives = clearing_price - exchange_fee

        return {
            "winner_id": winner.bidder_id,
            "winning_bid": winner.amount,
            "clearing_price": clearing_price,
            "exchange_fee": exchange_fee,
            "seller_receives": seller_receives,
            "bid_count": len(bids),
            "valid_bid_count": len(valid_bids)
        }
```

---

### Gap 17: Gas Cost Estimation

```python
# src/infrastructure/gas_estimation.py

class GasEstimator:
    """
    Estimate Sui/Walrus costs based on ADS Explorer data.

    Based on ~262M impressions with observed transaction patterns.
    """

    # Sui gas costs (in SUI, approximate)
    SUI_BASE_GAS = 0.001  # Base transaction cost
    SUI_PER_BYTE = 0.0000001  # Per byte of data

    # Walrus blob costs (in SUI, approximate)
    WALRUS_BASE_COST = 0.0005  # Minimum blob storage
    WALRUS_PER_KB = 0.00001  # Per KB stored

    def estimate_transaction_cost(self, payload_bytes: int) -> dict:
        """Estimate cost to write a transaction to Sui + Walrus."""

        # Sui object creation
        sui_gas = self.SUI_BASE_GAS + (payload_bytes * self.SUI_PER_BYTE)

        # Walrus blob storage
        payload_kb = payload_bytes / 1024
        walrus_cost = self.WALRUS_BASE_COST + (payload_kb * self.WALRUS_PER_KB)

        # Total in SUI
        total_sui = sui_gas + walrus_cost

        # Convert to USD (assuming $1.50/SUI)
        sui_price_usd = 1.50
        total_usd = total_sui * sui_price_usd

        return {
            "sui_gas": sui_gas,
            "walrus_cost": walrus_cost,
            "total_sui": total_sui,
            "total_usd": total_usd,
            "per_impression_usd": total_usd  # Assuming 1 tx per impression
        }

    def estimate_campaign_blockchain_cost(
        self,
        impressions: int,
        batch_size: int = 1000
    ) -> dict:
        """
        Estimate total blockchain cost for a campaign.

        Assumes batching: 1 transaction per batch_size impressions.
        """
        transactions = impressions // batch_size
        avg_payload_bytes = 500  # Typical bid request JSON

        per_tx = self.estimate_transaction_cost(avg_payload_bytes)

        return {
            "total_transactions": transactions,
            "total_sui": per_tx["total_sui"] * transactions,
            "total_usd": per_tx["total_usd"] * transactions,
            "cost_per_1000_impressions": per_tx["total_usd"],
            "comparison_exchange_fee_per_1000": 2.50,  # Typical $15 CPM * 15% = $2.25
        }
```

---

### Gap 18: Environment Variables Template

```bash
# .env.example

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=rtb_sim
POSTGRES_PASSWORD=your_password_here
POSTGRES_DB=rtb_simulation

# Redis
REDIS_URL=redis://localhost:6379

# InfluxDB
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your_influx_token
INFLUX_ORG=alkimi
INFLUX_BUCKET=rtb_metrics

# Grafana
GRAFANA_PASSWORD=your_grafana_password

# LLM (for IAB agents)
ANTHROPIC_API_KEY=your_anthropic_key
RTB_MOCK_LLM=false  # Set to true for cost-free testing

# Simulation
RTB_TIME_ACCELERATION=100
RTB_HALLUCINATION_RATE=0.05
RTB_CONTEXT_DECAY_RATE=0.02

# Scenarios
RTB_EXCHANGE_FEE_PCT=0.15
RTB_DEFAULT_FLOOR_CPM=10.0
```

---

### Gap 19: Narrative Logger Format

```python
# src/logging/narrative_logger.py

class NarrativeLogger:
    """
    Generates human-readable, content-ready log entries.

    Output format suitable for articles/documentation.
    """

    def log_deal(self, deal: dict) -> str:
        """Generate narrative for a deal."""
        return f"""
## Deal Completed: {deal['deal_id']}

**Buyer**: {deal['buyer_id']} was looking to reach {deal['target_audience']}
with a budget of ${deal['budget']:,.2f}.

**Seller**: {deal['seller_id']} offered {deal['impressions']:,} impressions
at ${deal['cpm']:.2f} CPM on their {deal['channel']} inventory.

**Outcome**: After {deal['negotiation_rounds']} rounds of negotiation,
the deal closed at ${deal['final_cpm']:.2f} CPM.

**Fees**:
- Buyer paid: ${deal['buyer_total']:,.2f}
- Seller received: ${deal['seller_revenue']:,.2f}
- Intermediary extracted: ${deal['exchange_fee']:,.2f} ({deal['fee_pct']:.1%})

**Scenario**: {deal['scenario']}
"""

    def log_hallucination(self, event: dict) -> str:
        """Generate narrative for a hallucination event."""
        return f"""
## Hallucination Detected

**Agent**: {event['agent_id']} ({event['agent_type']})

**What they claimed**: {event['claim']}
**Reality**: {event['actual']}

**Impact**: This led to {event['consequence']}

**Day**: Simulation day {event['day']}
"""

    def log_context_rot(self, event: dict) -> str:
        """Generate narrative for context rot."""
        return f"""
## Context Rot Event

**Agent**: {event['agent_id']} lost memory on day {event['day']}

**Lost context**:
{chr(10).join(f'- {k}' for k in event['lost_keys'])}

**Recovery attempt**: {'Successful' if event['recovered'] else 'Failed'}
**Recovery accuracy**: {event['accuracy']:.1%}

**Impact on campaign**: {event['impact']}
"""
```

---

### Gap 20: Git Conflict Resolution Strategy

```markdown
## Git Strategy for Parallel Polecats

### Branch Naming
- Each polecat works on: `polecat/<component>/<bead-id>`
- Example: `polecat/infra-db/gt-abc123`

### Merge Strategy
1. Each polecat creates a PR when done
2. Refinery (Gastown's merge queue) handles merges
3. Conflicts are resolved in order of completion
4. If conflict detected:
   - Later polecat must rebase
   - Or escalate to human (Mayor creates bead)

### Protected Files
These files require human approval to merge:
- `docker/docker-compose.yml`
- `src/config/*`
- `.env.example`

### Automated Merges
These can merge automatically:
- New files in `src/agents/`
- New files in `src/scenarios/`
- Test files
- Documentation
```

---

## Final Adversarial Review: Remaining Gaps

### Gap 21: Entry Point / CLI

```python
# src/cli.py
"""
Main entry point for simulation.

Usage:
    python -m src.cli run --scenario a,b,c --days 30 --buyers 5 --sellers 5
    python -m src.cli status
    python -m src.cli report --format markdown
"""

import typer
from src.orchestration.run_simulation import SimulationRunner
from src.metrics.report_generator import ReportGenerator

app = typer.Typer()

@app.command()
def run(
    scenarios: str = "a,b,c",
    days: int = 30,
    buyers: int = 5,
    sellers: int = 5,
    campaigns_per_buyer: int = 10,
    acceleration: float = 100.0,
    mock_llm: bool = False
):
    """Run the RTB simulation."""
    runner = SimulationRunner(
        scenarios=scenarios.split(","),
        days=days,
        buyers=buyers,
        sellers=sellers,
        campaigns_per_buyer=campaigns_per_buyer,
        time_acceleration=acceleration,
        mock_llm=mock_llm
    )
    runner.run()

@app.command()
def status():
    """Show simulation status."""
    # Query database for current state
    pass

@app.command()
def report(format: str = "markdown"):
    """Generate comparative report."""
    generator = ReportGenerator()
    generator.generate(format=format)

if __name__ == "__main__":
    app()
```

---

### Gap 22: IAB Repo Installation

```bash
# scripts/install_iab_repos.sh
#!/bin/bash
set -e

# Clone IAB repos as dependencies
mkdir -p vendor/iab

# Clone buyer-agent
if [ ! -d "vendor/iab/buyer-agent" ]; then
    git clone https://github.com/IABTechLab/buyer-agent vendor/iab/buyer-agent
fi

# Clone seller-agent
if [ ! -d "vendor/iab/seller-agent" ]; then
    git clone https://github.com/IABTechLab/seller-agent vendor/iab/seller-agent
fi

# Clone agentic-direct (MCP server)
if [ ! -d "vendor/iab/agentic-direct" ]; then
    git clone https://github.com/IABTechLab/agentic-direct vendor/iab/agentic-direct
fi

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/vendor/iab/buyer-agent/src"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/vendor/iab/seller-agent/src"

echo "IAB repos installed. PYTHONPATH updated."
```

---

### Gap 23: Error Handling & Recovery

```python
# src/infrastructure/error_handling.py

from functools import wraps
import asyncio
from typing import TypeVar, Callable
import structlog

logger = structlog.get_logger()
T = TypeVar('T')

class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0

def with_retry(config: RetryConfig = RetryConfig()):
    """Decorator for retrying failed operations."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(config.max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    logger.warning(
                        "Operation failed, retrying",
                        func=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

class TransactionManager:
    """Ensures atomic transactions with rollback."""

    async def execute_with_rollback(self, operations: list, rollback_ops: list):
        """Execute operations, rollback on failure."""
        completed = []
        try:
            for op in operations:
                await op()
                completed.append(op)
        except Exception as e:
            logger.error("Transaction failed, rolling back", error=str(e))
            for rollback in reversed(rollback_ops[:len(completed)]):
                try:
                    await rollback()
                except Exception as re:
                    logger.error("Rollback failed", error=str(re))
            raise
```

---

### Gap 24: Checkpoint & Resume

```python
# src/orchestration/checkpoint.py

import json
from datetime import datetime
from pathlib import Path

class SimulationCheckpoint:
    """
    Checkpoint simulation state for crash recovery.

    Creates checkpoint every hour or on significant events.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    async def save(self, state: dict) -> str:
        """Save simulation state to checkpoint."""
        checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.json"

        with open(checkpoint_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "state": state
            }, f, indent=2, default=str)

        # Keep only last 10 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        for old in checkpoints[:-10]:
            old.unlink()

        return checkpoint_id

    async def load_latest(self) -> dict:
        """Load most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        with open(checkpoints[-1]) as f:
            return json.load(f)

    async def resume_from(self, checkpoint_id: str) -> dict:
        """Resume from specific checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")

        with open(checkpoint_file) as f:
            return json.load(f)
```

---

### Gap 25: Per-Phase Acceptance Criteria

```markdown
## Phase Acceptance Criteria

### Phase 1: Infrastructure ✓ DONE when:
- [ ] `docker-compose up -d` starts all services without errors
- [ ] `psql` can connect and query `campaigns` table (empty OK)
- [ ] `redis-cli PING` returns PONG
- [ ] `curl localhost:8086/health` returns OK (InfluxDB)
- [ ] `curl localhost:3000` shows Grafana login
- [ ] `bd list` shows no errors (Beads initialized)

### Phase 2: Agent Adapters ✓ DONE when:
- [ ] `pytest tests/agents/test_buyer.py -v` passes
- [ ] `pytest tests/agents/test_seller.py -v` passes
- [ ] `pytest tests/agents/test_exchange.py -v` passes
- [ ] Mock mode works without LLM API calls
- [ ] Redis Streams show messages flowing

### Phase 3: Scenario Engines ✓ DONE when:
- [ ] `python -m src.cli run --scenario a --days 1 --mock-llm` completes
- [ ] `python -m src.cli run --scenario b --days 1 --mock-llm` completes
- [ ] `python -m src.cli run --scenario c --days 1 --mock-llm` completes
- [ ] Each scenario produces different fee extraction metrics
- [ ] Logs show scenario-specific behavior

### Phase 4: Orchestration ✓ DONE when:
- [ ] Campaigns map to Gastown convoys
- [ ] Ground truth database populated automatically
- [ ] Hallucination detection catches injected errors
- [ ] Context rot events logged correctly
- [ ] Event and narrative logs generated

### Phase 5: Simulation & Analysis ✓ DONE when:
- [ ] Full 30-day simulation completes for all 3 scenarios
- [ ] Comparative metrics show expected differences
- [ ] Grafana dashboards display all KPIs
- [ ] Report generator produces markdown output
- [ ] 5+ article-worthy findings documented
```

---

### Gap 26: Test Templates

```python
# tests/conftest.py
"""Shared pytest fixtures for all tests."""

import pytest
import asyncio
from src.infrastructure.database import Database
from src.infrastructure.redis_bus import RedisBus

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """In-memory test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:")
    await db.connect()
    await db.create_tables()
    yield db
    await db.disconnect()

@pytest.fixture
async def mock_redis():
    """Mock Redis for testing."""
    from unittest.mock import AsyncMock
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value="msg-123")
    redis.xread = AsyncMock(return_value=[])
    yield redis

@pytest.fixture
def sample_campaign():
    """Sample campaign for testing."""
    return {
        "id": "test-camp-001",
        "buyer_id": "buyer-001",
        "total_budget": 10000,
        "daily_budget": 333.33,
        "primary_kpi": "impressions",
        "target_impressions": 1000000,
        "target_cpm": 15.0,
        "channels": ["display"],
        "scenario": "A"
    }

# tests/integration/test_scenario_a.py
"""Integration tests for Scenario A (rent-seeking exchange)."""

import pytest
from src.scenarios.scenario_a import ScenarioA

@pytest.mark.asyncio
async def test_exchange_extracts_fees(test_db, mock_redis, sample_campaign):
    """Verify exchange agent extracts expected fees."""
    scenario = ScenarioA(db=test_db, redis=mock_redis, exchange_fee_pct=0.15)

    result = await scenario.run_single_deal(
        buyer_id="buyer-001",
        seller_id="seller-001",
        impressions=1000,
        cpm=20.0
    )

    assert result["exchange_fee"] == pytest.approx(3.0)  # 15% of $20
    assert result["seller_receives"] == pytest.approx(17.0)
    assert result["buyer_paid"] == 20.0
```

---

### Gap 27: Monitoring & Debugging

```python
# src/monitoring/health.py

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
import structlog

app = FastAPI()
logger = structlog.get_logger()

# Metrics
DEALS_COMPLETED = Counter('rtb_deals_completed', 'Total deals completed', ['scenario'])
DEAL_LATENCY = Histogram('rtb_deal_latency_seconds', 'Deal completion latency')
HALLUCINATIONS = Counter('rtb_hallucinations', 'Detected hallucinations', ['agent_type'])
CONTEXT_ROT = Counter('rtb_context_rot_events', 'Context rot events', ['scenario'])

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.get("/debug/agents")
async def debug_agents():
    """Show current agent states for debugging."""
    # Query Redis for agent states
    pass

@app.get("/debug/stuck")
async def debug_stuck():
    """Identify stuck agents."""
    # Find agents that haven't reported in >5 minutes
    pass
```

---

### Gap 28: Report Generation

```python
# src/metrics/report_generator.py

from datetime import datetime
import pandas as pd
from jinja2 import Template

REPORT_TEMPLATE = """
# IAB Agentic RTB Simulation Report
Generated: {{ generated_at }}

## Executive Summary
- Total campaigns: {{ total_campaigns }}
- Simulation duration: {{ duration_days }} days
- Total spend: ${{ total_spend | round(2) }}

## Fee Extraction Comparison

| Scenario | Total Spend | Intermediary Take | Take Rate |
|----------|-------------|-------------------|-----------|
{% for row in fee_comparison %}
| {{ row.scenario }} | ${{ row.spend | round(2) }} | ${{ row.take | round(2) }} | {{ row.rate | round(1) }}% |
{% endfor %}

## Key Finding: Fee Reduction with Alkimi
Moving from Scenario A (current state) to Scenario C (Alkimi ledger):
- Fee reduction: {{ fee_reduction | round(1) }}%
- Cost savings per $100K spend: ${{ savings_per_100k | round(2) }}

## Campaign Goal Achievement

| Scenario | Campaigns | Hit Goal | Success Rate |
|----------|-----------|----------|--------------|
{% for row in goal_achievement %}
| {{ row.scenario }} | {{ row.total }} | {{ row.hit }} | {{ row.rate | round(1) }}% |
{% endfor %}

## Context Rot Impact (Scenario B)
- Day 1 goal achievement: {{ day1_achievement | round(1) }}%
- Day 30 goal achievement: {{ day30_achievement | round(1) }}%
- Degradation: {{ degradation | round(1) }} percentage points

## Hallucination Analysis

| Scenario | Total Decisions | Hallucinations | Rate |
|----------|-----------------|----------------|------|
{% for row in hallucination_rates %}
| {{ row.scenario }} | {{ row.decisions }} | {{ row.hallucinations }} | {{ row.rate | round(2) }}% |
{% endfor %}

## Blockchain Cost Analysis (Scenario C)
- Total Sui gas: {{ sui_gas | round(4) }} SUI
- Total Walrus storage: {{ walrus_cost | round(4) }} SUI
- USD equivalent: ${{ blockchain_cost_usd | round(2) }}
- Cost per 1000 impressions: ${{ cost_per_1k | round(4) }}

## Conclusion
{{ conclusion }}
"""

class ReportGenerator:
    def generate(self, format: str = "markdown") -> str:
        # Query metrics from InfluxDB/PostgreSQL
        data = self._gather_data()

        template = Template(REPORT_TEMPLATE)
        return template.render(**data, generated_at=datetime.now().isoformat())
```

---

### Gap 29: Database Migration Strategy

```python
# src/infrastructure/migrations/env.py
"""Alembic migration environment."""

from alembic import context
from src.infrastructure.database import Base
from sqlalchemy import create_engine
import os

config = context.config
target_metadata = Base.metadata

def run_migrations_online():
    connectable = create_engine(os.getenv("DATABASE_URL"))
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()

# Usage:
# alembic revision --autogenerate -m "Add campaigns table"
# alembic upgrade head
```

---

### Gap 30: CrewAI Configuration

```python
# src/config/crewai_config.py
"""
CrewAI configuration for simulation agents.

The IAB agents use CrewAI with Anthropic models.
We override for simulation efficiency.
"""

import os

# Override IAB agent defaults for simulation
CREWAI_CONFIG = {
    # Use faster/cheaper model for simulation
    "manager_llm": os.getenv("RTB_MANAGER_MODEL", "anthropic/claude-3-haiku-20240307"),
    "agent_llm": os.getenv("RTB_AGENT_MODEL", "anthropic/claude-3-haiku-20240307"),

    # Reduce iterations for speed
    "max_iterations": int(os.getenv("RTB_MAX_ITERATIONS", "5")),

    # Enable memory only for Scenario C
    "memory_enabled": os.getenv("RTB_SCENARIO") == "C",

    # Verbose logging for debugging
    "verbose": os.getenv("RTB_VERBOSE", "true").lower() == "true",

    # Temperature settings
    "temperature": {
        "manager": 0.3,
        "specialist": 0.5,
        "functional": 0.2
    }
}

def apply_simulation_overrides():
    """Apply simulation-specific overrides to CrewAI."""
    os.environ["CREWAI_MANAGER_MODEL"] = CREWAI_CONFIG["manager_llm"]
    os.environ["CREWAI_AGENT_MODEL"] = CREWAI_CONFIG["agent_llm"]
    os.environ["CREWAI_MAX_ITERATIONS"] = str(CREWAI_CONFIG["max_iterations"])
```

---

## Final Checklist Before Execution

- [x] Architecture diagram with all components
- [x] Three scenarios clearly defined (A, B, C)
- [x] Simulation parameters (5 buyers, 5 sellers, 50 campaigns, 30 days)
- [x] Key metrics with SQL queries
- [x] Phase breakdown with parallelization
- [x] File structure complete
- [x] Docker Compose with all services
- [x] Python dependencies (pyproject.toml)
- [x] LLM configuration with cost management
- [x] Time acceleration implementation
- [x] Seed data generation
- [x] All agent decision trees
- [x] Ground truth schema
- [x] Hallucination injection mechanism
- [x] Context rot simulation
- [x] Message bus protocol
- [x] IAB adapter integration
- [x] Ledger schema with gas estimation
- [x] Environment variables
- [x] Narrative logger format
- [x] Git strategy for parallel polecats
- [x] CLI entry point
- [x] IAB repo installation script
- [x] Error handling with retry/rollback
- [x] Checkpoint and resume capability
- [x] Per-phase acceptance criteria
- [x] Test templates and fixtures
- [x] Monitoring and health endpoints
- [x] Report generation template
- [x] Database migration strategy
- [x] CrewAI configuration overrides

**Plan is ready for Gastown execution.**
