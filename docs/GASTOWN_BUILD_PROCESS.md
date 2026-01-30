# Gastown Build Process Documentation

> How the IAB Agentic Ecosystem Simulation was built using Gastown's parallel agent orchestration

---

## What is Gastown?

Gastown is a parallel AI agent orchestration system that enables multiple "polecat" (worker) agents to build software components simultaneously, coordinated by a "deacon" (supervisor) and "mayor" (coordinator).

### Key Components

| Component | Role |
|-----------|------|
| **Mayor** | High-level coordination, convoy management |
| **Deacon** | Supervises polecat workers, manages handoffs |
| **Polecat** | Individual worker agent building a component |
| **Beads** | State objects tracking work items |
| **Convoy** | Group of related beads working toward a goal |
| **Witness** | Monitors progress and quality |

---

## Build Timeline (from Git History)

The simulation was built in **5 phases**, with multiple polecats working in parallel:

### Phase 1: Infrastructure (Parallel)

| Commit | Polecat ID | Component |
|--------|------------|-----------|
| `cfd34b2` | - | Repository scaffold, pyproject.toml, docker-compose.yml |
| `5b69d49` | rs-uu0 | PostgreSQL schema for campaigns, deals, ground truth |
| `59fa47b` | rs-j5m | Redis Streams wrapper, A2A message routing |
| `814676d` | rs-s6q | InfluxDB + Grafana metrics infrastructure |
| `e7c610a` | rs-6lv | Custom Beads types: campaign, deal, bid, transaction |

### Phase 2: Agent Adapters (Parallel)

| Commit | Polecat ID | Component |
|--------|------------|-----------|
| `5195afc` | rs-93p | Buyer agent wrapper (IAB buyer-agent CrewAI flows) |
| `275ecd6` | rs-eu7 | Seller agent adapter |
| `aa2905b` | rs-3ev | Rent-seeking exchange with second-price auction |
| `81917c8` | rs-9bh | UCP embedding + hallucination injection/detection |

### Phase 3: Scenario Engines (Parallel)

| Commit | Polecat ID | Component |
|--------|------------|-----------|
| `d1445d8` | rs-6nld | Scenario A: Rent-seeking exchange simulation |
| `09e64ab` | rs-fl0o | Scenario B: IAB Pure A2A with context rot |
| `ab6b4e2` | rs-haoh | Scenario C: Alkimi ledger-backed exchange |

### Phase 4: Orchestration (Parallel)

| Commit | Polecat ID | Component |
|--------|------------|-----------|
| `e8336a7` | rs-umtc | Convoy sync for campaign-to-convoy mapping |
| `fcd1bfe` | rs-27rd | Ground truth repository for claim verification |
| `bad9877` | - | Orchestration logging (events + narratives) |

### Phase 5: Simulation & Analysis (Parallel)

| Commit | Polecat ID | Component |
|--------|------------|-----------|
| `9e0909a` | rs-6pzl | Simulation runner with time acceleration |
| `0042df8` | rs-spnr | KPI calculations, comparative reports |
| `868d7a5` | rs-in6w | Content generation module for article series |

### Phase 6: Research & Calibration (Sequential)

| Commit | Description |
|--------|-------------|
| `57842f0` | Reframe: context rot → cross-agent reconciliation |
| `96bc51d` | Calibrate discrepancy model to industry data (ISBA, ANA) |
| `aacf6d7` | Comprehensive research with Sui Seals + AdFi integration |
| `7adca57` | Final implementation plan |

---

## Polecat Naming Convention

Each polecat commit includes a tag like `(rs-XXXX)` indicating:
- `rs` = RTB Simulation project
- `XXXX` = Unique polecat/bead identifier

This allows tracking which worker built which component.

---

## Beads State Types

From `.beads/types.jsonl`:

```json
{"type": "campaign", "fields": {"id": "string", "buyer_id": "string", "budget": "number", "kpi": "string", "status": "string"}}
{"type": "deal", "fields": {"id": "string", "campaign_id": "string", "seller_id": "string", "impressions": "number", "cpm": "number", "scenario": "string"}}
{"type": "bid", "fields": {"id": "string", "campaign_id": "string", "seller_id": "string", "offered_cpm": "number", "status": "string"}}
{"type": "transaction", "fields": {"id": "string", "deal_id": "string", "buyer_spend": "number", "seller_revenue": "number", "exchange_fee": "number"}}
{"type": "hallucination", "fields": {"id": "string", "agent_id": "string", "claim_type": "string", "claimed_value": "string", "actual_value": "string"}}
{"type": "context-rot", "fields": {"id": "string", "agent_id": "string", "day": "number", "keys_lost": "array", "recovery_accuracy": "number"}}
```

---

## PRIME.md (Polecat Instructions)

Each polecat received context via `.beads/PRIME.md`:

```markdown
# RTB Simulation - Polecat Context

You are a worker agent in the IAB Agentic RTB Simulation project.

## Your Mission
Build components that simulate programmatic advertising across 3 scenarios:
- Scenario A: Current state with rent-seeking exchanges
- Scenario B: IAB Pure A2A (direct buyer↔seller, context rot)
- Scenario C: Alkimi ledger-backed (Beads = immutable records)

## Critical Rules
1. All state must be persisted
2. Use Beads for state
3. Log comprehensively
4. Test as you build
5. Follow the plan
```

---

## Build Efficiency

| Metric | Value |
|--------|-------|
| Total commits | 28 |
| Parallel phases | 5 |
| Sequential phases | 1 |
| Polecats used | 15+ |
| Build time (estimated) | ~4 hours |

The parallel architecture allowed multiple complex components to be built simultaneously, dramatically reducing total build time compared to sequential development.

---

## Key Insight: Agents Building Agent Simulations

This project demonstrates a meta-level concept: **AI agents (polecats) building a simulation to prove that AI agents need blockchain verification**.

The Gastown build process itself validates the thesis:
- Each polecat maintained private state during its work
- Coordination required explicit state synchronization (Beads)
- Without shared records, parallel work would have conflicted

---

*Documentation generated from git history and .beads/ artifacts*
