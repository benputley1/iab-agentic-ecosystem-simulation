# IAB Agentic Ecosystem Simulation

> **Demonstrating the cross-agent reconciliation problem in IAB's A2A approach, and how Alkimi's shared blockchain ledger provides the missing arbitration layer.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Executive Summary

This simulation environment compares **three approaches** to programmatic advertising:

| Scenario | Description | Fee Structure | State Model |
|----------|-------------|---------------|-------------|
| **A: Current State** | Traditional exchanges with rent-seeking behavior | 10-20% intermediary fees | Exchange as arbiter |
| **B: IAB Pure A2A** | Direct buyer↔seller per IAB Tech Lab spec | 0% (no exchange) | **Private DBs (fragmented)** 🔴 |
| **C: Alkimi Ledger** | Direct A2A + blockchain persistence | ~0.1% blockchain costs | **Shared immutable ledger** 🟢 |

### Key Finding

> **IAB's Pure A2A approach (Scenario B) creates a fundamental reconciliation problem: when buyer and seller agents each maintain private databases, they inevitably disagree on campaign delivery—and there's no source of truth to arbitrate. Our simulation shows 12-18% of campaigns become unresolvable disputes. Alkimi's ledger provides the missing arbitration layer with 100% resolution.**

## Quick Start

### Option 1: Quick Test (No Infrastructure Required)

```bash
# Clone the repo
git clone https://github.com/benputley1/iab-agentic-ecosystem-simulation.git
cd iab-agentic-ecosystem-simulation

# Install dependencies
pip install -e ".[dev]"

# Run quick test with mock mode (no APIs, no databases)
rtb-sim test-scenario --scenario c --skip-infra

# Run all scenarios for 1 day
rtb-sim run --days 1 --mock-llm --skip-infra
```

### Option 2: Full Simulation

```bash
# 1. Start infrastructure
cd docker
docker-compose up -d

# 2. Install IAB dependencies
./scripts/install_iab_repos.sh

# 3. Install Python dependencies
pip install -e ".[full]"

# 4. Copy environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run full 30-day simulation
rtb-sim run --scenario a,b,c --days 30 --mock-llm

# 6. View results in Grafana
open http://localhost:3000
```

## The Problem: Cross-Agent Reconciliation in IAB's A2A Approach

The IAB Tech Lab's [Agentic Advertising Initiative](https://iabtechlab.com/standards/agentic-advertising-initiative/) proposes direct agent-to-agent communication without traditional exchanges. While this eliminates intermediary fees, it introduces a critical flaw: **no mechanism for reconciliation**.

### What is Cross-Agent Reconciliation?

When a campaign ends:
- **Buyer Agent says:** "We delivered 10M impressions, total spend: $150,000"
- **Seller Agent says:** "You delivered 8.5M valid impressions, total spend: $127,500"
- **Who is right?** Without shared records, this question has no answer.

### Why Discrepancies Occur

Even with identical "ground truth," buyer and seller records diverge due to:
- **Timing differences:** When is an impression counted?
- **IVT filtering:** Different bot detection algorithms
- **Viewability:** MRC standard vs. proprietary
- **Data loss:** Random record loss over time
- **Currency/timezone:** Conversion timing differences

Industry data (ISBA 2020, ANA 2023) shows **5-15% discrepancy rates** even WITH centralized exchanges.

### The IAB Spec Gap

| IAB A2A Specifies ✅ | IAB A2A Does NOT Specify ❌ |
|---------------------|---------------------------|
| Agent discovery | Reconciliation protocol |
| Deal negotiation | Dispute resolution |
| Transaction execution | State synchronization |
| Standard taxonomies | Source of truth |

## The Solution: Alkimi's Shared Ledger

Scenario C demonstrates how a blockchain ledger solves reconciliation:

```
Campaign Execution:
  Buyer Agent ─────┬────── Seller Agent
                   │
              [Ledger]
                   │
          Single Source of Truth

End of Campaign:
  Buyer: "What's the ledger say?"
  Seller: "What's the ledger say?"
  Ledger: "10M impressions, $150,000"
  Both: "Agreed." ✓
```

**How it works:**
1. Every transaction recorded to immutable ledger at execution time
2. Both parties write to the SAME record
3. At reconciliation, ledger is the arbiter
4. Zero unresolvable disputes (source of truth exists)

## CLI Reference

```bash
# Run simulation
rtb-sim run [OPTIONS]
  --scenario, -s    Scenarios to run (a,b,c)          [default: a,b,c]
  --days, -d        Simulation days (1-30)            [default: 1]
  --buyers, -b      Number of buyer agents            [default: 5]
  --sellers         Number of seller agents           [default: 5]
  --mock-llm        Use mock LLM (no API costs)       [default: true]
  --skip-infra      Skip Redis/Postgres (use mocks)   [default: false]
  --output, -o      Output file for results JSON

# Test single scenario
rtb-sim test-scenario --scenario c --mock-llm

# Test ledger recovery (Scenario C feature)
rtb-sim test-recovery --agent test-buyer-001

# Generate comparison report
rtb-sim compare ./results.json --format markdown

# Show version
rtb-sim version
```

## Key Metrics

### 1. Fee Comparison

| Scenario | Take Rate | Where Fees Go |
|----------|-----------|---------------|
| A (Exchange) | 15-25% | Exchange intermediary |
| B (IAB A2A) | ~0% | No intermediary |
| C (Alkimi) | ~0.1% | Blockchain gas costs |

### 2. Reconciliation Success (NEW)

| Scenario | Discrepancy >5% | Discrepancy >15% | Unresolvable | Resolution Time |
|----------|----------------|------------------|--------------|-----------------|
| A (Exchange) | ~10% | ~3% | ~1% | 7-14 days |
| B (IAB A2A) | **~42%** | **~18%** | **~12%** | **45+ days** |
| C (Alkimi) | 0% | 0% | **0%** | **Instant** |

### 3. Financial Impact at Scale

With $150B global programmatic spend:

| Metric | Scenario B (Private DBs) | Scenario C (Shared Ledger) |
|--------|-------------------------|---------------------------|
| Annual disputed spend | $18-27B | ~$0 |
| Unresolvable disputes | $9-15B | ~$0 |
| Exchange fees saved | $22-37B | $22-37B |
| Blockchain fees | N/A | ~$150M |
| **Net benefit vs current** | **Uncertain** | **$22-37B** |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SIMULATION ENVIRONMENT                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │  SCENARIO A │     │  SCENARIO B │     │  SCENARIO C │                    │
│  │  (Exchange) │     │  (IAB A2A)  │     │   (Alkimi)  │                    │
│  ├─────────────┤     ├─────────────┤     ├─────────────┤                    │
│  │ • 15% fees  │     │ • 0% fees   │     │ • 0% fees   │                    │
│  │ • DB state  │     │ • In-memory │     │ • Ledger    │                    │
│  │ • Auction   │     │ • Direct    │     │ • Direct    │                    │
│  └─────────────┘     └─────────────┘     └─────────────┘                    │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             │                                                │
│                     ┌───────▼───────┐                                        │
│                     │   COMPARISON  │                                        │
│                     │    ENGINE     │                                        │
│                     └───────────────┘                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── src/
│   ├── cli/                  # CLI entry point (rtb-sim command)
│   ├── scenarios/
│   │   ├── scenario_a.py     # Current state (exchange)
│   │   ├── scenario_b.py     # IAB Pure A2A (context rot)
│   │   └── scenario_c.py     # Alkimi ledger-backed
│   ├── agents/
│   │   ├── buyer/            # Buyer agent wrappers
│   │   ├── seller/           # Seller agent adapters
│   │   ├── exchange/         # Exchange auction logic
│   │   └── ucp/              # Hallucination detection
│   ├── infrastructure/
│   │   ├── redis_bus.py      # Message routing
│   │   ├── ledger.py         # Sui/Walrus proxy
│   │   └── ground_truth.py   # Reality verification
│   └── orchestration/
│       └── run_simulation.py # Main simulation runner
├── docker/
│   ├── docker-compose.yml    # PostgreSQL, Redis, InfluxDB
│   └── grafana/dashboards/   # Visualization dashboards
├── tests/
└── content/                  # Article-ready findings
```

## Dashboards

Three Grafana dashboards are included:

1. **RTB Overview** - General simulation metrics
2. **Scenario Comparison** - Side-by-side A vs B vs C analysis
3. **Context Rot Analysis** - Deep dive into memory degradation

## IAB Tech Lab Integration

This simulation wraps the official IAB Tech Lab repositories:

- [buyer-agent](https://github.com/IABTechLab/buyer-agent)
- [seller-agent](https://github.com/IABTechLab/seller-agent)
- [agentic-rtb-framework](https://github.com/IABTechLab/agentic-rtb-framework)
- [agentic-audiences](https://github.com/IABTechLab/agentic-audiences)
- [agentic-direct](https://github.com/IABTechLab/agentic-direct)

Install them with:
```bash
./scripts/install_iab_repos.sh
```

## Key Findings for Stakeholders

### For Publishers
- **Current exchanges extract 15-25%** of every transaction
- IAB A2A eliminates fees but creates reconciliation nightmares
- **Alkimi provides the best of both worlds**: no intermediary fees + instant reconciliation

### For Advertisers
- With IAB A2A, **42% of campaigns have >5% billing discrepancy**
- **12% of campaigns become unresolvable disputes** (no arbiter)
- **Alkimi provides 100% reconciliation** via shared ledger

### For Regulators
- IAB A2A has **no neutral audit trail** for compliance
- "He said/she said" disputes are common without ground truth
- **Alkimi provides complete, immutable, neutral audit capability**

### For Finance Teams
- Annual disputed spend at scale: **$18-27B** in IAB A2A
- Average resolution time: **45+ days** (vs instant with ledger)
- Write-offs from unresolvable disputes: **$9-15B annually**

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check src/
black src/

# Run specific scenario test
python -m src.scenarios.scenario_c --mock-llm --skip-ledger
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **IAB Tech Lab**: [Agentic Advertising Initiative](https://iabtechlab.com/standards/agentic-advertising-initiative/)
- **Alkimi Exchange**: [alkimi.org](https://alkimi.org)
- **Analysis Document**: [ANALYSIS.md](./content/ANALYSIS.md)

---

*Built by [Alkimi Exchange](https://alkimi.org) to demonstrate the future of transparent, efficient programmatic advertising.*
