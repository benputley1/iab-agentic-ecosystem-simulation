# IAB Agentic Ecosystem Simulation

> **Demonstrating the fundamental challenges of IAB's A2A approach at scale, and how Alkimi's persistent blockchain ledger solves them definitively.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Executive Summary

This simulation environment compares **three approaches** to programmatic advertising:

| Scenario | Description | Fee Structure | State Persistence |
|----------|-------------|---------------|-------------------|
| **A: Current State** | Traditional exchanges with rent-seeking behavior | 10-20% intermediary fees | Centralized DB |
| **B: IAB Pure A2A** | Direct buyer↔seller per IAB Tech Lab spec | 0% (no exchange) | **In-memory only** 🔴 |
| **C: Alkimi Ledger** | Direct A2A + blockchain persistence | ~0.1% blockchain costs | **Immutable ledger** 🟢 |

### Key Finding

> **IAB's Pure A2A approach (Scenario B) suffers from "context rot" - agents progressively lose campaign context over time, leading to degraded performance and hallucinated decisions. Alkimi's ledger-backed approach (Scenario C) demonstrates zero context rot with perfect state recovery.**

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

## The Problem: Context Rot in IAB's A2A Approach

The IAB Tech Lab's [Agentic Advertising Initiative](https://iabtechlab.com/standards/agentic-advertising-initiative/) proposes direct agent-to-agent communication without traditional exchanges. While this eliminates intermediary fees, it introduces a critical flaw:

### What is Context Rot?

```
Day 1:  Agent Memory: [████████████████████] 100% - Full campaign context
Day 10: Agent Memory: [████████████░░░░░░░░]  82% - Some context lost
Day 20: Agent Memory: [████████░░░░░░░░░░░░]  67% - Significant degradation
Day 30: Agent Memory: [█████░░░░░░░░░░░░░░░]  55% - Campaign goals drift
```

**Causes:**
- In-memory state only (no persistent storage in IAB spec)
- Agent restarts wipe context completely
- LLM context windows have limits
- No single source of truth for disputes

**Impact:**
- Agents make decisions based on incomplete/stale data
- Campaign objectives drift from original goals
- No recovery mechanism when context is lost
- Regulatory compliance impossible (no audit trail)

## The Solution: Alkimi's Ledger-Backed Approach

Scenario C demonstrates how blockchain persistence solves these issues:

```
Day 1:  Agent Memory: [████████████████████] 100% ← Backed by ledger
Day 10: Agent Memory: [████████████████████] 100% ← Full recovery from ledger
Day 20: Agent Memory: [████████████████████] 100% ← Zero degradation
Day 30: Agent Memory: [████████████████████] 100% ← Perfect state maintained
```

**How it works:**
1. Every transaction recorded to immutable ledger (Sui/Walrus proxy)
2. Agent state fully recoverable from ledger at any time
3. Complete audit trail for compliance
4. Single source of truth for all parties

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

### 1. Fee Extraction Comparison

| Scenario | Take Rate | Where Fees Go |
|----------|-----------|---------------|
| A (Exchange) | 15-25% | Exchange intermediary |
| B (IAB A2A) | ~0% | No intermediary |
| C (Alkimi) | ~0.1% | Blockchain gas costs |

### 2. Context Rot Impact

| Scenario | Day 1 Memory | Day 30 Memory | Recovery Possible? |
|----------|--------------|---------------|-------------------|
| A | 100% | 100% (DB-backed) | Yes (centralized) |
| B | 100% | ~55% | **No** |
| C | 100% | 100% (ledger-backed) | **Yes (perfect)** |

### 3. Campaign Goal Achievement

Over a 30-day simulation:
- **Scenario A**: Stable performance, high fees reduce ROI
- **Scenario B**: Degrading performance, goals drift without persistence
- **Scenario C**: Stable performance, minimal fees, perfect audit trail

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
- IAB A2A eliminates this but introduces reliability risks
- **Alkimi provides the best of both worlds**: no intermediary fees + reliable state

### For Advertisers
- Campaign goals **drift by 30-45%** in pure A2A over 30 days
- Context rot means agents "forget" optimization learnings
- **Alkimi maintains perfect campaign fidelity** via immutable records

### For Regulators
- IAB A2A has **no audit trail** for compliance
- Disputes are unresolvable without ground truth
- **Alkimi provides complete, immutable audit capability**

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
