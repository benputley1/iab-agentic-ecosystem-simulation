# IAB Agentic Ecosystem Simulation

Simulation environment comparing THREE scenarios for programmatic advertising:

| Scenario | Description | Exchange Role | State Persistence |
|----------|-------------|---------------|-------------------|
| **A: Current State** | Rent-seeking exchanges | Exchange agent extracts 10-20% fees | Centralized DB |
| **B: IAB Pure A2A** | Direct buyer<->seller per IAB spec | No exchange (passive infrastructure) | In-memory (context rot) |
| **C: Alkimi Ledger** | Beads -> Walrus, internal ledger -> Sui | Decentralized audit trail | Immutable records |

## Quick Start

```bash
# 1. Start infrastructure
cd docker
docker-compose up -d

# 2. Install IAB dependencies
./scripts/install_iab_repos.sh

# 3. Install Python dependencies
pip install -e ".[dev]"

# 4. Copy environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run simulation (mock mode for testing)
rtb-sim run --scenario a,b,c --days 1 --mock-llm
```

## Key Metrics

1. **Fee Extraction Comparison** - How much do intermediaries extract?
2. **Campaign Goal Achievement** - What % hit their KPIs?
3. **Context Rot Impact** - How does performance degrade over 30 days?
4. **Hallucination Rate** - How often do agents decide on fabricated data?

## Project Structure

```
├── docker/                 # Docker Compose + PostgreSQL schemas
├── src/
│   ├── infrastructure/    # Database, Redis, Ledger clients
│   ├── agents/           # Buyer, Seller, Exchange, UCP adapters
│   ├── scenarios/        # Scenario A, B, C engines
│   ├── orchestration/    # Time control, ground truth
│   ├── metrics/          # InfluxDB, KPIs
│   └── logging/          # Event + narrative loggers
├── tests/
└── vendor/iab/           # Cloned IAB repos
```

## Documentation

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for detailed specifications.
