# RTB Simulation - Polecat Context

You are a worker agent in the IAB Agentic RTB Simulation project.

## Your Mission

Build components that simulate programmatic advertising across 3 scenarios:
- **Scenario A**: Current state with rent-seeking exchanges (15% fee extraction)
- **Scenario B**: IAB Pure A2A (direct buyer<->seller, context rot simulation)
- **Scenario C**: Alkimi ledger-backed (Beads = immutable records, Sui/Walrus proxy)

## Project Structure

```
iab-agentic-ecosystem-simulation/
├── docker/                 # Docker Compose + PostgreSQL schemas
├── src/
│   ├── infrastructure/    # Database, Redis, Ledger clients
│   ├── agents/           # Buyer, Seller, Exchange, UCP adapters
│   ├── scenarios/        # Scenario A, B, C engines
│   ├── orchestration/    # Time control, ground truth, convoy mapping
│   ├── metrics/          # InfluxDB, KPIs, dashboards
│   └── logging/          # Event + narrative loggers
├── tests/                # Integration + hallucination tests
└── vendor/iab/           # Cloned IAB repos
```

## Critical Rules

1. **All state must be persisted** - Never hold important data only in memory
2. **Use Beads for state** - Create/update beads for every significant event
3. **Log comprehensively** - Both event logs AND narrative logs
4. **Test as you build** - Each component needs integration tests
5. **Follow the plan** - See IMPLEMENTATION_PLAN.md for detailed specs

## Key Metrics to Track

- Fee extraction per scenario
- Campaign goal achievement rate
- Context rot degradation (Scenario B)
- Hallucination rate per agent type
- Blockchain costs vs exchange fees (Scenario C)

## Key Repos to Reference

- IAB buyer-agent: https://github.com/IABTechLab/buyer-agent
- IAB seller-agent: https://github.com/IABTechLab/seller-agent
- IAB agentic-rtb-framework: https://github.com/IABTechLab/agentic-rtb-framework
- IAB agentic-direct: https://github.com/IABTechLab/agentic-direct

## When Done

1. Run tests: `pytest tests/`
2. Update your bead: `bd update <id> --status done`
3. Push to branch: `git push origin <your-branch>`
