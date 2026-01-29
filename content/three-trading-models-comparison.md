---
title: "Three Trading Models: A Quantitative Comparison"
slug: three-trading-models-comparison
category: Analysis
keywords: ['programmatic', 'A2A trading', 'comparison', 'simulation']
description: "Exchange-based, A2A, and ledger-backed: we simulated all three trading models to measure performance, cost, and reliability."
generated: 2026-01-29T16:56:35.933902
---

# Three Trading Models: A Quantitative Comparison

*Exchange-based, A2A, and ledger-backed: we simulated all three trading models to measure performance, cost, and reliability.*

## Three Models, One Question

How should programmatic advertising work? We simulated three approaches:

1. **Scenario A: Exchange-Based** - Traditional model with centralized intermediary
2. **Scenario B: Pure A2A** - Direct agent-to-agent trading, no intermediary
3. **Scenario C: Ledger-Backed** - Direct trading with immutable record-keeping

## Methodology

We simulated 31 days of advertising activity with:

- 5 buyer agents (advertisers)
- 5 seller agents (publishers)
- 50 campaigns total
- Realistic market conditions and agent behavior

## Results Summary

| Metric | Exchange (A) | Pure A2A (B) | Ledger (C) |
|--------|-------------|--------------|------------|
| Deals | 504 | 547 | 478 |
| Spend | $453,079 | $403,521 | $442,687 |
| Fees | $67,962 | $0 | $0 |
| Infrastructure | $0 | $0 | $0.49 |
| Context Losses | 0 | 32 | 0 |
| Hallucinations | 0 | 7 | 0 |

## Key Findings

### 1,529 Deals Processed Across Three Trading Models

The simulation processed $1,299,286.97 in advertising transactions across 1,529 individual deals, demonstrating the viability of all three trading approaches.

Over 31 simulated days, each model successfully completed advertising transactions. Scenario A (exchange): 504 deals, $453,078.93. Scenario B (pure A2A): 547 deals, $403,520.67. Scenario C (ledger): 478 deals, $442,687.36. While all models achieved similar throughput, the critical differences lie in cost, reliability, and auditability.

**Key Data:**
- Total deals: 1,529
- Total spend: 1,299,286.97
- Scenario A deals: 504
- Scenario B deals: 547
- Scenario C deals: 478

## Conclusion

Each model has trade-offs:

- **Exchange**: Reliable but expensive (15% fees)
- **Pure A2A**: Free but unreliable (context rot, hallucinations)
- **Ledger**: Best of both - reliable, auditable, and cost-effective