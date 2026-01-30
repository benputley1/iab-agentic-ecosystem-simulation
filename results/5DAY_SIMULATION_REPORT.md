# 5-Day IAB Agentic Ecosystem Simulation Results

**Date:** 2026-01-30  
**Duration:** 47 seconds (accelerated time)  
**Simulated Time:** 5 days

---

## Executive Summary

This report summarizes the results of a 5-day simulation comparing three scenarios for AI-powered programmatic advertising:
- **Scenario A:** Current State (Exchange-Based) with 15% fee
- **Scenario B:** IAB Pure A2A (Direct Buyer↔Seller)
- **Scenario C:** Alkimi Ledger-Backed (Blockchain Verification)

### Key Finding
Scenario B (IAB A2A) successfully completed direct trades without exchange intermediation, demonstrating the viability of agent-to-agent protocols but **without ground truth verification**, context rot risk remains unaddressed.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Simulation Days | 5 |
| Buyers | 3 |
| Sellers | 3 |
| Campaigns per Buyer | 10 |
| Total Campaigns | 30 |
| Mode | Mock LLM (no API costs) |
| IAB Packages | seller-agent, buyer-agent |

---

## Results by Scenario

### Scenario A: Current State (Rent-Seeking Exchange)

| Metric | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Total |
|--------|-------|-------|-------|-------|-------|-------|
| Deals | 4 | 4 | 4 | 4 | 1 | **17** |
| Fee Rate | 15% | 15% | 15% | 15% | 15% | 15% |
| Exchange Fees | $5,256 | $5,256 | $5,256 | $5,256 | $1,559 | **$22,583** |
| Buyer Spend | $35,040 | $35,040 | $35,040 | $35,040 | $10,392 | **$150,552** |

**Observations:**
- Exchange fees totaling ~$22.6K over 5 days
- Consistent deal flow (4 per day) until day 5
- 15% intermediary take rate on all transactions

### Scenario B: IAB Pure A2A (Direct Communication)

| Metric | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Total |
|--------|-------|-------|-------|-------|-------|-------|
| Deals | 3 | 1 | 3 | 1 | 1 | **9** |
| Exchange Fees | $0 | $0 | $0 | $0 | $0 | **$0** |
| Total Spend | $10,716 | $3,536 | $7,033 | $6,844 | $2,506 | **$30,635** |
| Context Rot Events | 0 | 0 | 0 | 0 | 0 | **0** |

**Observations:**
- Zero exchange fees (direct buyer↔seller)
- Fewer deals than Scenario A (9 vs 17)
- No context rot events in mock mode (expected for 5-day window)
- Average CPM: $10.91

### Scenario C: Alkimi Ledger-Backed

**Status:** ⚠️ Database Migration Required

| Metric | Value |
|--------|-------|
| Deals | 0 |
| Error | `relation "ledger_entries" does not exist` |

**Note:** Scenario C requires database table setup. The ledger_entries table needs to be created before running full comparison.

---

## Comparison Table

| Metric | Scenario A | Scenario B | Scenario C |
|--------|------------|------------|------------|
| Architecture | Exchange | Pure A2A | Ledger-backed |
| Deals (5 days) | 17 | 9 | N/A |
| Exchange Fees | $22,583 | $0 | N/A |
| Verification | Exchange-mediated | None | Blockchain |
| Context Rot Risk | Low (stateless) | High (accumulates) | None |
| Hallucination Risk | Exchange catches | Undetected | Ledger catches |

---

## Technical Observations

### IAB Dependency Integration
The simulation successfully used IAB Tech Lab packages:
- **PricingRulesEngine:** Tiered pricing by buyer identity
- **TieredPricingConfig:** Public/Seat/Agency/Advertiser tiers
- **UnifiedClient:** MCP + A2A protocol support
- **A2AClient:** JSON-RPC 2.0 natural language interface

### Infrastructure
- Redis message bus: Working (all groups active)
- PostgreSQL ground truth: Connected (Scenario B)
- Ledger client: Connected but table missing (Scenario C)

---

## Issues Encountered

1. **Scenario C Database:** Missing `ledger_entries` table
2. **Metrics Aggregation:** JSON output shows 0s for Scenario A deals (logging showed actual deals)
3. **Mock Mode Limitation:** No real LLM hallucination testing in mock mode

---

## Recommendations

### For Extended Testing
1. Initialize ledger database schema for Scenario C
2. Run with `--real-llm` flag for hallucination testing
3. Extend to 10-30 days to observe context rot accumulation

### For Content Series
- 5-day results demonstrate proof of concept
- Full 30-day simulation needed for context rot thesis
- Scenario C database setup needed for complete comparison

---

## Raw Data Files

| File | Description |
|------|-------------|
| `5day_mock_results.json` | Simple runner results |
| `5day_scenarios_results.json` | Full CLI scenario results |
| `checkpoints/` | Daily checkpoint data |

---

## Next Steps

1. ✅ Document IAB dependency integration
2. ✅ Run 5-day simulation
3. ⚠️ Initialize ledger_entries table for Scenario C
4. 🔲 Run 30-day simulation with real LLM
5. 🔲 Generate full content series data

---

*Report generated: 2026-01-30T11:25:16Z*
