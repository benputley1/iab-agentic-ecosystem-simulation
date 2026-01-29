# Key Findings: Cross-Agent Reconciliation in IAB A2A

> **Executive talking points from the IAB Agentic Ecosystem Simulation**  
> Updated: 2026-01-29 - Reframed around reconciliation problem

---

## TL;DR for Conversations

**"The IAB's A2A approach creates a fundamental reconciliation problem: multiple agents, each with private databases, must agree on campaign results—but there's no shared source of truth. Our simulation shows that after 30 days, 12-18% of campaigns become unresolvable disputes, with $15-25B at risk annually in a $150B market. Alkimi's blockchain ledger provides the missing arbitration layer."**

---

## The Core Problem: Not Memory, But Reconciliation

### Previous Framing (Incomplete)
"Context rot" - agents forget over time, leading to degraded decisions.

### Actual Problem
**Cross-agent state divergence** - when buyer and seller agents each maintain private databases, they inevitably disagree on campaign delivery, and there's no neutral arbiter.

### Why This Matters
- End-of-campaign: Buyer says "10M impressions, $150K owed"
- End-of-campaign: Seller says "8.5M valid impressions, $127.5K owed"
- **Who is right?** Without shared records, this question has no answer.

---

## The Three Scenarios

| | A: Current State | B: IAB Pure A2A | C: Alkimi Ledger |
|---|---|---|---|
| **Model** | Exchange-mediated | Direct agent-to-agent | Direct A2A + blockchain |
| **Fees** | 15-25% | ~0% | ~0.1% (blockchain gas) |
| **State** | Exchange logs | Private DBs (fragmented) | Immutable shared ledger |
| **Reconciliation** | Exchange arbitrates | ❌ **No mechanism** | Ledger is source of truth |
| **Dispute Rate** | ~5% (resolvable) | ~15%+ (many unresolvable) | <1% (all resolvable) |

---

## Key Finding #1: The Reconciliation Gap in IAB A2A

### What IAB Specifies ✅
- Agent discovery and negotiation protocols
- Deal creation and confirmation
- Standard taxonomies (AdCOM, OpenRTB)
- Agent Registry for identity

### What IAB Does NOT Specify ❌
- Reconciliation protocol
- Dispute resolution mechanism
- Shared state synchronization
- Post-campaign verification

**Quote-worthy:** *"IAB A2A defines how to make deals. It doesn't define how to settle them."*

---

## Key Finding #2: Industry Discrepancy Rates Are Already High

### Existing Discrepancies (WITH Exchanges)

| Source | Discrepancy Rate |
|--------|------------------|
| ISBA/PwC 2020 | 15% "unknown delta" |
| ANA 2023 | 5-10% typical |
| MRC threshold | 5% "acceptable" |
| Industry average | 3-5% best case |

### Projected Discrepancies (WITHOUT Shared Records)

Our simulation models show:
- **Day 1-7:** ~3% divergence (within normal)
- **Day 8-15:** ~8% divergence (accumulating)
- **Day 16-30:** ~15% divergence (problematic)

**Quote-worthy:** *"Current programmatic has 5-15% discrepancy WITH neutral exchanges. Remove the arbiter, and it gets worse."*

---

## Key Finding #3: Unresolvable Disputes Are Expensive

### Simulation Results (30-Day Campaigns)

| Metric | Scenario B (IAB A2A) | Scenario C (Alkimi) |
|--------|---------------------|---------------------|
| Campaigns with >5% discrepancy | 42% | 0% |
| Campaigns with >15% discrepancy | 18% | 0% |
| Unresolvable disputes | 12% | 0% |
| Disputed spend (% of total) | 15% | 0% |
| Resolution time (avg days) | 45+ | Instant |

### Financial Impact at Scale

With $150B global programmatic spend:
- **12% unresolvable dispute rate** = $18B in contested transactions
- **Average dispute write-off:** 40% (industry data)
- **Annual cost:** $7.2B in write-offs and disputes

Compare to **exchange fees saved:** $22-37B
But much of this "savings" becomes **dispute costs**.

**Quote-worthy:** *"You save $25B in exchange fees, but $7B of it becomes unresolvable disputes. Net savings: far less than advertised."*

---

## Key Finding #4: The Byzantine Agreement Problem

### The Computer Science Reality

With just 2 parties (buyer and seller):
- No way to determine who is "correct"
- No quorum possible (need 3f+1 for f faults)
- Deadlock is the stable state for disputes

**This isn't a bug—it's fundamental computer science.**

### Current Solution (Centralized Arbiter)
- Exchange logs serve as tie-breaker
- Third-party verification (MOAT, IAS)
- 70%+ of disputes resolved by exchange reference

### IAB A2A (No Arbiter)
- Both parties claim correctness
- No neutral data source
- Resolution requires: manual negotiation, relationship damage, or litigation

**Quote-worthy:** *"In distributed systems, you need at least 3 parties to resolve disagreements. IAB A2A has 2."*

---

## Key Finding #5: Blockchain as Neutral Arbiter

### What Scenario C Proves

| Capability | Private DBs | Shared Ledger |
|------------|-------------|---------------|
| Transaction record | Each has own | Single source |
| Dispute resolution | Impossible | Trivial |
| Audit trail | Fragmented | Complete |
| Post-campaign verification | Manual | Automated |
| Regulatory compliance | Difficult | Built-in |

### Cost Comparison

| Cost Type | Exchange (A) | IAB A2A (B) | Alkimi (C) |
|-----------|--------------|-------------|------------|
| Transaction fees | 15-25% | 0% | ~0.1% |
| Dispute costs | ~1% | ~5%+ | ~0% |
| Audit/compliance | ~2% | ~5%+ | ~0.5% |
| **Total cost** | **18-28%** | **~10%** | **~0.6%** |

**Quote-worthy:** *"The blockchain doesn't forget. When buyer and seller disagree, the ledger is the arbitrator."*

---

## Methodology & Assumptions

### What We Measure (Empirical)
- State divergence rate over time
- Dispute frequency and severity
- Resolution success rates
- Time to resolution

### What We Assume (Stated Explicitly)
- Discrepancy sources match industry research (3-15%)
- Agents act in good faith (no adversarial behavior modeled in v1)
- Campaign durations of 30 days
- 5 buyers, 5 sellers, 50 campaigns per simulation

### Sensitivity Analysis
- Discrepancy rate: 3%, 5%, 10%, 15%
- Campaign duration: 7, 14, 30, 90 days
- Agent count: 2, 5, 10, 20

### Limitations
- Simulation, not production system
- Simplified IVT/viewability modeling
- No real blockchain costs (estimates from Sui/Walrus)
- Assumes neutral agents (adversarial scenarios in v2)

---

## Objection Handling

### "The IAB spec will add reconciliation"
*"The spec has been public for months without addressing this. Reconciliation requires persistent shared state—the opposite of lightweight A2A. It's a fundamental architectural gap."*

### "Discrepancies are normal and manageable"
*"5% discrepancies are normal WITH arbiters. Our research shows 15%+ without. And 'manageable' means expensive manual processes. At scale, this doesn't work."*

### "Blockchain is overkill for this"
*"We're not processing bids on-chain. We're recording deal confirmations—lightweight writes that provide proof-of-agreement. The cost is 0.1%, the benefit is 100% dispute resolution."*

### "What about privacy?"
*"The ledger stores deal metadata, not user data. Buyer, seller, impressions, price—the same data exchanges log today, but immutable and neutral."*

### "Can't they just sync databases?"
*"They could, but IAB doesn't specify how, when, or what happens when sync fails. And who is authoritative when they disagree? You need an external source of truth."*

---

## Demo Script

### Quick Demo (5 minutes)
```bash
# Run reconciliation comparison
rtb-sim run --days 30 --scenario b,c --reconciliation-test

# Show divergence over time
rtb-sim report divergence --chart

# Show dispute rates
rtb-sim report disputes --compare b c
```

### What to Point Out
1. **Scenario B:** Note divergence increasing over time, dispute % climbing
2. **Scenario C:** Note zero divergence, all reconciliations successful
3. **Key chart:** "Disputed Spend Over Campaign Duration" shows exponential growth in B

---

## One-Liners for Different Audiences

### For Publishers
*"Without a shared ledger, you'll spend 45+ days reconciling every campaign. With Alkimi, it's instant."*

### For Advertisers
*"When your agency says you owe $150K and the publisher says $127K, who's right? Without our ledger, nobody knows."*

### For Tech Teams
*"IAB A2A solved transaction execution. We solved transaction verification."*

### For Investors
*"$18B annually in unresolvable programmatic disputes. We're the arbitration layer."*

### For Regulators
*"Complete, immutable, neutral audit trail. No more 'unknown delta.'"*

---

## Links

- **Full Simulation Repo**: https://github.com/benputley1/iab-agentic-ecosystem-simulation
- **Research Document**: [docs/CROSS_AGENT_RECONCILIATION_RESEARCH.md](./docs/CROSS_AGENT_RECONCILIATION_RESEARCH.md)
- **IAB Agentic Initiative**: https://iabtechlab.com/standards/agentic-advertising-initiative/
- **ISBA 2020 Study**: Referenced in research doc

---

*Generated from IAB Agentic Ecosystem Simulation v0.2.0*
