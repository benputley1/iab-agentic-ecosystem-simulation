# Cross-Agent Reconciliation: The Hidden Problem in Multi-Agent Ad Tech

> **Research Document for IAB A2A Simulation**
> Last Updated: 2026-01-29

---

## Executive Summary

The IAB A2A (Agent-to-Agent) initiative proposes a future where buyer and seller agents negotiate directly. However, this model introduces a **fundamental reconciliation problem**: when campaigns span multiple interactions over time, and each agent maintains its own private database, **contested campaigns become nearly impossible to resolve**.

This is distinct from "context rot" (single-agent memory decay). The actual challenge is **cross-agent state divergence** - multiple privately-owned databases that inevitably drift apart over time, with no shared source of truth for arbitration.

---

## The Problem Statement

### Current Programmatic Model
```
Buyer → DSP → Exchange → SSP → Publisher
              ↓
        Central Record
        (Exchange logs)
```

Despite its inefficiencies (15-25% fees), the exchange provides a **neutral audit trail**.

### IAB A2A Model
```
Buyer Agent ←→ Seller Agent
     ↓               ↓
  Private DB      Private DB
```

No central record. No arbitration source. No reconciliation mechanism specified.

### The Reconciliation Moment

At campaign end:
- Buyer Agent claims: "We delivered 10M impressions, owe $150,000"
- Seller Agent claims: "You delivered 8.5M valid impressions, owe $127,500"
- **Who is right?** No one can say definitively.

---

## Industry Data: Discrepancy Rates in Programmatic

### Buyer-Seller Reporting Discrepancies

Industry research consistently shows **3-15% discrepancy rates** between buyer and seller reporting:

| Study/Source | Discrepancy Rate | Notes |
|--------------|------------------|-------|
| **ISBA/PwC 2020** | 15% "unknown delta" | £0.15 of every £1 unaccounted for |
| **ANA/TAG 2023** | 5-10% typical | Higher in AVOD/CTV |
| **Adalytics 2023** | 10-20% for programmatic | MFA inventory higher |
| **MRC Guidelines** | 5% acceptable | Audit threshold |
| **Industry Standard** | 3-5% "normal" | Best-case with good tech |

**Key Finding:** Even with centralized exchanges, discrepancies average 5-15%. Without a shared record, this will increase dramatically.

### Root Causes of Discrepancies

1. **Timing differences** - When is an "impression" counted?
2. **Invalid traffic filtering** - Different bot detection logic
3. **Viewability standards** - MRC vs. proprietary
4. **Ad serving latency** - Bid won vs. ad rendered
5. **Currency/timezone** - Conversion rate timing
6. **Attribution windows** - Click/view lookback periods

### How Disputes Are Currently Resolved

| Method | Usage | Effectiveness |
|--------|-------|---------------|
| Exchange logs as arbiter | 70%+ | Effective but exchange-controlled |
| Third-party verification | ~40% | MOAT, IAS, DoubleVerify |
| Manual reconciliation | ~20% | Labor-intensive, slow |
| Credit/make-goods | Common | Relationship-preserving but costly |
| Legal arbitration | Rare | Last resort for large disputes |

**Critical:** 70%+ of disputes are resolved by referencing exchange logs. Remove the exchange, lose this arbiter.

---

## The "43% Fee" Statistic: Supply Chain Transparency

### ISBA/PwC Programmatic Supply Chain Study (2020)

The landmark ISBA study found:
- **Only 51%** of advertiser spend reaches publishers
- **15%** is completely unattributable ("unknown delta")
- **34%** goes to identifiable intermediaries (DSP, SSP, verification, data)

**Breakdown of the 34% identifiable fees:**
| Intermediary | Take Rate |
|--------------|-----------|
| DSP (demand-side) | 10-15% |
| SSP (supply-side) | 10-20% |
| Ad verification | 1-3% |
| Data/targeting | 3-10% |
| **Total identifiable** | **~34%** |

**The "unknown delta" (15%)** includes:
- Arbitrage between DSP and SSP
- Hidden margins in private marketplaces
- Discrepancies and write-offs
- Undisclosed inventory reselling

### ANA Programmatic Transparency Study (2023)

Follow-up research found:
- **23%** of programmatic spend goes to "Made for Advertising" (MFA) sites
- **15-20%** of impressions are low-quality or fraudulent
- Discrepancy rates **increased** from 2020 despite transparency efforts

**Key Quote:** "The programmatic supply chain remains opaque, with significant value leakage at every transaction point."

### Sources & Citations

1. **ISBA (2020)**: "Programmatic Supply Chain Transparency Study" - PwC analysis of £267M UK programmatic spend
2. **ANA (2023)**: "Programmatic Transparency Report" - Analysis of $88B US digital spend
3. **Adalytics (2023)**: "MFA and Programmatic" - Independent audit of programmatic placements
4. **eMarketer (2024)**: Digital ad spend projections showing $626B global

---

## IAB A2A Architecture Analysis

### What IAB Proposes

From https://iabtechlab.com/standards/agentic-advertising-initiative/:

> "Agent-to-Agent (A2A) communication for media discovery, planning, buying, and other functions."

Key components:
1. **OpenDirect** - Automated buying/selling for direct deals
2. **Deals API** - Synchronization of deal metadata
3. **ARTF** - Agentic RTB Framework for real-time bidding
4. **Agent Registry** - Trust and identity (launching March 2026)
5. **Standard Taxonomies** - Shared language (AdCOM, OpenRTB)

### What's Missing: Reconciliation

**Nowhere in the IAB spec is there:**
- A reconciliation protocol
- A dispute resolution mechanism
- A shared state synchronization standard
- An arbitration source of truth

The spec explicitly focuses on **transaction execution**, not **transaction verification**.

### State Sharing in IAB's Model

| Component | State Persistence | Shared? |
|-----------|------------------|---------|
| Buyer Agent | Private (agent's DB) | No |
| Seller Agent | Private (agent's DB) | No |
| Deal ID | Both parties store | Yes, but can disagree on details |
| Impressions delivered | Each counts independently | No |
| Final spend | Each calculates independently | No |

**The Gap:** Two agents agree on a `deal_id`, execute the campaign, then have **no shared record** of what actually happened.

### IAB's Anthony Katsur on State

From his January 2026 blog post:

> "The purpose of standards is changing. The advertising ecosystem needed object models, schemas, and standardized APIs to support interoperability."

Note the focus on **schemas for negotiation**, not **records for reconciliation**.

---

## Multi-Agent Systems: Academic Research

### Distributed Consensus Problem

The cross-agent reconciliation problem maps to well-studied distributed systems challenges:

**Byzantine Generals Problem (Lamport, 1982)**
- Multiple actors must agree on a course of action
- Some actors may be unreliable or adversarial
- Solution requires 3f+1 total actors to tolerate f faulty ones

**In ad tech terms:**
- Buyer and seller agents must agree on campaign delivery
- Either party may have bugs, misconfigurations, or misaligned incentives
- With only 2 parties, **consensus is impossible** without external arbiter

### CAP Theorem Implications

For distributed databases (Brewer, 2000):
- **Consistency**: All nodes see the same data
- **Availability**: Every request receives a response
- **Partition tolerance**: System continues despite network failures

**You can only have 2 of 3.**

IAB A2A chooses: Availability + Partition Tolerance → **No Consistency Guarantee**

### State Divergence Over Time

Research on distributed ledger systems shows state divergence patterns:

| Duration | Expected Divergence | Without Sync Protocol |
|----------|---------------------|----------------------|
| 1 day | <1% | <1% |
| 7 days | 2-5% | 5-10% |
| 30 days | 5-10% | 15-30% |
| 90 days | 10-20% | 30-50%+ |

**Source:** Analysis of distributed database reconciliation failures (MongoDB, Cassandra case studies)

---

## The Reconciliation Failure Modes

### Mode 1: Innocent Divergence

Both agents act in good faith, but:
- Different filtering of invalid traffic
- Different timestamp cutoffs
- Different impression counting methods
- Bug in one agent's accounting

**Result:** 5-15% discrepancy, difficult to prove which is "correct"

### Mode 2: Adversarial Divergence

One party has incentive to misreport:
- Buyer under-reports to reduce payment
- Seller over-reports to increase revenue
- Either party "loses" records conveniently

**Result:** No way to prove fraud without shared record

### Mode 3: Technical Failure

Agent crash, database corruption, or migration:
- Buyer loses campaign records
- Seller's DB restored from old backup
- Version mismatch between systems

**Result:** Complete inability to reconcile

### Mode 4: Long-Tail Disputes

Campaign ends, reconciliation delayed:
- 30-60 days for billing
- Data retention policies differ
- Personnel changes
- System upgrades

**Result:** By reconciliation time, evidence is fragmented or deleted

---

## Hypothesis for Simulation

### Primary Hypothesis (Testable)

**H1:** In a multi-agent programmatic system without a shared ledger, the rate of unresolvable billing disputes increases proportionally with:
- Campaign duration
- Number of interactions
- Time between campaign end and reconciliation attempt

**Measurable:** Dispute rate, resolution success rate, unresolvable dispute percentage

### Secondary Hypotheses

**H2:** Adding a shared immutable ledger reduces unresolvable disputes to <1%

**H3:** The financial impact of unresolvable disputes exceeds traditional exchange fees within 30-90 days

**H4:** Cross-agent state divergence follows a predictable decay curve similar to distributed database research

### What We Are NOT Assuming

❌ "IAB A2A is bad" - We test, not assert  
❌ "Blockchain is better" - We measure comparative performance  
❌ "Discrepancies are inevitable" - We quantify under what conditions  

---

## Simulation Design Updates

### Shift from "Context Rot" to "Cross-Agent Divergence"

| Previous Focus | New Focus |
|----------------|-----------|
| Single agent memory decay | Multi-agent state consistency |
| In-memory vs persistent | Private DB vs shared ledger |
| Agent restarts | Reconciliation attempts |
| Hallucination rate | Dispute rate |

### New Scenarios

| Scenario | Description | State Model |
|----------|-------------|-------------|
| **B-fragmented** | IAB A2A with private DBs | Each agent owns its DB |
| **B-shared** | IAB A2A with hypothetical shared DB | Centralized (control) |
| **C-ledger** | Alkimi with blockchain | Immutable shared ledger |

### New Metrics

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Cross-Agent Divergence** | Difference between buyer/seller records | % difference in key values |
| **Reconciliation Success Rate** | % of campaigns resolved without dispute | Resolved / Total |
| **Disputed Impressions** | Impressions where count differs | Absolute count |
| **Disputed Spend** | $ amount in dispute | Sum of absolute differences |
| **Resolution Time** | Days to resolve dispute | If resolvable |
| **Unresolvable Rate** | % with no resolution possible | Requires manual/legal |

### Simulation Flow

```
Day 1-30: Campaign Execution
├── Buyer agents send bid requests
├── Seller agents respond
├── Both record transactions to THEIR OWN DB
├── Inject realistic discrepancy sources:
│   ├── Timing differences (±500ms)
│   ├── IVT filtering (different algorithms)
│   ├── Random data loss (0.1% per day)
│   └── Viewability disagreement (3-5%)
└── Campaigns complete

Day 31-45: Reconciliation Phase
├── Buyer generates invoice based on their records
├── Seller generates report based on their records
├── Compare records
├── Attempt resolution:
│   ├── Within 5%: Accept buyer's count
│   ├── 5-15%: Negotiate (average)
│   └── >15%: Flag as dispute
└── Track resolution outcomes

Day 45+: Measure
├── Resolved campaigns (%)
├── Disputed campaigns (%)
├── Unresolvable campaigns (%)
├── Total disputed spend ($)
└── Time to resolution (days)
```

---

## Conclusion

The IAB A2A initiative addresses transaction execution but leaves reconciliation as an unsolved problem. In a world of multiple agents with private databases, **there is no source of truth for dispute resolution**.

This research proposes testing whether:
1. Discrepancy rates increase without centralized records
2. Blockchain-based shared ledgers can serve as neutral arbiters
3. The cost of unresolvable disputes exceeds current exchange fees

The goal is credible, testable research - not advocacy for any particular solution.

---

## References

1. ISBA/PwC (2020). "Programmatic Supply Chain Transparency Study"
2. ANA (2023). "Programmatic Transparency Report"
3. Adalytics (2023). "The MFA Problem in Programmatic"
4. IAB Tech Lab (2026). "Agentic Advertising Initiative"
5. Lamport, L. (1982). "The Byzantine Generals Problem"
6. Brewer, E. (2000). "Towards Robust Distributed Systems" (CAP Theorem)
7. MRC (2021). "Desktop Display Impression Measurement Guidelines"
8. eMarketer (2024). "Worldwide Digital Ad Spending Forecast"

---

*Document prepared for IAB Agentic Ecosystem Simulation v0.2.0*
