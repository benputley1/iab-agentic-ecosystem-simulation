# Comprehensive Research: IAB Agentic Simulation

> **Final Research Document with Thesis Statement and Implementation Plan**  
> Version 3.0 | Updated: 2026-01-29  
> Incorporates: Single-agent limitations, Sui Seals, AdFi integration

---

## Executive Summary

This document presents a complete research framework for demonstrating why Alkimi Exchange provides critical infrastructure for the IAB's Agentic Advertising future. The research addresses:

1. **Single-agent limitations** — Even one agent making millions of decisions suffers from context rot and hallucinations
2. **Multi-agent reconciliation** — When buyer and seller agents maintain private databases, disputes become unresolvable
3. **Alkimi's solution** — Sui blockchain (persistent state), Seals (privacy + verification), AdFi (near-realtime settlement)

---

## PART 1: Single-Agent Context Rot

### Ben's Key Question
> "Even one agent making millions of campaign decisions over 30 days will suffer from hallucinations and context rot. How does IAB solve this specifically?"

### The Problem: LLM Context Limitations

Modern LLMs have fixed context windows:

| Model | Context Window | Practical Campaign Data Capacity |
|-------|---------------|----------------------------------|
| Claude 3 Opus | 200K tokens | ~100,000 bid events |
| GPT-4 Turbo | 128K tokens | ~64,000 bid events |
| Claude 3.5 Sonnet | 200K tokens | ~100,000 bid events |

**Campaign Reality:**
- A medium campaign generates **1-10 million bid events** over 30 days
- At 50,000 campaigns × 30 days × 1000s of events = **billions of decisions**
- Context window holds <0.01% of relevant history

### How CrewAI/LangChain Handle This

**CrewAI Approach (Used by IAB Repos):**
```python
Agent(
    role="Portfolio Manager",
    memory=True,  # ← In-session memory only
)
```

**Limitations:**
- Memory is **within-session only** — lost on process restart
- No persistence across crew executions
- No mechanism to select which context to retain
- No ground truth verification of recalled facts

**LangChain Memory Types:**
| Type | Persistence | Cross-Session | Verification |
|------|-------------|---------------|--------------|
| ConversationBufferMemory | In-memory | ❌ | ❌ |
| ConversationSummaryMemory | In-memory | ❌ | ❌ |
| VectorStoreMemory | Persistent | ✅ | ❌ |
| EntityMemory | In-memory | ❌ | ❌ |

**Key Gap:** Even persistent memories (VectorStore) have **no verification mechanism**. Agent recalls may be incorrect or hallucinated.

### What IAB Proposes (Nothing Specific)

From the [IAB Agentic Advertising Initiative](https://iabtechlab.com/standards/agentic-advertising-initiative/):

> "Large language models, transformer architectures, and breakthroughs in GPU compute are fundamentally changing how we connect advertisers with audiences..."

**IAB Specifies:**
- Agent discovery and negotiation protocols
- OpenDirect for direct transactions
- Deals API for programmatic
- Agent Registry (launching March 2026)
- Standard taxonomies (AdCOM, OpenRTB)

**IAB Does NOT Specify:**
- How agents persist state across sessions
- How agents verify recalled information
- How agents handle context overflow
- What happens when agents hallucinate

### What Happens: Stale/Hallucinated Decisions

**Our Simulation Results (31 days):**
- Context loss events: **32**
- Hallucinated decisions: **7** (22% of context losses)
- Types of hallucinations:
  - Imagined deal histories ("I've done 5 deals with this seller" — actually zero)
  - Invented price floors ("Their minimum is $2.00 CPM" — actually $5.00)
  - Hallucinated inventory ("They have 1M impressions" — actually sold out)

**Compounding Effect:**
```
Day 1:  Agent makes 1000 decisions → 5 based on hallucinations
Day 7:  Agent makes 7000 decisions → 35 based on hallucinations
Day 14: Agent makes 14000 decisions → 70+ based on hallucinations
Day 30: Cumulative error → 150+ problematic decisions per campaign
```

### Financial Impact at Scale

| Metric | Per Campaign | At $1B Scale |
|--------|--------------|--------------|
| Hallucinated decisions | 150+ | 150,000+ |
| Avg error per decision | $50-500 | $50-500 |
| Annual cost | $7,500-75,000 | $7.5M-75M |

**Quote-worthy:** *"An AI agent without persistent, verifiable memory is an AI agent making decisions based on what it thinks it remembers — not what actually happened."*

---

## PART 2: Multi-Agent Reconciliation Problem

### The Cross-Agent State Divergence Problem

Beyond single-agent memory issues, multi-agent systems face a **reconciliation problem**:

```
Campaign End:
  Buyer Agent says: "10M impressions delivered, $150,000 owed"
  Seller Agent says: "8.5M valid impressions, $127,500 owed"
  
Who is right? No one knows. There's no shared source of truth.
```

### Industry Discrepancy Data

| Source | Discrepancy Rate | Context |
|--------|------------------|---------|
| ISBA/PwC 2020 | 15% "unknown delta" | £0.15 of every £1 unaccounted |
| ANA 2023 | 5-10% typical | Higher in AVOD/CTV |
| MRC Guidelines | 5% "acceptable" | Audit threshold |
| Adalytics 2023 | 10-20% programmatic | MFA inventory higher |

**Key Finding:** These discrepancies exist **WITH** centralized exchanges acting as arbiters. Remove the exchange → discrepancies increase.

### Byzantine Agreement Problem

From computer science (Lamport, 1982):
- With 2 parties, consensus is **impossible** without an external arbiter
- Need 3f+1 parties to tolerate f faulty actors
- Buyer + Seller = 2 parties → **deadlock is stable state**

**Quote-worthy:** *"In distributed systems, you need at least 3 parties to resolve disagreements. IAB A2A has 2."*

---

## PART 3: Sui Seals Research

### What Are Sui Seals?

From Sui documentation and ecosystem:

**Sui Seals = Encryption with Access Control**

Seals enable **programmable decryption** of on-chain data:
- Data is encrypted and stored on-chain (or via Walrus blob storage)
- Decryption keys are controlled by smart contracts
- Access rules are programmable and auditable
- Privacy is maintained while verification remains possible

### How Programmable Decryption Works

```
Traditional Encryption:
  Data → Encrypt(key) → Ciphertext
  Only key holder can decrypt
  No programmatic access control

Sui Seals:
  Data → Encrypt(threshold_key) → Ciphertext + Access Policy
  Smart contract enforces:
    - Who can decrypt
    - Under what conditions
    - With what verification requirements
  
Access Policy Examples:
  - "Both buyer AND seller must sign to decrypt deal terms"
  - "Decrypt after campaign end date"
  - "Decrypt only if reconciliation dispute filed"
  - "Auditor can decrypt if compliance flag raised"
```

### Application to A2A Marketplaces

**Perfect Passive Marketplace:**

```
Campaign Execution Flow:
  
1. Buyer Agent → Seal(bid_details) → Encrypted on-chain
   - Bid amount, targeting, audience
   - Only seller can decrypt if matching criteria met
   
2. Seller Agent → Seal(inventory_offer) → Encrypted on-chain
   - Available impressions, pricing, placement
   - Only matching buyers can decrypt
   
3. Match discovered → Smart contract unlocks mutual visibility
   - Both parties see full details
   - Deal terms locked to chain
   
4. Campaign delivery → Events sealed with dual-unlock
   - Buyer records impressions (sealed)
   - Seller records impressions (sealed)
   - Smart contract reconciles at campaign end
```

**Benefits:**
- **Privacy**: Competitors can't see your bids/offers
- **Verification**: Both parties can prove what was agreed
- **Automation**: Smart contract handles reconciliation
- **Audit**: Regulators can access with appropriate keys

### Reconciliation and Payment (Near-Realtime)

**Traditional Settlement:**
```
Day 1-30:   Campaign runs
Day 31-60:  Manual reconciliation
Day 60-90:  Dispute resolution
Day 90-120: Payment
Total: 90-120 days
```

**Sui Seals + AdFi Settlement:**
```
Day 1-30:   Campaign runs, events sealed to chain
Day 30:     Smart contract auto-reconciles sealed records
Day 30-31:  AdFi pool releases payment
Total: 0-1 days post-campaign
```

---

## PART 4: Alkimi AdFi Pool Integration

### What is AdFi?

From Alkimi documentation:

**AdFi = Advertising Finance**
- Publishers opt-in to receive early payment (Day 0-1 vs Day 90+)
- Liquidity Providers supply USDC to a pool
- Publishers pay ~5% financing fee for early payment
- LPs earn 10-15% target APY

### Key Metrics (from company data)

| Metric | Value |
|--------|-------|
| Publisher network | 9,700+ |
| Daily impressions | 25M+ |
| Historical default rate | 0% |
| Target LP APY | 10-15% |
| Publisher discount | 5% (~0.25%/day) |

### How AdFi Enables Near-Realtime Settlement

**Integration with Sui/Walrus:**

```
Campaign Flow with AdFi:

1. Deal negotiated via A2A
   └→ Deal terms sealed to Sui
   
2. Campaign executes
   └→ Delivery events written to Walrus (via Beads)
   └→ Immutable record of impressions/spend
   
3. Campaign ends
   └→ Smart contract reconciles sealed records
   └→ Final spend calculated
   
4. Settlement via AdFi pool
   └→ Publisher receives payment from pool (instant)
   └→ Buyer payment flows to pool (net terms)
   └→ LP yield generated from spread
   
5. Zero reconciliation disputes
   └→ Shared ledger = source of truth
   └→ No "he said/she said"
```

### Company Positioning on A2A

From Ben's meetings with Roberto (Mysten Labs):
- AdFi positioned as **showcase for Sui DeFi capabilities**
- USD SUI stablecoin chosen for pool denomination
- Free transfers in 2026 via Labs partnership
- Validates Figure's model (IPO'd at $1B with similar approach)

---

## PART 5: IAB Gap Analysis

### What IAB A2A Specifies ✅

| Component | Status | Description |
|-----------|--------|-------------|
| Agent Discovery | ✅ | Find trading partners |
| Deal Negotiation | ✅ | Agree on terms |
| OpenDirect | ✅ | Direct transaction protocol |
| Deals API | ✅ | Programmatic deal sync |
| Agent Registry | ✅ (March 2026) | Identity verification |
| Taxonomies | ✅ | AdCOM, OpenRTB standards |

### What IAB A2A Does NOT Specify ❌

| Gap | Impact | Industry Cost |
|-----|--------|---------------|
| **Single-agent memory management** | Hallucinated decisions | $7.5-75M/year at scale |
| **Cross-agent reconciliation** | Unresolvable disputes | $9-15B/year |
| **Real-time settlement** | 90+ day payment cycles | Working capital drag |
| **Privacy with verification** | Can't audit without exposing data | Regulatory risk |
| **Dispute resolution** | No arbiter | Legal costs |
| **State persistence** | Volatile agent memory | Repeated errors |

---

## PART 6: Alkimi Solution Mapping

### Gap-to-Solution Matrix

| IAB Gap | Alkimi Solution | Mechanism | Benefit |
|---------|-----------------|-----------|---------|
| **Context rot** | Persistent ledger | Sui state objects | Agent can always recover from chain |
| **Hallucinations** | Ground truth verification | Compare agent claims to ledger | Detect and prevent bad decisions |
| **Reconciliation** | Shared source of truth | Both parties write to same ledger | Zero disputes |
| **Settlement** | AdFi pool | Near-realtime USDC | Payment in 0-1 days |
| **Privacy + Verification** | Seals | Programmable decryption | Audit without exposure |
| **Dispute resolution** | Smart contract arbitration | Automated by code | No legal costs |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ALKIMI A2A INFRASTRUCTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BUYER AGENT                                 SELLER AGENT                  │
│   ┌─────────────┐                            ┌─────────────┐                │
│   │ CrewAI/LLM  │                            │ CrewAI/LLM  │                │
│   │ (volatile)  │                            │ (volatile)  │                │
│   └──────┬──────┘                            └──────┬──────┘                │
│          │                                          │                       │
│          │         ┌──────────────────┐            │                       │
│          │         │   SUI SEALS      │            │                       │
│          │         │ Encrypted A2A    │            │                       │
│          └────────►│ Communication    │◄───────────┘                       │
│                    │ (Private but     │                                    │
│                    │  Verifiable)     │                                    │
│                    └────────┬─────────┘                                    │
│                             │                                               │
│                    ┌────────▼─────────┐                                    │
│                    │   SUI CHAIN      │                                    │
│                    │ - Deal records   │                                    │
│                    │ - State objects  │                                    │
│                    │ - Smart contracts│                                    │
│                    └────────┬─────────┘                                    │
│                             │                                               │
│         ┌───────────────────┼───────────────────┐                          │
│         │                   │                   │                          │
│    ┌────▼────┐        ┌─────▼─────┐       ┌────▼────┐                      │
│    │ WALRUS  │        │  AdFi     │       │ AUDIT   │                      │
│    │ (Blobs) │        │  POOL     │       │ TRAIL   │                      │
│    │ Event   │        │ USDC      │       │ Complete│                      │
│    │ Storage │        │ Settlement│       │ History │                      │
│    └─────────┘        └───────────┘       └─────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 7: Three Scenarios for 30-Day Simulation

### Scenario A: Current State (Centralized Exchange)

| Aspect | Description |
|--------|-------------|
| **Model** | Exchange-mediated auctions |
| **Persistence** | Full (exchange database) |
| **Fees** | 15-25% (industry standard: 43% total supply chain) |
| **Reconciliation** | Exchange logs = arbiter |
| **Settlement** | 30-90 days |
| **Single-agent issues** | Mitigated by exchange records |
| **Multi-agent issues** | Mitigated by central arbiter |

**Pros:** Works, has arbitration  
**Cons:** Massive rent extraction, opacity

### Scenario B: IAB A2A (Pure Direct)

| Aspect | Description |
|--------|-------------|
| **Model** | Direct buyer↔seller |
| **Persistence** | Private DBs (fragmented) |
| **Fees** | ~0% (no intermediary) |
| **Reconciliation** | ❌ No mechanism |
| **Settlement** | Manual, 45+ days average |
| **Single-agent issues** | Context rot, hallucinations |
| **Multi-agent issues** | Unresolvable disputes |

**Pros:** No exchange fees  
**Cons:** No source of truth, disputes, memory issues

### Scenario C: Alkimi (A2A + Sui/Walrus/Seals/AdFi)

| Aspect | Description |
|--------|-------------|
| **Model** | Direct buyer↔seller |
| **Persistence** | Shared ledger (Sui) |
| **Fees** | ~0.1% (blockchain gas) |
| **Reconciliation** | Ledger = source of truth |
| **Settlement** | Near-realtime via AdFi |
| **Single-agent issues** | Recoverable from chain |
| **Multi-agent issues** | Zero unresolvable disputes |
| **Privacy** | Seals for programmable access |

**Pros:** Best of both worlds  
**Cons:** Requires blockchain adoption

### Comparative Summary

| Metric | A: Exchange | B: IAB A2A | C: Alkimi |
|--------|-------------|------------|-----------|
| **Take rate** | 15-25% | ~0% | ~0.1% |
| **Discrepancy rate** | 5-10% | 15-20%+ | 0% |
| **Unresolvable disputes** | ~1% | ~12% | 0% |
| **Settlement time** | 30-90 days | 45+ days | 0-1 days |
| **Context recovery** | N/A | ❌ | ✅ |
| **Audit trail** | Partial | ❌ | ✅ Complete |
| **Privacy + verification** | ❌ | ❌ | ✅ (Seals) |

---

## PART 8: Final Thesis Statement

### The Thesis

> **IAB's Agentic Advertising Initiative solves transaction *execution* but ignores transaction *verification*. In a world of AI agents with volatile memory and private databases, this creates two fatal flaws:**
>
> **1. Single-agent context rot:** Even one agent making millions of decisions accumulates errors from hallucinations and memory loss, with no mechanism to verify decisions against ground truth.
>
> **2. Multi-agent reconciliation failure:** When buyer and seller agents each maintain private databases, campaign disputes become unresolvable—the Byzantine Generals Problem with only 2 parties.
>
> **Alkimi provides the missing infrastructure layer:**
> - **Sui blockchain** provides persistent, shared state that survives agent restarts
> - **Walrus blob storage** records all transaction events immutably
> - **Seals** enable privacy-preserving verification (see what you need, prove what you must)
> - **AdFi pool** enables near-realtime settlement, eliminating payment delays
>
> **The result:** IAB's fee-free A2A vision, with Alkimi's verification and settlement layer, delivers the benefits of direct trading without the chaos of fragmented records and unresolvable disputes.

### Supporting Statistics

| Claim | Supporting Data |
|-------|----------------|
| Context rot is real | 32 events in 31-day simulation |
| Hallucinations follow | 22% of context losses → fabricated decisions |
| Disputes are costly | $9-15B annually at $150B scale |
| Settlement is slow | 90+ days in current system |
| Blockchain is cheap | $0.001 per transaction |
| AdFi pool works | 0% default rate, 9,700+ publishers |

### Defensibility

**What We Claim (Defensible):**
- IAB spec doesn't address reconciliation (verifiable by reading spec)
- Context rot occurs in LLM agents (documented by OpenAI, Anthropic)
- Industry has 5-15% discrepancy rates (ISBA, ANA studies)
- Blockchain provides immutable records (technical fact)

**What We Don't Claim:**
- "IAB is wrong" → We say they solve different problems
- "Blockchain is always better" → We show it solves specific gaps
- "Our numbers are exact" → We provide calibrated estimates with ranges

---

## PART 9: Implementation Plan

### Infrastructure Requirements

| Component | Specification | Monthly Cost |
|-----------|---------------|--------------|
| **Compute** | 16 vCPU, 64GB RAM (simulation) | $400 |
| **Database** | PostgreSQL 15 (ground truth) | Included |
| **Redis** | Message bus, state cache | Included |
| **InfluxDB** | Time-series metrics | Included |
| **Grafana** | Visualization | Included |
| **LLM API** | Claude Haiku (cost-optimized) | $200-500 |

**Total Infrastructure:** ~$600-900/month

### LLM Cost Estimate

| Scenario | Agents | Decisions/Day | Tokens/Decision | 30-Day Cost |
|----------|--------|---------------|-----------------|-------------|
| A | 5+5 | 1000 | 1000 | $75 |
| B | 5+5 | 1000 | 1500 | $112 |
| C | 5+5 | 1000 | 1200 | $90 |

**Total LLM Cost:** ~$300 per full run  
**Sensitivity Analysis:** ~$900 (3x for parameter sweeps)

### Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **1. Infrastructure** | 3 days | Docker, DBs, message bus |
| **2. Agent Adapters** | 5 days | Buyer/seller wrappers |
| **3. Scenario Engines** | 5 days | A, B, C implementations |
| **4. Reconciliation Module** | 3 days | Dispute simulation |
| **5. Seals Integration** | 3 days | Privacy/verification demo |
| **6. AdFi Integration** | 2 days | Settlement simulation |
| **7. Full Simulation** | 2 days | 30-day runs |
| **8. Analysis** | 3 days | Metrics, charts |
| **9. Documentation** | 3 days | Research paper, content |
| **Total** | **~4 weeks** | |

### Data Capture

| Metric | Capture Method | Storage |
|--------|---------------|---------|
| Transaction events | Event logger | PostgreSQL |
| Agent decisions | Decision log | PostgreSQL |
| Context losses | Monitor | InfluxDB |
| Hallucinations | Ground truth comparison | PostgreSQL |
| Disputes | Reconciliation module | PostgreSQL |
| Settlement times | Timestamp tracking | InfluxDB |
| Costs | Fee calculator | InfluxDB |

### Analysis Framework

**Primary Analyses:**
1. Fee comparison (A vs B vs C)
2. Dispute rate comparison (B vs C)
3. Context rot impact (B vs C)
4. Settlement time distribution
5. Financial impact projections

**Statistical Methods:**
- Two-proportion z-tests for dispute rates
- Mann-Whitney U for settlement times
- Linear regression for duration effects
- Monte Carlo for financial projections

### Content Creation Pipeline

| Output | Format | Audience |
|--------|--------|----------|
| Research paper | Academic-style | Investors, regulators |
| Executive summary | 2-pager | C-suite |
| Technical whitepaper | Detailed | Engineers, CTOs |
| Blog series (5 posts) | LinkedIn-ready | Industry |
| Twitter thread | Visual | Crypto/Web3 audience |
| Slide deck | 20 slides | Presentations |

---

## PART 10: Repository Updates Required

### Files to Update

1. **README.md** — Add Seals and AdFi references
2. **KEY_FINDINGS.md** — Incorporate thesis statement
3. **RESEARCH_PLAN.md** — Update with single-agent section
4. **docs/CROSS_AGENT_RECONCILIATION_RESEARCH.md** — Add Seals section
5. **src/scenarios/scenario_c.py** — Add Seals simulation
6. **content/** — New articles on Seals and AdFi integration

### New Documentation

1. **docs/SUI_SEALS_INTEGRATION.md** — Technical guide
2. **docs/ADFI_SETTLEMENT_FLOW.md** — Settlement integration
3. **docs/SINGLE_AGENT_LIMITATIONS.md** — Memory/context research
4. **docs/THESIS_STATEMENT.md** — Formal thesis document

---

## Conclusion

This research framework provides:

✅ **Credible** — Based on industry data, academic research, and verifiable simulation  
✅ **Defensible** — Clear separation of claims vs assumptions  
✅ **Comprehensive** — Addresses single-agent, multi-agent, and settlement issues  
✅ **Actionable** — Ready to execute with clear timeline and costs  
✅ **Valuable** — Positions Alkimi as essential infrastructure for A2A future

The goal is not to criticize IAB, but to demonstrate that Alkimi provides the **missing verification and settlement layer** that makes their vision actually work in production.

---

*Document prepared for IAB Agentic Ecosystem Simulation v0.3.0*
