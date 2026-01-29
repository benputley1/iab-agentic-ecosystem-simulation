# AdFi Settlement Flow for A2A Advertising

> **How Alkimi's AdFi Pool Enables Near-Realtime Settlement**  
> Version 1.0 | 2026-01-29

---

## The Settlement Problem

### Current Industry Reality

| Stage | Time | Cumulative |
|-------|------|------------|
| Campaign execution | 30 days | 30 days |
| Data reconciliation | 15-30 days | 45-60 days |
| Dispute resolution | 15-30 days | 60-90 days |
| Payment processing | 30 days | 90-120 days |

**Publisher Reality:** Work done in January, paid in April/May.

### Why It Takes So Long

1. **No shared records** — Buyer and seller have different data
2. **Manual reconciliation** — Human review of discrepancies
3. **Net payment terms** — Advertisers pay on Net-60/90
4. **Dispute overhead** — Legal/commercial negotiations

---

## What is AdFi?

### Overview

**AdFi = Advertising Finance**

A receivables financing pool that:
- Provides publishers with early payment (Day 0-1)
- Uses stablecoin liquidity (USDC)
- Generates yield for liquidity providers
- Eliminates payment delays

### Key Metrics

| Metric | Value |
|--------|-------|
| Publisher network | 9,700+ |
| Daily impressions | 25M+ |
| Historical default rate | 0% |
| Target LP APY | 10-15% |
| Publisher discount | ~5% |
| Pool denomination | USD SUI stablecoin |

### Economic Model

```
Traditional Model:
  Publisher delivers ads → Waits 90 days → Gets $100

AdFi Model:
  Publisher delivers ads → Day 1: Gets $95 → Done
  
  The $5 difference:
    - ~3% goes to LPs as yield
    - ~2% covers operations/risk
```

---

## Integration with A2A and Sui

### The Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE A2A SETTLEMENT FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: DEAL NEGOTIATION (Day 0)                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Buyer Agent ←── A2A Protocol ──→ Seller Agent                       │  │
│  │                                                                       │  │
│  │  Deal terms negotiated:                                               │  │
│  │    - 5M impressions                                                   │  │
│  │    - $15 CPM                                                          │  │
│  │    - Total: $75,000                                                   │  │
│  │                                                                       │  │
│  │  Terms sealed to Sui → Immutable deal record created                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  PHASE 2: CAMPAIGN EXECUTION (Days 1-30)                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Every impression event:                                              │  │
│  │    1. Buyer Agent logs impression (local)                             │  │
│  │    2. Seller Agent logs impression (local)                            │  │
│  │    3. Both write sealed record to Walrus                              │  │
│  │                                                                       │  │
│  │  Daily summaries sealed to Sui:                                       │  │
│  │    - Day 1: 150K impressions, $2,250                                  │  │
│  │    - Day 2: 175K impressions, $2,625                                  │  │
│  │    - ...                                                              │  │
│  │    - Day 30: 180K impressions, $2,700                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  PHASE 3: AUTOMATED RECONCILIATION (Day 30)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Smart Contract:                                                      │  │
│  │    1. Unlock buyer's sealed records                                   │  │
│  │    2. Unlock seller's sealed records                                  │  │
│  │    3. Compare line by line                                            │  │
│  │    4. Calculate final amount: $75,000                                 │  │
│  │    5. Create sealed reconciliation record                             │  │
│  │                                                                       │  │
│  │  Result: Zero discrepancy (both read from same Walrus blobs)          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  PHASE 4: ADFI SETTLEMENT (Day 30-31)                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  PUBLISHER                 ADFI POOL                     ADVERTISER  │  │
│  │  ┌──────────┐             ┌────────────┐               ┌──────────┐  │  │
│  │  │ Request  │─────────────│            │               │          │  │  │
│  │  │ Early    │             │ Validate:  │               │ Net-90   │  │  │
│  │  │ Payment  │             │ - Sealed   │               │ Payment  │  │  │
│  │  └────┬─────┘             │   deal     │               │ Terms    │  │  │
│  │       │                   │ - Reconciled│               └────┬─────┘  │  │
│  │       │                   │   amount   │                    │        │  │
│  │       ▼                   │ - Publisher│                    │        │  │
│  │  ┌──────────┐             │   history  │                    │        │  │
│  │  │ RECEIVE  │◄────────────│            │                    │        │  │
│  │  │ $71,250  │ Day 1       │ Calculate: │                    │        │  │
│  │  │ (95%)    │             │ $75K × 95% │                    │        │  │
│  │  └──────────┘             │ = $71,250  │                    │        │  │
│  │                           │            │                    │        │  │
│  │                           │            │◄───────────────────│        │  │
│  │                           │ RECEIVE    │ Day 90             │        │  │
│  │                           │ $75,000    │                    │        │  │
│  │                           │ (100%)     │                    │        │  │
│  │                           │            │                    │        │  │
│  │                           │ LP YIELD:  │                    │        │  │
│  │                           │ $3,750     │                    │        │  │
│  │                           │ (5% on     │                    │        │  │
│  │                           │ 90 days)   │                    │        │  │
│  │                           └────────────┘                    │        │  │
│  │                                                             │        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How AdFi Validates Payments

### On-Chain Verification

```move
module adfi_settlement {
    public fun request_early_payment(
        publisher: address,
        sealed_deal: &SealedDeal,
        reconciliation: &ReconciliationRecord,
        pool: &mut AdFiPool
    ): USDC {
        // 1. Verify deal is reconciled
        assert!(reconciliation.status == RECONCILED, E_NOT_RECONCILED);
        
        // 2. Verify publisher is party to deal
        assert!(sealed_deal.seller == publisher, E_NOT_SELLER);
        
        // 3. Verify amount matches
        let amount = reconciliation.final_amount;
        
        // 4. Check publisher history (credit scoring)
        let credit_score = pool.get_credit_score(publisher);
        assert!(credit_score >= MIN_SCORE, E_LOW_CREDIT);
        
        // 5. Calculate early payment (apply discount)
        let discount_rate = pool.get_discount_rate(credit_score);
        let early_payment = amount * (1 - discount_rate);
        
        // 6. Transfer from pool
        pool.transfer_usdc(publisher, early_payment);
        
        // 7. Create receivable record
        pool.create_receivable(
            sealed_deal.buyer,
            amount,
            sealed_deal.due_date
        );
        
        early_payment
    }
}
```

### Risk Management

| Risk | Mitigation |
|------|------------|
| **Advertiser default** | Fortune 500 credit backing, 0% historical default |
| **Reconciliation dispute** | Shared ledger = no disputes |
| **Publisher fraud** | On-chain delivery verification |
| **Pool liquidity** | Staged rollout ($2M → $20M) |
| **Concentration risk** | Publisher caps, diversification requirements |

---

## Comparison: Traditional vs AdFi Settlement

| Aspect | Traditional | AdFi + Sui |
|--------|-------------|------------|
| **Settlement time** | 90-120 days | 0-1 days |
| **Reconciliation** | Manual, error-prone | Automated, exact |
| **Disputes** | Common (5-15%) | Zero |
| **Working capital** | Publisher bears cost | Pool provides liquidity |
| **Audit trail** | Fragmented | Complete on-chain |
| **Credit risk** | On publisher | On pool (diversified) |

---

## LP Economics

### Yield Calculation

```
Publisher discount: 5%
Payment timing: 90 days early
Annualized yield: 5% × (365/90) ≈ 20%

After pool expenses and reserves:
Target LP APY: 10-15%
```

### Risk-Adjusted Returns

| Scenario | Probability | Return |
|----------|-------------|--------|
| Normal operation | 95% | 10-15% APY |
| Minor defaults | 4% | 5-10% APY |
| Severe stress | 1% | 0-5% APY |

**Monte Carlo simulation (10,000 iterations):**
- Expected return: 12.3% APY
- 5th percentile: 8.1% APY
- Loss probability: 0.5-1.4%

---

## Pool Funding Strategy

### Staged Rollout

| Quarter | Target | Source | Purpose |
|---------|--------|--------|---------|
| Q1 2026 | $2M | Treasury | Prove model |
| Q2 2026 | $4M | First external LPs | Scale cautiously |
| Q3-Q4 2026 | $8-20M | Institutional LPs | Full scale |

### Why Start Small

1. **Prove economics** — Validate yield projections
2. **Build track record** — Establish 0% default history
3. **Refine operations** — Optimize smart contracts
4. **Manage risk** — Limit exposure during learning phase

---

## Technical Integration

### Smart Contract Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADFI SMART CONTRACTS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │  POOL MANAGER   │     │  CREDIT SCORER  │                   │
│  │                 │     │                 │                   │
│  │  - Deposit USDC │     │  - Publisher    │                   │
│  │  - Withdraw     │     │    history      │                   │
│  │  - Rebalance    │     │  - Default      │                   │
│  │                 │     │    prediction   │                   │
│  └────────┬────────┘     └────────┬────────┘                   │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       │                                         │
│              ┌────────▼────────┐                               │
│              │  SETTLEMENT     │                               │
│              │  ENGINE         │                               │
│              │                 │                               │
│              │  - Validate deal│                               │
│              │  - Calculate    │                               │
│              │    early payment│                               │
│              │  - Transfer USDC│                               │
│              │  - Create       │                               │
│              │    receivable   │                               │
│              └────────┬────────┘                               │
│                       │                                         │
│              ┌────────▼────────┐                               │
│              │  RECEIVABLES    │                               │
│              │  TRACKER        │                               │
│              │                 │                               │
│              │  - Outstanding  │                               │
│              │    amounts      │                               │
│              │  - Due dates    │                               │
│              │  - Collection   │                               │
│              │    status       │                               │
│              └─────────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Walrus Integration

```
Deal Records:
  - Stored as Walrus blobs
  - Referenced by Sui objects
  - Immutable after creation
  
Delivery Events:
  - Aggregated daily
  - Sealed and stored to Walrus
  - Used for reconciliation
  
Settlement Proofs:
  - Reconciliation record
  - Payment transaction
  - LP allocation
```

---

## Benefits Summary

### For Publishers

| Before AdFi | With AdFi |
|-------------|-----------|
| Wait 90+ days | Paid in 1 day |
| Cash flow pressure | Predictable revenue |
| Reconciliation disputes | Zero disputes |
| Working capital cost | 5% discount (vs cost of capital) |

### For Advertisers

| Before AdFi | With AdFi |
|-------------|-----------|
| Reconciliation overhead | Automated |
| Relationship friction | Smooth payments |
| Audit complexity | On-chain trail |

### For LPs

| Investment Type | AdFi |
|-----------------|------|
| Risk | 0% historical default |
| Return | 10-15% target APY |
| Liquidity | Continuous redemption |
| Transparency | On-chain visibility |

---

## Conclusion

AdFi transforms advertising settlement from:
- **90-120 day wait** → **0-1 day payment**
- **Manual reconciliation** → **Automated smart contracts**
- **Fragmented disputes** → **Shared ledger = zero disputes**
- **Publisher cash flow squeeze** → **Immediate liquidity**

This is the settlement layer that makes IAB's A2A vision actually work in production.

---

*Document prepared for IAB Agentic Ecosystem Simulation v0.3.0*
