# Scenario Comparison: Context Rot and Recovery

## Overview

This document describes how context rot and recovery mechanisms differ across the three simulation scenarios.

## The Core Insight

**All scenarios involve AI agents that experience context limitations.** The key difference is how each scenario handles recovery:

| Scenario | Description | Context Rot | Recovery Mechanism | Recovery Rate | Fees |
|----------|-------------|-------------|-------------------|---------------|------|
| **A** | Traditional Exchange | ✅ Yes | Exchange transaction logs | ~60% | 15% |
| **B** | Pure A2A (IAB spec) | ✅ Yes | None | 0% | 0% |
| **C** | Alkimi Ledger-Backed | ✅ Simulated | Immutable blockchain | 100% | 5% |

## Context Rot Model

### What is Context Rot?

AI agents have limited context windows. Over time:
- Transaction history gets truncated
- Partner relationship data decays
- Pending negotiations may be forgotten
- Agents may "hallucinate" based on stale data

### Parameters (Shared Base)

```python
decay_rate = 0.02          # 2% daily decay
restart_probability = 0.005 # 0.5% chance of full context wipe
grace_period_days = 3       # No decay first 3 days
```

By day 30: ~55% of original context remains (0.98^30 ≈ 0.545)

## Scenario Details

### Scenario A: Traditional Exchange

```
Agent Experience: Same context rot as B
Recovery: Exchange verifies transactions from logs
Result: ~60% of errors caught, net ~40% context loss
Cost: 15% exchange fees
```

**Real-world analog:** DSPs/SSPs using ML for optimization, with exchange providing audit trail and discrepancy resolution.

### Scenario B: Pure A2A

```
Agent Experience: Same context rot as A
Recovery: None (errors compound)
Result: 0% recovery, ~100% cumulative context loss
Cost: 0% fees (no intermediary)
```

**Real-world analog:** Direct agent-to-agent trading without verification layer. Each agent maintains own state with no reconciliation.

### Scenario C: Alkimi Ledger-Backed

```
Agent Experience: Context rot simulated to demonstrate recovery
Recovery: Full state recoverable from immutable ledger
Result: 100% recovery, 0% net context loss
Cost: 5% platform fee + minimal blockchain costs
```

**Real-world analog:** Alkimi's blockchain-backed exchange where transaction history is permanently recorded and always recoverable.

## Why This Matters

### For Buyers
- Scenario A: Pay 15% for partial protection against context issues
- Scenario B: Save 15% but risk compounding errors
- Scenario C: Pay 5% for complete protection

### For Sellers
- Scenario A: Rely on exchange for dispute resolution
- Scenario B: No third party for disputes
- Scenario C: Immutable record of all transactions

### For Campaign Performance
- Scenario A: ~60% context recovery → some campaign continuity
- Scenario B: 0% recovery → campaigns may drift from goals
- Scenario C: 100% recovery → campaigns stay on track

## Implementation

See `src/scenarios/context_rot.py` for the shared implementation.

Preset configurations:
- `SCENARIO_A_ROT_CONFIG` - Exchange recovery (~60%)
- `SCENARIO_B_ROT_CONFIG` - No recovery (0%)
- `SCENARIO_C_ROT_CONFIG` - Ledger recovery (100%)
