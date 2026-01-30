# Multi-Agent Isolation Architecture Plan

**Date:** 2026-01-30  
**Status:** Ready for Implementation  
**Priority:** Critical — current study has flawed architecture

---

## Problem Statement

The current Real Context Simulation has a fundamental flaw:

**Current:** Single shared `PricingAgent` with one `context_history` for all agents
**Reality:** Each agent should have isolated context, memory, and API calls

This matters because:
1. Context rot happens **per-agent**, not globally
2. Reconciliation failures occur when **buyer's memory ≠ seller's memory**
3. The IAB A2A problem is about **divergent state**, not just accumulation

---

## Required Changes

### 1. Isolated Agent Instances

```python
# CURRENT (wrong)
pricing_agent = PricingAgent()  # Shared across all buyers/sellers

# SHOULD BE
buyer_agents = {
    "buyer_1": PricingAgent(agent_id="buyer_1"),
    "buyer_2": PricingAgent(agent_id="buyer_2"),
    "buyer_3": PricingAgent(agent_id="buyer_3"),
}
seller_agents = {
    "seller_1": PricingAgent(agent_id="seller_1"),
    "seller_2": PricingAgent(agent_id="seller_2"),
    "seller_3": PricingAgent(agent_id="seller_3"),
}
```

### 2. Per-Agent Context Storage

Each agent needs:
- Own `context_history: List[Dict]`
- Own API call tracking (tokens, cost)
- Own hallucination metrics
- Own "database" (deal records they remember)

```python
@dataclass
class AgentState:
    agent_id: str
    agent_type: str  # "buyer" or "seller"
    context_history: List[Dict]  # LLM conversation history
    deal_records: Dict[str, DealRecord]  # Agent's "database"
    total_tokens: int
    total_cost: float
    hallucinations: int
```

### 3. Separate API Calls

Each pricing decision should be:
```python
# Buyer makes offer (buyer's context)
buyer_decision = buyer_agents["buyer_1"].make_bid(
    product=product,
    context=buyer_agents["buyer_1"].context_history
)

# Seller evaluates (seller's context - may differ!)
seller_decision = seller_agents["seller_1"].evaluate_offer(
    offer=buyer_decision,
    context=seller_agents["seller_1"].context_history
)
```

### 4. Divergent Database Records

When a deal completes:
```python
# Buyer records the deal (from buyer's perspective)
buyer_agents["buyer_1"].record_deal({
    "deal_id": deal_id,
    "agreed_cpm": buyer_decision.bid_cpm,  # What buyer thinks they agreed
    "impressions": buyer_impression_count,
    ...
})

# Seller records the deal (from seller's perspective)
seller_agents["seller_1"].record_deal({
    "deal_id": deal_id,
    "agreed_cpm": seller_accepted_cpm,  # What seller thinks they agreed
    "impressions": seller_impression_count,  # May differ!
    ...
})
```

### 5. Reconciliation Detection

After each deal, check for divergence:
```python
def check_reconciliation(buyer_record, seller_record) -> ReconciliationResult:
    discrepancies = []
    
    if buyer_record.agreed_cpm != seller_record.agreed_cpm:
        discrepancies.append(f"CPM mismatch: buyer={buyer_record.agreed_cpm}, seller={seller_record.agreed_cpm}")
    
    if buyer_record.impressions != seller_record.impressions:
        discrepancies.append(f"Impression mismatch: buyer={buyer_record.impressions}, seller={seller_record.impressions}")
    
    return ReconciliationResult(
        matched=len(discrepancies) == 0,
        discrepancies=discrepancies,
        buyer_context_tokens=buyer_agent.total_tokens,
        seller_context_tokens=seller_agent.total_tokens,
    )
```

---

## Impact on Results

### What We'll Measure

1. **Per-agent context accumulation**
   - Buyer 1: X tokens, Buyer 2: Y tokens, etc.
   - Different agents may hit pressure zones at different times

2. **Reconciliation failure rate**
   - % of deals where buyer and seller records don't match
   - Expected to increase as context pressure builds

3. **Divergence severity**
   - How far apart are the disagreeing values?
   - Small discrepancies vs major disputes

4. **Per-side hallucination rates**
   - Do buyers hallucinate more than sellers?
   - Does role affect context rot patterns?

### Expected Findings

- **Early days (1-20):** Low divergence, contexts in sync
- **Mid-run (20-40):** Divergence begins as contexts independently accumulate
- **Late run (40+):** Significant reconciliation failures, disputed deals

---

## Implementation Tasks

### Phase 1: Agent Isolation

1. [ ] Create `IsolatedAgent` class with own context/memory
2. [ ] Instantiate separate agents per buyer/seller
3. [ ] Update simulation loop to use per-agent instances
4. [ ] Track per-agent token counts and costs

### Phase 2: Database Separation

1. [ ] Create per-agent `DealRecord` storage
2. [ ] Each deal recorded from both perspectives
3. [ ] Add reconciliation check after each deal
4. [ ] Track discrepancy metrics

### Phase 3: Monitor Updates

1. [ ] Update `monitor_context_rot.py` to show per-agent stats
2. [ ] Add reconciliation failure tracking
3. [ ] Visualize divergence over time

### Phase 4: Re-run Study

1. [ ] Fresh 30-day run with isolated agents
2. [ ] Capture reconciliation metrics
3. [ ] Document where/when divergence appears

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/llm/pricing_agent.py` | Add `agent_id`, isolate context |
| `src/scenarios/real_context_scenario.py` | Create per-agent instances |
| `src/models/deal.py` | Add dual-record structure |
| `scripts/run_real_context_sim.py` | Update to multi-agent |
| `scripts/monitor_context_rot.py` | Per-agent + reconciliation view |

---

## Success Criteria

The refactored simulation should demonstrate:

1. ✅ Each agent has isolated context (visible in monitor)
2. ✅ Deals recorded from both buyer/seller perspectives
3. ✅ Reconciliation failures detected and logged
4. ✅ Divergence correlates with context pressure
5. ✅ Clear proof that A2A without shared ledger causes disputes

---

*This architecture correctly models the IAB A2A problem: independent agents with independent memories leading to irreconcilable states.*
