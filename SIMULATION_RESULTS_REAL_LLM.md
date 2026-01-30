# IAB Agentic Ecosystem Simulation - Real LLM Attempt Results

**Date:** 2026-01-30
**Simulation Configuration:**
- Mode: Real LLM (attempted)
- Duration: 30 days (attempted) / 5 days (actual)
- Scenarios: A (Current State Exchange), B (IAB Pure A2A)
- Buyers: 5 | Sellers: 5

## Executive Summary

**⚠️ CRITICAL FINDING: Real LLM mode falls back to mock behavior**

The simulation cannot run with actual LLM API calls because:
1. The IAB `ad_seller` SDK module is not installed
2. Seller adapters detect `--real-llm` flag but immediately fall back to mock when import fails
3. The `ad_seller` module is not available via PyPI or included in dependencies

### What This Means
- **NO Anthropic API calls were made** during this simulation
- All seller decisions use deterministic mock logic, not Claude responses
- The "Real LLM" label in output is misleading - it should say "Mock (ad_seller unavailable)"

## Bug Fix Applied

During this investigation, I fixed a bug in `scenario_b.py`:

```python
# Line 758: Changed from:
deal_type=DealType.PA,
# To:
deal_type=DealType.PRIVATE_AUCTION,
```

The `DealType.PA` didn't exist - the enum value is `DealType.PRIVATE_AUCTION = "PA"`.

## Simulation Results (Day 1 Sample)

### Scenario A: Current State (Rent-Seeking Exchange)
| Metric | Value |
|--------|-------|
| Total Deals | 0 |
| Total Impressions | 0 |
| Buyer Spend | $0.00 |
| Seller Revenue | $0.00 |
| Exchange Fees | $0.00 |
| Campaigns Started | 50 |
| Campaigns Completed | 0 |

**Issue:** Scenario A's exchange is not matching bids to responses. All bid responses become "orphan responses" because the exchange logic doesn't correlate them with pending requests.

### Scenario B: IAB Pure A2A
| Metric | Value |
|--------|-------|
| Total Deals | 8-15 (varies per run) |
| Total Impressions | ~4-8M |
| Buyer Spend | $31,316 - $48,329 |
| Seller Revenue | Same as spend (0% exchange fee) |
| Average CPM | $12.41 - $14.03 |
| Context Rot Events | 0 |
| Hallucination Rate | 0% |

**Note:** Despite `hallucination_rate=0.1` being configured, no hallucinations occurred because:
1. The hallucination injection requires actual LLM decisions to corrupt
2. Mock decisions are deterministic and don't go through the hallucination pipeline

## Technical Details

### Why Real LLM Doesn't Work

```python
# From src/agents/seller/adapter.py
try:
    from ad_seller import SellerAgent  # This import fails
except ImportError:
    logger.warning("seller_adapter.iab_init_failed", error="No module named 'ad_seller'", fallback="mock")
    # Falls back to mock behavior
```

The `ad_seller` module appears to be:
- A hypothetical IAB-specified SDK for seller agent behavior
- Not yet published or available
- Required for real agentic seller decisions

### Actual LLM Integration Points

The simulation *does* have LLM integration for:
1. **Buyer agents** via CrewAI (`src/agents/buyer/wrapper.py`)
   - Uses `anthropic/claude-3-haiku-20240307` model
   - Portfolio Manager, Research Analyst, Execution Specialist agents
   - BUT: These aren't being invoked in current flow

2. **Seller agents** require `ad_seller` SDK (unavailable)

### Dependencies Installed
- anthropic: 0.77.0
- crewai: 1.9.2  
- chromadb: 1.1.1

### Missing Dependencies
- `ad_seller` - IAB seller agent SDK (not available)

## API Costs

**Total Anthropic API Cost: $0.00**

No API calls were made because:
1. Seller agents fall back to mock (ad_seller unavailable)
2. Buyer CrewAI agents aren't being activated in current simulation flow
3. Scenario A doesn't reach deal-making stage
4. Scenario B uses mock negotiations

## Comparison: Mock vs "Real LLM" (Actually Also Mock)

Since both modes fell back to mock behavior, there's no meaningful difference:

| Aspect | Mock Mode | "Real LLM" Mode |
|--------|-----------|-----------------|
| Seller Decisions | Deterministic | Deterministic (fallback) |
| Buyer Decisions | Deterministic | Deterministic |
| Hallucinations | 0 | 0 |
| Context Rot | 0 | 0 |
| API Costs | $0 | $0 |

## Recommendations

### To Enable True Real LLM Mode:

1. **Install/Build ad_seller module:**
   - Check IAB Tech Lab for official SDK
   - Or implement custom seller agent using CrewAI directly

2. **Refactor seller adapter:**
   ```python
   # Instead of importing ad_seller, use CrewAI directly
   from crewai import Agent, Task, Crew
   
   seller_agent = Agent(
       role="Ad Inventory Seller",
       goal="Maximize revenue while maintaining quality inventory",
       llm=LLM(model="anthropic/claude-3-haiku-20240307"),
   )
   ```

3. **Fix Scenario A exchange matching:**
   - The exchange doesn't correlate requests with responses
   - All responses become "orphans" and no deals are made

4. **Activate buyer CrewAI agents:**
   - Currently configured but not invoked in trading flow
   - Need to wire up the agent hierarchy to actual bid decisions

### Estimated Real LLM Costs (if working):

Based on simulation activity levels:
- ~50 campaigns × 30 days = 1,500 campaign-days
- ~5-10 LLM calls per negotiation
- ~$0.0001 per Haiku call (input) + $0.0005 (output)
- **Estimated: $5-15 for 30-day simulation**

## Files Modified

1. `src/scenarios/scenario_b.py` - Fixed DealType.PA → DealType.PRIVATE_AUCTION bug

## Conclusion

The "Real LLM" mode cannot function as intended because the required `ad_seller` SDK module doesn't exist or isn't publicly available. The simulation falls back to mock behavior, making it identical to mock mode in practice.

**Action Required:** The simulation needs architectural changes to enable true LLM-powered agent decisions without depending on the unavailable IAB SDK.
