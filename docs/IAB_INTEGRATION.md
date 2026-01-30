# IAB Tech Lab Package Integration

This document explains how the IAB Tech Lab seller-agent and buyer-agent packages are integrated into the RTB simulation.

## Overview

The simulation uses two IAB Tech Lab packages:
- **ad_seller** - Publisher/SSP agent system
- **ad_buyer** - DSP/buyer agent system

Both packages connect to IAB's hosted OpenDirect server:
```
https://agentic-direct-server-hwgrypmndq-uk.a.run.app
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RTB Simulation                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │  IABSellerAdapter │         │  IABBuyerWrapper │              │
│  │  (iab_adapter.py) │         │  (iab_wrapper.py)│              │
│  └────────┬─────────┘         └────────┬─────────┘              │
│           │                            │                         │
│           ▼                            ▼                         │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │ PricingRulesEngine│         │   UnifiedClient  │              │
│  │ TieredPricingConfig│        │   A2AClient      │              │
│  │ (ad_seller pkg)   │         │   MCPClient      │              │
│  └──────────────────┘         └────────┬─────────┘              │
│                                        │                         │
└────────────────────────────────────────┼─────────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ IAB OpenDirect Server│
                              │ (MCP + A2A protocols)│
                              └──────────────────────┘
```

## Key Components

### Seller Side (`src/agents/seller/iab_adapter.py`)

The `IABSellerAdapter` wraps IAB seller-agent functionality:

```python
from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
from ad_seller.models.pricing_tiers import TieredPricingConfig
from ad_seller.models.buyer_identity import BuyerContext, BuyerIdentity, AccessTier

# Create pricing config with tiers
pricing_config = TieredPricingConfig(
    seller_organization_id=seller_id,
    global_floor_cpm=1.0,
)
pricing_engine = PricingRulesEngine(pricing_config)

# Calculate tiered price for a buyer
pricing_decision = pricing_engine.calculate_price(
    product_id=product.product_id,
    base_price=product.base_cpm,
    buyer_context=buyer_context,
    deal_type=DealType.PREFERRED_DEAL,
    volume=impressions_requested,
)
```

**Key Features:**
- Tiered pricing based on buyer identity (public, seat, agency, advertiser)
- Volume discounts
- Deal type-specific pricing
- Floor price enforcement

### Buyer Side (`src/agents/buyer/iab_wrapper.py`)

The `IABBuyerWrapper` uses the UnifiedClient for two protocols:

1. **MCP Protocol** - Direct tool calls (faster, deterministic)
```python
from ad_buyer.clients.unified_client import UnifiedClient, Protocol

client = UnifiedClient(
    base_url="https://agentic-direct-server-hwgrypmndq-uk.a.run.app",
    protocol=Protocol.MCP,
)
await client.connect()

# Direct tool call
result = await client.list_products()
result = await client.request_deal(product_id="...", deal_type="PD", ...)
```

2. **A2A Protocol** - Natural language requests (flexible, AI-interpreted)
```python
# Natural language query
result = await client.send_natural_language(
    "I'd like to book 500,000 impressions of premium display "
    "inventory at around $15 CPM for my Q1 campaign"
)
```

## Protocol Comparison

| Feature | MCP | A2A |
|---------|-----|-----|
| Speed | Fast | Slower |
| Cost | Lower | Higher (LLM calls) |
| Flexibility | Structured only | Natural language |
| Determinism | High | Variable |
| Error handling | Clear errors | May hallucinate |

## Running the Simulation

### Mock Mode (No API Costs)
```bash
cd /root/clawd/iab-sim-work
source .venv/bin/activate
python scripts/run_iab_simulation.py --mock --days 5
```

### Real LLM Mode
```bash
export ANTHROPIC_API_KEY=your_key
python scripts/run_iab_simulation.py --days 5
```

### With A2A Natural Language
```bash
python scripts/run_iab_simulation.py --days 5 --a2a
```

## Cost Estimation

Based on testing:
- **MCP calls**: ~$0.0005 per operation (no LLM interpretation)
- **A2A calls**: ~$0.01 per request (includes LLM processing)
- **Pricing engine**: ~$0.0001 per calculation (rule-based)

For a 30-day simulation with 5 buyers and 5 sellers:
- Mock mode: $0
- MCP-only: ~$0.50-1.00
- A2A mode: ~$5-10

## Hallucination Tracking

The simulation tracks hallucination events specific to A2A mode:
- Price misquotes
- Inventory availability errors
- Deal term discrepancies
- Context rot (memory degradation)

See `src/agents/ucp/hallucination.py` for the HallucinationManager.

## Integration Points

### Existing Simulation Infrastructure

The IAB adapters integrate with:
1. **Redis Bus** (`infrastructure/redis_bus.py`) - Message passing
2. **Message Schemas** (`infrastructure/message_schemas.py`) - BidRequest, BidResponse, DealConfirmation
3. **Inventory System** (`agents/seller/inventory.py`) - Product catalogs
4. **Scenario System** (`scenarios/`) - A, B, C scenario implementations

### Scenario B (Pure A2A)

Scenario B specifically uses these IAB components to demonstrate:
- Direct buyer↔seller communication
- Context rot over time
- No exchange intermediary
- Hallucination accumulation

## Files Changed

| File | Changes |
|------|---------|
| `src/agents/seller/iab_adapter.py` | New - IAB seller integration |
| `src/agents/buyer/iab_wrapper.py` | New - IAB buyer integration |
| `tests/test_iab_integration.py` | New - Integration tests |
| `scripts/run_iab_simulation.py` | New - Simulation runner |

## Testing

Run integration tests:
```bash
python tests/test_iab_integration.py
```

Expected output:
```
✓ PASS: Import seller adapter
✓ PASS: Import buyer wrapper
✓ PASS: Import IAB packages
✓ PASS: Mock evaluation flow
✓ PASS: IAB server connection
✓ PASS: Real LLM buyer discovery (skipped if no API key)
✓ PASS: A2A natural language (skipped if no API key)
```

## Troubleshooting

### "No module named 'ad_seller'"
Ensure the vendor packages are on the Python path:
```python
sys.path.insert(0, "vendor/iab/seller-agent/src")
sys.path.insert(0, "vendor/iab/buyer-agent/src")
```

### MCP Connection Timeout
The IAB server may have cold start latency. First connection can take 10-30 seconds.

### A2A Response Parsing
A2A responses are natural language - parsing may be inconsistent. Use MCP for deterministic results.
