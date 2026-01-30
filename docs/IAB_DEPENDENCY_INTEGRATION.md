# IAB Tech Lab Dependency Integration

> **Content Series Material**: This document details how the simulation integrates real IAB Tech Lab packages to ensure methodological credibility.

---

## Executive Summary

The IAB Agentic Ecosystem Simulation uses **official IAB Tech Lab open-source packages** for buyer and seller agent functionality. This is not a mock implementation—we're using the same code IAB envisions the industry adopting.

**Why this matters:** Any criticism that "you're not using real IAB specs" is preemptively addressed. The simulation runs on the actual IAB buyer-agent and seller-agent codebases.

---

## Packages Used

### 1. IAB Seller-Agent (`vendor/iab/seller-agent/`)

**Source:** IAB Tech Lab / Green Mountain Systems AI Inc.  
**Purpose:** Publisher/SSP agent system for programmatic advertising

#### Key Modules

| Module | Path | Purpose |
|--------|------|---------|
| `PricingRulesEngine` | `src/ad_seller/engines/pricing_rules_engine.py` | Tiered pricing by buyer identity |
| `TieredPricingConfig` | `src/ad_seller/models/pricing_tiers.py` | Configuration for public/seat/agency/advertiser tiers |
| `BuyerIdentity` | `src/ad_seller/models/buyer_identity.py` | Buyer authentication and access levels |
| `AccessTier` | `src/ad_seller/models/buyer_identity.py` | Enum for pricing tier levels |

#### PricingRulesEngine Capabilities

```python
from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
from ad_seller.models.pricing_tiers import TieredPricingConfig
from ad_seller.models.buyer_identity import BuyerContext, AccessTier

# Create seller pricing configuration
config = TieredPricingConfig(
    seller_organization_id="publisher-001",
    global_floor_cpm=1.0,
)

# Initialize the pricing engine
engine = PricingRulesEngine(config)

# Calculate price based on buyer identity
decision = engine.calculate_price(
    product_id="ctv-premium",
    base_price=35.0,
    buyer_context=buyer_context,  # Contains identity info
    deal_type=DealType.PREFERRED_DEAL,
    volume=5_000_000,
)
```

#### TieredPricingConfig Defaults

| Tier | Discount | Features |
|------|----------|----------|
| Public | 0% | Price ranges only, no negotiation |
| Seat | 5% | Fixed prices, custom deals |
| Agency | 10% | Negotiation, volume discounts, premium access |
| Advertiser | 15% | Best rates, full negotiation |

---

### 2. IAB Buyer-Agent (`vendor/iab/buyer-agent/`)

**Source:** IAB Tech Lab / Green Mountain Systems AI Inc.  
**Purpose:** DSP/buyer agent system for programmatic advertising

#### Key Modules

| Module | Path | Purpose |
|--------|------|---------|
| `UnifiedClient` | `src/ad_buyer/clients/unified_client.py` | Dual-protocol client (MCP + A2A) |
| `A2AClient` | `src/ad_buyer/clients/a2a_client.py` | Agent-to-Agent natural language protocol |
| `MCPClient` | `src/ad_buyer/clients/mcp_client.py` | Model Context Protocol for direct tool calls |
| `BuyerIdentity` | `src/ad_buyer/models/buyer_identity.py` | DSP seat/agency/advertiser identity |

#### UnifiedClient Protocol Options

```python
from ad_buyer.clients.unified_client import UnifiedClient, Protocol

# Option 1: MCP Protocol (Direct Tool Calls)
async with UnifiedClient(protocol=Protocol.MCP) as client:
    products = await client.list_products()
    deal = await client.request_deal(
        product_id="ctv-premium",
        deal_type="PD",
        impressions=1_000_000,
    )

# Option 2: A2A Protocol (Natural Language)
async with UnifiedClient(protocol=Protocol.A2A) as client:
    result = await client.send_natural_language(
        "I want premium CTV inventory under $30 CPM for Q1"
    )
```

#### A2AClient Details

The A2A client implements JSON-RPC 2.0 over HTTP with:
- Context management for multi-turn conversations
- Natural language → tool execution via AI interpretation
- Agent card discovery (`.well-known/agent-card.json`)

```python
from ad_buyer.clients.a2a_client import A2AClient

client = A2AClient(
    base_url="https://agentic-direct-server-hwgrypmndq-uk.a.run.app",
    agent_type="buyer",
)

# Natural language request
response = await client.send_message(
    "Book 500,000 impressions of display at $15 CPM"
)
```

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IAB Agentic Ecosystem Simulation                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐              ┌─────────────────────┐           │
│  │    Our Adapters     │              │    Our Adapters     │           │
│  │  IABSellerAdapter   │              │  IABBuyerWrapper    │           │
│  │  (iab_adapter.py)   │              │  (iab_wrapper.py)   │           │
│  └──────────┬──────────┘              └──────────┬──────────┘           │
│             │                                    │                       │
│  ┌──────────▼──────────────────────────────────▼──────────────┐        │
│  │               vendor/iab/ (IAB TECH LAB CODE)               │        │
│  ├─────────────────────────────────────────────────────────────┤        │
│  │                                                             │        │
│  │  seller-agent/                    buyer-agent/              │        │
│  │  ├── engines/                     ├── clients/              │        │
│  │  │   └── pricing_rules_engine.py  │   ├── unified_client.py │        │
│  │  ├── models/                      │   ├── a2a_client.py     │        │
│  │  │   ├── pricing_tiers.py         │   └── mcp_client.py     │        │
│  │  │   └── buyer_identity.py        ├── models/               │        │
│  │  └── tools/                       │   └── buyer_identity.py │        │
│  │      └── pricing/                 └── tools/                │        │
│  │                                       └── dsp/              │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                    │                                     │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  IAB OpenDirect     │
                          │  Server (Hosted)    │
                          │  MCP + A2A Endpoints│
                          └─────────────────────┘
```

---

## Python Path Setup

The vendor packages are loaded via Python path manipulation:

```python
import sys
import os

# Add IAB packages to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "vendor/iab/seller-agent/src"))
sys.path.insert(0, os.path.join(project_root, "vendor/iab/buyer-agent/src"))

# Now we can import IAB modules
from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
from ad_buyer.clients.unified_client import UnifiedClient
```

---

## Why This Matters

### 1. Removes "Not Using Real Specs" Criticism

The most common objection to simulation studies is that they don't reflect real-world implementations. By using IAB Tech Lab's own codebase:

- ✅ Pricing logic is exactly what IAB envisions
- ✅ Protocol handling (MCP/A2A) is official implementation
- ✅ Identity/tiering system matches spec

### 2. Demonstrates Fundamental Limitations

Using real code makes our findings about **context rot** and **hallucination** more credible:

- We're not strawmanning A2A protocols
- The hallucinations occur in the same code the industry would deploy
- Limitations are inherent to the architecture, not our implementation

### 3. Reproducibility

Anyone can:
1. Clone our repo
2. See we're using `vendor/iab/*` packages
3. Verify the code matches IAB Tech Lab repositories
4. Run the same simulation

---

## Verification

### Check Vendor Contents

```bash
cd /root/clawd/iab-sim-work
ls -la vendor/iab/
# buyer-agent/
# seller-agent/

# Verify key files exist
cat vendor/iab/seller-agent/src/ad_seller/engines/pricing_rules_engine.py | head -20
cat vendor/iab/buyer-agent/src/ad_buyer/clients/unified_client.py | head -20
```

### Test Imports

```bash
cd /root/clawd/iab-sim-work
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'vendor/iab/seller-agent/src')
sys.path.insert(0, 'vendor/iab/buyer-agent/src')

from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
from ad_seller.models.pricing_tiers import TieredPricingConfig
from ad_buyer.clients.unified_client import UnifiedClient
from ad_buyer.clients.a2a_client import A2AClient

print('✓ PricingRulesEngine imported')
print('✓ TieredPricingConfig imported')
print('✓ UnifiedClient imported')
print('✓ A2AClient imported')
print('All IAB dependencies verified!')
"
```

---

## Content Series Positioning

This integration step appears in **Article 2: "Building the Proof"** with framing:

> "To ensure our findings couldn't be dismissed as implementation artifacts, we used IAB Tech Lab's own open-source agent packages. The PricingRulesEngine, TieredPricingConfig, UnifiedClient, and A2AClient are the same code IAB envisions the industry adopting.
>
> This means our context rot findings and hallucination metrics reflect fundamental architectural limitations—not bugs in a mock implementation."

---

## Files Reference

| Component | Path | Purpose |
|-----------|------|---------|
| Seller Adapter | `src/agents/seller/iab_adapter.py` | Wraps IAB seller-agent |
| Buyer Wrapper | `src/agents/buyer/iab_wrapper.py` | Wraps IAB buyer-agent |
| Pricing Engine | `vendor/iab/seller-agent/src/ad_seller/engines/` | IAB official code |
| Client Library | `vendor/iab/buyer-agent/src/ad_buyer/clients/` | IAB official code |
| Integration Tests | `tests/test_iab_integration.py` | Verify IAB package usage |

---

*Document created: 2026-01-30*  
*Part of IAB Agentic Ecosystem Simulation content series*
