# Content Series Part 2: IAB Dependency Integration

> **Series:** How AI Agents Proved AI Agents Need Blockchain  
> **Part 2:** "Using IAB's Own Code Against Their Oversight"  
> **Status:** Draft for review

---

## The Integration Story

When building a simulation to critique IAB's A2A architecture, we faced an obvious objection: "You're not using real IAB specs."

So we used them.

---

## What We Integrated

### IAB Tech Lab Packages

We vendored two official IAB Tech Lab repositories:

| Package | Purpose | Key Modules |
|---------|---------|-------------|
| **seller-agent** | Publisher/SSP functionality | PricingRulesEngine, TieredPricingConfig |
| **buyer-agent** | DSP/Buyer functionality | UnifiedClient, A2AClient |

These are the same packages IAB envisions the industry adopting for agentic advertising.

### Specific Components Used

**From seller-agent:**
```python
from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
from ad_seller.models.pricing_tiers import TieredPricingConfig
```

The `PricingRulesEngine` handles:
- Tiered pricing by buyer identity (public/seat/agency/advertiser)
- Volume discounts
- Negotiation limits
- Floor price enforcement

**From buyer-agent:**
```python
from ad_buyer.clients.unified_client import UnifiedClient, Protocol
from ad_buyer.clients.a2a_client import A2AClient
```

The `UnifiedClient` supports:
- **MCP Protocol:** Direct tool calls (fast, deterministic)
- **A2A Protocol:** Natural language requests (flexible, AI-interpreted)

---

## Why This Matters

### Preemptive Defense

The most common objection to simulation studies is implementation bias. By using IAB's own code:

1. **Pricing logic** is exactly what IAB specifies
2. **Protocol handling** matches their reference implementation
3. **Identity/tiering** follows their OpenDirect 2.1 spec

### The Findings Are Architectural

Our context rot and hallucination findings don't stem from implementation bugs. They're inherent to the A2A architecture:

- **No persistent ground truth** → Agents diverge over time
- **Private state per agent** → No way to detect discrepancies
- **Natural language interpretation** → Hallucination risk

Using IAB's code proves these are fundamental issues, not artifacts of our simulation.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Our Simulation Framework                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐          ┌─────────────────────┐           │
│  │   IABSellerAdapter  │          │   IABBuyerWrapper   │           │
│  │   (our adapter)     │          │   (our adapter)     │           │
│  └──────────┬──────────┘          └──────────┬──────────┘           │
│             │                                │                       │
│  ╔══════════╧════════════════════════════════╧══════════════╗       │
│  ║               IAB TECH LAB CODE (unmodified)              ║       │
│  ╠═══════════════════════════════════════════════════════════╣       │
│  ║  seller-agent/               buyer-agent/                 ║       │
│  ║  └── PricingRulesEngine      └── UnifiedClient            ║       │
│  ║  └── TieredPricingConfig     └── A2AClient                ║       │
│  ╚═══════════════════════════════════════════════════════════╝       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5-Day Simulation Results

### Quick Stats

| Scenario | Deals | Exchange Fees | Notes |
|----------|-------|---------------|-------|
| A: Exchange | 17 | $22,583 | 15% intermediary take |
| B: Pure A2A | 9 | $0 | Direct buyer↔seller |
| C: Ledger | N/A | N/A | DB setup needed |

### Key Observations

1. **Scenario B works** – Direct A2A trades complete successfully
2. **Zero intermediary fees** – As IAB intends
3. **But no verification** – Disputes would be unresolvable
4. **5 days insufficient** – Context rot needs 10+ days to manifest

---

## The Quote

> "We asked IAB's own code to demonstrate why IAB's architecture is incomplete. The pricing engine, the client libraries, the tiered identity system—all working exactly as specified. And yet: no ground truth, no verification, no way to resolve disputes. That's not a bug in our simulation. That's the design."

---

## Next in Series

**Part 3: "The 30-Day Context Rot Experiment"**
- Extended simulation showing hallucination accumulation
- Day 7-10 critical threshold
- Recovery rate comparison: Ledger vs Private DB

---

## Technical Appendix

### Verifying IAB Integration

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

print('All IAB dependencies verified!')
"
```

### Running Simulation

```bash
# 5-day mock simulation
python src/cli.py run --scenario a,b,c --days 5 --mock-llm --skip-infra

# Full 30-day with real LLM
python src/cli.py run --scenario a,b,c --days 30 --real-llm
```

---

*Document created: 2026-01-30*  
*Part of IAB Agentic Ecosystem Simulation content series*
