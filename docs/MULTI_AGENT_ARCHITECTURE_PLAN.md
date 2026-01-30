# Multi-Agent Hierarchy Architecture Plan

## Overview

Rebuild simulation to accurately replicate IAB Tech Lab's proposed multi-agent advertising system.

## Target Architecture

### Buyer Agent System

```
Level 1: ORCHESTRATION (Claude Opus)
├── Portfolio Manager
│   ├── Budget allocation across campaigns
│   ├── Strategic decisions
│   ├── Channel optimization
│   └── Performance management

Level 2: CHANNEL SPECIALISTS (Claude Sonnet)
├── Branding Agent
├── Mobile App Agent
├── CTV Agent
├── Performance Agent
└── DSP Agent

Level 3: FUNCTIONAL AGENTS (Claude Sonnet)
├── Research Agent (ProductSearch, AvailsCheck)
├── Execution Agent (CreateOrder, CreateLine, BookLine)
├── Reporting Agent (Analytics, Attribution)
└── Audience Planner (AudienceDiscovery, Matching)
```

### Seller Agent System

```
Level 1: ORCHESTRATION (Claude Opus)
├── Inventory Manager
│   ├── Yield optimization
│   ├── Deal acceptance decisions
│   ├── Portfolio strategy
│   └── Cross-sell opportunities

Level 2: CHANNEL INVENTORY (Claude Sonnet)
├── Display Inventory
├── Video Inventory
├── CTV Inventory
├── Mobile App Inventory
└── Native Inventory

Level 3: FUNCTIONAL AGENTS (Claude Sonnet)
├── Pricing Agent (FloorManager, DiscountEngine)
├── Avails Agent (CapacityForecaster, AllocationManager)
├── Proposal Review Agent (CounterOfferBuilder)
├── Upsell Agent (Cross-sell logic)
└── Audience Validator (Coverage Calculator)
```

## Gastown Build Components

### Component 1: Agent Base Classes (rs-0100)
- `src/agents/base/orchestrator.py` - L1 Opus base
- `src/agents/base/specialist.py` - L2 Sonnet base
- `src/agents/base/functional.py` - L3 Sonnet base
- Context passing interface
- State management

### Component 2: Buyer L1 - Portfolio Manager (rs-0101)
- `src/agents/buyer/l1_portfolio_manager.py`
- Multi-campaign budget allocation
- Strategic decision making
- Channel optimization logic
- Integration with L2 specialists

### Component 3: Buyer L2 - Channel Specialists (rs-0102)
- `src/agents/buyer/l2_branding.py`
- `src/agents/buyer/l2_mobile_app.py`
- `src/agents/buyer/l2_ctv.py`
- `src/agents/buyer/l2_performance.py`
- `src/agents/buyer/l2_dsp.py`

### Component 4: Buyer L3 - Functional Agents (rs-0103)
- `src/agents/buyer/l3_research.py`
- `src/agents/buyer/l3_execution.py`
- `src/agents/buyer/l3_reporting.py`
- `src/agents/buyer/l3_audience_planner.py`

### Component 5: Seller L1 - Inventory Manager (rs-0104)
- `src/agents/seller/l1_inventory_manager.py`
- Yield optimization
- Deal acceptance logic
- Portfolio strategy

### Component 6: Seller L2 - Channel Inventory (rs-0105)
- `src/agents/seller/l2_display.py`
- `src/agents/seller/l2_video.py`
- `src/agents/seller/l2_ctv.py`
- `src/agents/seller/l2_mobile.py`
- `src/agents/seller/l2_native.py`

### Component 7: Seller L3 - Functional Agents (rs-0106)
- `src/agents/seller/l3_pricing.py`
- `src/agents/seller/l3_avails.py`
- `src/agents/seller/l3_proposal_review.py`
- `src/agents/seller/l3_upsell.py`
- `src/agents/seller/l3_audience_validator.py`

### Component 8: MCP Tool Integration (rs-0107)
- 33 OpenDirect tools
- Tool registry and dispatch
- Request/response handling

### Component 9: Protocol Handlers (rs-0108)
- A2A natural language protocol
- UCP/AA audience embeddings
- Inter-agent communication

### Component 10: Multi-Campaign State (rs-0109)
- Campaign portfolio state
- Cross-campaign context
- Budget pacing across campaigns
- Concurrent campaign handling

### Component 11: Context Flow Manager (rs-0110)
- L1 → L2 context passing
- L2 → L3 context passing
- State aggregation back up
- Context rot simulation at each level

### Component 12: Scenario Updates (rs-0111)
- Update Scenario A/B/C for multi-agent
- Multi-campaign test fixtures
- Measurement of inter-level context rot

## Context Rot Points

Each handoff is a potential failure point:

```
L1 (Opus) ──context──► L2 (Sonnet) ──context──► L3 (Sonnet)
     ▲                      │                      │
     └──────────────────────┴──────────────────────┘
                    state aggregation
```

**Scenario B (No Ledger):**
- Each level maintains private state
- Context degrades at each handoff
- Multi-campaign state diverges

**Scenario C (Alkimi Ledger):**
- Shared state on Sui blockchain
- Any agent can recover from ledger
- Cross-level verification

## Test Scenarios

1. **Single Campaign**: Baseline comparison
2. **5 Concurrent Campaigns**: Realistic load
3. **10 Concurrent Campaigns**: Stress test
4. **30-Day Multi-Campaign**: Full simulation

## Success Criteria

- Accurate replication of IAB architecture
- Real LLM calls at each level (Opus/Sonnet)
- Measurable context rot at each hierarchy level
- Multi-campaign state management
- Clear demonstration of ledger advantage
