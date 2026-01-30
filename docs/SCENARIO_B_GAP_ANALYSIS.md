# Scenario B Gap Analysis & Implementation Plan

> **Date:** 2026-01-30 | **Prepared by:** NJ | **Status:** Ready for Gastown

---

## Executive Summary

Build Scenario B (Direct A2A with Multi-Agent Hierarchy) end-to-end, then extend to Scenario C (Ledger-backed). Park Scenario A.

**Current State:** Codebase exists but has schema mismatches and incomplete wiring.
**Target State:** Working simulation matching IAB specs with real LLM calls (Claude Sonnet).

---

## 1. Critical Bugs (Must Fix)

### 1.1 DealRequest Schema Mismatch ❌

**Location:** `src/scenarios/multi_agent_scenario_b.py:455`

**Current (broken):**
```python
deal_request = DealRequest(
    deal_id=deal_id,              # ❌ Wrong field name
    buyer_id=buyer_system.buyer_id,
    buyer_tier=BuyerTier.AGENCY,
    impressions=target_impressions,
    offered_cpm=target_cpm,       # ❌ Wrong field name
    products=[],                   # ❌ Wrong type (should be product_id: str)
)
```

**Expected (from models.py):**
```python
deal_request = DealRequest(
    request_id=deal_id,            # ✅ Correct
    buyer_id=buyer_system.buyer_id,
    buyer_tier=BuyerTier.AGENCY,
    product_id="default-product",  # ✅ Required string
    impressions=target_impressions,
    max_cpm=target_cpm,            # ✅ Correct
    deal_type=DealTypeEnum.PREFERRED_DEAL,  # ✅ Required
    flight_dates=(start_date, end_date),     # ✅ Required
    audience_spec=AudienceSpec(),            # ✅ Required
)
```

**Files to fix:**
- `src/scenarios/multi_agent_scenario_b.py` line 455
- `src/orchestration/seller_system.py` line 542

### 1.2 Seller System Error Handler ❌

**Location:** `src/orchestration/seller_system.py:512`

**Current:**
```python
return DealDecision(
    deal_id=request.request_id,  # ❌ DealDecision expects request_id
    ...
)
```

**Should be:**
```python
return DealDecision(
    request_id=request.request_id,  # ✅ Correct
    ...
)
```

---

## 2. Missing Components for End-to-End

### 2.1 Real LLM Integration

**Status:** Partially implemented (mock mode works)

**Gaps:**
- [ ] Claude Sonnet client for agent decisions
- [ ] API key configuration in `.env`
- [ ] Token counting for context management
- [ ] Response parsing for structured outputs

**Location:** `src/llm/` (needs review)

### 2.2 A2A Protocol Implementation

**Status:** Stub exists

**Gaps:**
- [ ] Full offer/counter-offer flow
- [ ] Deal confirmation messages
- [ ] Error handling for protocol violations

**Location:** `src/protocols/a2a.py`

### 2.3 Context Rot Simulation

**Status:** Config exists, implementation partial

**Gaps:**
- [ ] Verify rot rates applied at each hierarchy level
- [ ] Track rot events in metrics
- [ ] Validate compounding error model

**Location:** `src/scenarios/context_rot.py`

### 2.4 Metrics Collection

**Status:** Basic structure exists

**Gaps:**
- [ ] Context integrity percentage at each level
- [ ] Hallucination detection and logging
- [ ] Deal success/failure tracking
- [ ] Cost accounting per scenario

**Location:** `src/metrics/`

---

## 3. Test Coverage

### Current Test Status

```
176 passed, 1 failed, 2 skipped
```

### Missing Tests

- [ ] End-to-end Scenario B test with real (mocked) LLM
- [ ] Context rot accumulation verification
- [ ] Multi-agent handoff integrity
- [ ] A2A protocol compliance

### Test to Fix

```
FAILED tests/agents/test_seller_adapter.py::TestSimulatedInventory::test_generate_catalog
```
Cause: Product type assertion failing - likely another schema mismatch.

---

## 4. Implementation Plan

### Phase 1: Fix Critical Bugs (1-2 hours)

| Task | File | Priority |
|------|------|----------|
| Fix DealRequest schema in scenario B | `multi_agent_scenario_b.py` | P0 |
| Fix DealRequest in seller_system | `seller_system.py` | P0 |
| Fix DealDecision field names | `seller_system.py` | P0 |
| Fix Product type in test | `test_seller_adapter.py` | P1 |

### Phase 2: Wire Real LLM (2-3 hours)

| Task | File | Priority |
|------|------|----------|
| Configure Claude Sonnet API | `.env` + `src/llm/client.py` | P0 |
| Implement structured output parsing | `src/llm/parsers.py` | P0 |
| Add token counting | `src/llm/token_counter.py` | P1 |
| Test LLM integration | `tests/llm/` | P1 |

### Phase 3: Complete Scenario B Flow (3-4 hours)

| Task | File | Priority |
|------|------|----------|
| Implement full deal cycle | `multi_agent_scenario_b.py` | P0 |
| Add context rot at each level | `multi_agent_scenario_b.py` | P0 |
| Complete metrics collection | `src/metrics/` | P1 |
| Add end-to-end test | `tests/scenarios/` | P1 |

### Phase 4: Test Suite (2 hours)

| Task | Priority |
|------|----------|
| Create comprehensive Scenario B test suite | P0 |
| Add mock LLM tests | P1 |
| Add context integrity tests | P1 |
| Verify all 30-day metrics captured | P1 |

### Phase 5: Extend to Scenario C (Framework Only)

Once Scenario B works end-to-end, the Scenario C extension adds:
- Ledger integration points (already stubbed)
- Verification layer hooks
- Recovery mechanism

---

## 5. Model Configuration

Per Ben's direction:

| Component | Model | Notes |
|-----------|-------|-------|
| Planning (this doc) | Opus 4.5 | ✅ Done |
| Agent decisions (Scenario B) | Claude Sonnet | Per IAB specs |
| Build/Deploy (Gastown) | Default selection | Rigs + Polecats |

---

## 6. Files to Modify

### Critical Path (P0)

1. `src/scenarios/multi_agent_scenario_b.py`
2. `src/orchestration/seller_system.py`
3. `src/agents/seller/models.py` (verify canonical)
4. `.env` (API keys)

### Supporting (P1)

1. `src/llm/client.py`
2. `src/metrics/collector.py`
3. `tests/scenarios/test_scenario_b.py`
4. `tests/agents/test_seller_adapter.py`

---

## 7. Success Criteria

| Metric | Target |
|--------|--------|
| Scenario B runs end-to-end | ✅ |
| Real LLM calls work | ✅ |
| Context rot measured | Per-level % |
| All unit tests pass | 100% |
| 30-day simulation completes | Full run |
| Results match expected degradation | ~22% integrity by Day 30 |

---

## 8. Command to Run

After fixes:

```bash
cd /root/clawd/iab-sim-work
source .venv/bin/activate

# Test infrastructure
PYTHONPATH=src pytest tests/ -v -x --ignore=tests/integration

# Run Scenario B end-to-end
PYTHONPATH=src python3 -m src.scenarios.multi_agent_scenario_b --days 1 --mock-llm

# Full 30-day simulation
PYTHONPATH=src python3 -m src.scenarios.multi_agent_scenario_b --days 30
```

---

*Ready for Gastown (Rigs + Polecats) to implement.*
