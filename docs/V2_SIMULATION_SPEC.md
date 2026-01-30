# V2 Simulation Spec: Context Window Hallucination Testing

> **Goal:** Measure intra-campaign hallucination rates caused by context window limitations in AI agents making ad buying/selling decisions.

## Executive Summary

V1 proved the reconciliation gap between agents with private databases. V2 tests the **single-agent reliability problem**: even one agent making millions of decisions accumulates errors from context overflow, memory loss, and hallucinations.

## Core Hypothesis

> As campaign duration increases, context window pressure causes non-linear growth in hallucination rate, with a critical threshold around Day 7-10 where decision quality degrades significantly.

---

## New Components

### 1. Token Pressure Engine (`src/pressure/token_tracker.py`)

Track and simulate context window limitations.

```python
class TokenPressureEngine:
    """Simulates context window pressure on agent decisions."""
    
    def __init__(self, model_context_limit: int = 200_000):
        self.context_limit = model_context_limit
        self.current_tokens = 0
        self.overflow_events = []
        self.compression_events = []
    
    def add_event(self, event: BidEvent) -> TokenPressureResult:
        """Add event to context, track overflow."""
        tokens = self.estimate_tokens(event)
        self.current_tokens += tokens
        
        if self.current_tokens > self.context_limit:
            return self.handle_overflow()
        return TokenPressureResult(overflow=False)
    
    def handle_overflow(self) -> TokenPressureResult:
        """Simulate context overflow - compression with information loss."""
        # Model: 20% information loss on each compression
        lost_events = self.compress_context(loss_rate=0.20)
        return TokenPressureResult(
            overflow=True,
            events_lost=lost_events,
            information_loss_pct=0.20
        )
```

**Metrics:**
- Overflow events per campaign
- Total information loss %
- Decision quality pre/post overflow

### 2. Decision Chain Tracker (`src/tracking/decision_chain.py`)

Track dependencies between decisions and detect cascading errors.

```python
class DecisionChainTracker:
    """Track decision dependencies and cascading errors."""
    
    def __init__(self, lookback_window: int = 100):
        self.decisions = []
        self.lookback = lookback_window
        self.reference_failures = []
    
    def record_decision(self, decision: AgentDecision) -> ChainResult:
        """Record decision and check reference integrity."""
        # Each decision must reference recent decisions
        references = decision.get_references()
        
        for ref in references:
            actual = self.get_actual_decision(ref.decision_id)
            if not self.verify_reference(ref, actual):
                self.reference_failures.append(ReferenceFailure(
                    decision_id=decision.id,
                    ref_id=ref.decision_id,
                    expected=ref.recalled_value,
                    actual=actual.value,
                    error_type=self.classify_error(ref, actual)
                ))
        
        self.decisions.append(decision)
        return ChainResult(
            cascading_errors=self.count_cascading_errors(decision)
        )
```

**Metrics:**
- Reference accuracy rate
- Cascading error chains (length, severity)
- Error type distribution

### 3. Hallucination Classifier (`src/hallucination/classifier.py`)

Classify types of hallucinations in agent decisions.

```python
class HallucinationType(Enum):
    BUDGET_DRIFT = "budget_drift"           # Misremembers spent amount
    FREQUENCY_VIOLATION = "frequency_cap"   # Loses user exposure tracking
    DEAL_INVENTION = "deal_invention"       # Invents deal terms
    CAMPAIGN_CROSS_CONTAMINATION = "cross_campaign"  # Wrong campaign attribution
    PHANTOM_INVENTORY = "phantom_inventory" # References non-existent supply
    PRICE_ANCHOR_ERROR = "price_anchor"     # Wrong price memory

class HallucinationClassifier:
    """Classify and track hallucination types."""
    
    def __init__(self, ground_truth: GroundTruthDB):
        self.ground_truth = ground_truth
        self.hallucinations = []
    
    def check_decision(self, decision: AgentDecision) -> HallucinationResult:
        """Compare decision against ground truth."""
        truth = self.ground_truth.get_state(decision.timestamp)
        
        errors = []
        
        # Budget drift check
        if decision.budget_remaining != truth.budget_remaining:
            drift = abs(decision.budget_remaining - truth.budget_remaining)
            if drift > truth.budget_remaining * 0.01:  # >1% drift
                errors.append(Hallucination(
                    type=HallucinationType.BUDGET_DRIFT,
                    expected=truth.budget_remaining,
                    actual=decision.budget_remaining,
                    severity=drift / truth.budget_remaining
                ))
        
        # ... other checks
        
        return HallucinationResult(errors=errors)
```

**Hallucination Types to Track:**

| Type | Description | Detection Method |
|------|-------------|------------------|
| Budget Drift | Agent misremembers spend | Compare to ground truth ledger |
| Frequency Violation | Exceeds frequency cap | Track actual user exposure |
| Deal Invention | Makes up deal terms | Verify against deal records |
| Cross-Contamination | Attributes to wrong campaign | Check campaign IDs |
| Phantom Inventory | Bids on non-existent supply | Verify publisher inventory |
| Price Anchor Error | Wrong floor price memory | Compare to actual floors |

### 4. Realistic Volume Generator (`src/volume/generator.py`)

Generate realistic bid volumes that stress context windows.

```python
class RealisticVolumeGenerator:
    """Generate realistic ad request volumes."""
    
    VOLUME_PROFILES = {
        "small_campaign": {"daily_requests": 10_000, "bid_rate": 0.3},
        "medium_campaign": {"daily_requests": 100_000, "bid_rate": 0.2},
        "large_campaign": {"daily_requests": 1_000_000, "bid_rate": 0.1},
        "enterprise_campaign": {"daily_requests": 10_000_000, "bid_rate": 0.05}
    }
    
    def generate_day(self, profile: str, day: int) -> List[BidRequest]:
        """Generate one day of bid requests."""
        config = self.VOLUME_PROFILES[profile]
        
        # Add temporal patterns (peak hours, weekday/weekend)
        hourly_weights = self.get_hourly_distribution(day)
        
        requests = []
        for hour in range(24):
            hour_volume = int(config["daily_requests"] / 24 * hourly_weights[hour])
            requests.extend(self.generate_hour(hour_volume, hour))
        
        return requests
```

**Volume Targets:**

| Profile | Daily Requests | 30-Day Total | Context Pressure |
|---------|----------------|--------------|------------------|
| Small | 10K | 300K | Low |
| Medium | 100K | 3M | Medium |
| Large | 1M | 30M | High |
| Enterprise | 10M | 300M | Extreme |

### 5. Agent Restart Simulator (`src/resilience/restart.py`)

Simulate agent crashes and state recovery.

```python
class AgentRestartSimulator:
    """Simulate agent crashes and measure state recovery."""
    
    def __init__(self, 
                 crash_probability: float = 0.01,  # Per-hour crash rate
                 recovery_modes: List[str] = ["private_db", "ledger"]):
        self.crash_prob = crash_probability
        self.modes = recovery_modes
        self.restart_events = []
    
    def maybe_crash(self, agent: Agent, hour: int) -> Optional[RestartEvent]:
        """Randomly crash agent and measure recovery."""
        if random.random() < self.crash_prob:
            # Save pre-crash state
            pre_state = agent.get_internal_state()
            
            # Simulate restart
            agent.restart()
            
            # Measure recovery accuracy for each mode
            results = {}
            for mode in self.modes:
                recovered_state = agent.recover_state(mode)
                accuracy = self.compare_states(pre_state, recovered_state)
                results[mode] = accuracy
            
            return RestartEvent(
                hour=hour,
                pre_state=pre_state,
                recovery_accuracy=results
            )
        return None
```

**Metrics:**
- State recovery accuracy (private DB vs ledger)
- Time to recovery
- Decisions affected by incomplete recovery

### 6. Ground Truth Database (`src/ground_truth/db.py`)

Authoritative record of all events for comparison.

```python
class GroundTruthDB:
    """Immutable record of actual events - the source of truth."""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def record_event(self, event: Event) -> None:
        """Record event to ground truth (immutable)."""
        self.conn.execute("""
            INSERT INTO events (
                event_id, timestamp, event_type, 
                campaign_id, amount, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, event.to_tuple())
    
    def get_campaign_state(self, campaign_id: str, at_time: datetime) -> CampaignState:
        """Get true campaign state at point in time."""
        result = self.conn.execute("""
            SELECT 
                SUM(CASE WHEN event_type = 'spend' THEN amount ELSE 0 END) as total_spend,
                COUNT(CASE WHEN event_type = 'impression' THEN 1 END) as impressions,
                MAX(timestamp) as last_event
            FROM events
            WHERE campaign_id = ? AND timestamp <= ?
        """, (campaign_id, at_time))
        return CampaignState.from_row(result.fetchone())
```

---

## New Simulation Modes

### Mode: `--context-pressure`

Test hallucination under context window stress.

```bash
rtb-sim run --days 30 --context-pressure \
    --volume-profile large \
    --context-limit 200000 \
    --compression-loss 0.20
```

### Mode: `--decision-chain`

Track decision dependencies and cascading errors.

```bash
rtb-sim run --days 30 --decision-chain \
    --lookback-window 100 \
    --track-references
```

### Mode: `--restart-test`

Simulate agent crashes and recovery.

```bash
rtb-sim run --days 30 --restart-test \
    --crash-probability 0.02 \
    --recovery-modes private_db,ledger
```

### Mode: `--full-v2`

Run all V2 features together.

```bash
rtb-sim run --days 30 --full-v2 \
    --volume-profile medium \
    --output results_v2.json
```

---

## New Reports

### Hallucination Rate Over Time

```
Day  | Decisions | Hallucinations | Rate   | Cumulative
-----|-----------|----------------|--------|------------
1    | 10,000    | 12             | 0.12%  | 0.12%
2    | 10,000    | 15             | 0.15%  | 0.14%
...
10   | 10,000    | 89             | 0.89%  | 0.45%
...
20   | 10,000    | 312            | 3.12%  | 1.23%
...
30   | 10,000    | 891            | 8.91%  | 2.87%
```

### Hallucination Type Distribution

```
Type                    | Count | % of Total | Avg Severity
------------------------|-------|------------|-------------
Budget Drift            | 423   | 35%        | 0.12
Price Anchor Error      | 289   | 24%        | 0.08
Frequency Violation     | 198   | 16%        | 0.15
Deal Invention          | 156   | 13%        | 0.45
Cross-Contamination     | 89    | 7%         | 0.32
Phantom Inventory       | 56    | 5%         | 0.28
```

### Context Overflow Impact

```
Overflow Event #3 (Day 8, Hour 14)
----------------------------------
Context tokens: 215,432 (limit: 200,000)
Compression: 20% events dropped
Events lost: 4,312

Decision quality before: 98.2%
Decision quality after: 91.4%
Recovery time: 847 decisions
```

### Recovery Comparison

```
Restart Event #2 (Day 12, Hour 9)
---------------------------------
Recovery Mode     | Accuracy | Time  | Decisions Affected
------------------|----------|-------|-------------------
Private DB        | 87.3%    | 2.3s  | 156 (12.7% errors)
Ledger (Alkimi)   | 99.8%    | 0.8s  | 3 (0.2% errors)
```

---

## Implementation Plan (Gastown)

### Phase 1: Infrastructure (Parallel)

| Component | Sub-Agent | Estimated Time |
|-----------|-----------|----------------|
| Token Pressure Engine | rs-0001 | 2 hours |
| Ground Truth DB | rs-0002 | 1 hour |
| Hallucination Classifier | rs-0003 | 2 hours |

### Phase 2: Generators (Parallel)

| Component | Sub-Agent | Estimated Time |
|-----------|-----------|----------------|
| Realistic Volume Generator | rs-0004 | 1.5 hours |
| Decision Chain Tracker | rs-0005 | 2 hours |
| Agent Restart Simulator | rs-0006 | 1.5 hours |

### Phase 3: Integration (Sequential)

| Component | Sub-Agent | Estimated Time |
|-----------|-----------|----------------|
| CLI Extensions | rs-0007 | 1 hour |
| Report Generator Updates | rs-0008 | 1.5 hours |
| Integration Tests | rs-0009 | 1 hour |

---

## Success Metrics

1. **Hallucination curve** — Demonstrate non-linear growth after Day 7-10
2. **Type distribution** — Budget drift and price errors most common
3. **Recovery comparison** — Ledger recovery >99% vs private DB <90%
4. **Context overflow** — Quantify decision quality degradation
5. **Cascading errors** — Show error propagation chains

---

## Files to Create

```
src/
├── pressure/
│   ├── __init__.py
│   └── token_tracker.py
├── tracking/
│   ├── __init__.py
│   └── decision_chain.py
├── hallucination/
│   ├── __init__.py
│   ├── classifier.py
│   └── types.py
├── volume/
│   ├── __init__.py
│   └── generator.py
├── resilience/
│   ├── __init__.py
│   └── restart.py
├── ground_truth/
│   ├── __init__.py
│   └── db.py
└── reports/
    └── v2_reports.py

tests/
├── test_token_pressure.py
├── test_decision_chain.py
├── test_hallucination.py
├── test_volume.py
├── test_restart.py
└── test_ground_truth.py
```

---

*Spec Version: 2.0.0*
*Author: NJ (Set Piece Coach)*
*Date: 2026-01-30*
