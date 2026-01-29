# Research Plan: Cross-Agent Reconciliation in Programmatic Advertising

> **Version 2.0** - Updated 2026-01-29  
> Reframed from "context rot" to "cross-agent reconciliation"

---

## Research Question

**In a multi-agent programmatic advertising system where buyer and seller agents maintain private databases, what is the rate of unresolvable campaign disputes, and how does this compare to systems with shared state?**

---

## Hypothesis Statements

### Primary Hypothesis (H1)

> Multi-agent advertising systems without shared state experience significantly higher rates of unresolvable billing disputes than systems with a neutral shared ledger.

**Operationalization:**
- **Unresolvable dispute**: A campaign where buyer and seller records differ by >15% and no resolution is achieved within 90 days
- **Significant**: p < 0.05, effect size > 20 percentage points

**Testable Prediction:**
- Scenario B (private DBs): >10% unresolvable dispute rate
- Scenario C (shared ledger): <1% unresolvable dispute rate

### Secondary Hypotheses

**H2: Dispute rate increases with campaign duration**
> Longer campaigns accumulate more discrepancies, leading to higher dispute rates.

Prediction: 30-day campaigns have 2x+ dispute rate vs 7-day campaigns.

**H3: Disputed spend exceeds exchange fee savings**
> The financial cost of unresolvable disputes in IAB A2A exceeds what would have been paid in exchange fees.

Prediction: At scale ($150B market), dispute costs > 5% of total spend.

**H4: Resolution time increases without shared state**
> Disputes take longer to resolve when there is no authoritative record.

Prediction: Average resolution time in Scenario B > 30 days vs < 1 day in Scenario C.

---

## Methodology

### Study Design

**Type:** Monte Carlo simulation with controlled variable comparison

**Independent Variable:**
- State architecture (private DBs vs shared ledger)

**Dependent Variables:**
- Unresolvable dispute rate (%)
- Total disputed spend ($)
- Average resolution time (days)
- Discrepancy distribution (%)

**Control Variables:**
- Same discrepancy injection rates across scenarios
- Same campaign parameters
- Same agent behaviors

### Simulation Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Campaigns per run | 50 | Statistical power |
| Campaign duration | 30 days | Industry standard flight |
| Buyers | 5 | Typical agency portfolio |
| Sellers | 5 | Typical publisher panel |
| Budget range | $25K - $100K | Mid-market campaigns |
| Impressions | 1.5M - 6M | Corresponding delivery |

### Discrepancy Injection (Calibrated to Industry Data)

Based on ISBA 2020, ANA 2023, and MRC guidelines:

| Source | Rate | Magnitude |
|--------|------|-----------|
| Timing differences | 3% of transactions | 1-5% variance |
| IVT disagreement | 5% of transactions | 2-8% variance |
| Viewability | 4% of transactions | 1-6% variance |
| Ad serving latency | 2% of transactions | 1-3% variance |
| Data loss | 0.1% per day | 0.1-1% variance |

**Expected aggregate discrepancy:** 5-15% (matching industry research)

### Resolution Model

| Discrepancy | Outcome | Resolution Time |
|-------------|---------|-----------------|
| <3% | Auto-accept | 1 day |
| 3-10% | Negotiated | 7-21 days |
| 10-15% | Formal dispute | 30-60 days |
| >15% | Often unresolvable | 45-90+ days |

**Calibration source:** Industry interviews, exchange dispute data

---

## Analysis Plan

### Primary Analysis

**Compare unresolvable dispute rates:**

```
H0: Rate(B) = Rate(C)
H1: Rate(B) > Rate(C)

Test: Two-proportion z-test
α = 0.05
Power analysis: n=50 campaigns provides 80% power to detect 15pp difference
```

### Secondary Analyses

1. **Duration effect:** Linear regression of dispute rate ~ campaign_days
2. **Financial impact:** Sum of disputed/unresolvable spend per scenario
3. **Time to resolution:** Mann-Whitney U test (non-parametric)
4. **Discrepancy accumulation:** Time series of daily divergence

### Sensitivity Analyses

Test robustness to parameter choices:

| Parameter | Range Tested |
|-----------|--------------|
| Discrepancy rate | 3%, 5%, 10%, 15% |
| Campaign duration | 7, 14, 30, 90 days |
| Agent count | 2, 5, 10, 20 |
| Resolution thresholds | ±50% of default |

---

## Separation of Assumptions vs Measurements

### Assumptions (Stated, Not Measured)

| Assumption | Source | Impact if Wrong |
|------------|--------|-----------------|
| Discrepancy rates match industry | ISBA, ANA studies | Results may not generalize |
| Agents act in good faith | Simplifying assumption | Adversarial behavior would increase disputes |
| Resolution thresholds are realistic | Industry interviews | Different thresholds → different rates |
| 30 days is representative campaign length | Industry standard | Different lengths → different accumulation |

### Measurements (Empirical, From Simulation)

| Measurement | Description |
|-------------|-------------|
| Unresolvable rate | % of campaigns with no resolution |
| Disputed spend | $ amount in dispute |
| Divergence over time | Daily delta between buyer/seller records |
| Resolution time | Days from campaign end to resolution |

### What We DO NOT Claim

❌ "IAB A2A is bad" → We show it has a reconciliation gap  
❌ "Blockchain solves everything" → We show it solves reconciliation  
❌ "Current exchanges are good" → They have high fees  
❌ "Our discrepancy rates are exact" → They are calibrated estimates  

---

## Expected Results

### Scenario B (Private DBs)

Based on industry discrepancy data and our simulation model:

| Metric | Expected Value | Range |
|--------|----------------|-------|
| Campaigns with >5% discrepancy | 40-50% | ±10% |
| Campaigns with >15% discrepancy | 15-20% | ±5% |
| Unresolvable disputes | 10-15% | ±5% |
| Disputed spend (% of total) | 12-18% | ±5% |
| Average resolution time | 30-45 days | ±15 days |

### Scenario C (Shared Ledger)

With authoritative record:

| Metric | Expected Value |
|--------|----------------|
| Campaigns with >5% discrepancy | 0% |
| Unresolvable disputes | 0% |
| Resolution time | 0 days (instant) |

### Financial Projection

At $150B global programmatic scale:

| Scenario | Annual Disputed Spend | Annual Unresolvable |
|----------|----------------------|---------------------|
| B (private DBs) | $18-27B | $9-15B |
| C (shared ledger) | ~$0 | ~$0 |

**Cost of blockchain fees:** ~0.1% = $150M  
**Net savings vs disputes:** $9-15B - $150M = **massive positive**

---

## Limitations

### Simulation vs Reality

1. **Simplified discrepancy model:** Real-world has more complex error sources
2. **No adversarial agents:** v2 will add intentional misreporting
3. **No network effects:** Real market has cascading disputes
4. **Fixed agent behavior:** Real agents adapt and learn

### Calibration Uncertainty

1. **Industry data is aggregate:** Individual campaign variance unknown
2. **Publication bias:** Studies may report worst cases
3. **Technology evolves:** 2020-2023 data may be outdated

### Generalization Limits

1. **Campaign types:** Model is for display/video, not all formats
2. **Market conditions:** Results assume current programmatic structure
3. **Geographic:** Data primarily from UK/US markets

---

## Deliverables

### Code Artifacts

1. `src/scenarios/reconciliation.py` - Core reconciliation simulation
2. `src/scenarios/scenario_b.py` - Updated with reconciliation metrics
3. `src/scenarios/scenario_c.py` - Updated with reconciliation metrics
4. `tests/test_reconciliation.py` - Validation tests

### Documentation

1. `docs/CROSS_AGENT_RECONCILIATION_RESEARCH.md` - Full research background
2. `KEY_FINDINGS.md` - Executive summary (updated)
3. `docs/RESEARCH_PLAN.md` - This document
4. `docs/METHODOLOGY.md` - Technical methodology details

### Results

1. `results/reconciliation_comparison.json` - Raw simulation output
2. `results/sensitivity_analysis.json` - Parameter sensitivity
3. `content/cross-agent-reconciliation-findings.md` - Article-ready

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Research & Design | Complete | This document |
| Implementation | 1 week | reconciliation.py, tests |
| Validation | 3 days | Test against industry data |
| Simulation Runs | 2 days | Primary + sensitivity |
| Analysis | 2 days | Results, charts |
| Documentation | 2 days | Findings, content |

---

## Quality Assurance

### Code Review Checklist

- [ ] Discrepancy injection matches industry rates
- [ ] Resolution logic matches described thresholds
- [ ] Metrics calculated correctly
- [ ] Random seed controls reproducibility
- [ ] Edge cases handled (0 impressions, 0 spend)

### Results Validation

- [ ] Scenario B dispute rate in expected range (10-20%)
- [ ] Scenario C shows 100% resolution
- [ ] Financial projections are reasonable
- [ ] Sensitivity analysis shows stability

### Documentation Standards

- [ ] All assumptions explicitly stated
- [ ] All data sources cited
- [ ] Limitations clearly documented
- [ ] No overclaiming of results

---

*Research plan prepared for credible, peer-reviewable simulation*
