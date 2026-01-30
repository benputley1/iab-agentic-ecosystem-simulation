"""
Comprehensive tests for the Hallucination Classifier.

Tests cover all hallucination types, edge cases, and severity calculations.
"""

import pytest
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, 'src')

from hallucination import (
    HallucinationClassifier,
    HallucinationType,
    AgentDecision,
    CampaignState,
    Hallucination,
    HallucinationResult,
    SeverityThresholds,
)


# =============================================================================
# Mock Ground Truth Database
# =============================================================================

class MockGroundTruthDB:
    """
    Mock implementation of GroundTruthDB for testing.
    
    Allows test cases to set up specific ground truth states
    and verify classifier behavior.
    """
    
    def __init__(self):
        self.campaigns: dict[str, CampaignState] = {}
        self.deals: dict[str, dict[str, Any]] = {}
        self.user_frequencies: dict[tuple[str, str], int] = {}  # (campaign_id, user_id) -> count
        self.inventory: set[str] = set()
        self.floor_prices: dict[str, float] = {}
    
    def set_campaign_state(self, state: CampaignState) -> None:
        """Set the ground truth state for a campaign."""
        self.campaigns[state.campaign_id] = state
    
    def set_deal(self, deal_id: str, terms: dict[str, Any]) -> None:
        """Set deal terms in ground truth."""
        self.deals[deal_id] = terms
    
    def set_user_frequency(self, campaign_id: str, user_id: str, count: int) -> None:
        """Set user frequency count."""
        self.user_frequencies[(campaign_id, user_id)] = count
    
    def add_inventory(self, inventory_id: str, floor_price: float = 0.0) -> None:
        """Add inventory to ground truth."""
        self.inventory.add(inventory_id)
        self.floor_prices[inventory_id] = floor_price
    
    # Protocol implementation methods
    
    def get_campaign_state(self, campaign_id: str, at_time: datetime) -> CampaignState:
        """Get campaign state (ignores time for simplicity in tests)."""
        if campaign_id in self.campaigns:
            return self.campaigns[campaign_id]
        # Return default empty state
        return CampaignState(
            campaign_id=campaign_id,
            budget_total=10000.0,
            budget_spent=0.0,
            budget_remaining=10000.0,
            impressions=0
        )
    
    def get_deal(self, deal_id: str) -> Optional[dict[str, Any]]:
        """Get deal by ID."""
        return self.deals.get(deal_id)
    
    def get_user_frequency(self, campaign_id: str, user_id: str, at_time: datetime) -> int:
        """Get user frequency count."""
        return self.user_frequencies.get((campaign_id, user_id), 0)
    
    def inventory_exists(self, inventory_id: str, at_time: datetime) -> bool:
        """Check if inventory exists."""
        return inventory_id in self.inventory
    
    def get_floor_price(self, inventory_id: str, at_time: datetime) -> Optional[float]:
        """Get floor price for inventory."""
        return self.floor_prices.get(inventory_id)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def ground_truth() -> MockGroundTruthDB:
    """Create a mock ground truth database."""
    db = MockGroundTruthDB()
    
    # Set up default campaign
    db.set_campaign_state(CampaignState(
        campaign_id="campaign_001",
        budget_total=10000.0,
        budget_spent=3500.0,
        budget_remaining=6500.0,
        impressions=1000,
        frequency_caps={"user_001": 5},
        active_deals={"deal_001": {"floor_price": 2.50, "priority": "high"}},
        floor_prices={"inv_001": 1.25},
        available_inventory={"inv_001", "inv_002"}
    ))
    
    # Set up some deals
    db.set_deal("deal_001", {"floor_price": 2.50, "priority": "high", "advertiser": "brand_x"})
    db.set_deal("deal_002", {"floor_price": 1.75, "priority": "medium"})
    
    # Set up user frequencies
    db.set_user_frequency("campaign_001", "user_001", 3)
    db.set_user_frequency("campaign_001", "user_002", 5)  # At cap
    
    # Set up inventory
    db.add_inventory("inv_001", floor_price=1.25)
    db.add_inventory("inv_002", floor_price=0.80)
    
    return db


@pytest.fixture
def classifier(ground_truth: MockGroundTruthDB) -> HallucinationClassifier:
    """Create a classifier with the mock ground truth."""
    return HallucinationClassifier(ground_truth)


@pytest.fixture
def base_decision() -> AgentDecision:
    """Create a base valid decision for testing."""
    return AgentDecision(
        id="decision_001",
        timestamp=datetime.now(),
        campaign_id="campaign_001",
        budget_remaining=6500.0,
        bid_amount=2.00
    )


# =============================================================================
# Test HallucinationType Enum
# =============================================================================

class TestHallucinationType:
    """Tests for HallucinationType enum."""
    
    def test_all_types_defined(self):
        """Verify all required hallucination types exist."""
        required_types = [
            "BUDGET_DRIFT",
            "FREQUENCY_VIOLATION", 
            "DEAL_INVENTION",
            "CAMPAIGN_CROSS_CONTAMINATION",
            "PHANTOM_INVENTORY",
            "PRICE_ANCHOR_ERROR"
        ]
        
        for type_name in required_types:
            assert hasattr(HallucinationType, type_name), f"Missing type: {type_name}"
    
    def test_type_values(self):
        """Verify enum values are strings."""
        assert HallucinationType.BUDGET_DRIFT.value == "budget_drift"
        assert HallucinationType.FREQUENCY_VIOLATION.value == "frequency_cap"
        assert HallucinationType.DEAL_INVENTION.value == "deal_invention"
        assert HallucinationType.CAMPAIGN_CROSS_CONTAMINATION.value == "cross_campaign"
        assert HallucinationType.PHANTOM_INVENTORY.value == "phantom_inventory"
        assert HallucinationType.PRICE_ANCHOR_ERROR.value == "price_anchor"


# =============================================================================
# Test Budget Drift Detection
# =============================================================================

class TestBudgetDrift:
    """Tests for budget drift hallucination detection."""
    
    def test_no_drift_within_threshold(self, classifier: HallucinationClassifier):
        """No hallucination when drift is within 1% threshold."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=6500.0 * 1.005,  # 0.5% drift - within threshold
        )
        
        result = classifier.check_decision(decision)
        
        budget_errors = [e for e in result.errors if e.type == HallucinationType.BUDGET_DRIFT]
        assert len(budget_errors) == 0
    
    def test_detects_drift_over_threshold(self, classifier: HallucinationClassifier):
        """Detects hallucination when drift exceeds 1% threshold."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=6500.0 * 1.05,  # 5% drift - over threshold
        )
        
        result = classifier.check_decision(decision)
        
        budget_errors = [e for e in result.errors if e.type == HallucinationType.BUDGET_DRIFT]
        assert len(budget_errors) == 1
        assert budget_errors[0].expected == 6500.0
        assert budget_errors[0].actual == 6500.0 * 1.05
        assert 0.04 < budget_errors[0].severity < 0.06
    
    def test_severe_budget_drift(self, classifier: HallucinationClassifier):
        """High severity for large budget drift."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=6500.0 * 2,  # 100% drift
        )
        
        result = classifier.check_decision(decision)
        
        budget_errors = [e for e in result.errors if e.type == HallucinationType.BUDGET_DRIFT]
        assert len(budget_errors) == 1
        assert budget_errors[0].severity == 1.0  # Capped at 1.0
    
    def test_budget_exhausted_but_agent_thinks_remaining(
        self, 
        classifier: HallucinationClassifier,
        ground_truth: MockGroundTruthDB
    ):
        """Detects hallucination when budget is exhausted but agent thinks money remains."""
        # Set budget to exhausted
        ground_truth.set_campaign_state(CampaignState(
            campaign_id="campaign_exhausted",
            budget_total=1000.0,
            budget_spent=1000.0,
            budget_remaining=0.0,
            impressions=500
        ))
        
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_exhausted",
            budget_remaining=500.0,  # Agent thinks money remains
        )
        
        result = classifier.check_decision(decision)
        
        budget_errors = [e for e in result.errors if e.type == HallucinationType.BUDGET_DRIFT]
        assert len(budget_errors) == 1
        assert budget_errors[0].severity == 1.0
    
    def test_negative_drift(self, classifier: HallucinationClassifier):
        """Detects underestimation of remaining budget."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=6500.0 * 0.90,  # Agent thinks 10% less
        )
        
        result = classifier.check_decision(decision)
        
        budget_errors = [e for e in result.errors if e.type == HallucinationType.BUDGET_DRIFT]
        assert len(budget_errors) == 1
        assert budget_errors[0].severity > 0.09
    
    def test_no_budget_field_no_check(self, classifier: HallucinationClassifier):
        """No check performed when budget_remaining is not set."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            # No budget_remaining set
        )
        
        result = classifier.check_decision(decision)
        
        budget_errors = [e for e in result.errors if e.type == HallucinationType.BUDGET_DRIFT]
        assert len(budget_errors) == 0


# =============================================================================
# Test Frequency Violation Detection
# =============================================================================

class TestFrequencyViolation:
    """Tests for frequency cap violation detection."""
    
    def test_correct_frequency_count(self, classifier: HallucinationClassifier):
        """No hallucination when frequency count is correct."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            user_id="user_001",
            user_exposure_count=3,  # Matches ground truth
        )
        
        result = classifier.check_decision(decision)
        
        freq_errors = [e for e in result.errors if e.type == HallucinationType.FREQUENCY_VIOLATION]
        assert len(freq_errors) == 0
    
    def test_detects_wrong_frequency_count(self, classifier: HallucinationClassifier):
        """Detects when agent has wrong exposure count."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            user_id="user_001",
            user_exposure_count=1,  # Agent thinks 1, actual is 3
        )
        
        result = classifier.check_decision(decision)
        
        freq_errors = [e for e in result.errors if e.type == HallucinationType.FREQUENCY_VIOLATION]
        assert len(freq_errors) == 1
        assert freq_errors[0].expected == 3
        assert freq_errors[0].actual == 1
    
    def test_high_severity_when_cap_violated(
        self, 
        classifier: HallucinationClassifier,
        ground_truth: MockGroundTruthDB
    ):
        """Higher severity when miscount leads to cap violation."""
        # User_002 is at cap (5 impressions)
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            user_id="user_002",
            user_exposure_count=2,  # Agent thinks 2, actual is 5 (at cap)
            frequency_cap=5,
        )
        
        result = classifier.check_decision(decision)
        
        freq_errors = [e for e in result.errors if e.type == HallucinationType.FREQUENCY_VIOLATION]
        assert len(freq_errors) == 1
        assert freq_errors[0].severity >= 0.5  # Higher severity for cap violation


# =============================================================================
# Test Deal Invention Detection
# =============================================================================

class TestDealInvention:
    """Tests for deal invention hallucination detection."""
    
    def test_valid_deal_no_error(self, classifier: HallucinationClassifier):
        """No hallucination when referencing valid deal."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            deal_id="deal_001",
        )
        
        result = classifier.check_decision(decision)
        
        deal_errors = [e for e in result.errors if e.type == HallucinationType.DEAL_INVENTION]
        assert len(deal_errors) == 0
    
    def test_detects_nonexistent_deal(self, classifier: HallucinationClassifier):
        """Detects reference to non-existent deal."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            deal_id="deal_phantom",  # Doesn't exist
        )
        
        result = classifier.check_decision(decision)
        
        deal_errors = [e for e in result.errors if e.type == HallucinationType.DEAL_INVENTION]
        assert len(deal_errors) == 1
        assert deal_errors[0].severity == 1.0
        assert "does not exist" in deal_errors[0].description
    
    def test_detects_wrong_deal_terms(self, classifier: HallucinationClassifier):
        """Detects when agent has wrong deal terms."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            deal_id="deal_001",
            deal_terms={"floor_price": 2.50, "priority": "low"}  # Wrong priority
        )
        
        result = classifier.check_decision(decision)
        
        deal_errors = [e for e in result.errors if e.type == HallucinationType.DEAL_INVENTION]
        assert len(deal_errors) == 1
        assert deal_errors[0].severity > 0
        assert deal_errors[0].severity < 1.0
    
    def test_detects_wrong_deal_floor_price(self, classifier: HallucinationClassifier):
        """Detects when agent has wrong deal floor price."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            deal_id="deal_001",
            deal_floor_price=3.00,  # Actual is 2.50
        )
        
        result = classifier.check_decision(decision)
        
        deal_errors = [e for e in result.errors if e.type == HallucinationType.DEAL_INVENTION]
        assert len(deal_errors) == 1
        assert deal_errors[0].expected == 2.50
        assert deal_errors[0].actual == 3.00


# =============================================================================
# Test Campaign Cross-Contamination Detection
# =============================================================================

class TestCrossContamination:
    """Tests for campaign cross-contamination detection."""
    
    def test_correct_campaign_no_error(self, classifier: HallucinationClassifier):
        """No error when campaign ID matches."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
        )
        
        result = classifier.check_decision(decision)
        
        cross_errors = [e for e in result.errors if e.type == HallucinationType.CAMPAIGN_CROSS_CONTAMINATION]
        assert len(cross_errors) == 0


# =============================================================================
# Test Phantom Inventory Detection
# =============================================================================

class TestPhantomInventory:
    """Tests for phantom inventory detection."""
    
    def test_valid_inventory_no_error(self, classifier: HallucinationClassifier):
        """No hallucination when referencing valid inventory."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            inventory_id="inv_001",
        )
        
        result = classifier.check_decision(decision)
        
        phantom_errors = [e for e in result.errors if e.type == HallucinationType.PHANTOM_INVENTORY]
        assert len(phantom_errors) == 0
    
    def test_detects_nonexistent_inventory(self, classifier: HallucinationClassifier):
        """Detects reference to non-existent inventory."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            inventory_id="inv_phantom",
        )
        
        result = classifier.check_decision(decision)
        
        phantom_errors = [e for e in result.errors if e.type == HallucinationType.PHANTOM_INVENTORY]
        assert len(phantom_errors) == 1
        assert phantom_errors[0].severity == 0.8


# =============================================================================
# Test Price Anchor Error Detection
# =============================================================================

class TestPriceAnchorError:
    """Tests for price anchor error detection."""
    
    def test_correct_floor_price_no_error(self, classifier: HallucinationClassifier):
        """No hallucination when floor price is correct."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            inventory_id="inv_001",
            expected_floor_price=1.25,  # Matches ground truth
        )
        
        result = classifier.check_decision(decision)
        
        price_errors = [e for e in result.errors if e.type == HallucinationType.PRICE_ANCHOR_ERROR]
        assert len(price_errors) == 0
    
    def test_price_within_threshold(self, classifier: HallucinationClassifier):
        """No error when price is within 5% threshold."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            inventory_id="inv_001",
            expected_floor_price=1.25 * 1.03,  # 3% off - within threshold
        )
        
        result = classifier.check_decision(decision)
        
        price_errors = [e for e in result.errors if e.type == HallucinationType.PRICE_ANCHOR_ERROR]
        assert len(price_errors) == 0
    
    def test_detects_wrong_floor_price(self, classifier: HallucinationClassifier):
        """Detects when agent has wrong floor price."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            inventory_id="inv_001",
            expected_floor_price=1.50,  # Actual is 1.25, 20% off
        )
        
        result = classifier.check_decision(decision)
        
        price_errors = [e for e in result.errors if e.type == HallucinationType.PRICE_ANCHOR_ERROR]
        assert len(price_errors) == 1
        assert price_errors[0].expected == 1.25
        assert price_errors[0].actual == 1.50
        assert price_errors[0].severity == pytest.approx(0.2, rel=0.01)


# =============================================================================
# Test HallucinationResult
# =============================================================================

class TestHallucinationResult:
    """Tests for HallucinationResult dataclass."""
    
    def test_has_hallucinations_true(self):
        """has_hallucinations returns True when errors exist."""
        result = HallucinationResult(
            decision_id="d001",
            timestamp=datetime.now(),
            errors=[
                Hallucination(
                    type=HallucinationType.BUDGET_DRIFT,
                    expected=100,
                    actual=150,
                    severity=0.5
                )
            ]
        )
        
        assert result.has_hallucinations is True
    
    def test_has_hallucinations_false(self):
        """has_hallucinations returns False when no errors."""
        result = HallucinationResult(
            decision_id="d001",
            timestamp=datetime.now(),
            errors=[]
        )
        
        assert result.has_hallucinations is False
    
    def test_total_severity(self):
        """total_severity sums all error severities."""
        result = HallucinationResult(
            decision_id="d001",
            timestamp=datetime.now(),
            errors=[
                Hallucination(type=HallucinationType.BUDGET_DRIFT, expected=0, actual=1, severity=0.3),
                Hallucination(type=HallucinationType.PRICE_ANCHOR_ERROR, expected=0, actual=1, severity=0.4),
            ]
        )
        
        assert result.total_severity == pytest.approx(0.7)
    
    def test_max_severity(self):
        """max_severity returns highest severity."""
        result = HallucinationResult(
            decision_id="d001",
            timestamp=datetime.now(),
            errors=[
                Hallucination(type=HallucinationType.BUDGET_DRIFT, expected=0, actual=1, severity=0.3),
                Hallucination(type=HallucinationType.DEAL_INVENTION, expected=0, actual=1, severity=0.9),
                Hallucination(type=HallucinationType.PRICE_ANCHOR_ERROR, expected=0, actual=1, severity=0.4),
            ]
        )
        
        assert result.max_severity == 0.9
    
    def test_hallucination_types(self):
        """hallucination_types returns set of types."""
        result = HallucinationResult(
            decision_id="d001",
            timestamp=datetime.now(),
            errors=[
                Hallucination(type=HallucinationType.BUDGET_DRIFT, expected=0, actual=1, severity=0.3),
                Hallucination(type=HallucinationType.BUDGET_DRIFT, expected=0, actual=1, severity=0.4),
                Hallucination(type=HallucinationType.PRICE_ANCHOR_ERROR, expected=0, actual=1, severity=0.5),
            ]
        )
        
        assert result.hallucination_types == {
            HallucinationType.BUDGET_DRIFT,
            HallucinationType.PRICE_ANCHOR_ERROR
        }


# =============================================================================
# Test Hallucination Dataclass
# =============================================================================

class TestHallucination:
    """Tests for Hallucination dataclass."""
    
    def test_valid_severity_range(self):
        """Accepts severity values in valid range."""
        h = Hallucination(
            type=HallucinationType.BUDGET_DRIFT,
            expected=100,
            actual=150,
            severity=0.5
        )
        assert h.severity == 0.5
    
    def test_severity_zero(self):
        """Accepts severity of 0."""
        h = Hallucination(
            type=HallucinationType.BUDGET_DRIFT,
            expected=100,
            actual=150,
            severity=0.0
        )
        assert h.severity == 0.0
    
    def test_severity_one(self):
        """Accepts severity of 1."""
        h = Hallucination(
            type=HallucinationType.BUDGET_DRIFT,
            expected=100,
            actual=150,
            severity=1.0
        )
        assert h.severity == 1.0
    
    def test_invalid_severity_above_one(self):
        """Rejects severity > 1."""
        with pytest.raises(ValueError, match="Severity must be between 0 and 1"):
            Hallucination(
                type=HallucinationType.BUDGET_DRIFT,
                expected=100,
                actual=150,
                severity=1.5
            )
    
    def test_invalid_severity_negative(self):
        """Rejects negative severity."""
        with pytest.raises(ValueError, match="Severity must be between 0 and 1"):
            Hallucination(
                type=HallucinationType.BUDGET_DRIFT,
                expected=100,
                actual=150,
                severity=-0.1
            )


# =============================================================================
# Test Classifier Statistics
# =============================================================================

class TestClassifierStats:
    """Tests for classifier statistics tracking."""
    
    def test_stats_tracking(self, classifier: HallucinationClassifier):
        """Stats are correctly tracked across decisions."""
        # First decision - no hallucinations
        decision1 = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=6500.0,
        )
        classifier.check_decision(decision1)
        
        # Second decision - with hallucination
        decision2 = AgentDecision(
            id="d002",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=10000.0,  # Wrong - causes drift
        )
        classifier.check_decision(decision2)
        
        stats = classifier.get_stats()
        assert stats.total_decisions_checked == 2
        assert stats.total_hallucinations == 1
        assert stats.hallucination_rate == 0.5
    
    def test_stats_by_type(self, classifier: HallucinationClassifier):
        """Stats correctly track hallucinations by type."""
        decisions = [
            AgentDecision(
                id=f"d{i}",
                timestamp=datetime.now(),
                campaign_id="campaign_001",
                budget_remaining=10000.0,  # Budget drift
            )
            for i in range(3)
        ]
        
        for d in decisions:
            classifier.check_decision(d)
        
        stats = classifier.get_stats()
        assert stats.hallucinations_by_type.get(HallucinationType.BUDGET_DRIFT, 0) == 3
    
    def test_reset_stats(self, classifier: HallucinationClassifier):
        """reset_stats clears statistics."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=10000.0,
        )
        classifier.check_decision(decision)
        
        classifier.reset_stats()
        
        stats = classifier.get_stats()
        assert stats.total_decisions_checked == 0


# =============================================================================
# Test Custom Thresholds
# =============================================================================

class TestCustomThresholds:
    """Tests for custom severity thresholds."""
    
    def test_custom_budget_threshold(self, ground_truth: MockGroundTruthDB):
        """Custom budget drift threshold works."""
        # More lenient threshold
        thresholds = SeverityThresholds(budget_drift_pct=0.10)  # 10%
        classifier = HallucinationClassifier(ground_truth, thresholds)
        
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=6500.0 * 1.05,  # 5% drift - within new threshold
        )
        
        result = classifier.check_decision(decision)
        
        budget_errors = [e for e in result.errors if e.type == HallucinationType.BUDGET_DRIFT]
        assert len(budget_errors) == 0
    
    def test_stricter_price_threshold(self, ground_truth: MockGroundTruthDB):
        """Stricter price anchor threshold works."""
        thresholds = SeverityThresholds(price_anchor_pct=0.01)  # 1%
        classifier = HallucinationClassifier(ground_truth, thresholds)
        
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            inventory_id="inv_001",
            expected_floor_price=1.25 * 1.03,  # 3% off - now triggers
        )
        
        result = classifier.check_decision(decision)
        
        price_errors = [e for e in result.errors if e.type == HallucinationType.PRICE_ANCHOR_ERROR]
        assert len(price_errors) == 1


# =============================================================================
# Test Summary Report
# =============================================================================

class TestHallucinationSummary:
    """Tests for hallucination summary generation."""
    
    def test_summary_generation(self, classifier: HallucinationClassifier):
        """get_hallucination_summary returns correct structure."""
        # Generate some hallucinations
        decisions = [
            AgentDecision(
                id="d001",
                timestamp=datetime.now(),
                campaign_id="campaign_001",
                budget_remaining=10000.0,  # Budget drift (high severity)
            ),
            AgentDecision(
                id="d002",
                timestamp=datetime.now(),
                campaign_id="campaign_001",
                inventory_id="inv_phantom",  # Phantom inventory
            ),
        ]
        
        for d in decisions:
            classifier.check_decision(d)
        
        summary = classifier.get_hallucination_summary()
        
        assert summary["total"] == 2
        assert "budget_drift" in summary["by_type"]
        assert "phantom_inventory" in summary["by_type"]
        assert "severity_distribution" in summary


# =============================================================================
# Test Multiple Hallucinations in Single Decision
# =============================================================================

class TestMultipleHallucinations:
    """Tests for detecting multiple hallucinations in one decision."""
    
    def test_multiple_errors_detected(self, classifier: HallucinationClassifier):
        """Multiple hallucination types can be detected in single decision."""
        decision = AgentDecision(
            id="d001",
            timestamp=datetime.now(),
            campaign_id="campaign_001",
            budget_remaining=10000.0,  # Budget drift
            inventory_id="inv_phantom",  # Phantom inventory
            deal_id="deal_phantom",  # Invented deal
        )
        
        result = classifier.check_decision(decision)
        
        assert result.has_hallucinations
        assert len(result.errors) >= 3
        
        types_detected = {e.type for e in result.errors}
        assert HallucinationType.BUDGET_DRIFT in types_detected
        assert HallucinationType.PHANTOM_INVENTORY in types_detected
        assert HallucinationType.DEAL_INVENTION in types_detected


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
