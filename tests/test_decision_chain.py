"""
Comprehensive tests for the Decision Chain Tracker component.

Tests cover:
- Basic decision recording
- Reference verification (correct and incorrect)
- Lookback window behavior and stale data detection
- Missing reference detection
- Cascading error detection and tracking
- Error classification and distribution
- Chain depth calculation
- Edge cases and error conditions
"""

import pytest
from datetime import datetime, timedelta
from src.tracking import (
    DecisionChainTracker,
    AgentDecision,
    DecisionReference,
    ReferenceFailure,
    ReferenceErrorType,
    ChainResult,
)


class TestDecisionChainTrackerInit:
    """Test tracker initialization."""
    
    def test_default_lookback_window(self):
        """Default lookback window should be 100."""
        tracker = DecisionChainTracker()
        assert tracker.lookback == 100
    
    def test_custom_lookback_window(self):
        """Should accept custom lookback window."""
        tracker = DecisionChainTracker(lookback_window=50)
        assert tracker.lookback == 50
    
    def test_invalid_lookback_window_zero(self):
        """Should reject lookback window of 0."""
        with pytest.raises(ValueError, match="at least 1"):
            DecisionChainTracker(lookback_window=0)
    
    def test_invalid_lookback_window_negative(self):
        """Should reject negative lookback window."""
        with pytest.raises(ValueError, match="at least 1"):
            DecisionChainTracker(lookback_window=-5)
    
    def test_empty_initial_state(self):
        """New tracker should have empty state."""
        tracker = DecisionChainTracker()
        assert tracker.total_decisions == 0
        assert tracker.total_failures == 0
        assert len(tracker.reference_failures) == 0


class TestBasicDecisionRecording:
    """Test basic decision recording without references."""
    
    def test_record_simple_decision(self):
        """Should record a decision without references."""
        tracker = DecisionChainTracker()
        decision = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        
        result = tracker.record_decision(decision)
        
        assert result.success is True
        assert result.decision_id == "d1"
        assert result.total_errors == 0
        assert tracker.total_decisions == 1
    
    def test_record_multiple_decisions(self):
        """Should record multiple decisions."""
        tracker = DecisionChainTracker()
        
        for i in range(5):
            decision = AgentDecision(
                id=f"d{i}",
                timestamp=datetime.utcnow(),
                agent_id="agent1",
                decision_type="bid",
                value=100.0 + i
            )
            tracker.record_decision(decision)
        
        assert tracker.total_decisions == 5
    
    def test_get_recorded_decision(self):
        """Should be able to retrieve recorded decision."""
        tracker = DecisionChainTracker()
        decision = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        tracker.record_decision(decision)
        
        retrieved = tracker.get_decision("d1")
        assert retrieved is not None
        assert retrieved.id == "d1"
        assert retrieved.value == 100.0
    
    def test_get_nonexistent_decision(self):
        """Should return None for nonexistent decision."""
        tracker = DecisionChainTracker()
        assert tracker.get_decision("nonexistent") is None


class TestReferenceVerification:
    """Test reference verification between decisions."""
    
    def test_valid_reference(self):
        """Should accept valid reference to previous decision."""
        tracker = DecisionChainTracker()
        
        # First decision
        d1 = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        tracker.record_decision(d1)
        
        # Second decision referencing first
        d2 = AgentDecision(
            id="d2",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=100.0)]
        )
        result = tracker.record_decision(d2)
        
        assert result.success is True
        assert result.total_errors == 0
    
    def test_wrong_value_reference(self):
        """Should detect when recalled value doesn't match actual."""
        tracker = DecisionChainTracker()
        
        # First decision with value 100
        d1 = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        tracker.record_decision(d1)
        
        # Second decision recalls wrong value
        d2 = AgentDecision(
            id="d2",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=90.0)]  # Wrong!
        )
        result = tracker.record_decision(d2)
        
        assert result.success is False
        assert result.total_errors == 1
        assert result.failures[0].error_type == ReferenceErrorType.WRONG_VALUE
        assert result.failures[0].expected == 90.0
        assert result.failures[0].actual == 100.0
    
    def test_missing_reference(self):
        """Should detect reference to non-existent decision."""
        tracker = DecisionChainTracker()
        
        decision = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0,
            references=[DecisionReference(decision_id="nonexistent", recalled_value=50.0)]
        )
        result = tracker.record_decision(decision)
        
        assert result.success is False
        assert result.total_errors == 1
        assert result.failures[0].error_type == ReferenceErrorType.MISSING_REFERENCE
        assert result.failures[0].actual is None
    
    def test_multiple_references_all_valid(self):
        """Should handle multiple valid references."""
        tracker = DecisionChainTracker()
        
        # Create base decisions
        for i in range(3):
            d = AgentDecision(
                id=f"d{i}",
                timestamp=datetime.utcnow(),
                agent_id="agent1",
                decision_type="bid",
                value=float(100 + i * 10)
            )
            tracker.record_decision(d)
        
        # Decision referencing all three
        d4 = AgentDecision(
            id="d4",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=200.0,
            references=[
                DecisionReference(decision_id="d0", recalled_value=100.0),
                DecisionReference(decision_id="d1", recalled_value=110.0),
                DecisionReference(decision_id="d2", recalled_value=120.0),
            ]
        )
        result = tracker.record_decision(d4)
        
        assert result.success is True
        assert result.total_errors == 0
    
    def test_multiple_references_mixed_validity(self):
        """Should detect errors in some references while others are valid."""
        tracker = DecisionChainTracker()
        
        # Create base decisions
        for i in range(2):
            d = AgentDecision(
                id=f"d{i}",
                timestamp=datetime.utcnow(),
                agent_id="agent1",
                decision_type="bid",
                value=float(100 + i * 10)
            )
            tracker.record_decision(d)
        
        # Decision with one valid and one invalid reference
        d3 = AgentDecision(
            id="d3",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=200.0,
            references=[
                DecisionReference(decision_id="d0", recalled_value=100.0),  # Valid
                DecisionReference(decision_id="d1", recalled_value=999.0),  # Wrong!
            ]
        )
        result = tracker.record_decision(d3)
        
        assert result.success is False
        assert result.total_errors == 1
        assert result.failures[0].ref_id == "d1"


class TestLookbackWindow:
    """Test lookback window and stale data detection."""
    
    def test_decisions_within_window(self):
        """Decisions within window should be retrievable."""
        tracker = DecisionChainTracker(lookback_window=5)
        
        for i in range(5):
            d = AgentDecision(
                id=f"d{i}",
                timestamp=datetime.utcnow(),
                agent_id="agent1",
                decision_type="bid",
                value=float(i)
            )
            tracker.record_decision(d)
        
        # All should be retrievable
        for i in range(5):
            assert tracker.get_decision(f"d{i}") is not None
    
    def test_decisions_evicted_from_window(self):
        """Old decisions should be evicted when window is full."""
        tracker = DecisionChainTracker(lookback_window=3)
        
        # Add 5 decisions to a window of 3
        for i in range(5):
            d = AgentDecision(
                id=f"d{i}",
                timestamp=datetime.utcnow(),
                agent_id="agent1",
                decision_type="bid",
                value=float(i)
            )
            tracker.record_decision(d)
        
        # First two should be evicted
        assert tracker.get_decision("d0") is None
        assert tracker.get_decision("d1") is None
        # Last three should exist
        assert tracker.get_decision("d2") is not None
        assert tracker.get_decision("d3") is not None
        assert tracker.get_decision("d4") is not None
    
    def test_stale_reference_detection(self):
        """Should detect reference to evicted (stale) decision."""
        tracker = DecisionChainTracker(lookback_window=3)
        
        # Fill the window
        for i in range(5):
            d = AgentDecision(
                id=f"d{i}",
                timestamp=datetime.utcnow(),
                agent_id="agent1",
                decision_type="bid",
                value=float(i)
            )
            tracker.record_decision(d)
        
        # Try to reference evicted decision d0
        d_new = AgentDecision(
            id="d5",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0,
            references=[DecisionReference(decision_id="d0", recalled_value=0.0)]
        )
        result = tracker.record_decision(d_new)
        
        assert result.success is False
        assert result.failures[0].error_type == ReferenceErrorType.STALE_DATA
    
    def test_window_maintains_correct_size(self):
        """Window should never exceed configured size."""
        tracker = DecisionChainTracker(lookback_window=10)
        
        for i in range(100):
            d = AgentDecision(
                id=f"d{i}",
                timestamp=datetime.utcnow(),
                agent_id="agent1",
                decision_type="bid",
                value=float(i)
            )
            tracker.record_decision(d)
        
        assert tracker.total_decisions == 10


class TestCascadingErrors:
    """Test cascading error detection."""
    
    def test_detect_cascading_error(self):
        """Should detect when error is caused by previous error."""
        tracker = DecisionChainTracker()
        
        # First decision (no errors)
        d1 = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        tracker.record_decision(d1)
        
        # Second decision with error
        d2 = AgentDecision(
            id="d2",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=50.0)]  # Wrong!
        )
        tracker.record_decision(d2)
        
        # Third decision references d2 (which had an error)
        d3 = AgentDecision(
            id="d3",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=120.0,
            references=[DecisionReference(decision_id="d2", recalled_value=999.0)]  # Wrong + cascading
        )
        result = tracker.record_decision(d3)
        
        assert result.success is False
        assert result.cascading_errors == 1
        assert result.failures[0].error_type == ReferenceErrorType.CASCADING_ERROR
        assert result.failures[0].caused_by == "d2"
    
    def test_no_cascading_when_reference_correct(self):
        """Correct reference to error decision should not be cascading."""
        tracker = DecisionChainTracker()
        
        # First decision
        d1 = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        tracker.record_decision(d1)
        
        # Second decision with error
        d2 = AgentDecision(
            id="d2",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=50.0)]  # Wrong!
        )
        tracker.record_decision(d2)
        
        # Third decision correctly references d2
        d3 = AgentDecision(
            id="d3",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=120.0,
            references=[DecisionReference(decision_id="d2", recalled_value=110.0)]  # Correct!
        )
        result = tracker.record_decision(d3)
        
        assert result.success is True
        assert result.cascading_errors == 0
    
    def test_count_cascading_errors_method(self):
        """count_cascading_errors should count references to error decisions."""
        tracker = DecisionChainTracker()
        
        # Create decisions with errors
        d1 = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        tracker.record_decision(d1)
        
        # d2 has an error
        d2 = AgentDecision(
            id="d2",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=50.0)]
        )
        tracker.record_decision(d2)
        
        # d3 has an error
        d3 = AgentDecision(
            id="d3",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=120.0,
            references=[DecisionReference(decision_id="d1", recalled_value=30.0)]
        )
        tracker.record_decision(d3)
        
        # New decision references both error decisions
        d4 = AgentDecision(
            id="d4",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=200.0,
            references=[
                DecisionReference(decision_id="d2", recalled_value=110.0),
                DecisionReference(decision_id="d3", recalled_value=120.0),
            ]
        )
        
        count = tracker.count_cascading_errors(d4)
        assert count == 2


class TestChainDepth:
    """Test chain depth calculation."""
    
    def test_chain_depth_no_references(self):
        """Decision without references should have depth 0."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        result = tracker.record_decision(d1)
        
        assert result.chain_depth == 0
    
    def test_chain_depth_one_level(self):
        """Reference to single decision should have depth 1."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(
            id="d1",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=100.0
        )
        tracker.record_decision(d1)
        
        d2 = AgentDecision(
            id="d2",
            timestamp=datetime.utcnow(),
            agent_id="agent1",
            decision_type="bid",
            value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=100.0)]
        )
        result = tracker.record_decision(d2)
        
        assert result.chain_depth == 1
    
    def test_chain_depth_nested(self):
        """Nested references should calculate correct depth."""
        tracker = DecisionChainTracker()
        
        # d1 -> d2 -> d3 -> d4 (depth of 3)
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=1.0)
        tracker.record_decision(d1)
        
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=2.0,
            references=[DecisionReference(decision_id="d1", recalled_value=1.0)]
        )
        tracker.record_decision(d2)
        
        d3 = AgentDecision(
            id="d3", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=3.0,
            references=[DecisionReference(decision_id="d2", recalled_value=2.0)]
        )
        tracker.record_decision(d3)
        
        d4 = AgentDecision(
            id="d4", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=4.0,
            references=[DecisionReference(decision_id="d3", recalled_value=3.0)]
        )
        result = tracker.record_decision(d4)
        
        assert result.chain_depth == 3


class TestErrorDistribution:
    """Test error distribution tracking."""
    
    def test_get_error_distribution(self):
        """Should track distribution of error types."""
        tracker = DecisionChainTracker(lookback_window=5)
        
        # Create base decision
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        # Add various error types
        # Wrong value error
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=50.0)]
        )
        tracker.record_decision(d2)
        
        # Missing reference error
        d3 = AgentDecision(
            id="d3", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=120.0,
            references=[DecisionReference(decision_id="nonexistent", recalled_value=50.0)]
        )
        tracker.record_decision(d3)
        
        distribution = tracker.get_error_distribution()
        
        assert distribution[ReferenceErrorType.WRONG_VALUE] == 1
        assert distribution[ReferenceErrorType.MISSING_REFERENCE] == 1
    
    def test_reference_accuracy_rate_all_correct(self):
        """Accuracy should be 1.0 when all references are correct."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=100.0)]
        )
        tracker.record_decision(d2)
        
        assert tracker.reference_accuracy_rate == 1.0
    
    def test_reference_accuracy_rate_with_errors(self):
        """Accuracy should decrease with errors."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        # 2 references: 1 correct, 1 wrong = 50% accuracy
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[
                DecisionReference(decision_id="d1", recalled_value=100.0),  # Correct
                DecisionReference(decision_id="d1", recalled_value=50.0, field_name="metadata"),  # Wrong
            ]
        )
        tracker.record_decision(d2)
        
        # 1 failure out of 2 references = 50%
        assert tracker.reference_accuracy_rate == 0.5
    
    def test_reference_accuracy_no_references(self):
        """Accuracy should be 1.0 when no references exist."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        assert tracker.reference_accuracy_rate == 1.0


class TestFieldReferences:
    """Test references to different fields."""
    
    def test_reference_to_metadata_field(self):
        """Should verify references to metadata fields."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(
            id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", 
            value=100.0,
            metadata={"budget": 5000.0, "campaign_id": "camp_123"}
        )
        tracker.record_decision(d1)
        
        # Valid reference to metadata
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=5000.0, field_name="budget")]
        )
        result = tracker.record_decision(d2)
        
        assert result.success is True
    
    def test_reference_to_wrong_metadata_value(self):
        """Should detect wrong metadata value."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(
            id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", 
            value=100.0,
            metadata={"budget": 5000.0}
        )
        tracker.record_decision(d1)
        
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=9999.0, field_name="budget")]
        )
        result = tracker.record_decision(d2)
        
        assert result.success is False
        assert result.failures[0].field_name == "budget"


class TestAgentDecisionDataclass:
    """Test AgentDecision dataclass functionality."""
    
    def test_add_reference(self):
        """Should be able to add references to decision."""
        d = AgentDecision(
            id="d1", timestamp=datetime.utcnow(), agent_id="a", 
            decision_type="bid", value=100.0
        )
        
        ref = DecisionReference(decision_id="d0", recalled_value=50.0)
        d.add_reference(ref)
        
        assert len(d.get_references()) == 1
        assert d.get_references()[0].decision_id == "d0"
    
    def test_default_empty_references(self):
        """Decision should have empty references by default."""
        d = AgentDecision(
            id="d1", timestamp=datetime.utcnow(), agent_id="a", 
            decision_type="bid", value=100.0
        )
        assert d.get_references() == []


class TestChainResult:
    """Test ChainResult dataclass."""
    
    def test_total_errors_property(self):
        """total_errors should count failures."""
        result = ChainResult(
            decision_id="d1",
            success=False,
            failures=[
                ReferenceFailure(
                    decision_id="d1", ref_id="d0", expected=1, actual=2,
                    error_type=ReferenceErrorType.WRONG_VALUE
                ),
                ReferenceFailure(
                    decision_id="d1", ref_id="d-1", expected=1, actual=None,
                    error_type=ReferenceErrorType.MISSING_REFERENCE
                ),
            ],
            cascading_errors=0
        )
        
        assert result.total_errors == 2
    
    def test_has_cascading_errors_true(self):
        """has_cascading_errors should return True when cascading > 0."""
        result = ChainResult(
            decision_id="d1", success=False, cascading_errors=1
        )
        assert result.has_cascading_errors is True
    
    def test_has_cascading_errors_false(self):
        """has_cascading_errors should return False when cascading == 0."""
        result = ChainResult(
            decision_id="d1", success=True, cascading_errors=0
        )
        assert result.has_cascading_errors is False


class TestReferenceFailure:
    """Test ReferenceFailure dataclass."""
    
    def test_is_cascading_by_type(self):
        """is_cascading should be True for CASCADING_ERROR type."""
        failure = ReferenceFailure(
            decision_id="d2", ref_id="d1", expected=1, actual=2,
            error_type=ReferenceErrorType.CASCADING_ERROR
        )
        assert failure.is_cascading is True
    
    def test_is_cascading_by_caused_by(self):
        """is_cascading should be True when caused_by is set."""
        failure = ReferenceFailure(
            decision_id="d2", ref_id="d1", expected=1, actual=2,
            error_type=ReferenceErrorType.WRONG_VALUE,
            caused_by="d0"
        )
        assert failure.is_cascading is True
    
    def test_is_cascading_false(self):
        """is_cascading should be False for non-cascading errors."""
        failure = ReferenceFailure(
            decision_id="d2", ref_id="d1", expected=1, actual=2,
            error_type=ReferenceErrorType.WRONG_VALUE
        )
        assert failure.is_cascading is False


class TestClearMethod:
    """Test the clear method."""
    
    def test_clear_resets_all_state(self):
        """clear should reset all internal state."""
        tracker = DecisionChainTracker(lookback_window=5)
        
        # Add some data
        for i in range(3):
            d = AgentDecision(
                id=f"d{i}", timestamp=datetime.utcnow(), 
                agent_id="a", decision_type="bid", value=float(i)
            )
            if i > 0:
                d.references.append(
                    DecisionReference(decision_id=f"d{i-1}", recalled_value=999.0)  # Wrong
                )
            tracker.record_decision(d)
        
        assert tracker.total_decisions > 0
        assert tracker.total_failures > 0
        
        tracker.clear()
        
        assert tracker.total_decisions == 0
        assert tracker.total_failures == 0
        assert len(tracker.reference_failures) == 0


class TestFloatComparison:
    """Test floating point value comparison."""
    
    def test_float_values_match_exactly(self):
        """Exact float values should match."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=100.0)]
        )
        result = tracker.record_decision(d2)
        
        assert result.success is True
    
    def test_float_values_close_enough(self):
        """Very close float values should match (floating point tolerance)."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        # Slightly off due to floating point
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=100.0 + 1e-12)]
        )
        result = tracker.record_decision(d2)
        
        assert result.success is True


class TestGetCascadingChains:
    """Test cascading chain retrieval."""
    
    def test_get_cascading_chains_simple(self):
        """Should build cascading error chain."""
        tracker = DecisionChainTracker()
        
        # d1 is correct
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        # d2 has error referencing d1
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=50.0)]
        )
        tracker.record_decision(d2)
        
        # d3 cascades from d2
        d3 = AgentDecision(
            id="d3", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=120.0,
            references=[DecisionReference(decision_id="d2", recalled_value=999.0)]
        )
        tracker.record_decision(d3)
        
        chains = tracker.get_cascading_chains()
        
        # Should have a chain starting from d2
        assert len(chains) >= 1
        # Chain should include d2 and d3
        flattened = [item for chain in chains for item in chain]
        assert "d2" in flattened
        assert "d3" in flattened


class TestClassifyError:
    """Test the classify_error utility method."""
    
    def test_classify_stale(self):
        """Should classify stale data correctly."""
        tracker = DecisionChainTracker(lookback_window=2)
        
        # Fill and overflow window
        for i in range(3):
            d = AgentDecision(id=f"d{i}", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=float(i))
            tracker.record_decision(d)
        
        ref = DecisionReference(decision_id="d0", recalled_value=0.0)
        error_type = tracker.classify_error(ref, None)
        
        assert error_type == ReferenceErrorType.STALE_DATA
    
    def test_classify_missing(self):
        """Should classify missing reference correctly."""
        tracker = DecisionChainTracker()
        
        ref = DecisionReference(decision_id="nonexistent", recalled_value=0.0)
        error_type = tracker.classify_error(ref, None)
        
        assert error_type == ReferenceErrorType.MISSING_REFERENCE


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_lookback_window_of_one(self):
        """Should work with minimal lookback window."""
        tracker = DecisionChainTracker(lookback_window=1)
        
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0)
        tracker.record_decision(d1)
        
        assert tracker.total_decisions == 1
        
        d2 = AgentDecision(id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0)
        tracker.record_decision(d2)
        
        assert tracker.total_decisions == 1
        assert tracker.get_decision("d1") is None
        assert tracker.get_decision("d2") is not None
    
    def test_self_reference(self):
        """Decision referencing itself should be missing (recorded after check)."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(
            id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=100.0,
            references=[DecisionReference(decision_id="d1", recalled_value=100.0)]
        )
        result = tracker.record_decision(d1)
        
        assert result.success is False
        assert result.failures[0].error_type == ReferenceErrorType.MISSING_REFERENCE
    
    def test_none_values(self):
        """Should handle None values in comparisons."""
        tracker = DecisionChainTracker()
        
        d1 = AgentDecision(id="d1", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=None)
        tracker.record_decision(d1)
        
        d2 = AgentDecision(
            id="d2", timestamp=datetime.utcnow(), agent_id="a", decision_type="bid", value=110.0,
            references=[DecisionReference(decision_id="d1", recalled_value=None)]
        )
        result = tracker.record_decision(d2)
        
        assert result.success is True
    
    def test_large_volume(self):
        """Should handle large number of decisions efficiently."""
        tracker = DecisionChainTracker(lookback_window=100)
        
        for i in range(1000):
            refs = []
            if i > 0:
                # Reference previous decision
                refs.append(DecisionReference(decision_id=f"d{i-1}", recalled_value=float(i-1)))
            
            d = AgentDecision(
                id=f"d{i}", timestamp=datetime.utcnow(), agent_id="a", 
                decision_type="bid", value=float(i),
                references=refs
            )
            tracker.record_decision(d)
        
        assert tracker.total_decisions == 100
        # Most recent 100 should be present
        for i in range(900, 1000):
            assert tracker.get_decision(f"d{i}") is not None
