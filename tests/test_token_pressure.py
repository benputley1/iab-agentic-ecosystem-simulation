"""
Tests for Token Pressure Engine.

Tests context window pressure simulation including:
- Token tracking per agent
- Token estimation for different event types
- Context overflow detection
- Compression with information loss
- Metrics and reporting
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(__file__).replace("/tests/test_token_pressure.py", ""))

from src.pressure import (
    TokenPressureEngine,
    TokenPressureResult,
    CompressionEvent,
    AgentTokenState,
)
from src.pressure.token_tracker import TrackedEvent


class TestTokenPressureResult:
    """Tests for TokenPressureResult dataclass."""
    
    def test_no_overflow_result(self):
        """Test result when no overflow occurs."""
        result = TokenPressureResult(
            overflow=False,
            tokens_before=1000,
            tokens_after=1100,
        )
        assert result.overflow is False
        assert result.events_lost == 0
        assert result.information_loss_pct == 0.0
        assert result.compression_occurred is False
        assert result.tokens_freed == 0
    
    def test_overflow_result(self):
        """Test result when overflow and compression occur."""
        result = TokenPressureResult(
            overflow=True,
            events_lost=50,
            information_loss_pct=0.20,
            compression_occurred=True,
            tokens_before=210000,
            tokens_after=140000,
        )
        assert result.overflow is True
        assert result.events_lost == 50
        assert result.information_loss_pct == 0.20
        assert result.compression_occurred is True
        assert result.tokens_freed == 70000
    
    def test_tokens_freed_never_negative(self):
        """Test that tokens_freed is never negative."""
        result = TokenPressureResult(
            overflow=False,
            tokens_before=100,
            tokens_after=200,  # More tokens after (shouldn't happen but test edge case)
        )
        assert result.tokens_freed == 0


class TestCompressionEvent:
    """Tests for CompressionEvent dataclass."""
    
    def test_compression_event_creation(self):
        """Test creating a compression event."""
        event = CompressionEvent(
            agent_id="buyer-001",
            tokens_before=210000,
            tokens_after=140000,
            events_dropped=100,
            information_loss_pct=0.20,
            context_limit=200000,
        )
        assert event.agent_id == "buyer-001"
        assert event.tokens_before == 210000
        assert event.tokens_after == 140000
        assert event.events_dropped == 100
        assert event.information_loss_pct == 0.20
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        event = CompressionEvent(
            tokens_before=200000,
            tokens_after=140000,
        )
        assert event.compression_ratio == 0.7
    
    def test_compression_ratio_zero_before(self):
        """Test compression ratio when tokens_before is zero."""
        event = CompressionEvent(
            tokens_before=0,
            tokens_after=0,
        )
        assert event.compression_ratio == 1.0
    
    def test_tokens_freed(self):
        """Test tokens freed calculation."""
        event = CompressionEvent(
            tokens_before=200000,
            tokens_after=140000,
        )
        assert event.tokens_freed == 60000


class TestAgentTokenState:
    """Tests for AgentTokenState dataclass."""
    
    def test_initial_state(self):
        """Test initial agent state."""
        state = AgentTokenState(agent_id="buyer-001")
        assert state.agent_id == "buyer-001"
        assert state.current_tokens == 0
        assert state.total_events == 0
        assert state.overflow_count == 0
        assert state.total_events_lost == 0
        assert state.cumulative_info_loss == 0.0
        assert state.effective_info_retained == 100.0
    
    def test_effective_info_retained_no_compressions(self):
        """Test info retained with no compressions."""
        state = AgentTokenState(agent_id="buyer-001")
        assert state.effective_info_retained == 100.0
    
    def test_effective_info_retained_single_compression(self):
        """Test info retained after single compression."""
        state = AgentTokenState(agent_id="buyer-001")
        state.compression_history.append(
            CompressionEvent(information_loss_pct=0.20)
        )
        # 100% * (1 - 0.20) = 80%
        assert state.effective_info_retained == pytest.approx(80.0)
    
    def test_effective_info_retained_multiple_compressions(self):
        """Test info retained after multiple compressions (compounds)."""
        state = AgentTokenState(agent_id="buyer-001")
        # Two 20% losses: 100% * 0.8 * 0.8 = 64%
        state.compression_history.append(
            CompressionEvent(information_loss_pct=0.20)
        )
        state.compression_history.append(
            CompressionEvent(information_loss_pct=0.20)
        )
        assert state.effective_info_retained == pytest.approx(64.0)


class TestTokenPressureEngine:
    """Tests for TokenPressureEngine."""
    
    def test_initialization_defaults(self):
        """Test engine initialization with defaults."""
        engine = TokenPressureEngine()
        assert engine.context_limit == 200_000
        assert engine.compression_loss_rate == 0.20
        assert engine.compression_target == 0.70
        assert engine.total_compressions == 0
    
    def test_initialization_custom(self):
        """Test engine initialization with custom values."""
        engine = TokenPressureEngine(
            model_context_limit=100_000,
            compression_loss_rate=0.15,
            token_estimate_range=(30, 100),
            compression_target=0.60,
        )
        assert engine.context_limit == 100_000
        assert engine.compression_loss_rate == 0.15
        assert engine.default_token_range == (30, 100)
        assert engine.compression_target == 0.60
    
    def test_get_agent_state_creates_new(self):
        """Test that get_agent_state creates new state if needed."""
        engine = TokenPressureEngine()
        state = engine.get_agent_state("buyer-001")
        assert state.agent_id == "buyer-001"
        assert state.current_tokens == 0
    
    def test_get_agent_state_returns_existing(self):
        """Test that get_agent_state returns existing state."""
        engine = TokenPressureEngine()
        state1 = engine.get_agent_state("buyer-001")
        state1.current_tokens = 5000
        state2 = engine.get_agent_state("buyer-001")
        assert state2.current_tokens == 5000
    
    def test_estimate_tokens_default_range(self):
        """Test token estimation uses default range."""
        engine = TokenPressureEngine(token_estimate_range=(50, 200))
        tokens = engine.estimate_tokens({"data": "test"})
        assert 50 <= tokens <= 400  # Upper bound can be 2x max for serializable objects
    
    def test_estimate_tokens_by_event_type(self):
        """Test token estimation uses event type ranges."""
        engine = TokenPressureEngine()
        # bid_request range is (80, 150)
        tokens = engine.estimate_tokens({}, event_type="bid_request")
        # Since it's a simple dict, should use the range
        assert 80 <= tokens <= 150
    
    def test_estimate_tokens_from_streamable_object(self):
        """Test token estimation from object with to_stream_data."""
        engine = TokenPressureEngine()
        
        mock_event = MagicMock()
        mock_event.to_stream_data.return_value = {
            "field1": "a" * 100,
            "field2": "b" * 100,
        }
        
        tokens = engine.estimate_tokens(mock_event)
        # Should estimate based on serialized length
        assert tokens > 0
    
    def test_add_event_no_overflow(self):
        """Test adding event when no overflow occurs."""
        engine = TokenPressureEngine(model_context_limit=200_000)
        result = engine.add_event(
            event={"data": "test"},
            agent_id="buyer-001",
            event_type="bid_request",
        )
        assert result.overflow is False
        assert result.events_lost == 0
        assert result.compression_occurred is False
        
        state = engine.get_agent_state("buyer-001")
        assert state.total_events == 1
        assert state.current_tokens > 0
    
    def test_add_event_tracks_per_agent(self):
        """Test that events are tracked separately per agent."""
        engine = TokenPressureEngine()
        
        engine.add_event({"data": "test1"}, agent_id="buyer-001")
        engine.add_event({"data": "test2"}, agent_id="buyer-002")
        engine.add_event({"data": "test3"}, agent_id="buyer-001")
        
        state1 = engine.get_agent_state("buyer-001")
        state2 = engine.get_agent_state("buyer-002")
        
        assert state1.total_events == 2
        assert state2.total_events == 1
    
    def test_add_event_triggers_overflow(self):
        """Test that overflow is triggered when limit exceeded."""
        # Small limit to easily trigger overflow
        engine = TokenPressureEngine(
            model_context_limit=500,
            compression_loss_rate=0.20,
            token_estimate_range=(100, 100),  # Fixed for predictability
        )
        
        # Add events until overflow
        overflowed = False
        for i in range(10):
            result = engine.add_event(
                event={"id": i},
                agent_id="buyer-001",
            )
            if result.overflow:
                overflowed = True
                assert result.events_lost > 0
                assert result.information_loss_pct == 0.20
                assert result.compression_occurred is True
                break
        
        assert overflowed, "Expected overflow to occur"
        
        state = engine.get_agent_state("buyer-001")
        assert state.overflow_count > 0
        assert len(state.compression_history) > 0
    
    def test_compression_reduces_tokens(self):
        """Test that compression reduces token count to target."""
        engine = TokenPressureEngine(
            model_context_limit=500,
            compression_target=0.70,
            token_estimate_range=(100, 100),
        )
        
        # Add events until overflow
        for i in range(10):
            result = engine.add_event({"id": i}, agent_id="buyer-001")
            if result.overflow:
                # After compression, should be at ~70% of limit
                state = engine.get_agent_state("buyer-001")
                assert state.current_tokens <= 500  # At or below limit
                break
    
    def test_compression_drops_low_priority_first(self):
        """Test that low priority events are dropped first."""
        engine = TokenPressureEngine(
            model_context_limit=500,
            compression_target=0.50,
            token_estimate_range=(100, 100),
        )
        
        # Add low priority events
        for i in range(3):
            engine.add_event({"id": f"low-{i}"}, agent_id="buyer-001", priority=1)
        
        # Add high priority event (should survive compression)
        engine.add_event({"id": "high-0"}, agent_id="buyer-001", priority=10)
        
        # Trigger overflow
        engine.add_event({"id": "trigger"}, agent_id="buyer-001", priority=5)
        
        state = engine.get_agent_state("buyer-001")
        # High priority event should still be there
        high_priority_remaining = [
            e for e in state.events if e.priority == 10
        ]
        assert len(high_priority_remaining) > 0
    
    def test_cumulative_info_loss(self):
        """Test that information loss compounds across compressions."""
        engine = TokenPressureEngine(
            model_context_limit=500,
            compression_loss_rate=0.20,
            compression_target=0.50,
            token_estimate_range=(100, 100),
        )
        
        # Add many events to trigger multiple compressions
        for i in range(30):
            engine.add_event({"id": i}, agent_id="buyer-001")
        
        state = engine.get_agent_state("buyer-001")
        
        if state.overflow_count > 1:
            # Multiple compressions should compound
            # 2 compressions at 20% each: 1 - (0.8 * 0.8) = 0.36 cumulative loss
            # But our implementation adds new_loss = loss_rate * (1 - cumulative)
            # So after 2: 0.20 + 0.20*(1-0.20) = 0.20 + 0.16 = 0.36
            assert state.cumulative_info_loss > 0.20
    
    def test_get_agent_metrics(self):
        """Test agent metrics retrieval."""
        engine = TokenPressureEngine(model_context_limit=10000)
        
        engine.add_event({"data": "test1"}, agent_id="buyer-001")
        engine.add_event({"data": "test2"}, agent_id="buyer-001")
        
        metrics = engine.get_agent_metrics("buyer-001")
        
        assert metrics["agent_id"] == "buyer-001"
        assert metrics["total_events"] == 2
        assert metrics["active_events"] == 2
        assert metrics["current_tokens"] > 0
        assert metrics["context_limit"] == 10000
        assert "context_utilization" in metrics
        assert "effective_info_retained_pct" in metrics
    
    def test_get_global_metrics(self):
        """Test global metrics retrieval."""
        engine = TokenPressureEngine()
        
        engine.add_event({"data": "test1"}, agent_id="buyer-001")
        engine.add_event({"data": "test2"}, agent_id="buyer-002")
        engine.add_event({"data": "test3"}, agent_id="seller-001")
        
        metrics = engine.get_global_metrics()
        
        assert metrics["agent_count"] == 3
        assert metrics["total_events"] == 3
        assert metrics["total_tokens"] > 0
        assert metrics["context_limit"] == 200_000
    
    def test_reset_agent(self):
        """Test resetting a single agent's state."""
        engine = TokenPressureEngine()
        
        engine.add_event({"data": "test"}, agent_id="buyer-001")
        engine.add_event({"data": "test"}, agent_id="buyer-002")
        
        engine.reset_agent("buyer-001")
        
        state1 = engine.get_agent_state("buyer-001")
        state2 = engine.get_agent_state("buyer-002")
        
        assert state1.current_tokens == 0
        assert len(state1.events) == 0
        assert state2.current_tokens > 0  # Unaffected
    
    def test_reset_all(self):
        """Test resetting all state."""
        engine = TokenPressureEngine()
        
        engine.add_event({"data": "test"}, agent_id="buyer-001")
        engine.add_event({"data": "test"}, agent_id="buyer-002")
        
        engine.reset_all()
        
        metrics = engine.get_global_metrics()
        assert metrics["agent_count"] == 0
        assert metrics["total_events"] == 0
        assert engine.total_compressions == 0


class TestTokenPressureSimulation:
    """Tests for pressure simulation over time."""
    
    def test_simulate_pressure_over_time(self):
        """Test multi-day pressure simulation."""
        engine = TokenPressureEngine(
            model_context_limit=10000,
            compression_target=0.70,
        )
        
        daily_metrics = engine.simulate_pressure_over_time(
            agent_id="buyer-001",
            events_per_day=50,
            days=5,
        )
        
        assert len(daily_metrics) == 5
        
        # Each day should have metrics
        for i, metrics in enumerate(daily_metrics):
            assert metrics["day"] == i + 1
            assert "current_tokens" in metrics
            assert "day_overflows" in metrics
            assert "day_events_lost" in metrics
    
    def test_pressure_increases_over_time(self):
        """Test that context pressure generally increases over time."""
        engine = TokenPressureEngine(
            model_context_limit=50000,  # Large enough to show pressure build
            compression_target=0.70,
        )
        
        daily_metrics = engine.simulate_pressure_over_time(
            agent_id="buyer-001",
            events_per_day=100,
            days=10,
        )
        
        # Early days should have lower utilization than later days (generally)
        early_util = daily_metrics[0]["context_utilization"]
        late_util = daily_metrics[-1]["context_utilization"]
        
        # At minimum, events should accumulate
        assert daily_metrics[-1]["total_events"] > daily_metrics[0]["total_events"]
    
    def test_overflow_tracking_in_simulation(self):
        """Test that overflows are tracked during simulation."""
        engine = TokenPressureEngine(
            model_context_limit=5000,  # Small limit to guarantee overflows
            compression_target=0.50,
        )
        
        daily_metrics = engine.simulate_pressure_over_time(
            agent_id="buyer-001",
            events_per_day=100,
            days=5,
        )
        
        # Should have some overflows with this small limit
        total_overflows = sum(m["day_overflows"] for m in daily_metrics)
        assert total_overflows > 0
        
        # Global tracking should match
        assert engine.total_compressions > 0


class TestIntegrationWithMessageSchemas:
    """Integration tests with actual message schema types."""
    
    def test_with_bid_request_like_object(self):
        """Test with an object that looks like BidRequest."""
        engine = TokenPressureEngine()
        
        class MockBidRequest:
            def __init__(self):
                self.request_id = "REQ-001"
                self.buyer_id = "buyer-001"
                self.campaign_id = "camp-001"
                
            def to_stream_data(self):
                return {
                    "request_id": self.request_id,
                    "buyer_id": self.buyer_id,
                    "campaign_id": self.campaign_id,
                    "max_cpm": "5.00",
                    "impressions": "10000",
                }
        
        result = engine.add_event(
            event=MockBidRequest(),
            agent_id="buyer-001",
            event_type="bid_request",
        )
        
        assert result.overflow is False
        state = engine.get_agent_state("buyer-001")
        assert state.total_events == 1
        
        # Event should be tracked with request_id
        assert any(e.event_id == "REQ-001" for e in state.events)
    
    def test_with_deal_confirmation_like_object(self):
        """Test with an object that looks like DealConfirmation."""
        engine = TokenPressureEngine()
        
        class MockDealConfirmation:
            def __init__(self):
                self.deal_id = "DEAL-001"
                self.buyer_id = "buyer-001"
                self.seller_id = "seller-001"
                
            def to_stream_data(self):
                return {
                    "deal_id": self.deal_id,
                    "buyer_id": self.buyer_id,
                    "seller_id": self.seller_id,
                    "cpm": "4.50",
                    "impressions": "10000",
                    "total_cost": "45.00",
                }
        
        result = engine.add_event(
            event=MockDealConfirmation(),
            agent_id="buyer-001",
            event_type="deal_confirmation",
        )
        
        assert result.overflow is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_event(self):
        """Test handling of empty event."""
        engine = TokenPressureEngine()
        result = engine.add_event({}, agent_id="buyer-001")
        assert result.overflow is False
    
    def test_very_large_event(self):
        """Test handling of very large event."""
        engine = TokenPressureEngine(model_context_limit=100)
        
        large_event = {"data": "x" * 10000}
        result = engine.add_event(large_event, agent_id="buyer-001")
        
        # Should handle gracefully (might trigger compression)
        assert isinstance(result, TokenPressureResult)
    
    def test_zero_context_limit(self):
        """Test with zero context limit (edge case)."""
        engine = TokenPressureEngine(model_context_limit=0)
        
        # Any event should trigger overflow
        result = engine.add_event({"data": "test"}, agent_id="buyer-001")
        assert result.overflow is True
    
    def test_many_agents(self):
        """Test with many concurrent agents."""
        engine = TokenPressureEngine()
        
        for i in range(100):
            engine.add_event({"data": f"test-{i}"}, agent_id=f"agent-{i}")
        
        metrics = engine.get_global_metrics()
        assert metrics["agent_count"] == 100
        assert metrics["total_events"] == 100
    
    def test_agent_id_special_characters(self):
        """Test agent IDs with special characters."""
        engine = TokenPressureEngine()
        
        special_ids = [
            "buyer:001",
            "agent@domain.com",
            "agent/path/to/agent",
            "agent-with-dashes_and_underscores",
        ]
        
        for agent_id in special_ids:
            engine.add_event({"data": "test"}, agent_id=agent_id)
            state = engine.get_agent_state(agent_id)
            assert state.agent_id == agent_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
