"""Tests for Token Pressure Engine."""
import pytest
from datetime import datetime
from src.pressure import TokenPressureEngine, TokenPressureResult, CompressionEvent


class TestTokenPressureEngine:
    """Test suite for TokenPressureEngine."""
    
    def test_initialization(self):
        """Test engine initializes with correct defaults."""
        engine = TokenPressureEngine()
        assert engine.context_limit == 200000
        assert engine.compression_loss == 0.20
        assert engine.current_tokens == 0
        assert engine.overflow_count == 0
        assert engine.total_events_lost == 0
        assert len(engine.events) == 0
    
    def test_custom_initialization(self):
        """Test engine initializes with custom parameters."""
        engine = TokenPressureEngine(context_limit=100000, compression_loss=0.30)
        assert engine.context_limit == 100000
        assert engine.compression_loss == 0.30
    
    def test_estimate_tokens_small_event(self):
        """Test token estimation for small events."""
        engine = TokenPressureEngine()
        small_event = {"type": "bid", "value": 100}
        tokens = engine.estimate_tokens(small_event)
        assert 50 <= tokens <= 200
        assert tokens >= 50  # Minimum base tokens
    
    def test_estimate_tokens_large_event(self):
        """Test token estimation for large events."""
        engine = TokenPressureEngine()
        large_event = {
            "type": "complex_event",
            "data": "x" * 1000,  # Large string
            "nested": {"key1": "value1", "key2": "value2"},
            "list": [1, 2, 3, 4, 5]
        }
        tokens = engine.estimate_tokens(large_event)
        assert tokens <= 200  # Should cap at 200
    
    def test_normal_event_addition(self):
        """Test adding events without overflow."""
        engine = TokenPressureEngine(context_limit=10000)
        
        # Add a few normal events
        for i in range(5):
            event = {"type": "bid", "id": i, "value": 100 + i}
            result = engine.add_event(event)
            
            assert isinstance(result, TokenPressureResult)
            assert result.overflow is False
            assert result.events_lost == 0
            assert result.info_loss_pct == 0.0
        
        assert len(engine.events) == 5
        assert engine.current_tokens > 0
        assert engine.overflow_count == 0
    
    def test_overflow_detection(self):
        """Test that overflow is detected when context limit exceeded."""
        # Small context limit to trigger overflow easily
        engine = TokenPressureEngine(context_limit=500, compression_loss=0.20)
        
        overflow_occurred = False
        
        # Add events until overflow
        for i in range(20):
            event = {"type": "event", "id": i, "data": "x" * 50}
            result = engine.add_event(event)
            
            if result.overflow:
                overflow_occurred = True
                assert result.events_lost > 0
                assert result.info_loss_pct > 0.0
                break
        
        assert overflow_occurred, "Overflow should have been triggered"
        assert engine.overflow_count > 0
    
    def test_compression_behavior(self):
        """Test that compression drops oldest 20% of events."""
        engine = TokenPressureEngine(context_limit=1000, compression_loss=0.20)
        
        # Add events until overflow
        initial_events = []
        for i in range(30):
            event = {"type": "event", "id": i, "sequence": i}
            initial_events.append(event)
            result = engine.add_event(event)
            
            if result.overflow:
                # Check that oldest events were dropped
                remaining_ids = [e.get('id') for e in engine.events]
                
                # The oldest events (lowest IDs) should be gone
                # At least some of the newest events should remain
                assert max(remaining_ids) == i  # Most recent should still be there
                assert result.events_lost > 0
                break
    
    def test_compression_event_recording(self):
        """Test that compression events are properly recorded."""
        engine = TokenPressureEngine(context_limit=500, compression_loss=0.20)
        
        # Trigger overflow
        for i in range(30):
            event = {"type": "event", "id": i, "data": "x" * 50}
            engine.add_event(event)
        
        # Should have at least one compression event
        assert len(engine.compression_history) > 0
        
        # Check compression event structure
        comp_event = engine.compression_history[0]
        assert isinstance(comp_event, CompressionEvent)
        assert isinstance(comp_event.timestamp, datetime)
        assert comp_event.tokens_before > comp_event.tokens_after
        assert comp_event.events_dropped > 0
    
    def test_stats_tracking(self):
        """Test that statistics are accurately tracked."""
        engine = TokenPressureEngine(context_limit=10000)
        
        # Add some events
        for i in range(10):
            event = {"type": "bid", "id": i}
            engine.add_event(event)
        
        stats = engine.get_stats()
        
        assert 'current_tokens' in stats
        assert 'overflow_count' in stats
        assert 'total_loss' in stats
        assert 'context_limit' in stats
        assert 'compression_loss' in stats
        assert 'current_events' in stats
        assert 'utilization_pct' in stats
        assert 'compression_history' in stats
        
        assert stats['current_tokens'] > 0
        assert stats['current_events'] == 10
        assert stats['context_limit'] == 10000
        assert 0 <= stats['utilization_pct'] <= 100
    
    def test_stats_after_compression(self):
        """Test that stats reflect compression events."""
        engine = TokenPressureEngine(context_limit=500, compression_loss=0.20)
        
        # Trigger overflow
        for i in range(30):
            event = {"type": "event", "id": i, "data": "x" * 50}
            engine.add_event(event)
        
        stats = engine.get_stats()
        
        assert stats['overflow_count'] > 0
        assert stats['total_loss'] > 0
        assert len(stats['compression_history']) > 0
        
        # Check compression history structure
        history_entry = stats['compression_history'][0]
        assert 'timestamp' in history_entry
        assert 'tokens_before' in history_entry
        assert 'tokens_after' in history_entry
        assert 'events_dropped' in history_entry
    
    def test_multiple_compressions(self):
        """Test that multiple compression cycles work correctly."""
        engine = TokenPressureEngine(context_limit=500, compression_loss=0.20)
        
        compression_count = 0
        
        # Add many events to trigger multiple compressions
        for i in range(100):
            event = {"type": "event", "id": i, "data": "x" * 50}
            result = engine.add_event(event)
            if result.overflow:
                compression_count += 1
        
        assert compression_count > 1, "Should have multiple compressions"
        assert len(engine.compression_history) == compression_count
        assert engine.overflow_count == compression_count
    
    def test_event_metadata_tracking(self):
        """Test that events are tracked with metadata."""
        engine = TokenPressureEngine()
        
        event = {"type": "bid", "value": 100}
        engine.add_event(event)
        
        tracked_event = engine.events[0]
        assert '_estimated_tokens' in tracked_event
        assert '_added_at' in tracked_event
        assert tracked_event['type'] == 'bid'
        assert tracked_event['value'] == 100
    
    def test_token_recalculation_after_compression(self):
        """Test that token counts are accurate after compression."""
        engine = TokenPressureEngine(context_limit=1000, compression_loss=0.20)
        
        # Add events until overflow
        for i in range(50):
            event = {"type": "event", "id": i, "data": "x" * 40}
            engine.add_event(event)
        
        # Manually verify token count matches sum of event tokens
        calculated_tokens = sum(
            e.get('_estimated_tokens', 0) for e in engine.events
        )
        assert engine.current_tokens == calculated_tokens
