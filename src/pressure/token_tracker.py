"""Token Pressure Engine - simulates context window limitations and compression effects."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any
import json


@dataclass
class TokenPressureResult:
    """Result of adding an event to the token tracker."""
    overflow: bool
    events_lost: int
    info_loss_pct: float


@dataclass
class CompressionEvent:
    """Record of a compression event."""
    timestamp: datetime
    tokens_before: int
    tokens_after: int
    events_dropped: int


class TokenPressureEngine:
    """
    Simulates token pressure and context window limitations.
    
    Tracks events and their estimated token counts, triggering compression
    when the context limit is reached by dropping the oldest events.
    """
    
    def __init__(self, context_limit: int = 200000, compression_loss: float = 0.20):
        """
        Initialize the Token Pressure Engine.
        
        Args:
            context_limit: Maximum number of tokens before compression
            compression_loss: Fraction of oldest events to drop during compression (0.20 = 20%)
        """
        self.context_limit = context_limit
        self.compression_loss = compression_loss
        self.events: List[Dict[str, Any]] = []
        self.current_tokens = 0
        self.overflow_count = 0
        self.total_events_lost = 0
        self.compression_history: List[CompressionEvent] = []
    
    def estimate_tokens(self, event_data: dict) -> int:
        """
        Estimate token count for an event based on its data size.
        
        Args:
            event_data: Dictionary containing event information
            
        Returns:
            Estimated token count (50-200 based on data complexity)
        """
        # Convert to JSON to get a rough size estimate
        json_str = json.dumps(event_data)
        
        # Base token count (minimum overhead)
        base_tokens = 50
        
        # Additional tokens based on content size
        # Roughly 4 characters per token (conservative estimate)
        content_tokens = len(json_str) // 4
        
        # Cap at 200 tokens per event as specified
        estimated = min(base_tokens + content_tokens, 200)
        
        return estimated
    
    def add_event(self, event: dict) -> TokenPressureResult:
        """
        Add an event to the tracker, handling overflow if necessary.
        
        Args:
            event: Event dictionary to track
            
        Returns:
            TokenPressureResult indicating if overflow occurred and information loss
        """
        # Estimate tokens for this event
        event_tokens = self.estimate_tokens(event)
        
        # Add event to our tracking
        event_with_tokens = {
            **event,
            '_estimated_tokens': event_tokens,
            '_added_at': datetime.now().isoformat()
        }
        self.events.append(event_with_tokens)
        self.current_tokens += event_tokens
        
        # Check if we've exceeded the context limit
        if self.current_tokens > self.context_limit:
            events_lost, info_loss_pct = self.handle_overflow()
            return TokenPressureResult(
                overflow=True,
                events_lost=events_lost,
                info_loss_pct=info_loss_pct
            )
        
        return TokenPressureResult(
            overflow=False,
            events_lost=0,
            info_loss_pct=0.0
        )
    
    def handle_overflow(self) -> tuple[int, float]:
        """
        Handle context overflow by compressing (dropping oldest events).
        
        Drops the oldest compression_loss percentage of events.
        
        Returns:
            Tuple of (events_dropped, info_loss_percentage)
        """
        tokens_before = self.current_tokens
        events_before = len(self.events)
        
        # Calculate how many events to drop (oldest compression_loss %)
        num_to_drop = max(1, int(len(self.events) * self.compression_loss))
        
        # Drop the oldest events
        dropped_events = self.events[:num_to_drop]
        self.events = self.events[num_to_drop:]
        
        # Recalculate token count
        tokens_dropped = sum(e.get('_estimated_tokens', 0) for e in dropped_events)
        self.current_tokens -= tokens_dropped
        
        # Track statistics
        self.overflow_count += 1
        self.total_events_lost += num_to_drop
        
        # Calculate information loss percentage
        info_loss_pct = (tokens_dropped / tokens_before * 100) if tokens_before > 0 else 0.0
        
        # Record this compression event
        compression_event = CompressionEvent(
            timestamp=datetime.now(),
            tokens_before=tokens_before,
            tokens_after=self.current_tokens,
            events_dropped=num_to_drop
        )
        self.compression_history.append(compression_event)
        
        return num_to_drop, info_loss_pct
    
    def get_stats(self) -> dict:
        """
        Get current statistics about token pressure and compression.
        
        Returns:
            Dictionary with current_tokens, overflow_count, total_loss, and compression history
        """
        return {
            'current_tokens': self.current_tokens,
            'overflow_count': self.overflow_count,
            'total_loss': self.total_events_lost,
            'context_limit': self.context_limit,
            'compression_loss': self.compression_loss,
            'current_events': len(self.events),
            'utilization_pct': (self.current_tokens / self.context_limit * 100) if self.context_limit > 0 else 0.0,
            'compression_history': [
                {
                    'timestamp': ce.timestamp.isoformat(),
                    'tokens_before': ce.tokens_before,
                    'tokens_after': ce.tokens_after,
                    'events_dropped': ce.events_dropped
                }
                for ce in self.compression_history
            ]
        }
