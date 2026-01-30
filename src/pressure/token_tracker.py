"""
Token Pressure Engine - Track and simulate context window limitations.

This module simulates how context window pressure affects AI agents making
ad buying/selling decisions over extended campaigns. Key features:
- Track cumulative token count per agent
- Estimate tokens for bid events
- Detect context overflow
- Simulate compression with configurable information loss
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable
from collections import defaultdict
import random
import uuid


@runtime_checkable
class TokenEstimable(Protocol):
    """Protocol for objects that can be converted to token estimates."""
    def to_stream_data(self) -> dict[str, Any]: ...


@dataclass
class TokenPressureResult:
    """Result of adding an event to the context tracker."""
    overflow: bool
    events_lost: int = 0
    information_loss_pct: float = 0.0
    compression_occurred: bool = False
    tokens_before: int = 0
    tokens_after: int = 0
    
    @property
    def tokens_freed(self) -> int:
        """Number of tokens freed by compression."""
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class CompressionEvent:
    """Record of a context compression event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    events_dropped: int = 0
    information_loss_pct: float = 0.0
    context_limit: int = 0
    
    @property
    def compression_ratio(self) -> float:
        """Ratio of tokens after vs before compression."""
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before
    
    @property
    def tokens_freed(self) -> int:
        """Number of tokens freed by this compression."""
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class TrackedEvent:
    """An event being tracked in the context window."""
    event_id: str
    agent_id: str
    tokens: int
    timestamp: datetime
    event_type: str
    priority: int = 1  # Higher = more important, less likely to be dropped
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentTokenState:
    """Token state for a single agent."""
    agent_id: str
    current_tokens: int = 0
    total_events: int = 0
    overflow_count: int = 0
    total_events_lost: int = 0
    cumulative_info_loss: float = 0.0
    events: list[TrackedEvent] = field(default_factory=list)
    compression_history: list[CompressionEvent] = field(default_factory=list)
    
    @property
    def effective_info_retained(self) -> float:
        """Percentage of original information retained after all compressions."""
        if not self.compression_history:
            return 100.0
        # Each compression compounds - multiply retention rates
        retained = 1.0
        for event in self.compression_history:
            retained *= (1.0 - event.information_loss_pct)
        return retained * 100.0


class TokenPressureEngine:
    """
    Simulates context window pressure on agent decisions.
    
    Tracks token accumulation across events, detects when context overflow
    occurs, and simulates compression with configurable information loss.
    
    Args:
        model_context_limit: Maximum tokens before overflow (default 200K)
        compression_loss_rate: Information loss per compression (default 0.20 = 20%)
        token_estimate_range: Min/max tokens per event (default 50-200)
        compression_target: Target context usage after compression (default 0.7 = 70%)
    """
    
    # Token estimation for different event types
    EVENT_TOKEN_ESTIMATES = {
        "bid_request": (80, 150),
        "bid_response": (60, 120),
        "deal_confirmation": (100, 200),
        "deal_rejection": (50, 80),
        "auction_result": (80, 160),
        "default": (50, 200),
    }
    
    def __init__(
        self,
        model_context_limit: int = 200_000,
        compression_loss_rate: float = 0.20,
        token_estimate_range: tuple[int, int] = (50, 200),
        compression_target: float = 0.70,
    ):
        self.context_limit = model_context_limit
        self.compression_loss_rate = compression_loss_rate
        self.default_token_range = token_estimate_range
        self.compression_target = compression_target
        
        # Per-agent tracking
        self._agent_states: dict[str, AgentTokenState] = {}
        
        # Global tracking
        self.overflow_events: list[CompressionEvent] = []
        self.total_compressions = 0
    
    def get_agent_state(self, agent_id: str) -> AgentTokenState:
        """Get or create token state for an agent."""
        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = AgentTokenState(agent_id=agent_id)
        return self._agent_states[agent_id]
    
    def estimate_tokens(self, event: Any, event_type: Optional[str] = None) -> int:
        """
        Estimate token count for an event.
        
        Uses event-type-specific ranges when known, otherwise defaults to 50-200.
        For objects with to_stream_data(), estimates based on serialized size.
        
        Args:
            event: The event to estimate tokens for
            event_type: Optional explicit event type for estimation
            
        Returns:
            Estimated token count
        """
        # Determine token range based on event type
        if event_type and event_type in self.EVENT_TOKEN_ESTIMATES:
            min_tokens, max_tokens = self.EVENT_TOKEN_ESTIMATES[event_type]
        elif hasattr(event, 'message_type'):
            msg_type = str(event.message_type.value if hasattr(event.message_type, 'value') else event.message_type)
            ranges = self.EVENT_TOKEN_ESTIMATES.get(msg_type, self.default_token_range)
            min_tokens, max_tokens = ranges
        else:
            min_tokens, max_tokens = self.default_token_range
        
        # If event has serializable data, use it to influence estimate
        if isinstance(event, TokenEstimable) or hasattr(event, 'to_stream_data'):
            try:
                data = event.to_stream_data()
                # Rough estimate: ~4 chars per token average
                serialized_len = sum(len(str(k)) + len(str(v)) for k, v in data.items())
                estimated = serialized_len // 4
                # Clamp to reasonable range
                return max(min_tokens, min(max_tokens * 2, estimated))
            except Exception:
                pass
        
        # Random within range for variety
        return random.randint(min_tokens, max_tokens)
    
    def add_event(
        self,
        event: Any,
        agent_id: str,
        event_type: Optional[str] = None,
        priority: int = 1,
    ) -> TokenPressureResult:
        """
        Add an event to the agent's context, tracking overflow.
        
        Args:
            event: The event to add (BidRequest, BidResponse, etc.)
            agent_id: ID of the agent whose context this belongs to
            event_type: Optional explicit event type for token estimation
            priority: Event importance (higher = less likely to be dropped)
            
        Returns:
            TokenPressureResult indicating if overflow occurred
        """
        state = self.get_agent_state(agent_id)
        tokens = self.estimate_tokens(event, event_type)
        
        tokens_before = state.current_tokens
        state.current_tokens += tokens
        state.total_events += 1
        
        # Create tracked event
        event_id = getattr(event, 'request_id', None) or \
                   getattr(event, 'response_id', None) or \
                   getattr(event, 'deal_id', None) or \
                   str(uuid.uuid4())
        
        tracked = TrackedEvent(
            event_id=event_id,
            agent_id=agent_id,
            tokens=tokens,
            timestamp=datetime.utcnow(),
            event_type=event_type or "default",
            priority=priority,
        )
        state.events.append(tracked)
        
        # Check for overflow
        if state.current_tokens > self.context_limit:
            return self._handle_overflow(state, tokens_before)
        
        return TokenPressureResult(
            overflow=False,
            tokens_before=tokens_before,
            tokens_after=state.current_tokens,
        )
    
    def _handle_overflow(self, state: AgentTokenState, tokens_before: int) -> TokenPressureResult:
        """
        Handle context overflow - compress with information loss.
        
        Simulates what happens when an agent's context window fills up:
        - Older, lower-priority events are dropped
        - Information is lost during compression
        - Decision quality may degrade
        
        Args:
            state: The agent's token state
            tokens_before: Token count before the overflow-triggering event
            
        Returns:
            TokenPressureResult with overflow details
        """
        state.overflow_count += 1
        
        # Target: reduce to compression_target of limit
        target_tokens = int(self.context_limit * self.compression_target)
        tokens_to_free = state.current_tokens - target_tokens
        
        # Sort events by priority (lowest first), then by age (oldest first)
        events_by_priority = sorted(
            state.events,
            key=lambda e: (e.priority, e.timestamp),
        )
        
        # Drop events until we're under target
        dropped_events = []
        freed_tokens = 0
        
        for event in events_by_priority:
            if freed_tokens >= tokens_to_free:
                break
            dropped_events.append(event)
            freed_tokens += event.tokens
        
        # Remove dropped events
        dropped_ids = {e.event_id for e in dropped_events}
        state.events = [e for e in state.events if e.event_id not in dropped_ids]
        state.current_tokens -= freed_tokens
        
        events_lost = len(dropped_events)
        state.total_events_lost += events_lost
        
        # Track cumulative information loss (compounds with each compression)
        # Each compression loses X% of REMAINING information
        new_loss = self.compression_loss_rate * (1 - state.cumulative_info_loss)
        state.cumulative_info_loss += new_loss
        
        # Record compression event
        compression = CompressionEvent(
            agent_id=state.agent_id,
            tokens_before=tokens_before,
            tokens_after=state.current_tokens,
            events_dropped=events_lost,
            information_loss_pct=self.compression_loss_rate,
            context_limit=self.context_limit,
        )
        state.compression_history.append(compression)
        self.overflow_events.append(compression)
        self.total_compressions += 1
        
        return TokenPressureResult(
            overflow=True,
            events_lost=events_lost,
            information_loss_pct=self.compression_loss_rate,
            compression_occurred=True,
            tokens_before=tokens_before,
            tokens_after=state.current_tokens,
        )
    
    def get_agent_metrics(self, agent_id: str) -> dict[str, Any]:
        """
        Get metrics for a specific agent.
        
        Returns:
            Dict with token usage, overflow counts, info loss, etc.
        """
        state = self.get_agent_state(agent_id)
        return {
            "agent_id": agent_id,
            "current_tokens": state.current_tokens,
            "context_limit": self.context_limit,
            "context_utilization": state.current_tokens / self.context_limit,
            "total_events": state.total_events,
            "active_events": len(state.events),
            "overflow_count": state.overflow_count,
            "total_events_lost": state.total_events_lost,
            "cumulative_info_loss_pct": state.cumulative_info_loss * 100,
            "effective_info_retained_pct": state.effective_info_retained,
            "compression_count": len(state.compression_history),
        }
    
    def get_global_metrics(self) -> dict[str, Any]:
        """
        Get global metrics across all agents.
        
        Returns:
            Dict with aggregate stats across all tracked agents
        """
        total_tokens = sum(s.current_tokens for s in self._agent_states.values())
        total_events = sum(s.total_events for s in self._agent_states.values())
        total_lost = sum(s.total_events_lost for s in self._agent_states.values())
        
        avg_info_loss = 0.0
        if self._agent_states:
            avg_info_loss = sum(
                s.cumulative_info_loss for s in self._agent_states.values()
            ) / len(self._agent_states)
        
        return {
            "agent_count": len(self._agent_states),
            "total_tokens": total_tokens,
            "total_events": total_events,
            "total_compressions": self.total_compressions,
            "total_events_lost": total_lost,
            "average_info_loss_pct": avg_info_loss * 100,
            "context_limit": self.context_limit,
            "compression_loss_rate": self.compression_loss_rate,
        }
    
    def reset_agent(self, agent_id: str) -> None:
        """Reset an agent's token state (e.g., after restart)."""
        if agent_id in self._agent_states:
            # Keep history but reset active state
            state = self._agent_states[agent_id]
            state.current_tokens = 0
            state.events = []
    
    def reset_all(self) -> None:
        """Reset all state (for new simulation runs)."""
        self._agent_states.clear()
        self.overflow_events.clear()
        self.total_compressions = 0
    
    def simulate_pressure_over_time(
        self,
        agent_id: str,
        events_per_day: int,
        days: int,
        event_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Simulate token pressure over multiple days.
        
        Useful for testing how context pressure builds over campaign duration.
        
        Args:
            agent_id: Agent to simulate for
            events_per_day: Number of events per simulated day
            days: Number of days to simulate
            event_types: Types of events to generate (random if None)
            
        Returns:
            List of daily metrics snapshots
        """
        if event_types is None:
            event_types = list(self.EVENT_TOKEN_ESTIMATES.keys())
        
        daily_metrics = []
        
        for day in range(1, days + 1):
            day_overflows = 0
            day_events_lost = 0
            
            for _ in range(events_per_day):
                event_type = random.choice(event_types)
                # Create a simple mock event
                mock_event = {"type": event_type, "day": day}
                
                result = self.add_event(
                    event=mock_event,
                    agent_id=agent_id,
                    event_type=event_type,
                )
                
                if result.overflow:
                    day_overflows += 1
                    day_events_lost += result.events_lost
            
            metrics = self.get_agent_metrics(agent_id)
            metrics["day"] = day
            metrics["day_overflows"] = day_overflows
            metrics["day_events_lost"] = day_events_lost
            daily_metrics.append(metrics)
        
        return daily_metrics
