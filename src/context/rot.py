"""
Context Rot Simulation.

Simulates realistic context degradation across agent hierarchy.

Types of rot:
1. Truncation: Old context pushed out when window fills
2. Decay: Information importance degrades over time
3. Restart: Agent restarts and loses in-memory context
4. Handoff: Context lost when passing between agent levels
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

if TYPE_CHECKING:
    from src.agents.base import AgentContext
    from .window import ContextEntry, ContextWindowManager

logger = structlog.get_logger(__name__)


class RotType(str, Enum):
    """Types of context rot."""
    TRUNCATION = "truncation"  # Old context pushed out
    DECAY = "decay"            # Gradual importance loss
    RESTART = "restart"        # Agent restart loses context
    HANDOFF = "handoff"        # Lost in transfer between levels


class DecayResult(BaseModel):
    """Result of applying context decay."""
    agent_id: str
    decay_rate: float
    
    # Before decay
    entries_before: int = 0
    tokens_before: int = 0
    average_importance_before: float = 0.0
    
    # After decay
    entries_after: int = 0
    tokens_after: int = 0
    average_importance_after: float = 0.0
    
    # Losses
    entries_pruned: int = 0
    tokens_lost: int = 0
    keys_affected: list[str] = Field(default_factory=list)
    
    # Metadata
    timestamp: float = Field(default_factory=time.time)
    
    @property
    def loss_rate(self) -> float:
        """Percentage of tokens lost."""
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_lost / self.tokens_before
    
    @property
    def importance_drop(self) -> float:
        """Drop in average importance."""
        if self.average_importance_before == 0:
            return 0.0
        return (
            self.average_importance_before - self.average_importance_after
        ) / self.average_importance_before


class HandoffResult(BaseModel):
    """Result of context handoff between agents."""
    handoff_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent_id: str
    to_agent_id: str
    scenario: str  # "A", "B", or "C"
    
    # Transfer stats
    tokens_sent: int = 0
    tokens_received: int = 0
    tokens_lost: int = 0
    
    items_sent: int = 0
    items_received: int = 0
    items_lost: int = 0
    
    # Lost item details
    lost_keys: list[str] = Field(default_factory=list)
    lost_priorities: dict[str, int] = Field(default_factory=dict)
    
    # Recovery info
    recoverable: bool = False
    recovery_source: str | None = None  # "ledger", "exchange_logs", None
    
    # Metadata
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    
    @property
    def loss_rate(self) -> float:
        """Percentage of context lost in handoff."""
        if self.tokens_sent == 0:
            return 0.0
        return self.tokens_lost / self.tokens_sent
    
    @property
    def integrity(self) -> float:
        """Context integrity after handoff (1.0 = perfect)."""
        return 1.0 - self.loss_rate


class ContextRotSimulator:
    """
    Simulate context rot across agent hierarchy.
    
    Models realistic information loss that occurs in multi-agent
    systems without persistent shared state.
    """
    
    def __init__(
        self,
        decay_rate: float = 0.05,        # 5% per day
        restart_probability: float = 0.01,  # 1% chance per day
        handoff_loss_rate: float = 0.10,    # 10% lost on handoff (Scenario B)
        min_importance_threshold: float = 0.1,  # Below this = prune
        seed: int | None = None
    ):
        """
        Initialize rot simulator.
        
        Args:
            decay_rate: Daily decay rate for context importance
            restart_probability: Daily probability of agent restart
            handoff_loss_rate: Base rate of loss during handoffs
            min_importance_threshold: Threshold below which entries are pruned
            seed: Random seed for reproducibility
        """
        self.decay_rate = decay_rate
        self.restart_probability = restart_probability
        self.handoff_loss_rate = handoff_loss_rate
        self.min_importance_threshold = min_importance_threshold
        
        if seed is not None:
            random.seed(seed)
        
        self.log = structlog.get_logger(__name__)
        
        # Track rot events
        self._decay_history: list[DecayResult] = []
        self._handoff_history: list[HandoffResult] = []
        self._restart_events: list[dict[str, Any]] = []
    
    async def apply_daily_decay(
        self,
        window: "ContextWindowManager",
        days: float = 1.0
    ) -> DecayResult:
        """
        Apply daily context decay.
        
        Simulates natural degradation of context importance over time.
        Older, less important information fades.
        
        Args:
            window: Context window to decay
            days: Number of days of decay to apply (can be fractional)
            
        Returns:
            DecayResult with before/after metrics
        """
        result = DecayResult(
            agent_id=window.agent_id,
            decay_rate=self.decay_rate * days,
            entries_before=len(window.history),
            tokens_before=window.current_tokens
        )
        
        # Calculate average importance before
        if window.history:
            result.average_importance_before = sum(
                e.effective_importance for e in window.history
            ) / len(window.history)
        
        # Apply decay to all entries
        effective_rate = self.decay_rate * days
        window.apply_decay(effective_rate)
        
        # Track affected keys
        result.keys_affected = [
            e.entry_id for e in window.history 
            if e.decay_factor < 1.0
        ]
        
        # Prune entries below threshold
        pruned = window.prune_decayed(self.min_importance_threshold)
        
        # Record results
        result.entries_pruned = len(pruned)
        result.tokens_lost = sum(e.tokens for e in pruned)
        result.entries_after = len(window.history)
        result.tokens_after = window.current_tokens
        
        if window.history:
            result.average_importance_after = sum(
                e.effective_importance for e in window.history
            ) / len(window.history)
        
        self._decay_history.append(result)
        
        self.log.info(
            "daily_decay_applied",
            agent_id=window.agent_id,
            days=days,
            entries_pruned=result.entries_pruned,
            tokens_lost=result.tokens_lost,
            importance_drop=result.importance_drop
        )
        
        return result
    
    async def check_restart(self, agent_id: str, days: float = 1.0) -> bool:
        """
        Check if agent experiences a restart event.
        
        Restarts cause complete loss of in-memory context.
        This is a random event based on restart_probability.
        
        Args:
            agent_id: Agent to check
            days: Number of days to check over
            
        Returns:
            True if restart occurred
        """
        # Probability of at least one restart over the period
        # P(restart) = 1 - (1 - daily_prob)^days
        no_restart_prob = (1 - self.restart_probability) ** days
        restart_occurred = random.random() > no_restart_prob
        
        if restart_occurred:
            self._restart_events.append({
                "agent_id": agent_id,
                "timestamp": time.time(),
                "days_elapsed": days
            })
            
            self.log.warning(
                "agent_restart_event",
                agent_id=agent_id,
                days=days,
                probability=self.restart_probability
            )
        
        return restart_occurred
    
    async def simulate_handoff(
        self,
        context: "AgentContext",
        from_agent_id: str,
        to_agent_id: str,
        scenario: str
    ) -> HandoffResult:
        """
        Simulate context handoff between agent levels.
        
        Different scenarios have different loss characteristics:
        - Scenario A: ~5% loss, recoverable via exchange logs
        - Scenario B: ~10% loss, not recoverable
        - Scenario C: 0% loss, full ledger backup
        
        Args:
            context: Context being handed off
            from_agent_id: Source agent
            to_agent_id: Target agent
            scenario: "A", "B", or "C"
            
        Returns:
            HandoffResult with transfer details
        """
        start_time = time.time()
        
        result = HandoffResult(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            scenario=scenario,
            tokens_sent=context.current_tokens,
            items_sent=len(context.items)
        )
        
        if scenario == "C":
            # Scenario C: Full preservation via ledger
            result.tokens_received = context.current_tokens
            result.items_received = len(context.items)
            result.tokens_lost = 0
            result.items_lost = 0
            result.recoverable = True
            result.recovery_source = "ledger"
            
        elif scenario == "A":
            # Scenario A: Partial loss, recoverable via exchange logs
            loss_rate = self.handoff_loss_rate * 0.5  # 5% instead of 10%
            
            items_lost = 0
            tokens_lost = 0
            
            for key, item in context.items.items():
                # Only lose medium/low priority items
                if item.priority.value in ["medium", "low"]:
                    if random.random() < loss_rate:
                        items_lost += 1
                        tokens_lost += item.token_estimate
                        result.lost_keys.append(key)
                        priority = item.priority.value
                        result.lost_priorities[priority] = (
                            result.lost_priorities.get(priority, 0) + 1
                        )
            
            result.items_lost = items_lost
            result.tokens_lost = tokens_lost
            result.items_received = len(context.items) - items_lost
            result.tokens_received = context.current_tokens - tokens_lost
            result.recoverable = True
            result.recovery_source = "exchange_logs"
            
        else:
            # Scenario B: Full loss rate, not recoverable
            loss_rate = self.handoff_loss_rate
            
            items_lost = 0
            tokens_lost = 0
            
            for key, item in context.items.items():
                # Critical items never lost, high items rarely lost
                if item.priority.value == "critical":
                    continue
                elif item.priority.value == "high":
                    effective_rate = loss_rate * 0.3  # 30% of base rate
                elif item.priority.value == "medium":
                    effective_rate = loss_rate * 0.7  # 70% of base rate
                else:
                    effective_rate = loss_rate  # Full rate for low
                
                if random.random() < effective_rate:
                    items_lost += 1
                    tokens_lost += item.token_estimate
                    result.lost_keys.append(key)
                    priority = item.priority.value
                    result.lost_priorities[priority] = (
                        result.lost_priorities.get(priority, 0) + 1
                    )
            
            result.items_lost = items_lost
            result.tokens_lost = tokens_lost
            result.items_received = len(context.items) - items_lost
            result.tokens_received = context.current_tokens - tokens_lost
            result.recoverable = False
            result.recovery_source = None
        
        result.duration_ms = (time.time() - start_time) * 1000
        self._handoff_history.append(result)
        
        self.log.info(
            "handoff_simulated",
            scenario=scenario,
            from_agent=from_agent_id,
            to_agent=to_agent_id,
            loss_rate=result.loss_rate,
            items_lost=result.items_lost,
            recoverable=result.recoverable
        )
        
        return result
    
    async def simulate_truncation(
        self,
        window: "ContextWindowManager",
        new_tokens: int
    ) -> list["ContextEntry"]:
        """
        Simulate context truncation when window overflows.
        
        This is a direct simulation of what happens when
        context exceeds model limits.
        
        Args:
            window: Context window
            new_tokens: Tokens that need to fit
            
        Returns:
            List of truncated entries
        """
        if window.can_fit(new_tokens):
            return []
        
        tokens_needed = new_tokens - window.available_tokens
        truncated = window.truncate_oldest(tokens_needed)
        
        self.log.info(
            "truncation_simulated",
            window_id=window.window_id,
            new_tokens=new_tokens,
            entries_truncated=len(truncated),
            tokens_freed=sum(e.tokens for e in truncated)
        )
        
        return truncated
    
    def get_decay_history(self) -> list[DecayResult]:
        """Get history of decay events."""
        return self._decay_history.copy()
    
    def get_handoff_history(self) -> list[HandoffResult]:
        """Get history of handoff events."""
        return self._handoff_history.copy()
    
    def get_restart_events(self) -> list[dict[str, Any]]:
        """Get history of restart events."""
        return self._restart_events.copy()
    
    def get_total_tokens_lost(self) -> int:
        """Calculate total tokens lost to all rot types."""
        decay_loss = sum(r.tokens_lost for r in self._decay_history)
        handoff_loss = sum(r.tokens_lost for r in self._handoff_history)
        return decay_loss + handoff_loss
    
    def get_average_handoff_loss(self) -> float:
        """Calculate average loss rate across handoffs."""
        if not self._handoff_history:
            return 0.0
        return sum(h.loss_rate for h in self._handoff_history) / len(self._handoff_history)
    
    def clear_history(self) -> None:
        """Clear all rot history."""
        self._decay_history.clear()
        self._handoff_history.clear()
        self._restart_events.clear()
