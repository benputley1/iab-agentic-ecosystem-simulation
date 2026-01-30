"""
Context Management for Multi-Agent Hierarchy.

Handles context passing between agent levels, token tracking,
and context degradation simulation for measuring information loss.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class ContextPriority(str, Enum):
    """Priority levels for context items."""
    CRITICAL = "critical"  # Must preserve - goals, constraints
    HIGH = "high"          # Important - recent decisions, key state
    MEDIUM = "medium"      # Useful - historical context
    LOW = "low"            # Nice to have - verbose details


class ContextItem(BaseModel):
    """Individual piece of context with metadata."""
    key: str
    value: Any
    priority: ContextPriority = ContextPriority.MEDIUM
    token_estimate: int = 0
    timestamp: float = Field(default_factory=time.time)
    source_agent: str | None = None
    
    def content_hash(self) -> str:
        """Hash of content for change detection."""
        return hashlib.sha256(str(self.value).encode()).hexdigest()[:16]


class AgentContext(BaseModel):
    """
    Context container for agent communication.
    
    Tracks what information is being passed between agent levels,
    with token budgets and priority-based degradation.
    """
    context_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_context_id: str | None = None
    
    # Source information
    source_level: int = 0  # 1=Orchestrator, 2=Specialist, 3=Functional
    source_agent_id: str = ""
    target_agent_id: str | None = None
    
    # Context items by priority
    items: dict[str, ContextItem] = Field(default_factory=dict)
    
    # Token tracking
    token_budget: int = 8000  # Max tokens for this context
    current_tokens: int = 0
    
    # Task information
    task_description: str = ""
    task_constraints: list[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)
    
    def add_item(
        self,
        key: str,
        value: Any,
        priority: ContextPriority = ContextPriority.MEDIUM,
        token_estimate: int | None = None,
        source_agent: str | None = None
    ) -> bool:
        """
        Add an item to context if within budget.
        
        Returns True if added, False if would exceed budget.
        """
        if token_estimate is None:
            # Rough estimate: 4 chars per token
            token_estimate = len(str(value)) // 4 + 1
        
        # Check budget
        if self.current_tokens + token_estimate > self.token_budget:
            logger.warning(
                "context_budget_exceeded",
                key=key,
                needed=token_estimate,
                available=self.token_budget - self.current_tokens
            )
            return False
        
        item = ContextItem(
            key=key,
            value=value,
            priority=priority,
            token_estimate=token_estimate,
            source_agent=source_agent or self.source_agent_id
        )
        self.items[key] = item
        self.current_tokens += token_estimate
        self.last_updated = time.time()
        return True
    
    def get_item(self, key: str) -> Any | None:
        """Get item value by key."""
        item = self.items.get(key)
        return item.value if item else None
    
    def remove_item(self, key: str) -> bool:
        """Remove an item from context."""
        if key in self.items:
            self.current_tokens -= self.items[key].token_estimate
            del self.items[key]
            return True
        return False
    
    def get_by_priority(self, priority: ContextPriority) -> dict[str, Any]:
        """Get all items at a specific priority level."""
        return {
            key: item.value 
            for key, item in self.items.items() 
            if item.priority == priority
        }
    
    def to_prompt_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for prompt injection."""
        return {
            "task": self.task_description,
            "constraints": self.task_constraints,
            "context": {key: item.value for key, item in self.items.items()}
        }


class ContextWindow(BaseModel):
    """
    Tracks token usage within a conversation window.
    
    Used to monitor context consumption and trigger
    summarization/pruning when approaching limits.
    """
    window_id: str = Field(default_factory=lambda: str(uuid4()))
    max_tokens: int = 100000  # Model-dependent
    
    # Token accounting
    system_tokens: int = 0
    context_tokens: int = 0
    history_tokens: int = 0
    reserved_output: int = 4000  # Reserve for response
    
    # History of token usage
    usage_history: list[dict[str, int]] = Field(default_factory=list)
    
    @property
    def used_tokens(self) -> int:
        """Total tokens currently used."""
        return self.system_tokens + self.context_tokens + self.history_tokens
    
    @property
    def available_tokens(self) -> int:
        """Tokens available for new content."""
        return self.max_tokens - self.used_tokens - self.reserved_output
    
    @property
    def utilization(self) -> float:
        """Percentage of window used (0.0-1.0)."""
        return self.used_tokens / self.max_tokens
    
    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record a conversation turn's token usage."""
        self.usage_history.append({
            "input": input_tokens,
            "output": output_tokens,
            "timestamp": time.time()
        })
        self.history_tokens += input_tokens + output_tokens
    
    def needs_summarization(self, threshold: float = 0.7) -> bool:
        """Check if history should be summarized."""
        return self.utilization > threshold


class ContextPassingProtocol(Protocol):
    """Protocol for context passing between agent levels."""
    
    def context_to_child(
        self,
        parent_context: AgentContext,
        child_agent_id: str,
        task: str,
        include_priorities: list[ContextPriority] | None = None
    ) -> AgentContext:
        """Create child context from parent."""
        ...
    
    def context_from_child(
        self,
        parent_context: AgentContext,
        child_context: AgentContext,
        results: Any
    ) -> AgentContext:
        """Merge child results back into parent context."""
        ...


class StandardContextPassing:
    """
    Standard implementation of context passing.
    
    Simulates realistic context degradation at each handoff.
    """
    
    def __init__(
        self,
        degradation_rate: float = 0.1,
        child_budget_ratio: float = 0.6
    ):
        """
        Initialize context passing.
        
        Args:
            degradation_rate: Percentage of low-priority items to drop (0.0-1.0)
            child_budget_ratio: Ratio of parent budget to give child
        """
        self.degradation_rate = degradation_rate
        self.child_budget_ratio = child_budget_ratio
        self.log = structlog.get_logger(__name__)
    
    def context_to_child(
        self,
        parent_context: AgentContext,
        child_agent_id: str,
        task: str,
        include_priorities: list[ContextPriority] | None = None
    ) -> AgentContext:
        """
        Create child context from parent with controlled degradation.
        
        Higher priority items are preserved, lower priorities may be dropped.
        """
        if include_priorities is None:
            include_priorities = list(ContextPriority)
        
        child_budget = int(parent_context.token_budget * self.child_budget_ratio)
        
        child = AgentContext(
            parent_context_id=parent_context.context_id,
            source_level=parent_context.source_level + 1,
            source_agent_id=parent_context.source_agent_id,
            target_agent_id=child_agent_id,
            token_budget=child_budget,
            task_description=task,
            task_constraints=parent_context.task_constraints.copy()
        )
        
        # Copy items by priority (highest first)
        for priority in [ContextPriority.CRITICAL, ContextPriority.HIGH, 
                         ContextPriority.MEDIUM, ContextPriority.LOW]:
            if priority not in include_priorities:
                continue
                
            items = [
                (k, v) for k, v in parent_context.items.items()
                if v.priority == priority
            ]
            
            for key, item in items:
                # Apply degradation to low priority items
                if priority == ContextPriority.LOW:
                    import random
                    if random.random() < self.degradation_rate:
                        self.log.debug(
                            "context_degraded",
                            key=key,
                            priority=priority
                        )
                        continue
                
                # Try to add (respects budget)
                if not child.add_item(
                    key=key,
                    value=item.value,
                    priority=item.priority,
                    token_estimate=item.token_estimate,
                    source_agent=item.source_agent
                ):
                    self.log.debug(
                        "context_truncated",
                        key=key,
                        child_budget=child_budget
                    )
                    break  # Budget exhausted
        
        self.log.info(
            "context_passed_to_child",
            parent_id=parent_context.context_id,
            child_id=child.context_id,
            parent_items=len(parent_context.items),
            child_items=len(child.items),
            child_tokens=child.current_tokens
        )
        
        return child
    
    def context_from_child(
        self,
        parent_context: AgentContext,
        child_context: AgentContext,
        results: Any
    ) -> AgentContext:
        """
        Merge child results back into parent context.
        
        Child execution results are added to parent, key findings promoted.
        """
        # Add child results as new context item
        result_key = f"child_result_{child_context.target_agent_id}"
        parent_context.add_item(
            key=result_key,
            value=results,
            priority=ContextPriority.HIGH,
            source_agent=child_context.target_agent_id
        )
        
        # Promote any critical findings from child
        for key, item in child_context.items.items():
            if item.priority == ContextPriority.CRITICAL:
                if key not in parent_context.items:
                    parent_context.add_item(
                        key=f"child_{key}",
                        value=item.value,
                        priority=ContextPriority.HIGH,
                        source_agent=item.source_agent
                    )
        
        self.log.info(
            "context_merged_from_child",
            parent_id=parent_context.context_id,
            child_id=child_context.context_id,
            new_parent_items=len(parent_context.items)
        )
        
        return parent_context


@dataclass
class ContextDegradationMetrics:
    """Metrics for measuring context loss between levels."""
    items_passed: int = 0
    items_received: int = 0
    tokens_passed: int = 0
    tokens_received: int = 0
    critical_items_lost: int = 0
    high_items_lost: int = 0
    medium_items_lost: int = 0
    low_items_lost: int = 0
    
    @property
    def item_loss_rate(self) -> float:
        """Percentage of items lost."""
        if self.items_passed == 0:
            return 0.0
        return 1.0 - (self.items_received / self.items_passed)
    
    @property
    def token_loss_rate(self) -> float:
        """Percentage of tokens lost."""
        if self.tokens_passed == 0:
            return 0.0
        return 1.0 - (self.tokens_received / self.tokens_passed)


def measure_degradation(
    parent: AgentContext,
    child: AgentContext
) -> ContextDegradationMetrics:
    """
    Measure context degradation between parent and child.
    
    Used to quantify information loss at each hierarchy level.
    """
    metrics = ContextDegradationMetrics(
        items_passed=len(parent.items),
        items_received=len(child.items),
        tokens_passed=parent.current_tokens,
        tokens_received=child.current_tokens
    )
    
    # Count lost items by priority
    parent_keys = set(parent.items.keys())
    child_keys = set(child.items.keys())
    lost_keys = parent_keys - child_keys
    
    for key in lost_keys:
        item = parent.items[key]
        if item.priority == ContextPriority.CRITICAL:
            metrics.critical_items_lost += 1
        elif item.priority == ContextPriority.HIGH:
            metrics.high_items_lost += 1
        elif item.priority == ContextPriority.MEDIUM:
            metrics.medium_items_lost += 1
        else:
            metrics.low_items_lost += 1
    
    return metrics
