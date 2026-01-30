"""
Context Flow Manager for Inter-Level Communication.

Manages context flow between agent hierarchy levels with
scenario-aware handling for different infrastructure setups.

L1 (Opus) ──context──► L2 (Sonnet) ──context──► L3 (Sonnet)
     ▲                      │                      │
     └──────────────────────┴──────────────────────┘
                    state aggregation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

if TYPE_CHECKING:
    from src.agents.base import AgentContext, ContextItem

logger = structlog.get_logger(__name__)


class FlowDirection(str, Enum):
    """Direction of context flow."""
    DOWN = "down"      # L1 -> L2 -> L3
    UP = "up"          # L3 -> L2 -> L1
    LATERAL = "lateral"  # Same level (e.g., L2 to L2 coordination)


class ContextPassResult(BaseModel):
    """Result of passing context between levels."""
    success: bool
    from_agent_id: str
    to_agent_id: str
    direction: FlowDirection
    
    # Token accounting
    tokens_sent: int = 0
    tokens_received: int = 0
    tokens_lost: int = 0
    
    # Item accounting
    items_sent: int = 0
    items_received: int = 0
    items_truncated: int = 0
    items_degraded: int = 0
    
    # Scenario-specific
    scenario: str = ""  # "A", "B", or "C"
    ledger_checkpoint: str | None = None  # For Scenario C
    exchange_log_id: str | None = None  # For Scenario A
    
    # Timing
    pass_duration_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    
    # Errors
    error: str | None = None
    
    @property
    def loss_rate(self) -> float:
        """Percentage of context lost in transfer."""
        if self.tokens_sent == 0:
            return 0.0
        return self.tokens_lost / self.tokens_sent
    
    @property
    def integrity(self) -> float:
        """Context integrity after transfer (1.0 = perfect)."""
        return 1.0 - self.loss_rate


class AggregatedContext(BaseModel):
    """
    Context aggregated from multiple child agents.
    
    Used when L2 specialists or L3 functionals report
    back to their parent orchestrator.
    """
    aggregation_id: str = Field(default_factory=lambda: str(uuid4()))
    target_agent_id: str
    
    # Source agents
    source_agent_ids: list[str] = Field(default_factory=list)
    source_results: dict[str, Any] = Field(default_factory=dict)
    
    # Merged context
    merged_items: dict[str, Any] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    
    # Token summary
    total_source_tokens: int = 0
    merged_tokens: int = 0
    compression_ratio: float = 1.0
    
    # Metadata
    created_at: float = Field(default_factory=time.time)
    scenario: str = ""
    
    def add_source(
        self,
        agent_id: str,
        result: Any,
        tokens: int = 0
    ) -> None:
        """Add a source agent's result to the aggregation."""
        self.source_agent_ids.append(agent_id)
        self.source_results[agent_id] = result
        self.total_source_tokens += tokens
    
    def record_conflict(
        self,
        key: str,
        values: dict[str, Any],
        resolution: Any,
        resolution_method: str
    ) -> None:
        """Record a conflict that occurred during aggregation."""
        self.conflicts.append({
            "key": key,
            "values": values,
            "resolution": resolution,
            "method": resolution_method,
            "timestamp": time.time()
        })


@dataclass
class AgentRef:
    """Reference to an agent in the hierarchy."""
    agent_id: str
    level: int  # 1, 2, or 3
    role: str   # e.g., "portfolio_manager", "ctv_specialist", "pricing"
    
    @property
    def is_orchestrator(self) -> bool:
        return self.level == 1
    
    @property
    def is_specialist(self) -> bool:
        return self.level == 2
    
    @property
    def is_functional(self) -> bool:
        return self.level == 3


class ContextFlowManager:
    """
    Manages context flow between agent hierarchy levels.
    
    Handles the complexities of passing context between:
    - L1 Orchestrator (Claude Opus) - Strategic decisions
    - L2 Specialists (Claude Sonnet) - Channel expertise
    - L3 Functional (Claude Sonnet) - Tool execution
    
    Different scenarios have different context preservation:
    - Scenario A: Exchange mediates, ~60% recovery via logs
    - Scenario B: Direct communication, ~10% loss per handoff
    - Scenario C: Ledger-backed, 0% loss (full recovery)
    """
    
    def __init__(
        self,
        scenario: str = "B",
        ledger_client: Any | None = None,
        exchange_logger: Any | None = None,
        base_degradation_rate: float = 0.10,
        child_budget_ratio: float = 0.6
    ):
        """
        Initialize context flow manager.
        
        Args:
            scenario: Which scenario to simulate ("A", "B", or "C")
            ledger_client: Client for ledger operations (Scenario C)
            exchange_logger: Logger for exchange operations (Scenario A)
            base_degradation_rate: Base rate of context loss per handoff
            child_budget_ratio: Ratio of parent budget given to children
        """
        self.scenario = scenario
        self.ledger_client = ledger_client
        self.exchange_logger = exchange_logger
        self.base_degradation_rate = base_degradation_rate
        self.child_budget_ratio = child_budget_ratio
        self.log = structlog.get_logger(__name__)
        
        # Flow history for debugging/metrics
        self._flow_history: list[ContextPassResult] = []
        self._aggregation_history: list[AggregatedContext] = []
    
    async def pass_context_down(
        self,
        from_agent: AgentRef,
        to_agent: AgentRef,
        context: "AgentContext",
        task: str,
        constraints: list[str] | None = None
    ) -> ContextPassResult:
        """
        Pass context from higher to lower level agent.
        
        In Scenario B: Context may be truncated/lost (~10% per handoff)
        In Scenario C: Full context preserved via ledger checkpoint
        In Scenario A: Context logged to exchange for partial recovery
        
        Args:
            from_agent: Source agent reference
            to_agent: Target agent reference
            context: Context to pass down
            task: Task description for the child agent
            constraints: Additional constraints
            
        Returns:
            ContextPassResult with details of the transfer
        """
        start_time = time.time()
        
        self.log.info(
            "context_flow_down",
            from_agent=from_agent.agent_id,
            to_agent=to_agent.agent_id,
            scenario=self.scenario,
            tokens=context.current_tokens
        )
        
        result = ContextPassResult(
            success=False,
            from_agent_id=from_agent.agent_id,
            to_agent_id=to_agent.agent_id,
            direction=FlowDirection.DOWN,
            scenario=self.scenario,
            tokens_sent=context.current_tokens,
            items_sent=len(context.items)
        )
        
        try:
            if self.scenario == "C":
                # Scenario C: Full preservation via ledger
                result = await self._pass_with_ledger(
                    from_agent, to_agent, context, task, constraints, result
                )
            elif self.scenario == "A":
                # Scenario A: Exchange-mediated with logging
                result = await self._pass_with_exchange_log(
                    from_agent, to_agent, context, task, constraints, result
                )
            else:
                # Scenario B: Direct pass with degradation
                result = await self._pass_direct(
                    from_agent, to_agent, context, task, constraints, result
                )
            
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            self.log.error(
                "context_flow_error",
                error=str(e),
                from_agent=from_agent.agent_id,
                to_agent=to_agent.agent_id
            )
        
        result.pass_duration_ms = (time.time() - start_time) * 1000
        self._flow_history.append(result)
        
        return result
    
    async def _pass_with_ledger(
        self,
        from_agent: AgentRef,
        to_agent: AgentRef,
        context: "AgentContext",
        task: str,
        constraints: list[str] | None,
        result: ContextPassResult
    ) -> ContextPassResult:
        """
        Pass context with ledger checkpoint (Scenario C).
        
        Creates a checkpoint on the ledger that can be used
        for full recovery if the child agent loses context.
        """
        # Create ledger checkpoint
        checkpoint_id = f"ctx_{from_agent.agent_id}_{to_agent.agent_id}_{int(time.time())}"
        
        if self.ledger_client:
            # Store full context on ledger
            await self.ledger_client.store_context(
                checkpoint_id=checkpoint_id,
                context=context.model_dump(),
                from_agent=from_agent.agent_id,
                to_agent=to_agent.agent_id
            )
        
        result.ledger_checkpoint = checkpoint_id
        
        # Full context preserved - no loss
        result.tokens_received = context.current_tokens
        result.items_received = len(context.items)
        result.tokens_lost = 0
        result.items_truncated = 0
        result.items_degraded = 0
        
        self.log.info(
            "context_checkpointed_to_ledger",
            checkpoint_id=checkpoint_id,
            tokens=context.current_tokens
        )
        
        return result
    
    async def _pass_with_exchange_log(
        self,
        from_agent: AgentRef,
        to_agent: AgentRef,
        context: "AgentContext",
        task: str,
        constraints: list[str] | None,
        result: ContextPassResult
    ) -> ContextPassResult:
        """
        Pass context with exchange logging (Scenario A).
        
        Logs the context pass to the exchange, enabling
        partial recovery (~60%) if context is lost.
        """
        import random
        
        # Log to exchange
        log_id = f"exlog_{from_agent.agent_id}_{to_agent.agent_id}_{int(time.time())}"
        
        if self.exchange_logger:
            await self.exchange_logger.log_context_pass(
                log_id=log_id,
                context=context.model_dump(),
                from_agent=from_agent.agent_id,
                to_agent=to_agent.agent_id,
                task=task
            )
        
        result.exchange_log_id = log_id
        
        # Apply moderate degradation (less than Scenario B)
        # Exchange helps preserve ~90% on handoff
        degradation_rate = self.base_degradation_rate * 0.5  # 5% instead of 10%
        
        items_degraded = 0
        tokens_lost = 0
        
        for key, item in list(context.items.items()):
            # Only degrade medium/low priority items
            if item.priority.value in ["medium", "low"]:
                if random.random() < degradation_rate:
                    items_degraded += 1
                    tokens_lost += item.token_estimate
        
        result.tokens_received = context.current_tokens - tokens_lost
        result.items_received = len(context.items) - items_degraded
        result.tokens_lost = tokens_lost
        result.items_degraded = items_degraded
        
        self.log.info(
            "context_logged_to_exchange",
            log_id=log_id,
            tokens_preserved=result.tokens_received,
            items_degraded=items_degraded
        )
        
        return result
    
    async def _pass_direct(
        self,
        from_agent: AgentRef,
        to_agent: AgentRef,
        context: "AgentContext",
        task: str,
        constraints: list[str] | None,
        result: ContextPassResult
    ) -> ContextPassResult:
        """
        Direct context pass with degradation (Scenario B).
        
        No external recovery mechanism - context lost is gone.
        ~10% loss per handoff.
        """
        import random
        
        items_degraded = 0
        items_truncated = 0
        tokens_lost = 0
        
        # Calculate child budget
        child_budget = int(context.token_budget * self.child_budget_ratio)
        child_tokens = 0
        
        # Process items by priority (highest first)
        priority_order = ["critical", "high", "medium", "low"]
        
        for priority in priority_order:
            for key, item in list(context.items.items()):
                if item.priority.value != priority:
                    continue
                
                # Check budget
                if child_tokens + item.token_estimate > child_budget:
                    items_truncated += 1
                    tokens_lost += item.token_estimate
                    continue
                
                # Apply degradation (only to low/medium priority)
                if priority in ["medium", "low"]:
                    if random.random() < self.base_degradation_rate:
                        items_degraded += 1
                        tokens_lost += item.token_estimate
                        continue
                
                child_tokens += item.token_estimate
        
        result.tokens_received = child_tokens
        result.items_received = len(context.items) - items_degraded - items_truncated
        result.tokens_lost = tokens_lost
        result.items_truncated = items_truncated
        result.items_degraded = items_degraded
        
        self.log.info(
            "context_passed_direct",
            tokens_preserved=result.tokens_received,
            tokens_lost=tokens_lost,
            items_degraded=items_degraded,
            items_truncated=items_truncated
        )
        
        return result
    
    async def aggregate_context_up(
        self,
        from_agents: list[AgentRef],
        to_agent: AgentRef,
        results: list[dict[str, Any]],
        contexts: list["AgentContext"] | None = None
    ) -> AggregatedContext:
        """
        Aggregate results from lower level back to higher.
        
        Combines results from multiple child agents (e.g., multiple
        L3 functionals reporting to an L2 specialist, or multiple
        L2 specialists reporting to L1 orchestrator).
        
        Args:
            from_agents: List of source agents
            to_agent: Target parent agent
            results: Results from each source agent
            contexts: Optional contexts from each source
            
        Returns:
            AggregatedContext with merged results and conflict info
        """
        aggregation = AggregatedContext(
            target_agent_id=to_agent.agent_id,
            scenario=self.scenario
        )
        
        self.log.info(
            "aggregating_context_up",
            from_agents=[a.agent_id for a in from_agents],
            to_agent=to_agent.agent_id,
            result_count=len(results)
        )
        
        # Add each source's results
        for i, agent in enumerate(from_agents):
            result = results[i] if i < len(results) else {}
            tokens = contexts[i].current_tokens if contexts and i < len(contexts) else 0
            aggregation.add_source(agent.agent_id, result, tokens)
        
        # Merge items, detecting conflicts
        all_keys: set[str] = set()
        key_values: dict[str, dict[str, Any]] = {}
        
        for agent_id, result in aggregation.source_results.items():
            if isinstance(result, dict):
                for key, value in result.items():
                    all_keys.add(key)
                    if key not in key_values:
                        key_values[key] = {}
                    key_values[key][agent_id] = value
        
        # Process each key
        for key in all_keys:
            values = key_values.get(key, {})
            
            if len(values) == 1:
                # Single source - no conflict
                aggregation.merged_items[key] = list(values.values())[0]
            else:
                # Multiple sources - check for conflict
                unique_values = set(str(v) for v in values.values())
                
                if len(unique_values) == 1:
                    # Same value from all sources
                    aggregation.merged_items[key] = list(values.values())[0]
                else:
                    # Conflict! Use recency-based resolution
                    resolution = self._resolve_conflict(key, values)
                    aggregation.merged_items[key] = resolution
                    aggregation.record_conflict(
                        key=key,
                        values=values,
                        resolution=resolution,
                        resolution_method="recency"
                    )
        
        # Calculate compression
        if aggregation.total_source_tokens > 0:
            aggregation.merged_tokens = sum(
                len(str(v)) // 4 + 1 
                for v in aggregation.merged_items.values()
            )
            aggregation.compression_ratio = (
                aggregation.merged_tokens / aggregation.total_source_tokens
            )
        
        self._aggregation_history.append(aggregation)
        
        self.log.info(
            "context_aggregated",
            source_count=len(from_agents),
            merged_items=len(aggregation.merged_items),
            conflicts=len(aggregation.conflicts),
            compression_ratio=aggregation.compression_ratio
        )
        
        return aggregation
    
    def _resolve_conflict(
        self,
        key: str,
        values: dict[str, Any]
    ) -> Any:
        """
        Resolve conflicting values from multiple agents.
        
        Current strategy: Use last value (recency wins).
        Future: Could use voting, priority, or other strategies.
        """
        # Simple: return the last value
        return list(values.values())[-1]
    
    def get_flow_history(self) -> list[ContextPassResult]:
        """Get history of context flows."""
        return self._flow_history.copy()
    
    def get_aggregation_history(self) -> list[AggregatedContext]:
        """Get history of context aggregations."""
        return self._aggregation_history.copy()
    
    def get_average_loss_rate(self) -> float:
        """Calculate average context loss rate across all flows."""
        if not self._flow_history:
            return 0.0
        return sum(f.loss_rate for f in self._flow_history) / len(self._flow_history)
    
    def clear_history(self) -> None:
        """Clear flow and aggregation history."""
        self._flow_history.clear()
        self._aggregation_history.clear()
