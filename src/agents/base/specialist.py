"""
L2 Specialist Agent Base Class.

Specialist agents handle channel-specific expertise and coordinate
L3 functional agents for their domain. Uses Claude Sonnet for
efficient channel-focused decision making.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar
from uuid import uuid4

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
import structlog

from .context import (
    AgentContext,
    ContextPriority,
    ContextWindow,
    StandardContextPassing,
    ContextDegradationMetrics,
    measure_degradation,
)
from .state import (
    AgentState,
    StateManager,
    StateBackend,
    VolatileStateBackend,
)
from .functional import FunctionalAgent

logger = structlog.get_logger(__name__)

# Default model for L2 specialist agents
L2_MODEL = "claude-sonnet-4-20250514"


# =============================================================================
# Shared Models for L2 Specialists (used by both buyer and seller)
# =============================================================================

class FitScore(BaseModel):
    """Score indicating how well something fits criteria."""
    score: float = Field(ge=0.0, le=1.0, description="Fit score 0-1")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    factors: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class Campaign(BaseModel):
    """Campaign representation for specialist agents."""
    campaign_id: str
    name: str
    objective: str = ""
    budget: float = 0.0
    target_impressions: int = 0
    target_cpm: float = 0.0
    start_date: str | None = None
    end_date: str | None = None
    targeting: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Task to delegate to L3 functional agents."""
    task_id: str
    task_type: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    context: dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    """Result from L3 functional agent execution."""
    task_id: str
    success: bool
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


class AvailsRequest(BaseModel):
    """Request to check inventory availability."""
    channel: str
    impressions_needed: int
    start_date: str | None = None
    end_date: str | None = None
    targeting: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class AvailsResponse(BaseModel):
    """Response with availability information."""
    available: bool
    impressions_available: int = 0
    recommended_allocation: int = 0
    constraints_met: bool = True
    notes: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class PricingRequest(BaseModel):
    """Request for pricing information."""
    channel: str
    impressions: int
    deal_type: str = "open"  # open, pmp, pg
    buyer_tier: str = "public"
    targeting: dict[str, Any] = Field(default_factory=dict)


class PricingResponse(BaseModel):
    """Response with pricing information."""
    base_cpm: float
    floor_cpm: float
    recommended_cpm: float
    discount_applied: float = 0.0
    pricing_factors: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class DelegationRequest(BaseModel):
    """Request to delegate work to an L3 functional agent."""
    functional_agent_id: str
    task: str
    context_additions: dict[str, Any] = Field(default_factory=dict)
    priority: ContextPriority = ContextPriority.HIGH


class DelegationResult(BaseModel):
    """Result from L3 functional agent delegation."""
    request: DelegationRequest
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    degradation_metrics: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0


class SpecialistAgentState(AgentState):
    """State specific to L2 specialist agents."""
    agent_type: str = "specialist"
    
    # Channel-specific state
    channel: str = ""
    channel_config: dict[str, Any] = Field(default_factory=dict)
    
    # Delegation tracking
    delegations_made: int = 0
    successful_delegations: int = 0
    failed_delegations: int = 0
    
    # Context metrics
    total_context_passed_tokens: int = 0
    total_context_received_tokens: int = 0
    context_degradation_events: int = 0
    
    # Token tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    def record_delegation(self, result: DelegationResult) -> None:
        """Record a delegation result."""
        self.delegations_made += 1
        if result.success:
            self.successful_delegations += 1
        else:
            self.failed_delegations += 1
        self.updated_at = time.time()
    
    def record_context_degradation(
        self,
        tokens_passed: int,
        tokens_received: int
    ) -> None:
        """Record context passing metrics."""
        self.total_context_passed_tokens += tokens_passed
        self.total_context_received_tokens += tokens_received
        if tokens_received < tokens_passed:
            self.context_degradation_events += 1
        self.updated_at = time.time()


class SpecialistAgent(ABC):
    """
    L2 Specialist Agent Base Class.
    
    Specialist agents handle channel-specific expertise (e.g., CTV, Mobile, Display).
    They receive strategic context from L1 orchestrators, make channel-specific
    decisions, and coordinate L3 functional agents for execution.
    
    Uses Claude Sonnet for cost-effective channel expertise.
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "SpecialistAgent",
        channel: str = "generic",
        anthropic_client: AsyncAnthropic | None = None,
        model: str = L2_MODEL,
        state_backend: StateBackend | None = None,
        context_passing: StandardContextPassing | None = None
    ):
        self.agent_id = agent_id or str(uuid4())
        self.name = name
        self.channel = channel
        self.model = model
        
        # Anthropic client
        self.client = anthropic_client or AsyncAnthropic()
        
        # State management
        self.state_backend = state_backend or VolatileStateBackend()
        self.state_manager = StateManager(self.state_backend)
        self.state: SpecialistAgentState | None = None
        
        # Context management
        self.context_passing = context_passing or StandardContextPassing(
            degradation_rate=0.1,
            child_budget_ratio=0.6
        )
        self.context_window = ContextWindow(max_tokens=100000)
        
        # L3 functional agents
        self._functional_agents: dict[str, FunctionalAgent] = {}
        
        # Logging
        self.log = structlog.get_logger(__name__).bind(
            agent_id=self.agent_id,
            agent_name=self.name,
            channel=self.channel,
            level="L2"
        )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this specialist agent."""
        pass
    
    @abstractmethod
    def get_channel_capabilities(self) -> list[str]:
        """Return list of capabilities for this channel."""
        pass
    
    @abstractmethod
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str
    ) -> list[DelegationRequest]:
        """
        Plan what L3 agents to invoke for the given objective.
        
        Subclasses implement channel-specific planning logic.
        """
        pass
    
    def register_functional_agent(
        self,
        agent: FunctionalAgent
    ) -> None:
        """Register an L3 functional agent."""
        self._functional_agents[agent.agent_id] = agent
        self.log.info(
            "functional_agent_registered",
            functional_id=agent.agent_id,
            functional_name=agent.name
        )
    
    async def initialize(self) -> None:
        """Initialize agent state."""
        self.state = SpecialistAgentState(
            agent_id=self.agent_id,
            agent_type="specialist",
            channel=self.channel,
            data={
                "name": self.name,
                "model": self.model,
                "capabilities": self.get_channel_capabilities(),
                "functional_agents": list(self._functional_agents.keys())
            }
        )
        await self.state_manager.backend.save(self.state)
        
        # Initialize functional agents
        for agent in self._functional_agents.values():
            await agent.initialize()
        
        self.log.info(
            "agent_initialized",
            capabilities=self.get_channel_capabilities(),
            functional_count=len(self._functional_agents)
        )
    
    async def execute(
        self,
        context: AgentContext,
        objective: str
    ) -> dict[str, Any]:
        """
        Execute channel-specific objective.
        
        Args:
            context: Context passed down from L1 orchestrator
            objective: Channel-specific objective to achieve
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        if self.state is None:
            await self.initialize()
        
        self.log.info(
            "execution_started",
            objective=objective[:100],
            context_items=len(context.items)
        )
        
        # Phase 1: Strategic analysis using LLM
        analysis = await self._analyze_objective(context, objective)
        
        # Phase 2: Plan delegations
        delegation_requests = await self.plan_execution(context, objective)
        
        self.log.info(
            "delegation_plan_created",
            delegation_count=len(delegation_requests)
        )
        
        # Phase 3: Execute delegations to L3 agents
        delegation_results = []
        for request in delegation_requests:
            result = await self._execute_delegation(context, request)
            delegation_results.append(result)
            
            if self.state:
                self.state.record_delegation(result)
        
        # Phase 4: Synthesize results
        synthesis = await self._synthesize_results(
            context,
            objective,
            analysis,
            delegation_results
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        result = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "channel": self.channel,
            "objective": objective,
            "success": all(r.success for r in delegation_results) if delegation_results else True,
            "analysis": analysis,
            "delegations": [r.model_dump() for r in delegation_results],
            "synthesis": synthesis,
            "execution_time_ms": execution_time,
            "token_usage": {
                "input_tokens": self.state.total_input_tokens if self.state else 0,
                "output_tokens": self.state.total_output_tokens if self.state else 0
            }
        }
        
        self.log.info(
            "execution_completed",
            success=result["success"],
            delegations=len(delegation_results),
            time_ms=execution_time
        )
        
        return result
    
    async def _analyze_objective(
        self,
        context: AgentContext,
        objective: str
    ) -> dict[str, Any]:
        """Use LLM to analyze objective and determine approach."""
        
        context_summary = self._summarize_context(context)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this objective for {self.channel} channel:

Context:
{context_summary}

Objective: {objective}

Provide:
1. Key considerations for this channel
2. Recommended approach
3. Potential risks or constraints
4. Which functional capabilities are needed

Respond in JSON format with keys: considerations, approach, risks, required_capabilities"""
                    }
                ]
            )
            
            # Track tokens
            self.context_window.record_usage(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            
            if self.state:
                self.state.total_input_tokens += response.usage.input_tokens
                self.state.total_output_tokens += response.usage.output_tokens
            
            # Parse response
            text = response.content[0].text
            
            # Try to extract JSON
            import json
            try:
                # Find JSON in response
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
            
            return {"raw_analysis": text}
            
        except Exception as e:
            self.log.error("analysis_failed", error=str(e))
            return {"error": str(e)}
    
    async def _execute_delegation(
        self,
        parent_context: AgentContext,
        request: DelegationRequest
    ) -> DelegationResult:
        """Execute a delegation to an L3 functional agent."""
        start_time = time.time()
        
        agent = self._functional_agents.get(request.functional_agent_id)
        if not agent:
            return DelegationResult(
                request=request,
                success=False,
                result={"error": f"Unknown functional agent: {request.functional_agent_id}"}
            )
        
        # Create child context with degradation
        child_context = self.context_passing.context_to_child(
            parent_context=parent_context,
            child_agent_id=request.functional_agent_id,
            task=request.task
        )
        
        # Add request-specific context
        for key, value in request.context_additions.items():
            child_context.add_item(
                key=key,
                value=value,
                priority=request.priority
            )
        
        # Measure degradation
        degradation = measure_degradation(parent_context, child_context)
        
        if self.state:
            self.state.record_context_degradation(
                parent_context.current_tokens,
                child_context.current_tokens
            )
        
        try:
            # Execute functional agent
            result = await agent.execute(child_context, request.task)
            
            # Merge results back into parent context
            self.context_passing.context_from_child(
                parent_context,
                child_context,
                result
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return DelegationResult(
                request=request,
                success=result.get("success", False),
                result=result,
                degradation_metrics={
                    "item_loss_rate": degradation.item_loss_rate,
                    "token_loss_rate": degradation.token_loss_rate,
                    "critical_items_lost": degradation.critical_items_lost
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.log.error(
                "delegation_failed",
                functional_id=request.functional_agent_id,
                error=str(e)
            )
            return DelegationResult(
                request=request,
                success=False,
                result={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _synthesize_results(
        self,
        context: AgentContext,
        objective: str,
        analysis: dict[str, Any],
        delegation_results: list[DelegationResult]
    ) -> dict[str, Any]:
        """Synthesize delegation results into cohesive response."""
        
        # Build results summary
        results_summary = []
        for dr in delegation_results:
            results_summary.append({
                "task": dr.request.task,
                "success": dr.success,
                "result": dr.result.get("final_response", dr.result),
                "context_degradation": dr.degradation_metrics
            })
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f"""Synthesize results for objective: {objective}

Initial Analysis:
{analysis}

Delegation Results:
{results_summary}

Provide a synthesis that:
1. Summarizes what was accomplished
2. Notes any failures or issues
3. Provides recommendations for the orchestrator
4. Highlights any context loss that affected results

Respond in JSON format with keys: summary, issues, recommendations, context_notes"""
                    }
                ]
            )
            
            # Track tokens
            if self.state:
                self.state.total_input_tokens += response.usage.input_tokens
                self.state.total_output_tokens += response.usage.output_tokens
            
            text = response.content[0].text
            
            import json
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
            
            return {"raw_synthesis": text}
            
        except Exception as e:
            self.log.error("synthesis_failed", error=str(e))
            return {"error": str(e)}
    
    def _summarize_context(self, context: AgentContext) -> str:
        """Create text summary of context for LLM."""
        parts = []
        
        if context.task_description:
            parts.append(f"Task: {context.task_description}")
        
        if context.task_constraints:
            parts.append(f"Constraints: {', '.join(context.task_constraints)}")
        
        for priority in ContextPriority:
            items = context.get_by_priority(priority)
            if items:
                parts.append(f"\n{priority.value.upper()}:")
                for key, value in items.items():
                    parts.append(f"  {key}: {value}")
        
        return "\n".join(parts)
    
    async def cleanup(self) -> None:
        """Cleanup agent and child resources."""
        # Cleanup functional agents
        for agent in self._functional_agents.values():
            await agent.cleanup()
        
        if self.state:
            await self.state_manager.create_snapshot(
                self.state,
                description="Specialist cleanup snapshot",
                recovery_point=True
            )
        
        self.log.info("agent_cleanup_complete")
