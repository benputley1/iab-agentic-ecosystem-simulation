"""
L1 Orchestrator Agent Base Class.

Orchestrator agents sit at the top of the hierarchy, making strategic
decisions and coordinating L2 specialist agents. Uses Claude Opus
for complex, high-stakes decision making.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any
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
from .specialist import SpecialistAgent

logger = structlog.get_logger(__name__)

# Default model for L1 orchestrator agents
L1_MODEL = "claude-sonnet-4-20250514"


class CampaignState(BaseModel):
    """State for a single campaign within the portfolio."""
    campaign_id: str
    name: str
    budget_total: float
    budget_spent: float = 0
    budget_remaining: float = 0
    objectives: list[str] = Field(default_factory=list)
    channel_allocations: dict[str, float] = Field(default_factory=dict)
    status: str = "active"
    priority: int = 1
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize budget_remaining if not set."""
        if self.budget_remaining == 0:
            self.budget_remaining = self.budget_total


class StrategicDecision(BaseModel):
    """A strategic decision made by the orchestrator."""
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_type: str  # e.g., "budget_allocation", "channel_selection", "campaign_priority"
    description: str
    rationale: str
    impact: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class SpecialistAssignment(BaseModel):
    """Assignment of work to an L2 specialist."""
    specialist_id: str
    channel: str
    objective: str
    context_items: dict[str, Any] = Field(default_factory=dict)
    priority: ContextPriority = ContextPriority.HIGH
    budget_allocation: float = 0


class AssignmentResult(BaseModel):
    """Result from L2 specialist assignment."""
    assignment: SpecialistAssignment
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    degradation_metrics: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0


class OrchestratorAgentState(AgentState):
    """State specific to L1 orchestrator agents."""
    agent_type: str = "orchestrator"
    
    # Portfolio state
    campaigns: dict[str, CampaignState] = Field(default_factory=dict)
    total_budget: float = 0
    total_spent: float = 0
    
    # Strategic decisions
    decisions: list[StrategicDecision] = Field(default_factory=list)
    
    # Coordination tracking
    assignments_made: int = 0
    successful_assignments: int = 0
    failed_assignments: int = 0
    
    # Context aggregation metrics
    total_context_sent_tokens: int = 0
    total_context_received_tokens: int = 0
    context_rot_events: int = 0  # Times context loss affected outcomes
    
    # Token tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    def add_campaign(self, campaign: CampaignState) -> None:
        """Add a campaign to the portfolio."""
        self.campaigns[campaign.campaign_id] = campaign
        self.total_budget += campaign.budget_total
        self.updated_at = time.time()
    
    def record_decision(self, decision: StrategicDecision) -> None:
        """Record a strategic decision."""
        self.decisions.append(decision)
        self.updated_at = time.time()
    
    def record_assignment(self, result: AssignmentResult) -> None:
        """Record an assignment result."""
        self.assignments_made += 1
        if result.success:
            self.successful_assignments += 1
        else:
            self.failed_assignments += 1
        self.updated_at = time.time()
    
    def get_active_campaigns(self) -> list[CampaignState]:
        """Get all active campaigns."""
        return [c for c in self.campaigns.values() if c.status == "active"]


class OrchestratorAgent(ABC):
    """
    L1 Orchestrator Agent Base Class.
    
    Orchestrators are the strategic decision-makers at the top of the hierarchy.
    They manage campaign portfolios, allocate budgets, and coordinate L2
    specialist agents across different channels.
    
    Uses Claude Opus for sophisticated strategic reasoning.
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "OrchestratorAgent",
        anthropic_client: AsyncAnthropic | None = None,
        model: str = L1_MODEL,
        state_backend: StateBackend | None = None,
        context_passing: StandardContextPassing | None = None
    ):
        self.agent_id = agent_id or str(uuid4())
        self.name = name
        self.model = model
        
        # Anthropic client
        self.client = anthropic_client or AsyncAnthropic()
        
        # State management
        self.state_backend = state_backend or VolatileStateBackend()
        self.state_manager = StateManager(self.state_backend)
        self.state: OrchestratorAgentState | None = None
        
        # Context management
        self.context_passing = context_passing or StandardContextPassing(
            degradation_rate=0.05,  # Lower degradation at L1->L2
            child_budget_ratio=0.7
        )
        self.context_window = ContextWindow(max_tokens=200000)  # Opus has larger window
        
        # L2 specialist agents
        self._specialists: dict[str, SpecialistAgent] = {}
        self._channel_specialists: dict[str, str] = {}  # channel -> specialist_id
        
        # Logging
        self.log = structlog.get_logger(__name__).bind(
            agent_id=self.agent_id,
            agent_name=self.name,
            level="L1"
        )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this orchestrator agent."""
        pass
    
    @abstractmethod
    async def make_strategic_decision(
        self,
        context: AgentContext,
        decision_request: str
    ) -> StrategicDecision:
        """
        Make a strategic decision using Opus.
        
        Subclasses implement domain-specific strategic logic.
        """
        pass
    
    @abstractmethod
    async def plan_specialist_assignments(
        self,
        context: AgentContext,
        campaign: CampaignState
    ) -> list[SpecialistAssignment]:
        """
        Plan assignments to L2 specialists for a campaign.
        
        Subclasses implement campaign-specific planning.
        """
        pass
    
    def register_specialist(
        self,
        agent: SpecialistAgent
    ) -> None:
        """Register an L2 specialist agent."""
        self._specialists[agent.agent_id] = agent
        self._channel_specialists[agent.channel] = agent.agent_id
        self.log.info(
            "specialist_registered",
            specialist_id=agent.agent_id,
            specialist_name=agent.name,
            channel=agent.channel
        )
    
    def get_specialist_for_channel(self, channel: str) -> SpecialistAgent | None:
        """Get the specialist agent for a channel."""
        specialist_id = self._channel_specialists.get(channel)
        if specialist_id:
            return self._specialists.get(specialist_id)
        return None
    
    async def initialize(self) -> None:
        """Initialize orchestrator state."""
        self.state = OrchestratorAgentState(
            agent_id=self.agent_id,
            data={
                "name": self.name,
                "model": self.model,
                "specialists": list(self._specialists.keys()),
                "channels": list(self._channel_specialists.keys())
            }
        )
        await self.state_manager.backend.save(self.state)
        
        # Initialize specialists
        for specialist in self._specialists.values():
            await specialist.initialize()
        
        self.log.info(
            "orchestrator_initialized",
            specialists=len(self._specialists),
            channels=list(self._channel_specialists.keys())
        )
    
    async def add_campaign(self, campaign: CampaignState) -> None:
        """Add a campaign to the portfolio."""
        if self.state is None:
            await self.initialize()
        
        self.state.add_campaign(campaign)
        
        self.log.info(
            "campaign_added",
            campaign_id=campaign.campaign_id,
            name=campaign.name,
            budget=campaign.budget_total
        )
    
    async def execute_campaign(
        self,
        campaign_id: str,
        objective: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a campaign through the agent hierarchy.
        
        Args:
            campaign_id: ID of campaign to execute
            objective: Optional specific objective (uses campaign objectives if not provided)
            
        Returns:
            Dictionary with execution results and metrics
        """
        start_time = time.time()
        
        if self.state is None:
            await self.initialize()
        
        campaign = self.state.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign not found: {campaign_id}")
        
        self.log.info(
            "campaign_execution_started",
            campaign_id=campaign_id,
            name=campaign.name,
            budget=campaign.budget_total
        )
        
        # Create root context for this campaign
        context = AgentContext(
            source_level=1,
            source_agent_id=self.agent_id,
            token_budget=int(self.context_window.available_tokens * 0.5),
            task_description=objective or f"Execute campaign: {campaign.name}",
            task_constraints=campaign.objectives
        )
        
        # Add campaign state to context
        context.add_item(
            key="campaign",
            value=campaign.model_dump(),
            priority=ContextPriority.CRITICAL
        )
        
        # Phase 1: Strategic analysis
        strategy = await self._develop_strategy(context, campaign)
        
        # Record strategic decision
        decision = StrategicDecision(
            decision_type="campaign_strategy",
            description=f"Strategy for campaign {campaign.name}",
            rationale=strategy.get("rationale", ""),
            impact=strategy.get("expected_outcomes", {})
        )
        self.state.record_decision(decision)
        
        # Phase 2: Plan specialist assignments
        assignments = await self.plan_specialist_assignments(context, campaign)
        
        self.log.info(
            "specialist_assignments_planned",
            count=len(assignments),
            channels=[a.channel for a in assignments]
        )
        
        # Phase 3: Execute assignments in parallel or sequence
        assignment_results = await self._execute_assignments(context, assignments)
        
        # Phase 4: Aggregate results and update state
        aggregation = await self._aggregate_results(
            context,
            campaign,
            strategy,
            assignment_results
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Update campaign state
        campaign.updated_at = time.time()
        
        result = {
            "orchestrator_id": self.agent_id,
            "orchestrator_name": self.name,
            "campaign_id": campaign_id,
            "campaign_name": campaign.name,
            "success": all(r.success for r in assignment_results) if assignment_results else True,
            "strategy": strategy,
            "assignments": [r.model_dump() for r in assignment_results],
            "aggregation": aggregation,
            "execution_time_ms": execution_time,
            "context_metrics": {
                "root_context_tokens": context.current_tokens,
                "total_degradation_events": sum(
                    1 for r in assignment_results 
                    if r.degradation_metrics.get("item_loss_rate", 0) > 0
                )
            },
            "token_usage": {
                "input_tokens": self.state.total_input_tokens,
                "output_tokens": self.state.total_output_tokens
            }
        }
        
        self.log.info(
            "campaign_execution_completed",
            campaign_id=campaign_id,
            success=result["success"],
            assignments_count=len(assignment_results),
            time_ms=execution_time
        )
        
        return result
    
    async def _develop_strategy(
        self,
        context: AgentContext,
        campaign: CampaignState
    ) -> dict[str, Any]:
        """Develop strategic approach using Opus."""
        
        context_summary = self._build_context_summary(context)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f"""Develop a strategic approach for this campaign:

Campaign: {campaign.name}
Budget: ${campaign.budget_total:,.2f}
Objectives: {', '.join(campaign.objectives)}
Current Allocations: {campaign.channel_allocations}

Context:
{context_summary}

Available Channels: {list(self._channel_specialists.keys())}

Provide strategic analysis including:
1. Recommended channel allocation (percentages that sum to 100)
2. Prioritization of objectives
3. Key success factors
4. Risk assessment
5. Rationale for the approach
6. Expected outcomes

Respond in JSON format with keys: channel_allocation, objective_priority, success_factors, risks, rationale, expected_outcomes"""
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
            
            text = response.content[0].text
            
            import json
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
            
            return {"raw_strategy": text}
            
        except Exception as e:
            self.log.error("strategy_development_failed", error=str(e))
            return {"error": str(e)}
    
    async def _execute_assignments(
        self,
        context: AgentContext,
        assignments: list[SpecialistAssignment]
    ) -> list[AssignmentResult]:
        """Execute specialist assignments."""
        results = []
        
        # Execute in parallel for efficiency
        async def execute_single(assignment: SpecialistAssignment) -> AssignmentResult:
            return await self._execute_single_assignment(context, assignment)
        
        # Run all assignments concurrently
        tasks = [execute_single(a) for a in assignments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log.error(
                    "assignment_exception",
                    assignment=assignments[i].specialist_id,
                    error=str(result)
                )
                processed_results.append(AssignmentResult(
                    assignment=assignments[i],
                    success=False,
                    result={"error": str(result)}
                ))
            else:
                processed_results.append(result)
            
            if self.state:
                self.state.record_assignment(processed_results[-1])
        
        return processed_results
    
    async def _execute_single_assignment(
        self,
        parent_context: AgentContext,
        assignment: SpecialistAssignment
    ) -> AssignmentResult:
        """Execute a single specialist assignment."""
        start_time = time.time()
        
        specialist = self._specialists.get(assignment.specialist_id)
        if not specialist:
            return AssignmentResult(
                assignment=assignment,
                success=False,
                result={"error": f"Unknown specialist: {assignment.specialist_id}"}
            )
        
        # Create child context with controlled degradation
        child_context = self.context_passing.context_to_child(
            parent_context=parent_context,
            child_agent_id=assignment.specialist_id,
            task=assignment.objective
        )
        
        # Add assignment-specific context
        for key, value in assignment.context_items.items():
            child_context.add_item(
                key=key,
                value=value,
                priority=assignment.priority
            )
        
        # Track budget allocation
        child_context.add_item(
            key="budget_allocation",
            value=assignment.budget_allocation,
            priority=ContextPriority.CRITICAL
        )
        
        # Measure degradation
        degradation = measure_degradation(parent_context, child_context)
        
        if self.state:
            self.state.total_context_sent_tokens += parent_context.current_tokens
            self.state.total_context_received_tokens += child_context.current_tokens
        
        try:
            # Execute specialist
            result = await specialist.execute(child_context, assignment.objective)
            
            # Merge results back
            self.context_passing.context_from_child(
                parent_context,
                child_context,
                result
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return AssignmentResult(
                assignment=assignment,
                success=result.get("success", False),
                result=result,
                degradation_metrics={
                    "item_loss_rate": degradation.item_loss_rate,
                    "token_loss_rate": degradation.token_loss_rate,
                    "critical_items_lost": degradation.critical_items_lost,
                    "high_items_lost": degradation.high_items_lost
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.log.error(
                "assignment_execution_failed",
                specialist_id=assignment.specialist_id,
                error=str(e)
            )
            return AssignmentResult(
                assignment=assignment,
                success=False,
                result={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _aggregate_results(
        self,
        context: AgentContext,
        campaign: CampaignState,
        strategy: dict[str, Any],
        assignment_results: list[AssignmentResult]
    ) -> dict[str, Any]:
        """Aggregate results from all specialists using Opus."""
        
        # Build results summary
        results_summary = []
        total_degradation = 0
        
        for ar in assignment_results:
            results_summary.append({
                "channel": ar.assignment.channel,
                "success": ar.success,
                "synthesis": ar.result.get("synthesis", ar.result),
                "degradation": ar.degradation_metrics
            })
            total_degradation += ar.degradation_metrics.get("item_loss_rate", 0)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f"""Aggregate results for campaign: {campaign.name}

Original Strategy:
{strategy}

Specialist Results:
{results_summary}

Total Context Degradation: {total_degradation:.2%} average item loss

Provide aggregated analysis including:
1. Overall campaign outcome assessment
2. Channel-by-channel performance summary
3. Impact of context degradation on results
4. Lessons learned
5. Recommendations for future campaigns
6. Budget utilization assessment

Respond in JSON format with keys: outcome_assessment, channel_performance, degradation_impact, lessons, recommendations, budget_assessment"""
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
            
            return {"raw_aggregation": text}
            
        except Exception as e:
            self.log.error("aggregation_failed", error=str(e))
            return {"error": str(e)}
    
    def _build_context_summary(self, context: AgentContext) -> str:
        """Build text summary of context for LLM."""
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
                    # Truncate long values
                    str_value = str(value)
                    if len(str_value) > 500:
                        str_value = str_value[:500] + "..."
                    parts.append(f"  {key}: {str_value}")
        
        return "\n".join(parts)
    
    async def get_portfolio_status(self) -> dict[str, Any]:
        """Get current portfolio status."""
        if self.state is None:
            return {"status": "uninitialized"}
        
        return {
            "orchestrator_id": self.agent_id,
            "orchestrator_name": self.name,
            "campaigns": {
                cid: {
                    "name": c.name,
                    "status": c.status,
                    "budget_total": c.budget_total,
                    "budget_spent": c.budget_spent,
                    "budget_remaining": c.budget_remaining
                }
                for cid, c in self.state.campaigns.items()
            },
            "total_budget": self.state.total_budget,
            "total_spent": self.state.total_spent,
            "decisions_count": len(self.state.decisions),
            "specialists": list(self._specialists.keys()),
            "channels": list(self._channel_specialists.keys())
        }
    
    async def cleanup(self) -> None:
        """Cleanup orchestrator and all child agents."""
        # Cleanup all specialists
        for specialist in self._specialists.values():
            await specialist.cleanup()
        
        if self.state:
            await self.state_manager.create_snapshot(
                self.state,
                description="Orchestrator cleanup snapshot",
                recovery_point=True
            )
        
        self.log.info("orchestrator_cleanup_complete")
