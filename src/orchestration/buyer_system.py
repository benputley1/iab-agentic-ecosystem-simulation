"""
Buyer Agent System - Complete hierarchy with all levels.

Provides a unified interface for the buyer-side agent hierarchy:
- L1: Portfolio Manager (strategic decisions)
- L2: Channel Specialists (display, video, CTV, mobile, native)
- L3: Functional Agents (research, execution, reporting, audience)

This system coordinates the full hierarchy for multi-campaign management
and integrates with context flow management for tracking context rot.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..agents.buyer.l1_portfolio_manager import PortfolioManager, create_portfolio_manager
from ..agents.buyer.l2_performance import PerformanceSpecialist
from ..agents.buyer.l2_branding import BrandingSpecialist
from ..agents.buyer.l2_ctv import CTVSpecialist
from ..agents.buyer.l2_mobile_app import MobileAppSpecialist
from ..agents.buyer.l2_dsp import DSPSpecialist
from ..agents.buyer.l3_research import ResearchAgent
from ..agents.buyer.l3_execution import ExecutionAgent
from ..agents.buyer.l3_reporting import ReportingAgent
from ..agents.buyer.l3_audience_planner import AudiencePlannerAgent
from ..agents.buyer.models import (
    Campaign,
    CampaignObjectives,
    CampaignStatus,
    AudienceSpec,
    BudgetAllocation,
    ChannelSelection,
    PortfolioState,
    SpecialistTask,
    SpecialistResult,
    Channel,
)
from ..protocols.inter_level import (
    InterLevelProtocol,
    AgentContext,
    Task,
    Result,
    ResultStatus,
    ContextSerializer,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextFlowConfig:
    """Configuration for context flow between agent levels."""
    
    # Token limits for context passing
    l1_to_l2_limit: int = 8000
    l2_to_l3_limit: int = 4000
    l3_to_l2_limit: int = 2000
    l2_to_l1_limit: int = 4000
    
    # Context rot simulation
    enable_context_rot: bool = False
    rot_rate_per_level: float = 0.05  # 5% context loss per level
    
    # Recovery configuration (varies by scenario)
    recovery_enabled: bool = False
    recovery_accuracy: float = 0.0


@dataclass
class HierarchyMetrics:
    """Metrics for the full agent hierarchy execution."""
    
    total_l1_decisions: int = 0
    total_l2_tasks: int = 0
    total_l3_operations: int = 0
    
    # Token tracking
    l1_tokens_used: int = 0
    l2_tokens_used: int = 0
    l3_tokens_used: int = 0
    
    # Context flow tracking
    context_handoffs: int = 0
    context_rot_events: int = 0
    context_recovered: int = 0
    
    # Performance
    total_execution_time_ms: float = 0.0
    avg_level_latency_ms: dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0, 3: 0.0})
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "l1_decisions": self.total_l1_decisions,
            "l2_tasks": self.total_l2_tasks,
            "l3_operations": self.total_l3_operations,
            "l1_tokens": self.l1_tokens_used,
            "l2_tokens": self.l2_tokens_used,
            "l3_tokens": self.l3_tokens_used,
            "context_handoffs": self.context_handoffs,
            "context_rot_events": self.context_rot_events,
            "context_recovered": self.context_recovered,
            "total_time_ms": self.total_execution_time_ms,
            "avg_latency_ms": self.avg_level_latency_ms,
        }


@dataclass
class CampaignResult:
    """Result of processing a campaign through the hierarchy."""
    
    campaign_id: str
    success: bool
    
    # Budget execution
    budget_allocated: float = 0.0
    spend: float = 0.0
    
    # Delivery
    impressions_secured: int = 0
    deals_made: int = 0
    
    # Channel breakdown
    channel_results: dict[str, SpecialistResult] = field(default_factory=dict)
    
    # Context tracking
    context_preserved_pct: float = 100.0  # How much context was retained
    
    # Errors
    errors: list[str] = field(default_factory=list)
    
    # Metrics
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "success": self.success,
            "budget_allocated": self.budget_allocated,
            "spend": self.spend,
            "impressions_secured": self.impressions_secured,
            "deals_made": self.deals_made,
            "context_preserved_pct": self.context_preserved_pct,
            "errors": self.errors,
            "execution_time_ms": self.execution_time_ms,
        }


class ContextFlowManager:
    """Manages context flow between agent hierarchy levels.
    
    Tracks context as it flows L1 -> L2 -> L3 and back, simulating
    context rot at each handoff when enabled.
    """
    
    def __init__(self, config: Optional[ContextFlowConfig] = None):
        """Initialize context flow manager.
        
        Args:
            config: Context flow configuration
        """
        self.config = config or ContextFlowConfig()
        self._serializer = ContextSerializer()
        self._context_history: list[dict] = []
        
        # Track context state
        self._current_context: Optional[AgentContext] = None
        self._context_tokens_original: int = 0
        self._context_tokens_current: int = 0
    
    def create_initial_context(
        self,
        campaign: Campaign,
        agent_id: str,
    ) -> AgentContext:
        """Create initial context for campaign processing.
        
        Args:
            campaign: Campaign to create context for
            agent_id: Agent ID for context
            
        Returns:
            Initial AgentContext
        """
        context = AgentContext.create(
            agent_id=agent_id,
            level=1,
            working_memory={
                "campaign": campaign.to_dict(),
                "campaign_id": campaign.campaign_id,
                "total_budget": campaign.total_budget,
                "objectives": campaign.objectives.to_dict(),
                "audience": campaign.audience.to_dict(),
            },
            constraints={
                "budget_limit": campaign.total_budget,
                "cpm_target": campaign.objectives.cpm_target,
                "reach_target": campaign.objectives.reach_target,
            },
            metadata={
                "created_at": datetime.utcnow().isoformat(),
                "scenario": "multi_agent",
            },
        )
        
        self._current_context = context
        self._context_tokens_original = self._serializer.to_tokens(context)
        self._context_tokens_current = self._context_tokens_original
        
        return context
    
    def pass_context_down(
        self,
        context: AgentContext,
        from_level: int,
        to_level: int,
    ) -> AgentContext:
        """Pass context from higher to lower level.
        
        May apply context rot if enabled.
        
        Args:
            context: Context to pass
            from_level: Source level
            to_level: Target level
            
        Returns:
            Potentially truncated context
        """
        # Determine token limit for this transition
        if from_level == 1 and to_level == 2:
            limit = self.config.l1_to_l2_limit
        elif from_level == 2 and to_level == 3:
            limit = self.config.l2_to_l3_limit
        else:
            limit = 4000  # Default
        
        # Truncate to limit
        truncated = self._serializer.truncate_to_limit(context, limit)
        
        # Apply context rot if enabled
        if self.config.enable_context_rot:
            truncated = self._apply_rot(truncated)
        
        # Track
        self._context_history.append({
            "action": "pass_down",
            "from_level": from_level,
            "to_level": to_level,
            "tokens_before": context.token_count,
            "tokens_after": truncated.token_count,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        self._context_tokens_current = truncated.token_count
        
        return truncated
    
    def pass_context_up(
        self,
        context: AgentContext,
        result: Result,
        from_level: int,
        to_level: int,
    ) -> AgentContext:
        """Pass context and result from lower to higher level.
        
        Args:
            context: Context to pass
            result: Task result
            from_level: Source level
            to_level: Target level
            
        Returns:
            Updated context
        """
        # Determine token limit
        if from_level == 3 and to_level == 2:
            limit = self.config.l3_to_l2_limit
        elif from_level == 2 and to_level == 1:
            limit = self.config.l2_to_l1_limit
        else:
            limit = 4000
        
        # Merge result into context
        updated_context = AgentContext(
            context_id=context.context_id,
            agent_id=context.agent_id,
            level=to_level,
            conversation_history=context.conversation_history.copy(),
            working_memory={
                **context.working_memory,
                f"l{from_level}_result": result.output,
                f"l{from_level}_status": result.status.value,
            },
            constraints=context.constraints.copy(),
            metadata={
                **context.metadata,
                f"l{from_level}_completed_at": datetime.utcnow().isoformat(),
            },
        )
        
        # Truncate
        truncated = self._serializer.truncate_to_limit(updated_context, limit)
        
        # Track
        self._context_history.append({
            "action": "pass_up",
            "from_level": from_level,
            "to_level": to_level,
            "result_status": result.status.value,
            "tokens_after": truncated.token_count,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return truncated
    
    def _apply_rot(self, context: AgentContext) -> AgentContext:
        """Apply context rot simulation.
        
        Args:
            context: Context to rot
            
        Returns:
            Rotted context
        """
        import random
        
        rot_rate = self.config.rot_rate_per_level
        
        # Randomly drop some working memory keys
        rotted_memory = {}
        for key, value in context.working_memory.items():
            if random.random() > rot_rate:
                rotted_memory[key] = value
        
        # Randomly drop some conversation history
        rotted_history = [
            msg for msg in context.conversation_history
            if random.random() > rot_rate
        ]
        
        return AgentContext(
            context_id=context.context_id,
            agent_id=context.agent_id,
            level=context.level,
            conversation_history=rotted_history,
            working_memory=rotted_memory,
            constraints=context.constraints,
            metadata={
                **context.metadata,
                "rot_applied": True,
                "rot_rate": rot_rate,
            },
        )
    
    def get_context_preservation(self) -> float:
        """Get percentage of original context preserved."""
        if self._context_tokens_original == 0:
            return 100.0
        return (self._context_tokens_current / self._context_tokens_original) * 100
    
    def get_history(self) -> list[dict]:
        """Get context flow history."""
        return self._context_history.copy()


class BuyerAgentSystem:
    """Complete buyer agent system with all hierarchy levels.
    
    Coordinates:
    - L1: Portfolio Manager (strategic orchestration)
    - L2: Channel Specialists (display, video, CTV, mobile, native)
    - L3: Functional Agents (research, execution, reporting, audience)
    
    Example:
        ```python
        system = BuyerAgentSystem(buyer_id="buyer-001")
        await system.initialize()
        
        result = await system.process_campaign(campaign)
        ```
    """
    
    def __init__(
        self,
        buyer_id: Optional[str] = None,
        scenario: str = "A",
        context_config: Optional[ContextFlowConfig] = None,
        mock_llm: bool = False,
    ):
        """Initialize the buyer agent system.
        
        Args:
            buyer_id: Unique identifier for this buyer system
            scenario: Simulation scenario (A, B, or C)
            context_config: Context flow configuration
            mock_llm: Use mock LLM responses (for testing)
        """
        self.buyer_id = buyer_id or f"buyer-{uuid.uuid4().hex[:8]}"
        self.scenario = scenario
        self.mock_llm = mock_llm
        
        # Context management
        self.context_manager = ContextFlowManager(context_config)
        
        # Hierarchy components (lazy initialized)
        self._l1_portfolio_manager: Optional[PortfolioManager] = None
        self._l2_specialists: dict[str, Any] = {}
        self._l3_functional: dict[str, Any] = {}
        
        # Inter-level protocol
        self._protocols: dict[int, InterLevelProtocol] = {}
        
        # Metrics
        self.metrics = HierarchyMetrics()
        
        # State
        self._initialized = False
        
        logger.info(f"BuyerAgentSystem created: {self.buyer_id}")
    
    @property
    def l1_portfolio_manager(self) -> PortfolioManager:
        """Get L1 Portfolio Manager."""
        if self._l1_portfolio_manager is None:
            raise RuntimeError("System not initialized. Call initialize() first.")
        return self._l1_portfolio_manager
    
    @property
    def l2_specialists(self) -> dict[str, Any]:
        """Get L2 channel specialists."""
        return self._l2_specialists
    
    @property
    def l3_functional(self) -> dict[str, Any]:
        """Get L3 functional agents."""
        return self._l3_functional
    
    async def initialize(self) -> None:
        """Initialize all hierarchy components.
        
        Creates and connects:
        - L1 Portfolio Manager
        - L2 Channel Specialists
        - L3 Functional Agents
        - Inter-level protocols
        """
        if self._initialized:
            return
        
        logger.info(f"Initializing BuyerAgentSystem {self.buyer_id}")
        
        # Create L1 Portfolio Manager
        self._l1_portfolio_manager = create_portfolio_manager(
            agent_id=f"{self.buyer_id}-l1-pm",
            scenario=self.scenario,
        )
        await self._l1_portfolio_manager.initialize()
        
        # Create L2 Channel Specialists
        self._l2_specialists = {
            Channel.DISPLAY.value: DSPSpecialist(
                agent_id=f"{self.buyer_id}-l2-display",
            ),
            Channel.VIDEO.value: PerformanceSpecialist(
                agent_id=f"{self.buyer_id}-l2-video",
            ),
            Channel.CTV.value: CTVSpecialist(
                agent_id=f"{self.buyer_id}-l2-ctv",
            ),
            Channel.MOBILE_APP.value: MobileAppSpecialist(
                agent_id=f"{self.buyer_id}-l2-mobile",
            ),
            Channel.NATIVE.value: BrandingSpecialist(
                agent_id=f"{self.buyer_id}-l2-native",
            ),
        }
        
        # Initialize L2 specialists
        for specialist in self._l2_specialists.values():
            await specialist.initialize()
        
        # Create L3 Functional Agents
        self._l3_functional = {
            "research": ResearchAgent(
                agent_id=f"{self.buyer_id}-l3-research",
            ),
            "execution": ExecutionAgent(
                agent_id=f"{self.buyer_id}-l3-execution",
            ),
            "reporting": ReportingAgent(
                agent_id=f"{self.buyer_id}-l3-reporting",
            ),
            "audience": AudiencePlannerAgent(
                agent_id=f"{self.buyer_id}-l3-audience",
            ),
        }
        
        # Initialize L3 agents
        for agent in self._l3_functional.values():
            await agent.initialize()
        
        # Create inter-level protocols
        self._protocols[1] = InterLevelProtocol(
            agent_id=self._l1_portfolio_manager.agent_id,
            agent_level=1,
        )
        
        for name, specialist in self._l2_specialists.items():
            self._protocols[1].register_subordinate(specialist.agent_id, 2)
        
        self._initialized = True
        logger.info(f"BuyerAgentSystem {self.buyer_id} initialized")
    
    async def process_campaign(self, campaign: Campaign) -> CampaignResult:
        """Process a campaign through the full hierarchy.
        
        Flow:
        1. L1 allocates budget and selects channels
        2. L2 specialists create channel plans
        3. L3 agents execute functional tasks
        4. Results flow back up through hierarchy
        
        Args:
            campaign: Campaign to process
            
        Returns:
            CampaignResult with execution details
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        start_time = datetime.utcnow()
        result = CampaignResult(campaign_id=campaign.campaign_id, success=False)
        
        try:
            # Create initial context
            context = self.context_manager.create_initial_context(
                campaign=campaign,
                agent_id=self._l1_portfolio_manager.agent_id,
            )
            
            # L1: Portfolio Manager - Strategic allocation
            logger.info(f"L1 processing campaign {campaign.campaign_id}")
            self.metrics.total_l1_decisions += 1
            
            # Add campaign to portfolio manager
            self._l1_portfolio_manager.add_l1_campaign(campaign)
            
            # Get budget allocation
            allocation = await self._l1_portfolio_manager.allocate_budget([campaign])
            result.budget_allocated = allocation.total_allocated
            
            # Get channel selections
            channel_selections = await self._l1_portfolio_manager.select_channels(campaign)
            
            # L2: Delegate to channel specialists
            for selection in channel_selections:
                if not selection.selected:
                    continue
                
                channel = selection.channel
                specialist = self._l2_specialists.get(channel)
                
                if not specialist:
                    logger.warning(f"No specialist for channel: {channel}")
                    continue
                
                # Pass context down L1 -> L2
                l2_context = self.context_manager.pass_context_down(context, 1, 2)
                self.metrics.context_handoffs += 1
                
                logger.info(f"L2 {channel} processing campaign {campaign.campaign_id}")
                self.metrics.total_l2_tasks += 1
                
                # Calculate budget for this channel
                channel_budget = campaign.remaining_budget * selection.allocation_pct
                
                # Create task for specialist
                task = Task.create(
                    name=f"execute_{channel}_buy",
                    description=f"Execute {channel} buying for campaign {campaign.campaign_id}",
                    task_type="channel_execution",
                    created_by=self._l1_portfolio_manager.agent_id,
                    parameters={
                        "campaign_id": campaign.campaign_id,
                        "budget": channel_budget,
                        "cpm_target": campaign.objectives.cpm_target,
                        "reach_target": int(campaign.objectives.reach_target * selection.allocation_pct),
                    },
                    context=l2_context,
                )
                
                # Execute L2 specialist
                specialist_result = await self._execute_l2_task(
                    specialist=specialist,
                    task=task,
                    context=l2_context,
                )
                
                # Store channel result
                result.channel_results[channel] = specialist_result
                
                # Aggregate results
                if specialist_result.success:
                    result.impressions_secured += specialist_result.impressions_secured
                    result.spend += specialist_result.spend
                    result.deals_made += len(specialist_result.deals)
            
            # Update campaign state
            campaign.spend += result.spend
            campaign.impressions_delivered += result.impressions_secured
            campaign.deals_made += result.deals_made
            
            # Calculate context preservation
            result.context_preserved_pct = self.context_manager.get_context_preservation()
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Campaign processing failed: {e}", exc_info=True)
            result.errors.append(str(e))
        
        # Calculate execution time
        end_time = datetime.utcnow()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        self.metrics.total_execution_time_ms += result.execution_time_ms
        
        return result
    
    async def _execute_l2_task(
        self,
        specialist: Any,
        task: Task,
        context: AgentContext,
    ) -> SpecialistResult:
        """Execute a task with an L2 specialist.
        
        Args:
            specialist: L2 specialist agent
            task: Task to execute
            context: Agent context
            
        Returns:
            SpecialistResult
        """
        try:
            # Get L3 functional agents to help
            research_agent = self._l3_functional.get("research")
            execution_agent = self._l3_functional.get("execution")
            
            # Pass context L2 -> L3
            l3_context = self.context_manager.pass_context_down(context, 2, 3)
            self.metrics.context_handoffs += 1
            
            # L3: Research inventory
            self.metrics.total_l3_operations += 1
            if research_agent:
                research_result = await research_agent.discover_inventory(
                    channel=task.parameters.get("channel", "display"),
                    audience=task.context.working_memory.get("audience", {}),
                    budget=task.parameters.get("budget", 0),
                )
            else:
                research_result = {"inventory": [], "recommendations": []}
            
            # L3: Execute buying
            self.metrics.total_l3_operations += 1
            if execution_agent:
                execution_result = await execution_agent.execute_buy(
                    inventory=research_result.get("inventory", []),
                    budget=task.parameters.get("budget", 0),
                    cpm_target=task.parameters.get("cpm_target", 15.0),
                )
            else:
                # Mock execution
                impressions = int(task.parameters.get("budget", 0) / 0.015)  # $15 CPM
                execution_result = {
                    "impressions": impressions,
                    "spend": task.parameters.get("budget", 0),
                    "deals": [],
                }
            
            return SpecialistResult(
                task_id=task.task_id,
                campaign_id=task.parameters.get("campaign_id", ""),
                channel=task.parameters.get("channel", "display"),
                success=True,
                impressions_secured=execution_result.get("impressions", 0),
                spend=execution_result.get("spend", 0.0),
                deals=execution_result.get("deals", []),
            )
            
        except Exception as e:
            logger.error(f"L2 task execution failed: {e}")
            return SpecialistResult(
                task_id=task.task_id,
                campaign_id=task.parameters.get("campaign_id", ""),
                channel=task.parameters.get("channel", "display"),
                success=False,
                error=str(e),
            )
    
    async def process_multi_campaign(
        self,
        campaigns: list[Campaign],
    ) -> list[CampaignResult]:
        """Process multiple campaigns through the hierarchy.
        
        Uses portfolio-level coordination for budget allocation
        across campaigns.
        
        Args:
            campaigns: List of campaigns to process
            
        Returns:
            List of CampaignResult
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        results = []
        
        # First, have L1 do portfolio-level allocation
        for campaign in campaigns:
            self._l1_portfolio_manager.add_l1_campaign(campaign)
        
        # Get allocation across all campaigns
        allocation = await self._l1_portfolio_manager.allocate_budget(campaigns)
        
        # Process each campaign
        for campaign in campaigns:
            result = await self.process_campaign(campaign)
            results.append(result)
        
        return results
    
    def get_metrics(self) -> dict:
        """Get hierarchy execution metrics."""
        return self.metrics.to_dict()
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state from L1."""
        if self._l1_portfolio_manager:
            return self._l1_portfolio_manager.portfolio
        return PortfolioState(portfolio_id="empty")
    
    async def shutdown(self) -> None:
        """Shutdown the buyer agent system."""
        logger.info(f"Shutting down BuyerAgentSystem {self.buyer_id}")
        
        # Shutdown L3 agents
        for agent in self._l3_functional.values():
            if hasattr(agent, "shutdown"):
                await agent.shutdown()
        
        # Shutdown L2 specialists
        for specialist in self._l2_specialists.values():
            if hasattr(specialist, "shutdown"):
                await specialist.shutdown()
        
        # Shutdown L1
        if self._l1_portfolio_manager and hasattr(self._l1_portfolio_manager, "shutdown"):
            await self._l1_portfolio_manager.shutdown()
        
        self._initialized = False
        logger.info(f"BuyerAgentSystem {self.buyer_id} shutdown complete")


async def create_buyer_system(
    buyer_id: Optional[str] = None,
    scenario: str = "A",
    context_config: Optional[ContextFlowConfig] = None,
    mock_llm: bool = False,
) -> BuyerAgentSystem:
    """Create and initialize a buyer agent system.
    
    Args:
        buyer_id: Optional buyer ID
        scenario: Simulation scenario
        context_config: Context flow configuration
        mock_llm: Use mock LLM
        
    Returns:
        Initialized BuyerAgentSystem
    """
    system = BuyerAgentSystem(
        buyer_id=buyer_id,
        scenario=scenario,
        context_config=context_config,
        mock_llm=mock_llm,
    )
    await system.initialize()
    return system
