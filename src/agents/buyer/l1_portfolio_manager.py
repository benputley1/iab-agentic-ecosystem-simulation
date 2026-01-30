"""Level 1 Buyer Orchestrator: Portfolio Manager.

The Portfolio Manager is the strategic brain of the buyer side, coordinating
multiple campaigns and delegating to L2 channel specialists.

Responsibilities:
- Multi-campaign budget allocation
- Strategic channel decisions
- Performance management
- L2 specialist coordination
"""

import json
import logging
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base.orchestrator import (
    OrchestratorAgent,
    OrchestratorAgentState,
    CampaignState,
    StrategicDecision,
    SpecialistAssignment,
    AssignmentResult,
    L1_MODEL,
)
from ..base.context import AgentContext, ContextPriority
from .models import (
    Campaign,
    CampaignObjectives,
    AudienceSpec,
    BudgetAllocation,
    ChannelSelection,
    PortfolioState,
    SpecialistTask,
    SpecialistResult,
    Channel,
)
from .prompts.portfolio_manager import (
    PORTFOLIO_MANAGER_SYSTEM_PROMPT,
    BUDGET_ALLOCATION_PROMPT,
    CHANNEL_SELECTION_PROMPT,
    PERFORMANCE_REVIEW_PROMPT,
    MULTI_CAMPAIGN_COORDINATION_PROMPT,
    BUDGET_ALLOCATION_SCHEMA,
    CHANNEL_SELECTION_SCHEMA,
)

logger = logging.getLogger(__name__)


class PortfolioManager(OrchestratorAgent):
    """Level 1 Buyer Orchestrator using Claude Opus.
    
    The Portfolio Manager makes strategic decisions about budget allocation,
    channel selection, and coordinates L2 channel specialists to execute
    the buying strategy.
    
    Example:
        ```python
        manager = PortfolioManager(agent_id="buyer-pm-001")
        await manager.initialize()
        
        # Add campaign
        campaign = CampaignState(
            campaign_id="camp-001",
            name="Q1 Awareness",
            budget_total=50000.0,
            objectives=["reach", "awareness"]
        )
        await manager.add_campaign(campaign)
        
        # Execute
        result = await manager.execute_campaign("camp-001")
        ```
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "PortfolioManager",
        **kwargs
    ):
        """Initialize the Portfolio Manager.
        
        Args:
            agent_id: Unique identifier for this manager
            name: Display name for the manager
            **kwargs: Additional args passed to OrchestratorAgent
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            **kwargs
        )
        self._portfolio = PortfolioState(portfolio_id=f"portfolio-{self.agent_id}")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for budget allocation decisions."""
        return PORTFOLIO_MANAGER_SYSTEM_PROMPT
    
    @property
    def portfolio(self) -> PortfolioState:
        """Get current portfolio state (L1Campaign-based view)."""
        return self._portfolio
    
    def add_l1_campaign(self, campaign: Campaign) -> None:
        """Add a Campaign (from models.py) to the portfolio.
        
        This converts the Campaign to CampaignState and adds both
        to their respective state objects.
        
        Args:
            campaign: Campaign to add
        """
        # Add to portfolio state (our domain model)
        self._portfolio.add_campaign(campaign)
        
        # Convert to CampaignState for base class
        campaign_state = CampaignState(
            campaign_id=campaign.campaign_id,
            name=campaign.name,
            budget_total=campaign.total_budget,
            budget_spent=campaign.spend,
            budget_remaining=campaign.remaining_budget,
            objectives=list(campaign.objectives.channel_mix.keys()) if campaign.objectives.channel_mix else ["reach"],
            channel_allocations=campaign.objectives.channel_mix,
            status=campaign.status.value,
            priority=campaign.priority,
        )
        
        # Add to base class state
        if self.state:
            self.state.add_campaign(campaign_state)
        
        logger.info(f"Added campaign {campaign.campaign_id} to portfolio")
    
    async def make_strategic_decision(
        self,
        context: AgentContext,
        decision_request: str
    ) -> StrategicDecision:
        """Make a strategic decision using Opus.
        
        Args:
            context: Agent context with relevant data
            decision_request: Description of the decision needed
            
        Returns:
            StrategicDecision with reasoning and impact
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f"""Make a strategic decision for the following request:

{decision_request}

Context:
Task: {context.task_description}
Constraints: {context.task_constraints}

Respond in JSON format with keys: decision_type, description, rationale, impact (dict with expected effects)"""
                    }
                ]
            )
            
            # Track tokens
            if self.state:
                self.state.total_input_tokens += response.usage.input_tokens
                self.state.total_output_tokens += response.usage.output_tokens
            
            text = response.content[0].text
            
            # Parse JSON response
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(text[start:end])
                    return StrategicDecision(
                        decision_type=data.get("decision_type", "strategic"),
                        description=data.get("description", decision_request),
                        rationale=data.get("rationale", ""),
                        impact=data.get("impact", {})
                    )
            except json.JSONDecodeError:
                pass
            
            # Fallback if parsing fails
            return StrategicDecision(
                decision_type="strategic",
                description=decision_request,
                rationale=text,
                impact={}
            )
            
        except Exception as e:
            logger.error(f"Strategic decision failed: {e}")
            return StrategicDecision(
                decision_type="error",
                description=decision_request,
                rationale=f"Decision failed: {e}",
                impact={"error": True}
            )
    
    async def plan_specialist_assignments(
        self,
        context: AgentContext,
        campaign: CampaignState
    ) -> List[SpecialistAssignment]:
        """Plan assignments to L2 specialists for a campaign.
        
        Args:
            context: Agent context
            campaign: Campaign to plan for
            
        Returns:
            List of specialist assignments
        """
        assignments = []
        
        # Get channel allocations from campaign
        channel_allocations = campaign.channel_allocations or {}
        
        if not channel_allocations:
            # Default to display if no allocations specified
            channel_allocations = {"display": 1.0}
        
        # Create assignments for each channel
        for channel, allocation_pct in channel_allocations.items():
            specialist = self.get_specialist_for_channel(channel)
            
            if specialist:
                budget_for_channel = campaign.budget_remaining * allocation_pct
                
                assignment = SpecialistAssignment(
                    specialist_id=specialist.agent_id,
                    channel=channel,
                    objective=f"Execute {channel} buying for campaign {campaign.name}",
                    context_items={
                        "campaign_id": campaign.campaign_id,
                        "campaign_name": campaign.name,
                        "objectives": campaign.objectives,
                    },
                    priority=ContextPriority.HIGH if campaign.priority <= 2 else ContextPriority.NORMAL,
                    budget_allocation=budget_for_channel
                )
                assignments.append(assignment)
                
                logger.info(
                    f"Planned {channel} assignment: ${budget_for_channel:,.2f} "
                    f"for campaign {campaign.campaign_id}"
                )
            else:
                logger.warning(f"No specialist for channel: {channel}")
        
        return assignments
    
    async def allocate_budget(self, campaigns: List[Campaign]) -> BudgetAllocation:
        """Decide how to split budget across campaigns and channels.
        
        Uses Claude Opus to make strategic allocation decisions based on
        campaign objectives, performance history, and market conditions.
        
        Args:
            campaigns: List of campaigns to allocate budget for
            
        Returns:
            Budget allocation with reasoning
        """
        if not campaigns:
            return BudgetAllocation(reasoning="No campaigns to allocate")
        
        # Prepare portfolio summary
        total_budget = sum(c.remaining_budget for c in campaigns)
        portfolio_summary = f"Total campaigns: {len(campaigns)}, Total budget: ${total_budget:,.2f}"
        
        # Prepare campaign details
        campaigns_json = json.dumps([c.to_dict() for c in campaigns], indent=2)
        
        # Performance context (simplified for now)
        performance_context = "Initial allocation - no historical performance data available."
        
        # Format the prompt
        prompt = BUDGET_ALLOCATION_PROMPT.format(
            portfolio_summary=portfolio_summary,
            campaigns_json=campaigns_json,
            total_budget=total_budget,
            performance_context=performance_context,
        )
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Track tokens
            if self.state:
                self.state.total_input_tokens += response.usage.input_tokens
                self.state.total_output_tokens += response.usage.output_tokens
            
            text = response.content[0].text
            
            # Parse JSON
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(text[start:end])
                    allocation = BudgetAllocation(
                        allocations=data.get("allocations", {}),
                        reasoning=data.get("reasoning", ""),
                        total_allocated=data.get("total_allocated", 0.0),
                    )
                    self._portfolio.current_allocation = allocation
                    logger.info(f"Budget allocation completed: ${allocation.total_allocated:,.2f}")
                    return allocation
            except json.JSONDecodeError:
                pass
            
            return self._default_allocation(campaigns)
            
        except Exception as e:
            logger.error(f"Budget allocation failed: {e}")
            return self._default_allocation(campaigns)
    
    def _default_allocation(self, campaigns: List[Campaign]) -> BudgetAllocation:
        """Create a default even allocation when LLM fails."""
        allocations = {}
        total = 0.0
        
        for campaign in campaigns:
            channel_mix = campaign.objectives.channel_mix
            if not channel_mix:
                channel_mix = {Channel.DISPLAY.value: 1.0}
            
            allocations[campaign.campaign_id] = {}
            for channel, pct in channel_mix.items():
                amount = campaign.remaining_budget * pct
                allocations[campaign.campaign_id][channel] = amount
                total += amount
        
        return BudgetAllocation(
            allocations=allocations,
            reasoning="Default even allocation (LLM fallback)",
            total_allocated=total,
        )
    
    async def select_channels(self, campaign: Campaign) -> List[ChannelSelection]:
        """Strategic decision: which channels for this campaign?
        
        Args:
            campaign: Campaign to select channels for
            
        Returns:
            List of channel selections with rationale
        """
        campaign_json = json.dumps(campaign.to_dict(), indent=2)
        audience_json = json.dumps(campaign.audience.to_dict(), indent=2)
        
        channels_info = """
Available Channels:
- display: Standard display banners, CPM range $8-20
- video: Pre-roll/mid-roll video, CPM range $15-35
- ctv: Connected TV streaming, CPM range $25-50
- mobile_app: In-app advertising, CPM range $10-25
- native: Native content integration, CPM range $12-30
"""
        
        channel_performance = "No historical data available for new campaign."
        
        prompt = CHANNEL_SELECTION_PROMPT.format(
            campaign_json=campaign_json,
            reach_target=campaign.objectives.reach_target,
            frequency_cap=campaign.objectives.frequency_cap,
            cpm_target=campaign.objectives.cpm_target,
            viewability_target=campaign.objectives.viewability_target,
            audience_json=audience_json,
            channels_info=channels_info,
            channel_performance=channel_performance,
        )
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )
            
            if self.state:
                self.state.total_input_tokens += response.usage.input_tokens
                self.state.total_output_tokens += response.usage.output_tokens
            
            text = response.content[0].text
            
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(text[start:end])
                    selections = []
                    for ch in data.get("selected_channels", []):
                        selections.append(ChannelSelection(
                            channel=ch.get("channel", ""),
                            selected=True,
                            allocation_pct=ch.get("allocation_pct", 0.0),
                            rationale=ch.get("rationale", ""),
                            expected_reach=ch.get("expected_reach", 0),
                            expected_cpm=ch.get("expected_cpm", 0.0),
                        ))
                    logger.info(f"Channel selection for {campaign.campaign_id}: {len(selections)} channels")
                    return selections
            except json.JSONDecodeError:
                pass
            
            return [ChannelSelection(
                channel=Channel.DISPLAY.value,
                selected=True,
                allocation_pct=1.0,
                rationale="Default fallback to display",
            )]
            
        except Exception as e:
            logger.error(f"Channel selection failed: {e}")
            return [ChannelSelection(
                channel=Channel.DISPLAY.value,
                selected=True,
                allocation_pct=1.0,
                rationale="Default fallback to display",
            )]
    
    async def aggregate_results(
        self, 
        specialist_results: List[SpecialistResult]
    ) -> PortfolioState:
        """Aggregate L2 results back into portfolio view.
        
        Args:
            specialist_results: Results from L2 specialists
            
        Returns:
            Updated portfolio state
        """
        for result in specialist_results:
            if not result.success:
                logger.warning(f"Task {result.task_id} failed: {result.error}")
                continue
            
            # Update campaign state
            campaign = self._portfolio.campaigns.get(result.campaign_id)
            if campaign:
                campaign.spend += result.spend
                campaign.impressions_delivered += result.impressions_secured
                campaign.deals_made += len(result.deals)
        
        # Recalculate portfolio totals
        self._portfolio._recalculate_totals()
        
        logger.info(
            f"Portfolio aggregation complete: "
            f"spend=${self._portfolio.total_spend:,.2f}, "
            f"impressions={self._portfolio.total_impressions:,}"
        )
        
        return self._portfolio


def create_portfolio_manager(
    agent_id: Optional[str] = None,
    scenario: str = "A",
    **kwargs
) -> PortfolioManager:
    """Create a Portfolio Manager instance.
    
    Args:
        agent_id: Optional custom agent ID
        scenario: Simulation scenario (A, B, or C)
        **kwargs: Additional args passed to PortfolioManager
        
    Returns:
        Configured PortfolioManager instance
    """
    if agent_id is None:
        agent_id = f"portfolio-manager-{uuid.uuid4().hex[:8]}"
    
    manager = PortfolioManager(
        agent_id=agent_id,
        name=f"PortfolioManager-{scenario}",
        **kwargs
    )
    
    return manager
