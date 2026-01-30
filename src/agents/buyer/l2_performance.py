"""Performance Specialist Agent for L2 hierarchy.

Focuses on performance/conversion campaigns with emphasis on:
- ROAS optimization
- Conversion tracking
- Retargeting strategies
- Attribution modeling

Author: Alkimi Exchange
"""

from typing import Any

from anthropic import AsyncAnthropic

from ..base.context import AgentContext, ContextPriority
from ..base.state import StateBackend
from ..base.specialist import (
    SpecialistAgent,
    DelegationRequest,
    StandardContextPassing,
)


class PerformanceSpecialist(SpecialistAgent):
    """Performance/conversion campaign optimization specialist.
    
    This L2 specialist focuses on lower-funnel performance campaigns
    where ROAS, conversions, and measurable outcomes are primary KPIs.
    Expertise in attribution, retargeting, and conversion optimization.
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        anthropic_client: AsyncAnthropic | None = None,
        state_backend: StateBackend | None = None,
        context_passing: StandardContextPassing | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            name="PerformanceSpecialist",
            channel="performance",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
            context_passing=context_passing,
        )
    
    def get_system_prompt(self) -> str:
        """System prompt for performance expertise."""
        return """You are a Performance Marketing Specialist with deep expertise in 
conversion-focused advertising campaigns. You work at the L2 level of a 
multi-agent advertising hierarchy, receiving strategic direction from 
the L1 Portfolio Manager and delegating execution tasks to L3 functional agents.

Your core responsibilities:
1. ANALYZE performance inventory opportunities for ROAS potential
2. PLAN channel execution optimized for conversions
3. DELEGATE research and execution tasks to L3 agents
4. SYNTHESIZE results and recommendations for L1

Your expertise areas:
- Return on Ad Spend (ROAS) optimization
- Conversion tracking and pixel implementation
- Retargeting and remarketing strategies
- Multi-touch attribution modeling
- Dynamic creative optimization (DCO)
- Landing page and conversion funnel analysis
- Bid optimization algorithms
- Customer acquisition cost (CAC) management
- Incrementality measurement

Decision principles:
- ROAS is the north star - every dollar must work
- Attribution model choice dramatically affects perceived performance
- Retargeting has diminishing returns - watch frequency
- View-through conversions need careful valuation
- Lower-funnel users need different creative than prospecting
- Test incrementality to avoid paying for organic conversions

When analyzing objectives:
- Inventory with proven conversion performance scores highest
- Consider proximity to purchase intent
- Retargeting pools from 1st party data are premium
- Commerce/shopping contexts convert better
- Mobile web vs app has different conversion patterns

When planning execution:
- Layer audiences by funnel stage (prospect, consider, retarget)
- Recommend frequency caps by audience segment
- Plan for creative testing cadence
- Include conversion pixel verification
- Budget allocation: prospecting vs retargeting mix
- Consider lookback window alignment with sales cycle

Always provide structured JSON responses as requested.
"""
    
    def get_channel_capabilities(self) -> list[str]:
        """Performance channel capabilities."""
        return [
            "roas_optimization",
            "conversion_tracking",
            "retargeting_strategies",
            "attribution_modeling",
            "dynamic_creative",
            "bid_optimization",
            "cac_management",
            "incrementality_testing",
            "funnel_analysis",
            "pixel_implementation",
            "audience_segmentation",
            "commerce_media",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str,
    ) -> list[DelegationRequest]:
        """Plan L3 delegations for performance objectives.
        
        Args:
            context: Context from L1 orchestrator
            objective: Performance objective to achieve
            
        Returns:
            List of delegation requests for L3 agents
        """
        delegations = []
        
        # 1. Research for conversion-focused inventory
        if "research" in self._functional_agents or "l3_research" in self._functional_agents:
            agent_id = "research" if "research" in self._functional_agents else "l3_research"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Find conversion-optimized inventory for: {objective}",
                context_additions={
                    "channel": "performance",
                    "optimize_for": "conversions",
                    "inventory_types": ["commerce", "retargeting", "prospecting"],
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 2. Audience planning with funnel segmentation
        if "audience" in self._functional_agents or "l3_audience" in self._functional_agents:
            agent_id = "audience" if "audience" in self._functional_agents else "l3_audience"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Plan conversion funnel audiences for: {objective}",
                context_additions={
                    "funnel_stages": ["prospecting", "consideration", "retargeting"],
                    "lookalike_sources": ["converters", "high_value_customers"],
                    "exclusion_lists": ["recent_converters"],
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 3. Execution with ROAS focus
        if "execution" in self._functional_agents or "l3_execution" in self._functional_agents:
            agent_id = "execution" if "execution" in self._functional_agents else "l3_execution"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Execute performance campaign: {objective}",
                context_additions={
                    "bid_strategy": "target_roas",
                    "conversion_tracking": True,
                    "frequency_caps": {
                        "prospecting": {"daily": 5},
                        "retargeting": {"daily": 10},
                    },
                    "attribution_window": "7d_click_1d_view",
                },
                priority=ContextPriority.MEDIUM,
            ))
        
        # 4. Reporting for ROAS tracking
        if "reporting" in self._functional_agents or "l3_reporting" in self._functional_agents:
            agent_id = "reporting" if "reporting" in self._functional_agents else "l3_reporting"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Set up performance reporting for: {objective}",
                context_additions={
                    "metrics": ["roas", "cpa", "conversion_rate", "ltv"],
                    "attribution_model": "data_driven",
                    "incrementality_test": True,
                },
                priority=ContextPriority.LOW,
            ))
        
        return delegations


def create_performance_specialist(**kwargs) -> PerformanceSpecialist:
    """Factory function to create a PerformanceSpecialist.
    
    Args:
        **kwargs: Arguments passed to PerformanceSpecialist constructor
        
    Returns:
        Configured PerformanceSpecialist instance
    """
    return PerformanceSpecialist(**kwargs)
