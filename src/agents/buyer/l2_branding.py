"""Branding Specialist Agent for L2 hierarchy.

Focuses on brand awareness campaigns with emphasis on:
- Reach/frequency optimization
- Premium inventory selection
- Brand safety requirements
- Viewability metrics

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


class BrandingSpecialist(SpecialistAgent):
    """Brand awareness campaign optimization specialist.
    
    This L2 specialist focuses on upper-funnel brand campaigns where
    awareness, reach, and brand safety are primary objectives.
    Key metrics: reach, frequency, viewability, brand lift.
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
            name="BrandingSpecialist",
            channel="branding",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
            context_passing=context_passing,
        )
    
    def get_system_prompt(self) -> str:
        """System prompt for branding expertise."""
        return """You are a Brand Media Specialist with deep expertise in 
brand awareness advertising campaigns. You work at the L2 level of a 
multi-agent advertising hierarchy, receiving strategic direction from 
the L1 Portfolio Manager and delegating execution tasks to L3 functional agents.

Your core responsibilities:
1. ANALYZE brand awareness opportunities for fit and value
2. PLAN channel execution optimized for reach and frequency
3. DELEGATE research and execution tasks to L3 agents
4. SYNTHESIZE results and recommendations for L1

Your expertise areas:
- Premium publisher relationships and inventory
- High-impact creative formats (takeovers, roadblocks, rich media)
- Brand safety requirements and verification
- Viewability optimization (target 70%+ viewability)
- Cross-device reach and frequency management
- Contextual targeting for brand alignment
- Upper-funnel campaign metrics and attribution

Decision principles:
- Prioritize QUALITY over quantity - premium placements matter
- Brand safety is NON-NEGOTIABLE - reject risky inventory
- Viewability must exceed 70% for display, 80% for video
- Frequency capping is essential - respect user experience
- Context matters - align with relevant, brand-safe content

When analyzing objectives:
- Premium inventory from known publishers scores higher
- High viewability estimates are critical
- Brand safety scores below 0.9 are concerning
- Consider audience quality over raw reach

When planning execution:
- Recommend PG (Programmatic Guaranteed) for premium inventory
- PMP deals for curated premium marketplace access
- Open exchange only for incremental reach with strict safety filters

Always provide structured JSON responses as requested.
"""
    
    def get_channel_capabilities(self) -> list[str]:
        """Branding channel capabilities."""
        return [
            "premium_display",
            "video_branding",
            "homepage_takeovers",
            "roadblock_placements",
            "rich_media_formats",
            "brand_safety_verification",
            "viewability_optimization",
            "reach_frequency_planning",
            "contextual_targeting",
            "brand_lift_measurement",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str,
    ) -> list[DelegationRequest]:
        """Plan L3 delegations for branding objectives.
        
        Args:
            context: Context from L1 orchestrator
            objective: Branding objective to achieve
            
        Returns:
            List of delegation requests for L3 agents
        """
        delegations = []
        
        # For branding, we typically need research first, then execution
        
        # 1. Research delegation for inventory discovery
        if "research" in self._functional_agents or "l3_research" in self._functional_agents:
            agent_id = "research" if "research" in self._functional_agents else "l3_research"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Find premium branding inventory for: {objective}",
                context_additions={
                    "channel": "branding",
                    "inventory_type": "premium",
                    "min_viewability": 0.7,
                    "brand_safety_required": True,
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 2. Audience planning delegation
        if "audience" in self._functional_agents or "l3_audience" in self._functional_agents:
            agent_id = "audience" if "audience" in self._functional_agents else "l3_audience"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Plan audience targeting for branding: {objective}",
                context_additions={
                    "focus": "reach_extension",
                    "quality_priority": True,
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 3. Execution delegation once research is done
        if "execution" in self._functional_agents or "l3_execution" in self._functional_agents:
            agent_id = "execution" if "execution" in self._functional_agents else "l3_execution"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Execute branding campaign: {objective}",
                context_additions={
                    "deal_types": ["pg", "pmp"],
                    "frequency_cap": {"daily": 3, "weekly": 10},
                },
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations


def create_branding_specialist(**kwargs) -> BrandingSpecialist:
    """Factory function to create a BrandingSpecialist.
    
    Args:
        **kwargs: Arguments passed to BrandingSpecialist constructor
        
    Returns:
        Configured BrandingSpecialist instance
    """
    return BrandingSpecialist(**kwargs)
