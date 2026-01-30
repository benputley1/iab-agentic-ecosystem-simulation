"""CTV Specialist Agent for L2 hierarchy.

Focuses on Connected TV campaigns with emphasis on:
- Household reach
- Streaming platform selection
- PG vs PMP decisions
- Cross-device attribution

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


class CTVSpecialist(SpecialistAgent):
    """Connected TV campaign optimization specialist.
    
    This L2 specialist focuses on CTV/OTT campaigns where household
    reach, premium streaming content, and cross-device measurement
    are key considerations.
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
            name="CTVSpecialist",
            channel="ctv",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
            context_passing=context_passing,
        )
    
    def get_system_prompt(self) -> str:
        """System prompt for CTV expertise."""
        return """You are a Connected TV (CTV) Advertising Specialist with deep expertise in 
streaming and OTT video advertising campaigns. You work at the L2 level of a 
multi-agent advertising hierarchy, receiving strategic direction from 
the L1 Portfolio Manager and delegating execution tasks to L3 functional agents.

Your core responsibilities:
1. ANALYZE CTV inventory opportunities for reach and quality
2. PLAN channel execution for streaming campaigns
3. DELEGATE research and execution tasks to L3 agents
4. SYNTHESIZE results and recommendations for L1

Your expertise areas:
- Household reach optimization
- Streaming platform selection (Hulu, Roku, Amazon, etc.)
- Programmatic Guaranteed (PG) vs PMP negotiations
- Premium long-form video inventory
- Cross-device attribution and measurement
- Frequency capping at household level
- Content genre targeting
- Device targeting (Smart TVs, streaming sticks, gaming consoles)
- Linear TV + CTV holistic planning

Decision principles:
- CTV is PREMIUM - don't treat it like display
- Household-level frequency is critical - avoid over-exposure
- Full-episode players (FEP) >> short-form content
- Verify inventory is actually TV-screen delivery
- Premium streaming apps have better completion rates
- Consider cord-cutters vs cord-stackers in targeting

When analyzing objectives:
- Premium AVOD/FAST channels from recognized platforms score highest
- Verify 100% completion rate expectations for non-skippable
- Check for household-level targeting and deduplication
- Consider co-viewing - multiple viewers per impression
- Device transparency is essential - reject blind CTV

When planning execution:
- Recommend PG deals for premium tentpole content
- PMP for quality scale across multiple platforms
- Reserve open exchange for incremental reach only
- Plan for 15s and 30s creative mix
- Include measurement for brand lift and tune-in

Always provide structured JSON responses as requested.
"""
    
    def get_channel_capabilities(self) -> list[str]:
        """CTV channel capabilities."""
        return [
            "household_reach",
            "streaming_platform_selection",
            "programmatic_guaranteed",
            "pmp_deals",
            "long_form_video",
            "cross_device_attribution",
            "frequency_management",
            "content_targeting",
            "device_targeting",
            "avod_fast_channels",
            "completion_rate_optimization",
            "co_viewing_measurement",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str,
    ) -> list[DelegationRequest]:
        """Plan L3 delegations for CTV objectives.
        
        Args:
            context: Context from L1 orchestrator
            objective: CTV objective to achieve
            
        Returns:
            List of delegation requests for L3 agents
        """
        delegations = []
        
        # 1. Research for premium streaming inventory
        if "research" in self._functional_agents or "l3_research" in self._functional_agents:
            agent_id = "research" if "research" in self._functional_agents else "l3_research"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Find premium CTV/streaming inventory for: {objective}",
                context_additions={
                    "channel": "ctv",
                    "inventory_types": ["avod", "fast", "premium_ott"],
                    "device_types": ["smart_tv", "streaming_device", "gaming_console"],
                    "require_transparency": True,
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 2. Audience planning with household focus
        if "audience" in self._functional_agents or "l3_audience" in self._functional_agents:
            agent_id = "audience" if "audience" in self._functional_agents else "l3_audience"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Plan household audience targeting for CTV: {objective}",
                context_additions={
                    "targeting_level": "household",
                    "deduplication": True,
                    "co_viewing_factor": 1.5,
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 3. Execution with PG/PMP focus
        if "execution" in self._functional_agents or "l3_execution" in self._functional_agents:
            agent_id = "execution" if "execution" in self._functional_agents else "l3_execution"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Execute CTV campaign: {objective}",
                context_additions={
                    "deal_types": ["pg", "pmp"],
                    "frequency_cap": {"household_daily": 2, "household_weekly": 6},
                    "creative_lengths": ["15s", "30s"],
                    "completion_target": 0.95,
                },
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations


def create_ctv_specialist(**kwargs) -> CTVSpecialist:
    """Factory function to create a CTVSpecialist.
    
    Args:
        **kwargs: Arguments passed to CTVSpecialist constructor
        
    Returns:
        Configured CTVSpecialist instance
    """
    return CTVSpecialist(**kwargs)
