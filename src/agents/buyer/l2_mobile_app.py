"""Mobile App Specialist Agent for L2 hierarchy.

Focuses on mobile app install campaigns with emphasis on:
- CPI optimization
- App store targeting
- Attribution tracking
- In-app ad formats

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


class MobileAppSpecialist(SpecialistAgent):
    """Mobile app install campaign optimization specialist.
    
    This L2 specialist focuses on app install and engagement campaigns
    where cost-per-install (CPI) and lifetime value (LTV) are key metrics.
    Expertise in mobile SDKs, attribution, and in-app formats.
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
            name="MobileAppSpecialist",
            channel="mobile_app",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
            context_passing=context_passing,
        )
    
    def get_system_prompt(self) -> str:
        """System prompt for mobile app expertise."""
        return """You are a Mobile App Marketing Specialist with deep expertise in 
app install and engagement advertising campaigns. You work at the L2 level of a 
multi-agent advertising hierarchy, receiving strategic direction from 
the L1 Portfolio Manager and delegating execution tasks to L3 functional agents.

Your core responsibilities:
1. ANALYZE mobile app inventory opportunities for CPI efficiency
2. PLAN channel execution optimized for installs and ROAS
3. DELEGATE research and execution tasks to L3 agents
4. SYNTHESIZE results and recommendations for L1

Your expertise areas:
- Cost-per-install (CPI) optimization
- App store targeting (iOS/Android segmentation)
- Mobile measurement partner (MMP) attribution
- In-app ad formats (interstitial, rewarded, native)
- User acquisition funnels
- Post-install event optimization
- Fraud detection and prevention
- SDK integrations (AppsFlyer, Adjust, Branch)

Decision principles:
- CPI must align with LTV expectations
- iOS vs Android requires different strategies (ATT considerations)
- Rewarded video often drives highest quality users
- In-app inventory quality varies widely - verify sources
- Attribution windows matter - consider your MMP settings
- Fraud is rampant - prioritize verified inventory sources

When analyzing objectives:
- In-app inventory from SDK-verified sources preferred
- Consider platform mix (iOS premium but smaller, Android scale)
- Rewarded formats typically drive engaged users
- Interstitials need frequency capping
- Gaming inventory often performs well for gaming apps

When planning execution:
- Split budgets across iOS/Android based on LTV ratios
- Recommend A/B testing creative formats
- Include retargeting for lapsed users
- Plan for post-install event optimization
- Consider seasonal patterns in app installs

Always provide structured JSON responses as requested.
"""
    
    def get_channel_capabilities(self) -> list[str]:
        """Mobile app channel capabilities."""
        return [
            "cpi_optimization",
            "app_store_targeting",
            "attribution_tracking",
            "interstitial_ads",
            "rewarded_video",
            "native_in_app",
            "playable_ads",
            "fraud_prevention",
            "ios_campaigns",
            "android_campaigns",
            "post_install_events",
            "ltv_optimization",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str,
    ) -> list[DelegationRequest]:
        """Plan L3 delegations for mobile app objectives.
        
        Args:
            context: Context from L1 orchestrator
            objective: Mobile app objective to achieve
            
        Returns:
            List of delegation requests for L3 agents
        """
        delegations = []
        
        # 1. Research delegation for inventory discovery
        if "research" in self._functional_agents or "l3_research" in self._functional_agents:
            agent_id = "research" if "research" in self._functional_agents else "l3_research"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Find mobile app inventory for: {objective}",
                context_additions={
                    "channel": "mobile_app",
                    "inventory_types": ["rewarded_video", "interstitial", "native"],
                    "fraud_filter": True,
                    "sdk_verified": True,
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 2. Audience planning - especially for lookalike modeling
        if "audience" in self._functional_agents or "l3_audience" in self._functional_agents:
            agent_id = "audience" if "audience" in self._functional_agents else "l3_audience"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Plan audience targeting for app installs: {objective}",
                context_additions={
                    "include_lookalikes": True,
                    "platform_split": {"ios": 0.4, "android": 0.6},
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 3. Execution with focus on post-install events
        if "execution" in self._functional_agents or "l3_execution" in self._functional_agents:
            agent_id = "execution" if "execution" in self._functional_agents else "l3_execution"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Execute mobile app campaign: {objective}",
                context_additions={
                    "optimize_for": "install",
                    "secondary_event": "first_purchase",
                    "attribution_window": "7d_click_1d_view",
                },
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations


def create_mobile_app_specialist(**kwargs) -> MobileAppSpecialist:
    """Factory function to create a MobileAppSpecialist.
    
    Args:
        **kwargs: Arguments passed to MobileAppSpecialist constructor
        
    Returns:
        Configured MobileAppSpecialist instance
    """
    return MobileAppSpecialist(**kwargs)
