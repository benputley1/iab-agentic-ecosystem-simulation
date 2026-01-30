"""DSP Specialist Agent for L2 hierarchy.

Focuses on DSP-specific optimization with emphasis on:
- Real-time bidding strategy
- Inventory source selection
- Deal prioritization
- DSP feature utilization

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


class DSPSpecialist(SpecialistAgent):
    """DSP-specific optimization specialist.
    
    This L2 specialist focuses on maximizing DSP platform capabilities
    including RTB bidding strategies, inventory source management,
    deal prioritization, and platform-specific feature utilization.
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
            name="DSPSpecialist",
            channel="dsp",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
            context_passing=context_passing,
        )
    
    def get_system_prompt(self) -> str:
        """System prompt for DSP expertise."""
        return """You are a DSP Platform Specialist with deep expertise in 
demand-side platform optimization and real-time bidding strategies. 
You work at the L2 level of a multi-agent advertising hierarchy, 
receiving strategic direction from the L1 Portfolio Manager and 
delegating execution tasks to L3 functional agents.

Your core responsibilities:
1. ANALYZE inventory sources and supply paths for efficiency
2. PLAN RTB and deal execution strategies
3. DELEGATE research and execution tasks to L3 agents
4. SYNTHESIZE results and recommendations for L1

Your expertise areas:
- Real-time bidding (RTB) strategy and optimization
- Supply path optimization (SPO)
- Deal ID prioritization and management
- DSP algorithm tuning and bid shading
- Inventory source selection and curation
- Private marketplace (PMP) setup and management
- Deal troubleshooting and optimization
- Cross-DSP strategy coordination
- Bid response time optimization

Decision principles:
- SUPPLY PATH matters - shortest path wins
- Deal priority order: PG > PMP > Open
- Bid shading saves money without losing wins
- Inventory quality varies by SSP - know your sources
- Duplicate auctions waste budget - deduplicate
- Win rate targets matter for pacing
- First-price vs second-price requires different strategies

When analyzing objectives:
- Direct publisher paths score highest (fewer hops)
- Known SSPs with quality reputations preferred
- Consider bid density and competition levels
- Deal IDs with performance history score higher
- Watch for inventory arbitrage patterns

When planning execution:
- Structure deal priority waterfall correctly
- Recommend SPO exclusions for known bad actors
- Plan bid shading parameters by inventory type
- Include deal testing allocation for new PMPs
- Set appropriate win rate targets for pacing
- Consider time-of-day bidding adjustments

Always provide structured JSON responses as requested.
"""
    
    def get_channel_capabilities(self) -> list[str]:
        """DSP channel capabilities."""
        return [
            "rtb_optimization",
            "supply_path_optimization",
            "deal_prioritization",
            "bid_shading",
            "inventory_curation",
            "pmp_management",
            "deal_troubleshooting",
            "bid_response_optimization",
            "win_rate_management",
            "ssp_evaluation",
            "auction_dynamics",
            "pacing_algorithms",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str,
    ) -> list[DelegationRequest]:
        """Plan L3 delegations for DSP objectives.
        
        Args:
            context: Context from L1 orchestrator
            objective: DSP objective to achieve
            
        Returns:
            List of delegation requests for L3 agents
        """
        delegations = []
        
        # 1. Research for supply path analysis
        if "research" in self._functional_agents or "l3_research" in self._functional_agents:
            agent_id = "research" if "research" in self._functional_agents else "l3_research"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Analyze supply paths and inventory sources for: {objective}",
                context_additions={
                    "channel": "dsp",
                    "analyze": ["supply_paths", "ssp_quality", "auction_dynamics"],
                    "spo_enabled": True,
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 2. Execution with deal prioritization
        if "execution" in self._functional_agents or "l3_execution" in self._functional_agents:
            agent_id = "execution" if "execution" in self._functional_agents else "l3_execution"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Configure DSP execution for: {objective}",
                context_additions={
                    "deal_priority": ["pg", "pmp", "open"],
                    "bid_shading": True,
                    "win_rate_target": 0.15,
                    "deduplication": True,
                },
                priority=ContextPriority.HIGH,
            ))
        
        # 3. Reporting for bid landscape analysis
        if "reporting" in self._functional_agents or "l3_reporting" in self._functional_agents:
            agent_id = "reporting" if "reporting" in self._functional_agents else "l3_reporting"
            delegations.append(DelegationRequest(
                functional_agent_id=agent_id,
                task=f"Set up DSP performance reporting for: {objective}",
                context_additions={
                    "metrics": ["win_rate", "avg_bid", "ssp_breakdown", "deal_performance"],
                    "bid_landscape": True,
                },
                priority=ContextPriority.LOW,
            ))
        
        return delegations


def create_dsp_specialist(**kwargs) -> DSPSpecialist:
    """Factory function to create a DSPSpecialist.
    
    Args:
        **kwargs: Arguments passed to DSPSpecialist constructor
        
    Returns:
        Configured DSPSpecialist instance
    """
    return DSPSpecialist(**kwargs)
