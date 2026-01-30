"""Buyer agent implementations for RTB simulation.

This module provides buyer agents that wrap IAB buyer-agent CrewAI flows
to work within the RTB simulation environment.

Key Components:
    - BuyerAgentWrapper: Main wrapper adapting IAB flows to simulation
    - Campaign: Campaign configuration and state
    - BidStrategy: Bidding strategy types

Usage:
    ```python
    from src.agents.buyer import create_buyer_agent, Campaign, BidStrategy

    async with await create_buyer_agent(
        buyer_id="buyer-001",
        scenario="B",
        bid_strategy=BidStrategy.TARGET_CPM,
    ) as buyer:
        buyer.add_campaign(Campaign(
            campaign_id="camp-001",
            name="Q1 Branding",
            budget=10000.0,
            target_impressions=1000000,
            target_cpm=15.0,
            channel="display",
        ))

        deals = await buyer.run_bidding_cycle()
        print(f"Made {len(deals)} deals")
    ```

Scenarios:
    - A: Traditional exchange with 15% fee
    - B: A2A direct (context rot simulation)
    - C: Ledger-backed (Beads/Sui/Walrus)
"""

from .config import buyer_settings, BuyerAgentSettings
from .strategies import (
    BidStrategy,
    BidDecision,
    BidCalculator,
    get_strategy,
)
from .wrapper import (
    BuyerAgentWrapper,
    Campaign,
    BuyerState,
    create_buyer_agent,
)
from .tools import (
    SimulationClient,
    SimDiscoverInventoryTool,
    SimRequestDealTool,
    SimCheckAvailsTool,
)
# L1 Portfolio Manager
from .l1_portfolio_manager import PortfolioManager, create_portfolio_manager
from .models import (
    Campaign as L1Campaign,
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
# L2 Channel Specialists
from .l2_branding import BrandingSpecialist, create_branding_specialist
from .l2_mobile_app import MobileAppSpecialist, create_mobile_app_specialist
from .l2_ctv import CTVSpecialist, create_ctv_specialist
from .l2_performance import PerformanceSpecialist, create_performance_specialist
from .l2_dsp import DSPSpecialist, create_dsp_specialist
# L3 Functional Agents
from .l3_base import (
    FunctionalAgent,
    ToolResult,
    ToolExecutionStatus,
    AgentContext,
)
from .l3_tools import (
    BUYER_TOOLS,
    SearchCriteria,
    Product,
    AvailsResult,
    OrderSpec,
    Order,
    Deal,
    BookingConfirmation,
    Metrics,
    Attribution,
    CampaignBrief,
    AudienceSegment,
    CoverageEstimate,
    get_tool_schema,
    get_tools_for_agent,
)
from .l3_research import ResearchAgent
from .l3_execution import ExecutionAgent
from .l3_reporting import ReportingAgent
from .l3_audience_planner import AudiencePlannerAgent

__all__ = [
    # Configuration
    "buyer_settings",
    "BuyerAgentSettings",
    # Strategies
    "BidStrategy",
    "BidDecision",
    "BidCalculator",
    "get_strategy",
    # Wrapper
    "BuyerAgentWrapper",
    "Campaign",
    "BuyerState",
    "create_buyer_agent",
    # Tools
    "SimulationClient",
    "SimDiscoverInventoryTool",
    "SimRequestDealTool",
    "SimCheckAvailsTool",
    # L1 Portfolio Manager
    "PortfolioManager",
    "create_portfolio_manager",
    "L1Campaign",
    "CampaignObjectives",
    "CampaignStatus",
    "AudienceSpec",
    "BudgetAllocation",
    "ChannelSelection",
    "PortfolioState",
    "SpecialistTask",
    "SpecialistResult",
    "Channel",
    # L2 Channel Specialists
    "BrandingSpecialist",
    "create_branding_specialist",
    "MobileAppSpecialist",
    "create_mobile_app_specialist",
    "CTVSpecialist",
    "create_ctv_specialist",
    "PerformanceSpecialist",
    "create_performance_specialist",
    "DSPSpecialist",
    "create_dsp_specialist",
    # L3 Base
    "FunctionalAgent",
    "ToolResult",
    "ToolExecutionStatus",
    "AgentContext",
    # L3 Data Types
    "BUYER_TOOLS",
    "SearchCriteria",
    "Product",
    "AvailsResult",
    "OrderSpec",
    "Order",
    "Deal",
    "BookingConfirmation",
    "Metrics",
    "Attribution",
    "CampaignBrief",
    "AudienceSegment",
    "CoverageEstimate",
    "get_tool_schema",
    "get_tools_for_agent",
    # L3 Functional Agents
    "ResearchAgent",
    "ExecutionAgent",
    "ReportingAgent",
    "AudiencePlannerAgent",
]
