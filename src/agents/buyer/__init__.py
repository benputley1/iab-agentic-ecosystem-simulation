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
]
