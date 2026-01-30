"""Buyer Agent Wrapper for RTB Simulation.

This module wraps the IAB buyer-agent CrewAI flows to work within
the RTB simulation environment, replacing OpenDirect communication
with Redis Streams-based A2A messaging.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum

from crewai import Agent, Crew, Process, Task, LLM

from .config import buyer_settings
from .tools.sim_client import SimulationClient
from .tools.sim_tools import (
    SimDiscoverInventoryTool,
    SimRequestDealTool,
    SimCheckAvailsTool,
)
from infrastructure.redis_bus import RedisBus
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    CONSUMER_GROUPS,
)


class BidStrategy(str, Enum):
    """Bidding strategies for the buyer agent."""

    TARGET_CPM = "target_cpm"  # Bid at target CPM
    MAXIMIZE_REACH = "maximize_reach"  # Bid aggressively to win
    FLOOR_PLUS = "floor_plus"  # Bid just above floor


@dataclass
class Campaign:
    """Campaign configuration for the buyer agent."""

    campaign_id: str
    name: str
    budget: float
    target_impressions: int
    target_cpm: float
    channel: str = "display"
    targeting: dict = field(default_factory=dict)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Runtime state
    impressions_delivered: int = 0
    spend: float = 0.0

    @property
    def remaining_budget(self) -> float:
        """Budget remaining after spend."""
        return self.budget - self.spend

    @property
    def remaining_impressions(self) -> int:
        """Impressions remaining to reach target."""
        return max(0, self.target_impressions - self.impressions_delivered)

    @property
    def is_active(self) -> bool:
        """Whether campaign still has budget and impressions to deliver."""
        return self.remaining_budget > 0 and self.remaining_impressions > 0


@dataclass
class BuyerState:
    """State tracking for a buyer agent."""

    buyer_id: str
    campaigns: dict[str, Campaign] = field(default_factory=dict)
    deals_made: list[DealConfirmation] = field(default_factory=list)
    total_spend: float = 0.0
    total_impressions: int = 0

    # Context rot tracking (Scenario B)
    context_memory: dict[str, Any] = field(default_factory=dict)
    context_rot_events: int = 0


class BuyerAgentWrapper:
    """Wrapper for IAB buyer-agent CrewAI flows.

    This class adapts the IAB buyer-agent to work within the RTB simulation:
    - Replaces OpenDirect client with Redis Streams messaging
    - Provides simulation-specific tools
    - Supports all three scenarios (A, B, C)
    - Tracks buyer state and metrics
    """

    def __init__(
        self,
        buyer_id: str,
        scenario: str = "A",
        bid_strategy: BidStrategy = BidStrategy.TARGET_CPM,
        redis_bus: Optional[RedisBus] = None,
        mock_llm: bool = True,
    ):
        """Initialize buyer agent wrapper.

        Args:
            buyer_id: Unique identifier for this buyer
            scenario: Simulation scenario ("A", "B", or "C")
            bid_strategy: Bidding strategy to use
            redis_bus: Optional pre-configured Redis bus
            mock_llm: If True, use mock LLM (no API calls)
        """
        self.buyer_id = buyer_id
        self.scenario = scenario
        self.bid_strategy = bid_strategy
        self._bus = redis_bus
        self._owned_bus = False
        self.mock_llm = mock_llm

        # State tracking
        self.state = BuyerState(buyer_id=buyer_id)

        # Simulation client (initialized on connect)
        self._client: Optional[SimulationClient] = None

        # CrewAI components (initialized on connect)
        self._tools: list = []
        self._agents: dict[str, Agent] = {}
        self._crews: dict[str, Crew] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> "BuyerAgentWrapper":
        """Connect to Redis and initialize CrewAI components.

        Returns:
            Self for context manager chaining
        """
        # Connect to Redis
        if self._bus is None:
            from infrastructure.redis_bus import create_redis_bus

            self._bus = await create_redis_bus(
                consumer_id=f"buyer-{self.buyer_id}"
            )
            self._owned_bus = True

        # Create simulation client
        self._client = SimulationClient(
            buyer_id=self.buyer_id,
            redis_bus=self._bus,
            scenario=self.scenario,
        )
        await self._client.connect()

        # Initialize CrewAI tools and agents
        self._init_tools()
        self._init_agents()
        self._init_crews()

        return self

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        if self._client:
            await self._client.disconnect()
            self._client = None

        if self._bus and self._owned_bus:
            await self._bus.disconnect()
            self._bus = None

    async def __aenter__(self) -> "BuyerAgentWrapper":
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Campaign Management
    # -------------------------------------------------------------------------

    def add_campaign(self, campaign: Campaign) -> None:
        """Add a campaign to this buyer.

        Args:
            campaign: Campaign to add
        """
        self.state.campaigns[campaign.campaign_id] = campaign

    def get_active_campaigns(self) -> list[Campaign]:
        """Get all active campaigns.

        Returns:
            List of campaigns with remaining budget/impressions
        """
        return [c for c in self.state.campaigns.values() if c.is_active]

    # -------------------------------------------------------------------------
    # Bidding (Main Loop)
    # -------------------------------------------------------------------------

    async def run_bidding_cycle(
        self,
        max_iterations: int = 10,
    ) -> list[DealConfirmation]:
        """Run a full bidding cycle for active campaigns.

        Args:
            max_iterations: Maximum bid attempts per campaign

        Returns:
            List of deals made in this cycle
        """
        if not self._client:
            raise RuntimeError("Not connected")

        deals = []
        active_campaigns = self.get_active_campaigns()

        for campaign in active_campaigns:
            campaign_deals = await self._bid_for_campaign(
                campaign,
                max_iterations=max_iterations,
            )
            deals.extend(campaign_deals)

        return deals

    async def _bid_for_campaign(
        self,
        campaign: Campaign,
        max_iterations: int = 10,
    ) -> list[DealConfirmation]:
        """Run bidding for a single campaign.

        Args:
            campaign: Campaign to bid for
            max_iterations: Maximum bid attempts

        Returns:
            List of deals made for this campaign
        """
        deals = []

        for _ in range(max_iterations):
            if not campaign.is_active:
                break

            # Discover inventory
            inventory = await self._discover_inventory(campaign)
            if not inventory:
                break

            # Select best seller
            best = self._select_best_offer(campaign, inventory)
            if not best:
                break

            # Calculate bid
            bid_cpm = self._calculate_bid(campaign, best)

            # Request deal
            result = await self._client.request_deal(
                seller_id=best["seller_id"],
                campaign_id=campaign.campaign_id,
                impressions=min(
                    campaign.remaining_impressions,
                    best.get("available_impressions", 100000),
                ),
                max_cpm=bid_cpm,
                channel=campaign.channel,
                targeting=campaign.targeting,
            )

            if result.success and result.data:
                deal = result.data
                deals.append(deal)
                self._record_deal(campaign, deal)

                # Apply context rot in Scenario B
                if self.scenario == "B":
                    self._apply_context_rot()

        return deals

    async def _discover_inventory(
        self,
        campaign: Campaign,
    ) -> list[dict]:
        """Discover available inventory for a campaign.

        Args:
            campaign: Campaign to find inventory for

        Returns:
            List of inventory offers
        """
        if not self._client:
            return []

        result = await self._client.search_products(
            filters={
                "channel": campaign.channel,
                "maxPrice": campaign.target_cpm * buyer_settings.max_cpm_multiplier,
                "minImpressions": min(10000, campaign.remaining_impressions),
            }
        )

        if not result.success or not result.data:
            return []

        return [
            {
                "seller_id": item.seller_id,
                "product_id": item.product_id,
                "cpm": item.base_cpm,
                "available_impressions": item.available_impressions,
                "floor_price": item.floor_price,
                "deal_type": item.deal_type,
            }
            for item in result.data
        ]

    def _select_best_offer(
        self,
        campaign: Campaign,
        inventory: list[dict],
    ) -> Optional[dict]:
        """Select the best inventory offer for a campaign.

        Args:
            campaign: Campaign being fulfilled
            inventory: Available inventory offers

        Returns:
            Best offer or None if none acceptable
        """
        # Filter by max CPM
        max_cpm = campaign.target_cpm * buyer_settings.max_cpm_multiplier
        valid = [i for i in inventory if i["cpm"] <= max_cpm]

        if not valid:
            return None

        # Sort by strategy
        if self.bid_strategy == BidStrategy.TARGET_CPM:
            # Prefer CPMs closest to target
            valid.sort(key=lambda x: abs(x["cpm"] - campaign.target_cpm))
        elif self.bid_strategy == BidStrategy.MAXIMIZE_REACH:
            # Prefer highest impression volume
            valid.sort(key=lambda x: -x["available_impressions"])
        else:  # FLOOR_PLUS
            # Prefer lowest CPM
            valid.sort(key=lambda x: x["cpm"])

        return valid[0] if valid else None

    def _calculate_bid(
        self,
        campaign: Campaign,
        offer: dict,
    ) -> float:
        """Calculate bid CPM based on strategy.

        Args:
            campaign: Campaign being bid for
            offer: Inventory offer

        Returns:
            Bid CPM
        """
        if self.bid_strategy == BidStrategy.TARGET_CPM:
            return min(campaign.target_cpm, offer["cpm"])

        elif self.bid_strategy == BidStrategy.MAXIMIZE_REACH:
            # Bid at offer price (aggressive)
            return offer["cpm"]

        else:  # FLOOR_PLUS
            # Bid just above floor
            floor = offer.get("floor_price", offer["cpm"] * 0.8)
            return floor * 1.05

    def _record_deal(self, campaign: Campaign, deal: DealConfirmation) -> None:
        """Record a deal in state.

        Args:
            campaign: Campaign the deal is for
            deal: Deal confirmation
        """
        # Update campaign
        campaign.impressions_delivered += deal.impressions
        campaign.spend += deal.total_cost

        # Update buyer state
        self.state.deals_made.append(deal)
        self.state.total_spend += deal.total_cost
        self.state.total_impressions += deal.impressions

    def _apply_context_rot(self) -> None:
        """Apply context rot in Scenario B.

        Simulates memory degradation where the agent "forgets"
        some context over time.
        """
        if random.random() < buyer_settings.context_decay_rate:
            self.state.context_rot_events += 1
            # Remove random context keys
            if self.state.context_memory:
                keys = list(self.state.context_memory.keys())
                if keys:
                    del self.state.context_memory[random.choice(keys)]

    # -------------------------------------------------------------------------
    # CrewAI Integration (for hierarchical agent orchestration)
    # -------------------------------------------------------------------------

    def _init_tools(self) -> None:
        """Initialize CrewAI tools."""
        if not self._client:
            return

        self._tools = [
            SimDiscoverInventoryTool(client=self._client),
            SimRequestDealTool(client=self._client),
            SimCheckAvailsTool(client=self._client),
        ]

    def _init_agents(self) -> None:
        """Initialize CrewAI agents.

        Creates agents modeled after the IAB buyer-agent hierarchy:
        - Portfolio Manager (Level 1)
        - Channel Specialists (Level 2)
        - Functional Agents (Level 3)
        """
        if self.mock_llm:
            # Use a lightweight model config for testing
            llm = LLM(
                model="claude-sonnet-4-20250514",
                temperature=0.1,
            )
        else:
            llm = LLM(
                model=buyer_settings.default_llm_model,
                temperature=buyer_settings.llm_temperature,
            )

        # Level 1: Portfolio Manager
        self._agents["portfolio_manager"] = Agent(
            role="Portfolio Manager",
            goal="Optimize budget allocation and coordinate channel specialists",
            backstory=f"""You manage advertising campaigns for buyer {self.buyer_id}.
Your job is to allocate budget across channels and coordinate specialists
to achieve campaign KPIs. You work in RTB simulation scenario {self.scenario}.""",
            llm=llm,
            tools=[],
            allow_delegation=True,
            verbose=buyer_settings.crew_verbose,
            memory=buyer_settings.crew_memory_enabled,
        )

        # Level 3: Research Agent
        self._agents["research"] = Agent(
            role="Inventory Research Analyst",
            goal="Discover and evaluate optimal advertising inventory",
            backstory="""You search for advertising inventory that matches
campaign requirements. You evaluate pricing, quality, and availability
to provide recommendations to channel specialists.""",
            llm=llm,
            tools=self._tools[:2],  # discover_inventory, request_deal
            allow_delegation=False,
            verbose=buyer_settings.crew_verbose,
            memory=buyer_settings.crew_memory_enabled,
        )

        # Level 3: Execution Agent
        self._agents["execution"] = Agent(
            role="Campaign Execution Specialist",
            goal="Execute advertising orders with precision",
            backstory="""You handle deal execution after inventory is selected.
You ensure all parameters are correct before booking and manage the
complete booking lifecycle.""",
            llm=llm,
            tools=self._tools[1:],  # request_deal, check_avails
            allow_delegation=False,
            verbose=buyer_settings.crew_verbose,
            memory=buyer_settings.crew_memory_enabled,
        )

    def _init_crews(self) -> None:
        """Initialize CrewAI crews."""
        if not self._agents:
            return

        # Research crew for inventory discovery
        self._crews["research"] = Crew(
            agents=[self._agents["research"]],
            tasks=[],  # Tasks added dynamically
            process=Process.sequential,
            verbose=buyer_settings.crew_verbose,
            memory=buyer_settings.crew_memory_enabled,
        )

        # Execution crew for deal booking
        self._crews["execution"] = Crew(
            agents=[self._agents["execution"]],
            tasks=[],
            process=Process.sequential,
            verbose=buyer_settings.crew_verbose,
            memory=buyer_settings.crew_memory_enabled,
        )

    async def run_crew_bidding(
        self,
        campaign: Campaign,
    ) -> str:
        """Run CrewAI hierarchical bidding for a campaign.

        This uses the full CrewAI agent hierarchy for complex
        decision-making, useful for comparing agent approaches.

        Args:
            campaign: Campaign to bid for

        Returns:
            Crew execution result
        """
        if not self._crews or not self._agents:
            return "CrewAI not initialized"

        # Create research task
        research_task = Task(
            description=f"""
Research available inventory for campaign:
- Campaign ID: {campaign.campaign_id}
- Name: {campaign.name}
- Channel: {campaign.channel}
- Budget Remaining: ${campaign.remaining_budget:.2f}
- Impressions Needed: {campaign.remaining_impressions:,}
- Target CPM: ${campaign.target_cpm:.2f}
- Scenario: {self.scenario}

Find inventory matching these requirements and recommend the best options.
""",
            expected_output="List of recommended inventory with pricing and rationale",
            agent=self._agents["research"],
        )

        # Run research crew
        self._crews["research"].tasks = [research_task]
        result = self._crews["research"].kickoff()

        return str(result)


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


async def create_buyer_agent(
    buyer_id: str,
    scenario: str = "A",
    bid_strategy: BidStrategy = BidStrategy.TARGET_CPM,
    redis_bus: Optional[RedisBus] = None,
    mock_llm: bool = True,
) -> BuyerAgentWrapper:
    """Create and connect a buyer agent.

    Args:
        buyer_id: Unique identifier for this buyer
        scenario: Simulation scenario ("A", "B", or "C")
        bid_strategy: Bidding strategy to use
        redis_bus: Optional pre-configured Redis bus
        mock_llm: If True, use mock LLM (no API calls)

    Returns:
        Connected BuyerAgentWrapper
    """
    agent = BuyerAgentWrapper(
        buyer_id=buyer_id,
        scenario=scenario,
        bid_strategy=bid_strategy,
        redis_bus=redis_bus,
        mock_llm=mock_llm,
    )
    await agent.connect()
    return agent
