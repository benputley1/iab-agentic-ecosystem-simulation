"""
Scenario A: Current State with Rent-Seeking Exchange.

Models today's programmatic ecosystem with exchange intermediaries
that extract 10-20% fees (default 15%) from every transaction.

IMPORTANT (v2 update): Buyer and seller agents in this scenario ALSO
experience context rot, just like in Scenario B. The key difference is:
- Exchange provides partial state verification via transaction logs
- Exchange catches ~60% of agent errors through reconciliation
- But fees are charged for this intermediary service (15%)

This models reality where:
- DSPs use AI/ML for bidding decisions (subject to context limits)
- SSPs use AI/ML for floor price optimization (subject to context limits)
- Exchange provides SOME verification but not agent memory
- Discrepancies still occur, but exchange can arbitrate

Flow:
1. Buyer agents submit bid requests (may have corrupted context)
2. Exchange receives, validates, and forwards to sellers
3. Sellers respond with offers (may have corrupted context)
4. Exchange runs second-price auction + validates data
5. Exchange catches some hallucinated data via its records
6. Exchange extracts fee and creates deal
7. Marked-up price passed to buyer
"""

import asyncio
from datetime import datetime
from typing import Optional
import structlog

from .base import BaseScenario, ScenarioConfig, ScenarioMetrics
from .context_rot import (
    ContextRotSimulator,
    ContextRotConfig,
    AgentMemory,
    RecoverySource,
    SCENARIO_A_ROT_CONFIG,
)
from agents.buyer.wrapper import BuyerAgentWrapper, Campaign, BidStrategy
from agents.seller.adapter import SellerAgentAdapter
from agents.exchange.auction import RentSeekingExchange
from agents.exchange.fees import FeeConfig
from agents.ucp.hallucination import HallucinationManager
from infrastructure.redis_bus import RedisBus, create_redis_bus
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    STREAMS,
    CONSUMER_GROUPS,
)


logger = structlog.get_logger()


class ScenarioA(BaseScenario):
    """
    Scenario A: Rent-seeking exchange with fee extraction.

    This scenario simulates the current programmatic advertising ecosystem
    where exchanges act as intermediaries and extract significant fees
    (typically 10-20%) from every transaction.

    UPDATED (v2): Now includes context rot simulation for buyer/seller agents.
    The exchange provides partial recovery through transaction log verification,
    catching approximately 60% of agent errors.

    Key characteristics:
    - Exchange mediates all buyer-seller communication
    - Second-price auction determines winner
    - Exchange extracts configurable fee (default 15%)
    - Buyer sees marked-up price
    - Seller receives price minus fee
    - **Agents experience context rot (new in v2)**
    - **Exchange catches ~60% of errors via verification (new in v2)**
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        redis_bus: Optional[RedisBus] = None,
        fee_pct: Optional[float] = None,
        context_rot_config: Optional[ContextRotConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Scenario A.

        Args:
            config: Scenario configuration
            redis_bus: Optional pre-configured Redis bus
            fee_pct: Override fee percentage (default: config.exchange_fee_pct)
            context_rot_config: Configuration for context rot simulation
            seed: Random seed for reproducibility
        """
        config = config or ScenarioConfig()
        super().__init__(
            scenario_id="A",
            scenario_name="Current State (Rent-Seeking Exchange)",
            config=config,
            redis_bus=redis_bus,
        )

        # Fee configuration
        self.fee_pct = fee_pct or config.exchange_fee_pct
        self.fee_config = FeeConfig(base_fee_pct=self.fee_pct)

        # Context rot simulation (v2 addition)
        # Exchange provides ~60% recovery via transaction log verification
        self._context_rot = ContextRotSimulator(
            context_rot_config or SCENARIO_A_ROT_CONFIG,
            seed=seed,
        )
        
        # Agent memories (subject to context rot)
        self._buyer_memories: dict[str, AgentMemory] = {}
        self._seller_memories: dict[str, AgentMemory] = {}
        
        # Hallucination tracking (agents may have corrupted data)
        # But exchange catches some via verification
        self._hallucination_mgr = HallucinationManager(
            scenario="A",
            injection_rate=config.hallucination_rate * 0.4,  # Lower rate due to exchange validation
        )

        # Agents (initialized in setup)
        self._buyers: list[BuyerAgentWrapper] = []
        self._sellers: list[SellerAgentAdapter] = []
        self._exchange: Optional[RentSeekingExchange] = None

        # Background tasks
        self._seller_tasks: list[asyncio.Task] = []
        self._exchange_task: Optional[asyncio.Task] = None

    async def setup(self) -> None:
        """Set up Scenario A: buyers, sellers, and exchange."""
        logger.info(
            "scenario_a.setup",
            buyers=self.config.num_buyers,
            sellers=self.config.num_sellers,
            fee_pct=self.fee_pct * 100,
        )

        # Connect to Redis
        bus = await self.connect_bus()

        # Ensure consumer groups exist
        await bus.ensure_consumer_group(
            STREAMS["bid_requests"],
            CONSUMER_GROUPS["sellers"],
        )
        await bus.ensure_consumer_group(
            STREAMS["bid_requests"],
            CONSUMER_GROUPS["exchange"],
        )
        await bus.ensure_consumer_group(
            STREAMS["bid_responses"],
            CONSUMER_GROUPS["exchange"],
        )

        # Initialize exchange agent
        self._exchange = RentSeekingExchange(
            bus=bus,
            fee_config=self.fee_config,
            exchange_id="exchange-a-001",
        )

        # Initialize seller agents
        for i in range(self.config.num_sellers):
            seller_id = f"seller-{i+1:03d}"
            seller = SellerAgentAdapter(
                seller_id=seller_id,
                scenario="A",
                mock_llm=self.config.mock_llm,
            )
            await seller.connect()
            self._sellers.append(seller)

        # Initialize buyer agents
        for i in range(self.config.num_buyers):
            buyer_id = f"buyer-{i+1:03d}"
            buyer = BuyerAgentWrapper(
                buyer_id=buyer_id,
                scenario="A",
                bid_strategy=BidStrategy.TARGET_CPM,
                redis_bus=bus,
                mock_llm=self.config.mock_llm,
            )
            await buyer.connect()
            self._buyers.append(buyer)

            # Add campaigns for this buyer
            self._create_campaigns_for_buyer(buyer, i)

        # Update metrics
        self.metrics.campaigns_started = sum(
            len(b.state.campaigns) for b in self._buyers
        )

        logger.info(
            "scenario_a.setup_complete",
            buyers=len(self._buyers),
            sellers=len(self._sellers),
            total_campaigns=self.metrics.campaigns_started,
        )

    def _create_campaigns_for_buyer(
        self,
        buyer: BuyerAgentWrapper,
        buyer_index: int,
    ) -> None:
        """Create campaigns for a buyer."""
        import random

        random.seed(buyer_index)  # Reproducible campaigns

        for j in range(self.config.campaigns_per_buyer):
            budget = random.choice([10000, 25000, 50000, 100000])
            campaign = Campaign(
                campaign_id=f"camp-{buyer.buyer_id}-{j+1:03d}",
                name=f"Q1 Campaign {j+1}",
                budget=budget,
                target_impressions=int(budget / 0.015 * 1000),  # ~$15 CPM target
                target_cpm=random.uniform(10, 25),
                channel=random.choice(["display", "video", "ctv"]),
                targeting={
                    "segments": random.sample(
                        ["sports", "tech", "luxury", "auto", "travel"],
                        k=random.randint(1, 3),
                    )
                },
            )
            buyer.add_campaign(campaign)

    def get_or_create_buyer_memory(self, buyer_id: str) -> AgentMemory:
        """Get or create memory for a buyer agent."""
        if buyer_id not in self._buyer_memories:
            self._buyer_memories[buyer_id] = AgentMemory(
                agent_id=buyer_id,
                agent_type="buyer",
            )
        return self._buyer_memories[buyer_id]

    def get_or_create_seller_memory(self, seller_id: str) -> AgentMemory:
        """Get or create memory for a seller agent."""
        if seller_id not in self._seller_memories:
            self._seller_memories[seller_id] = AgentMemory(
                agent_id=seller_id,
                agent_type="seller",
            )
        return self._seller_memories[seller_id]

    async def apply_daily_context_rot(self) -> dict[str, int]:
        """
        Apply context rot to all agent memories at end of day.
        
        In Scenario A, exchange provides ~60% recovery via transaction logs,
        so net context loss is lower than Scenario B.

        Returns:
            Dict mapping agent_id to net_keys_lost count
        """
        results = {}

        # Apply to buyers
        for buyer_id, memory in self._buyer_memories.items():
            # Check for restart first
            restart_event = self._context_rot.check_restart(memory, self.current_day)
            if restart_event:
                net_loss = restart_event.keys_lost - restart_event.keys_recovered
                results[buyer_id] = net_loss
                await self.record_context_rot(
                    agent_id=buyer_id,
                    keys_lost=restart_event.keys_lost,
                    is_decay=False,
                    agent_type="buyer",
                    recovery_attempted=restart_event.recovery_attempted,
                    recovery_successful=restart_event.recovery_successful,
                    recovery_accuracy=restart_event.recovery_accuracy,
                    recovery_source=restart_event.recovery_source.value,
                )
            else:
                # Apply gradual decay
                decay_event = self._context_rot.apply_daily_decay(
                    memory, self.current_day
                )
                if decay_event.keys_lost > 0:
                    net_loss = decay_event.keys_lost - decay_event.keys_recovered
                    results[buyer_id] = net_loss
                    await self.record_context_rot(
                        agent_id=buyer_id,
                        keys_lost=decay_event.keys_lost,
                        is_decay=True,
                        agent_type="buyer",
                        keys_lost_names=decay_event.keys_lost_names,
                        recovery_attempted=decay_event.recovery_attempted,
                        recovery_successful=decay_event.recovery_successful,
                        recovery_accuracy=decay_event.recovery_accuracy,
                        recovery_source=decay_event.recovery_source.value,
                    )

        # Apply to sellers
        for seller_id, memory in self._seller_memories.items():
            restart_event = self._context_rot.check_restart(memory, self.current_day)
            if restart_event:
                net_loss = restart_event.keys_lost - restart_event.keys_recovered
                results[seller_id] = net_loss
                await self.record_context_rot(
                    agent_id=seller_id,
                    keys_lost=restart_event.keys_lost,
                    is_decay=False,
                    agent_type="seller",
                    recovery_attempted=restart_event.recovery_attempted,
                    recovery_successful=restart_event.recovery_successful,
                    recovery_accuracy=restart_event.recovery_accuracy,
                    recovery_source=restart_event.recovery_source.value,
                )
            else:
                decay_event = self._context_rot.apply_daily_decay(
                    memory, self.current_day
                )
                if decay_event.keys_lost > 0:
                    net_loss = decay_event.keys_lost - decay_event.keys_recovered
                    results[seller_id] = net_loss
                    await self.record_context_rot(
                        agent_id=seller_id,
                        keys_lost=decay_event.keys_lost,
                        is_decay=True,
                        agent_type="seller",
                        keys_lost_names=decay_event.keys_lost_names,
                        recovery_attempted=decay_event.recovery_attempted,
                        recovery_successful=decay_event.recovery_successful,
                        recovery_accuracy=decay_event.recovery_accuracy,
                        recovery_source=decay_event.recovery_source.value,
                    )

        return results

    async def run_day(self, day: int) -> list[DealConfirmation]:
        """
        Run one simulation day for Scenario A.

        Orchestrates:
        1. Buyers submit bid requests (may have context rot)
        2. Sellers respond via exchange (may have context rot)
        3. Exchange validates and runs auctions
        4. Exchange catches some errors via transaction logs (~60%)
        5. Deals are created with fee extraction
        6. End-of-day context rot applied to all agents
        """
        self.current_day = day
        deals = []

        logger.info(
            "scenario_a.day_start",
            day=day,
            active_campaigns=sum(
                len(b.get_active_campaigns()) for b in self._buyers
            ),
        )

        # Start seller response handlers
        seller_tasks = []
        for seller in self._sellers:
            task = asyncio.create_task(
                seller.run(max_iterations=self.config.num_buyers * 2)
            )
            seller_tasks.append(task)

        # Process each buyer's campaigns
        for buyer in self._buyers:
            active_campaigns = buyer.get_active_campaigns()

            for campaign in active_campaigns:
                # Submit bid request
                request = self._create_bid_request(buyer, campaign)

                # Register request with exchange for response correlation
                # This is needed because the exchange needs to match responses
                # back to their original requests
                self._exchange._pending_requests[request.request_id] = request

                await self._bus.publish_bid_request(request)

                logger.debug(
                    "scenario_a.request_submitted",
                    buyer_id=buyer.buyer_id,
                    campaign_id=campaign.campaign_id,
                    request_id=request.request_id,
                    max_cpm=request.max_cpm,
                )

        # Wait for sellers to process requests
        await asyncio.sleep(
            self.config.bid_collection_timeout_ms / 1000
        )

        # Collect responses and run auctions
        response_messages = await self._bus.read_bid_responses(
            group=CONSUMER_GROUPS["exchange"],
            count=100,
            block_ms=self.config.bid_collection_timeout_ms,
        )

        for msg_id, response in response_messages:
            await self._exchange.handle_bid_response(response, msg_id)

        # Run all pending auctions
        auction_deals = await self._exchange.process_pending_auctions(
            min_bids=1,
            max_wait_ms=self.config.bid_collection_timeout_ms,
        )

        deals.extend(auction_deals)

        # Update buyer campaign state with completed deals
        for deal in deals:
            self._update_buyer_state(deal)

        # Cancel seller tasks
        for task in seller_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Apply end-of-day context rot to agents
        rot_results = await self.apply_daily_context_rot()
        context_rot_events = len([v for v in rot_results.values() if v > 0])
        net_keys_lost = sum(rot_results.values())

        # Log day summary
        day_spend = sum(d.total_cost for d in deals)
        day_fees = sum(d.exchange_fee for d in deals)

        # Update scenario metrics with context rot info
        self.metrics.context_rot_events += context_rot_events
        self.metrics.keys_lost_total += net_keys_lost

        logger.info(
            "scenario_a.day_complete",
            day=day,
            deals=len(deals),
            spend=round(day_spend, 2),
            fees=round(day_fees, 2),
            fee_rate=round(day_fees / day_spend * 100, 2) if day_spend > 0 else 0,
            context_rot_events=context_rot_events,
            net_keys_lost=net_keys_lost,
            recovery_source="exchange",
        )

        return deals

    def _create_bid_request(
        self,
        buyer: BuyerAgentWrapper,
        campaign: Campaign,
    ) -> BidRequest:
        """Create a bid request for a campaign."""
        return BidRequest(
            buyer_id=buyer.buyer_id,
            campaign_id=campaign.campaign_id,
            channel=campaign.channel,
            impressions_requested=min(
                campaign.remaining_impressions,
                1000000,  # Max 1M per request
            ),
            max_cpm=campaign.target_cpm,
            targeting=campaign.targeting,
        )

    def _update_buyer_state(self, deal: DealConfirmation) -> None:
        """Update buyer campaign state with completed deal."""
        for buyer in self._buyers:
            if buyer.buyer_id == deal.buyer_id:
                campaign = buyer.state.campaigns.get(deal.request_id)
                if campaign:
                    campaign.impressions_delivered += deal.impressions
                    campaign.spend += deal.total_cost
                break

    async def teardown(self) -> None:
        """Clean up Scenario A resources."""
        logger.info("scenario_a.teardown")

        # Disconnect buyers
        for buyer in self._buyers:
            await buyer.disconnect()
        self._buyers.clear()

        # Disconnect sellers
        for seller in self._sellers:
            await seller.disconnect()
        self._sellers.clear()

        # Clear agent memories
        self._buyer_memories.clear()
        self._seller_memories.clear()

        # Disconnect bus
        await self.disconnect_bus()

        # Calculate final metrics
        self._calculate_final_metrics()

    def get_memory_summary(self) -> dict:
        """Get summary of all agent memories and context rot."""
        return {
            "buyers": {
                agent_id: memory.to_dict()
                for agent_id, memory in self._buyer_memories.items()
            },
            "sellers": {
                agent_id: memory.to_dict()
                for agent_id, memory in self._seller_memories.items()
            },
            "total_rot_events": sum(
                m.rot_events for m in self._buyer_memories.values()
            ) + sum(m.rot_events for m in self._seller_memories.values()),
            "total_keys_lost": sum(
                m.keys_lost_total for m in self._buyer_memories.values()
            ) + sum(m.keys_lost_total for m in self._seller_memories.values()),
            "total_keys_recovered": sum(
                m.keys_recovered_total for m in self._buyer_memories.values()
            ) + sum(m.keys_recovered_total for m in self._seller_memories.values()),
            "recovery_source": "exchange",
            "recovery_accuracy": 0.60,  # Exchange catches ~60% of errors
        }

    def get_context_rot_summary(self) -> dict:
        """Get context rot simulation summary."""
        return self._context_rot.get_summary()

    def _calculate_final_metrics(self) -> None:
        """Calculate final scenario metrics."""
        # Count completed campaigns
        completed = 0
        total_goal_pct = 0.0

        for buyer in self._buyers:
            for campaign in buyer.state.campaigns.values():
                goal_pct = min(
                    100.0,
                    (campaign.impressions_delivered / campaign.target_impressions) * 100
                    if campaign.target_impressions > 0
                    else 0,
                )
                total_goal_pct += goal_pct
                if campaign.impressions_delivered >= campaign.target_impressions:
                    completed += 1

        self.metrics.campaigns_completed = completed
        if self.metrics.campaigns_started > 0:
            self.metrics.goal_achievement_rate = (
                total_goal_pct / self.metrics.campaigns_started
            )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


async def run_scenario_a(
    days: int = 30,
    buyers: int = 5,
    sellers: int = 5,
    campaigns_per_buyer: int = 10,
    fee_pct: float = 0.15,
    mock_llm: bool = True,
) -> ScenarioMetrics:
    """
    Run Scenario A simulation.

    Args:
        days: Number of simulation days
        buyers: Number of buyer agents
        sellers: Number of seller agents
        campaigns_per_buyer: Campaigns per buyer
        fee_pct: Exchange fee percentage (default: 15%)
        mock_llm: Use mock LLM (no API costs)

    Returns:
        Collected metrics
    """
    config = ScenarioConfig(
        num_buyers=buyers,
        num_sellers=sellers,
        campaigns_per_buyer=campaigns_per_buyer,
        simulation_days=days,
        mock_llm=mock_llm,
        exchange_fee_pct=fee_pct,
    )

    scenario = ScenarioA(config=config, fee_pct=fee_pct)
    return await scenario.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Scenario A: Rent-Seeking Exchange"
    )
    parser.add_argument("--days", type=int, default=30, help="Simulation days")
    parser.add_argument("--buyers", type=int, default=5, help="Number of buyers")
    parser.add_argument("--sellers", type=int, default=5, help="Number of sellers")
    parser.add_argument(
        "--campaigns-per-buyer",
        type=int,
        default=10,
        help="Campaigns per buyer",
    )
    parser.add_argument(
        "--fee-pct",
        type=float,
        default=0.15,
        help="Exchange fee percentage",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run mini test (1 day, 1 buyer, 1 seller)",
    )

    args = parser.parse_args()

    if args.test:
        # Quick test mode
        metrics = asyncio.run(
            run_scenario_a(
                days=1,
                buyers=1,
                sellers=1,
                campaigns_per_buyer=2,
                mock_llm=True,
            )
        )
    else:
        metrics = asyncio.run(
            run_scenario_a(
                days=args.days,
                buyers=args.buyers,
                sellers=args.sellers,
                campaigns_per_buyer=args.campaigns_per_buyer,
                fee_pct=args.fee_pct,
                mock_llm=args.mock_llm,
            )
        )

    print("\n" + "=" * 60)
    print("SCENARIO A RESULTS: Rent-Seeking Exchange")
    print("=" * 60)
    for key, value in metrics.to_dict().items():
        print(f"{key:30s}: {value}")
