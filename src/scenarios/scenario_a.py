"""
Scenario A: Current State with Rent-Seeking Exchange.

Models today's programmatic ecosystem with exchange intermediaries
that extract 10-20% fees (default 15%) from every transaction.

Flow:
1. Buyer agents submit bid requests
2. Exchange receives and forwards to sellers
3. Sellers respond with offers
4. Exchange runs second-price auction
5. Exchange extracts fee and creates deal
6. Marked-up price passed to buyer
"""

import asyncio
from datetime import datetime
from typing import Optional
import structlog

from .base import BaseScenario, ScenarioConfig, ScenarioMetrics
from ..agents.buyer.wrapper import BuyerAgentWrapper, Campaign, BidStrategy
from ..agents.seller.adapter import SellerAgentAdapter
from ..agents.exchange.auction import RentSeekingExchange
from ..agents.exchange.fees import FeeConfig
from ..infrastructure.redis_bus import RedisBus, create_redis_bus
from ..infrastructure.message_schemas import (
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

    Key characteristics:
    - Exchange mediates all buyer-seller communication
    - Second-price auction determines winner
    - Exchange extracts configurable fee (default 15%)
    - Buyer sees marked-up price
    - Seller receives price minus fee
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        redis_bus: Optional[RedisBus] = None,
        fee_pct: Optional[float] = None,
    ):
        """
        Initialize Scenario A.

        Args:
            config: Scenario configuration
            redis_bus: Optional pre-configured Redis bus
            fee_pct: Override fee percentage (default: config.exchange_fee_pct)
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

    async def run_day(self, day: int) -> list[DealConfirmation]:
        """
        Run one simulation day for Scenario A.

        Orchestrates:
        1. Buyers submit bid requests
        2. Sellers respond via exchange
        3. Exchange runs auctions
        4. Deals are created with fee extraction
        """
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

        # Log day summary
        day_spend = sum(d.total_cost for d in deals)
        day_fees = sum(d.exchange_fee for d in deals)

        logger.info(
            "scenario_a.day_complete",
            day=day,
            deals=len(deals),
            spend=round(day_spend, 2),
            fees=round(day_fees, 2),
            fee_rate=round(day_fees / day_spend * 100, 2) if day_spend > 0 else 0,
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

        # Disconnect bus
        await self.disconnect_bus()

        # Calculate final metrics
        self._calculate_final_metrics()

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
