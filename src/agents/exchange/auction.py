"""
Second-price auction implementation for Scenario A rent-seeking exchange.

Implements the auction mechanics where:
1. Multiple bids are collected for inventory
2. Winner is determined by highest bid
3. Winner pays second-highest price (+ exchange fee)
4. Exchange extracts configurable 10-20% fee
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import defaultdict
import structlog

from ...infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    CONSUMER_GROUPS,
)
from ...infrastructure.redis_bus import RedisBus
from .fees import FeeConfig, calculate_markup_cpm

logger = structlog.get_logger()


@dataclass
class AuctionBid:
    """A bid in the auction."""
    response: BidResponse
    request: BidRequest
    effective_cpm: float  # Price buyer is willing to pay
    message_id: str  # Redis message ID for acknowledgment


@dataclass
class AuctionResult:
    """Result of running a second-price auction."""
    winner: Optional[AuctionBid]
    winning_price: float  # Second-price (what winner pays)
    original_price: float  # Winner's original bid
    all_bids: list[AuctionBid]
    auction_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def bid_count(self) -> int:
        return len(self.all_bids)

    @property
    def had_competition(self) -> bool:
        return self.bid_count > 1


class SecondPriceAuction:
    """
    Second-price (Vickrey) auction implementation.

    Winner pays the price of the second-highest bid, not their own bid.
    This incentivizes truthful bidding.
    """

    def __init__(self, auction_id: str, floor_price: float = 0.0):
        """
        Initialize auction.

        Args:
            auction_id: Unique identifier for this auction
            floor_price: Minimum price (if no second bid, winner pays floor)
        """
        self.auction_id = auction_id
        self.floor_price = floor_price
        self.bids: list[AuctionBid] = []

    def add_bid(self, bid: AuctionBid) -> None:
        """Add a bid to the auction."""
        self.bids.append(bid)
        logger.debug(
            "auction.bid_added",
            auction_id=self.auction_id,
            seller_id=bid.response.seller_id,
            cpm=bid.effective_cpm,
            total_bids=len(self.bids),
        )

    def run(self) -> AuctionResult:
        """
        Execute the auction and determine winner.

        Returns:
            AuctionResult with winner and pricing
        """
        if not self.bids:
            logger.info("auction.no_bids", auction_id=self.auction_id)
            return AuctionResult(
                winner=None,
                winning_price=0.0,
                original_price=0.0,
                all_bids=[],
                auction_id=self.auction_id,
            )

        # Sort by effective CPM (highest first)
        sorted_bids = sorted(
            self.bids,
            key=lambda b: b.effective_cpm,
            reverse=True,
        )

        winner = sorted_bids[0]
        original_price = winner.effective_cpm

        # Second-price: winner pays second-highest bid or floor
        if len(sorted_bids) > 1:
            winning_price = sorted_bids[1].effective_cpm
        else:
            winning_price = max(self.floor_price, original_price * 0.8)  # 80% if solo

        logger.info(
            "auction.completed",
            auction_id=self.auction_id,
            winner_seller=winner.response.seller_id,
            original_bid=original_price,
            winning_price=winning_price,
            bid_count=len(self.bids),
            competition=len(sorted_bids) > 1,
        )

        return AuctionResult(
            winner=winner,
            winning_price=winning_price,
            original_price=original_price,
            all_bids=sorted_bids,
            auction_id=self.auction_id,
        )


class RentSeekingExchange:
    """
    Scenario A: Rent-seeking exchange agent.

    Sits between buyers and sellers, runs auctions, and extracts fees.
    Implements second-price auction with configurable 10-20% fee extraction.
    """

    def __init__(
        self,
        bus: RedisBus,
        fee_config: Optional[FeeConfig] = None,
        exchange_id: str = "exchange-001",
    ):
        """
        Initialize exchange agent.

        Args:
            bus: Redis bus for message routing
            fee_config: Fee extraction configuration (default: 15%)
            exchange_id: Unique identifier for this exchange
        """
        self.bus = bus
        self.fee_config = fee_config or FeeConfig()
        self.exchange_id = exchange_id

        # Track pending auctions by request_id
        self._pending_requests: dict[str, BidRequest] = {}
        self._collected_responses: dict[str, list[AuctionBid]] = defaultdict(list)
        self._response_message_ids: dict[str, list[str]] = defaultdict(list)

    async def handle_bid_request(
        self,
        request: BidRequest,
        message_id: str,
    ) -> None:
        """
        Process incoming bid request from buyer.

        Forward to all sellers and track for auction.

        Args:
            request: Buyer's bid request
            message_id: Redis message ID
        """
        self._pending_requests[request.request_id] = request

        logger.info(
            "exchange.request_received",
            exchange_id=self.exchange_id,
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            campaign_id=request.campaign_id,
            max_cpm=request.max_cpm,
        )

        # Publish to main bid_requests stream for sellers
        await self.bus.publish_bid_request(request)

        # Acknowledge the original request
        await self.bus.ack_bid_requests(
            CONSUMER_GROUPS["exchange"],
            message_id,
        )

    async def handle_bid_response(
        self,
        response: BidResponse,
        message_id: str,
    ) -> None:
        """
        Process incoming bid response from seller.

        Collect for auction processing.

        Args:
            response: Seller's bid response
            message_id: Redis message ID
        """
        request = self._pending_requests.get(response.request_id)
        if not request:
            logger.warning(
                "exchange.orphan_response",
                exchange_id=self.exchange_id,
                response_id=response.response_id,
                request_id=response.request_id,
            )
            return

        bid = AuctionBid(
            response=response,
            request=request,
            effective_cpm=response.offered_cpm,
            message_id=message_id,
        )

        self._collected_responses[response.request_id].append(bid)
        self._response_message_ids[response.request_id].append(message_id)

        logger.info(
            "exchange.response_collected",
            exchange_id=self.exchange_id,
            request_id=response.request_id,
            seller_id=response.seller_id,
            offered_cpm=response.offered_cpm,
            collected_count=len(self._collected_responses[response.request_id]),
        )

    async def run_auction(
        self,
        request_id: str,
        floor_price: Optional[float] = None,
    ) -> Optional[DealConfirmation]:
        """
        Execute auction for a request and create deal if winner found.

        Args:
            request_id: Request to run auction for
            floor_price: Minimum acceptable price

        Returns:
            DealConfirmation if auction successful, None otherwise
        """
        request = self._pending_requests.get(request_id)
        bids = self._collected_responses.get(request_id, [])

        if not request or not bids:
            logger.warning(
                "exchange.auction_skipped",
                exchange_id=self.exchange_id,
                request_id=request_id,
                has_request=request is not None,
                bid_count=len(bids),
            )
            return None

        # Create and run auction
        auction = SecondPriceAuction(
            auction_id=f"AUCTION-{request_id[:8]}",
            floor_price=floor_price or request.floor_price or 0.0,
        )

        for bid in bids:
            auction.add_bid(bid)

        result = auction.run()

        if not result.winner:
            return None

        # Calculate effective fee for this transaction
        fee_pct = self.fee_config.get_effective_fee(
            buyer_id=request.buyer_id,
            seller_id=result.winner.response.seller_id,
        )

        # Create deal at second-price
        deal = DealConfirmation.from_deal(
            request=request,
            response=result.winner.response,
            scenario="A",
            exchange_fee_pct=fee_pct,
        )

        # Adjust CPM to second-price (not winner's original bid)
        deal.cpm = result.winning_price
        deal.total_cost = (deal.impressions / 1000) * deal.cpm
        deal.exchange_fee = deal.total_cost * fee_pct

        # Publish deal
        await self.bus.publish_deal(deal)

        # Route modified response back to buyer (with marked-up price)
        marked_up_cpm = calculate_markup_cpm(result.winning_price, fee_pct)
        buyer_response = BidResponse(
            response_id=result.winner.response.response_id,
            request_id=request_id,
            seller_id=result.winner.response.seller_id,
            offered_cpm=marked_up_cpm,  # Buyer sees marked-up price
            available_impressions=result.winner.response.available_impressions,
            deal_type=result.winner.response.deal_type,
            deal_id=deal.deal_id,
        )
        await self.bus.route_to_buyer(buyer_response, request.buyer_id)

        logger.info(
            "exchange.deal_created",
            exchange_id=self.exchange_id,
            deal_id=deal.deal_id,
            buyer_id=deal.buyer_id,
            seller_id=deal.seller_id,
            impressions=deal.impressions,
            second_price_cpm=result.winning_price,
            buyer_pays_cpm=marked_up_cpm,
            exchange_fee=deal.exchange_fee,
            fee_pct=fee_pct * 100,
            seller_revenue=deal.seller_revenue,
        )

        # Acknowledge all collected responses
        if self._response_message_ids[request_id]:
            await self.bus.ack_bid_responses(
                CONSUMER_GROUPS["exchange"],
                *self._response_message_ids[request_id],
            )

        # Clean up tracking
        self._cleanup_request(request_id)

        return deal

    def _cleanup_request(self, request_id: str) -> None:
        """Remove request from tracking after auction."""
        self._pending_requests.pop(request_id, None)
        self._collected_responses.pop(request_id, None)
        self._response_message_ids.pop(request_id, None)

    async def process_pending_auctions(
        self,
        min_bids: int = 1,
        max_wait_ms: int = 5000,
    ) -> list[DealConfirmation]:
        """
        Process all pending auctions that have enough bids.

        Args:
            min_bids: Minimum bids required to run auction
            max_wait_ms: Maximum time to wait for more bids

        Returns:
            List of created deals
        """
        deals = []

        for request_id in list(self._collected_responses.keys()):
            bids = self._collected_responses[request_id]
            if len(bids) >= min_bids:
                deal = await self.run_auction(request_id)
                if deal:
                    deals.append(deal)

        return deals

    def get_stats(self) -> dict:
        """Get exchange statistics."""
        return {
            "exchange_id": self.exchange_id,
            "pending_requests": len(self._pending_requests),
            "collected_responses": sum(
                len(bids) for bids in self._collected_responses.values()
            ),
            "fee_config": {
                "base_pct": self.fee_config.base_fee_pct * 100,
                "min_pct": self.fee_config.min_fee_pct * 100,
                "max_pct": self.fee_config.max_fee_pct * 100,
            },
        }
