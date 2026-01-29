"""Tests for second-price auction and rent-seeking exchange."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    CONSUMER_GROUPS,
)
from src.agents.exchange.auction import (
    AuctionBid,
    AuctionResult,
    SecondPriceAuction,
    RentSeekingExchange,
)
from src.agents.exchange.fees import FeeConfig


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def bid_request():
    """Sample bid request."""
    return BidRequest(
        request_id="req-001",
        buyer_id="buyer-001",
        campaign_id="camp-001",
        channel="display",
        impressions_requested=10000,
        max_cpm=20.0,
        targeting={"geo": ["US"]},
        floor_price=5.0,
    )


@pytest.fixture
def high_bid_response(bid_request):
    """High bid response."""
    return BidResponse(
        response_id="resp-high",
        request_id=bid_request.request_id,
        seller_id="seller-premium",
        offered_cpm=15.0,
        available_impressions=10000,
        deal_type=DealType.OPEN_AUCTION,
    )


@pytest.fixture
def medium_bid_response(bid_request):
    """Medium bid response."""
    return BidResponse(
        response_id="resp-medium",
        request_id=bid_request.request_id,
        seller_id="seller-standard",
        offered_cpm=12.0,
        available_impressions=8000,
        deal_type=DealType.OPEN_AUCTION,
    )


@pytest.fixture
def low_bid_response(bid_request):
    """Low bid response."""
    return BidResponse(
        response_id="resp-low",
        request_id=bid_request.request_id,
        seller_id="seller-budget",
        offered_cpm=8.0,
        available_impressions=15000,
        deal_type=DealType.OPEN_AUCTION,
    )


@pytest.fixture
def mock_redis_bus():
    """Mock Redis bus for testing."""
    bus = MagicMock()
    bus.publish_bid_request = AsyncMock(return_value="msg-001")
    bus.publish_bid_response = AsyncMock(return_value="msg-002")
    bus.publish_deal = AsyncMock(return_value="msg-003")
    bus.route_to_buyer = AsyncMock(return_value="msg-004")
    bus.ack_bid_requests = AsyncMock(return_value=1)
    bus.ack_bid_responses = AsyncMock(return_value=1)
    return bus


# -------------------------------------------------------------------------
# SecondPriceAuction Tests
# -------------------------------------------------------------------------


class TestSecondPriceAuction:
    """Tests for second-price auction mechanics."""

    def test_empty_auction(self):
        """Auction with no bids returns no winner."""
        auction = SecondPriceAuction(auction_id="test-001")
        result = auction.run()

        assert result.winner is None
        assert result.winning_price == 0.0
        assert result.bid_count == 0

    def test_single_bid_auction(self, bid_request, high_bid_response):
        """Single bid wins at 80% of own bid (no second price)."""
        auction = SecondPriceAuction(auction_id="test-001", floor_price=5.0)

        bid = AuctionBid(
            response=high_bid_response,
            request=bid_request,
            effective_cpm=15.0,
            message_id="msg-001",
        )
        auction.add_bid(bid)

        result = auction.run()

        assert result.winner is not None
        assert result.winner.response.seller_id == "seller-premium"
        assert result.original_price == 15.0
        # Single bid pays max(floor, 80% of bid)
        assert result.winning_price == 12.0  # 15 * 0.8
        assert not result.had_competition

    def test_single_bid_floor_price(self, bid_request, low_bid_response):
        """Single bid pays floor if higher than 80% of bid."""
        auction = SecondPriceAuction(auction_id="test-001", floor_price=7.0)

        bid = AuctionBid(
            response=low_bid_response,
            request=bid_request,
            effective_cpm=8.0,
            message_id="msg-001",
        )
        auction.add_bid(bid)

        result = auction.run()

        assert result.winning_price == 7.0  # Floor > 8*0.8=6.4

    def test_two_bid_auction(
        self, bid_request, high_bid_response, medium_bid_response
    ):
        """Winner pays second-highest bid."""
        auction = SecondPriceAuction(auction_id="test-001")

        auction.add_bid(AuctionBid(
            response=high_bid_response,
            request=bid_request,
            effective_cpm=15.0,
            message_id="msg-001",
        ))
        auction.add_bid(AuctionBid(
            response=medium_bid_response,
            request=bid_request,
            effective_cpm=12.0,
            message_id="msg-002",
        ))

        result = auction.run()

        assert result.winner.response.seller_id == "seller-premium"
        assert result.original_price == 15.0
        assert result.winning_price == 12.0  # Second-highest bid
        assert result.had_competition

    def test_three_bid_auction(
        self, bid_request, high_bid_response, medium_bid_response, low_bid_response
    ):
        """Winner pays second-highest of three bids."""
        auction = SecondPriceAuction(auction_id="test-001")

        auction.add_bid(AuctionBid(
            response=low_bid_response,
            request=bid_request,
            effective_cpm=8.0,
            message_id="msg-003",
        ))
        auction.add_bid(AuctionBid(
            response=high_bid_response,
            request=bid_request,
            effective_cpm=15.0,
            message_id="msg-001",
        ))
        auction.add_bid(AuctionBid(
            response=medium_bid_response,
            request=bid_request,
            effective_cpm=12.0,
            message_id="msg-002",
        ))

        result = auction.run()

        assert result.winner.response.seller_id == "seller-premium"
        assert result.winning_price == 12.0
        assert result.bid_count == 3
        # All bids sorted by price
        assert result.all_bids[0].effective_cpm == 15.0
        assert result.all_bids[1].effective_cpm == 12.0
        assert result.all_bids[2].effective_cpm == 8.0

    def test_tied_bids(self, bid_request):
        """Tied bids - first received wins (stable sort)."""
        auction = SecondPriceAuction(auction_id="test-001")

        resp1 = BidResponse(
            response_id="resp-1",
            request_id=bid_request.request_id,
            seller_id="seller-A",
            offered_cpm=10.0,
            available_impressions=5000,
            deal_type=DealType.OPEN_AUCTION,
        )
        resp2 = BidResponse(
            response_id="resp-2",
            request_id=bid_request.request_id,
            seller_id="seller-B",
            offered_cpm=10.0,
            available_impressions=5000,
            deal_type=DealType.OPEN_AUCTION,
        )

        auction.add_bid(AuctionBid(
            response=resp1,
            request=bid_request,
            effective_cpm=10.0,
            message_id="msg-001",
        ))
        auction.add_bid(AuctionBid(
            response=resp2,
            request=bid_request,
            effective_cpm=10.0,
            message_id="msg-002",
        ))

        result = auction.run()

        # Winner pays same as second (tied)
        assert result.winning_price == 10.0


# -------------------------------------------------------------------------
# RentSeekingExchange Tests
# -------------------------------------------------------------------------


class TestRentSeekingExchange:
    """Tests for rent-seeking exchange agent."""

    @pytest.mark.asyncio
    async def test_handle_bid_request(self, mock_redis_bus, bid_request):
        """Exchange forwards bid request and tracks it."""
        exchange = RentSeekingExchange(
            bus=mock_redis_bus,
            exchange_id="exchange-test",
        )

        await exchange.handle_bid_request(bid_request, "msg-original")

        # Verify request published
        mock_redis_bus.publish_bid_request.assert_called_once_with(bid_request)

        # Verify acknowledgment
        mock_redis_bus.ack_bid_requests.assert_called_once()

        # Verify tracking
        assert bid_request.request_id in exchange._pending_requests

    @pytest.mark.asyncio
    async def test_handle_bid_response(
        self, mock_redis_bus, bid_request, high_bid_response
    ):
        """Exchange collects bid responses."""
        exchange = RentSeekingExchange(bus=mock_redis_bus)

        # First register the request
        await exchange.handle_bid_request(bid_request, "msg-req")

        # Then handle response
        await exchange.handle_bid_response(high_bid_response, "msg-resp")

        # Verify collected
        responses = exchange._collected_responses[bid_request.request_id]
        assert len(responses) == 1
        assert responses[0].response.seller_id == "seller-premium"

    @pytest.mark.asyncio
    async def test_orphan_response_ignored(self, mock_redis_bus, high_bid_response):
        """Response without matching request is ignored."""
        exchange = RentSeekingExchange(bus=mock_redis_bus)

        # Handle response without prior request
        await exchange.handle_bid_response(high_bid_response, "msg-orphan")

        # Should not be collected
        assert len(exchange._collected_responses) == 0

    @pytest.mark.asyncio
    async def test_run_auction_with_fee(
        self, mock_redis_bus, bid_request, high_bid_response, medium_bid_response
    ):
        """Exchange runs auction and applies fee."""
        fee_config = FeeConfig(base_fee_pct=0.15)
        exchange = RentSeekingExchange(
            bus=mock_redis_bus,
            fee_config=fee_config,
        )

        # Setup auction
        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(high_bid_response, "msg-high")
        await exchange.handle_bid_response(medium_bid_response, "msg-medium")

        # Run auction
        deal = await exchange.run_auction(bid_request.request_id)

        assert deal is not None
        assert deal.scenario == "A"
        assert deal.seller_id == "seller-premium"
        assert deal.cpm == 12.0  # Second-price
        assert deal.exchange_fee > 0
        assert deal.fee_percentage == pytest.approx(15.0, rel=0.01)

        # Verify deal published
        mock_redis_bus.publish_deal.assert_called_once()

        # Verify buyer gets marked-up price
        mock_redis_bus.route_to_buyer.assert_called_once()
        routed_response = mock_redis_bus.route_to_buyer.call_args[0][0]
        assert routed_response.offered_cpm == pytest.approx(13.8, rel=0.01)  # 12 * 1.15

    @pytest.mark.asyncio
    async def test_auction_cleanup(
        self, mock_redis_bus, bid_request, high_bid_response
    ):
        """Auction cleans up tracking after completion."""
        exchange = RentSeekingExchange(bus=mock_redis_bus)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(high_bid_response, "msg-resp")

        # Before auction
        assert bid_request.request_id in exchange._pending_requests

        await exchange.run_auction(bid_request.request_id)

        # After auction - cleaned up
        assert bid_request.request_id not in exchange._pending_requests
        assert bid_request.request_id not in exchange._collected_responses

    @pytest.mark.asyncio
    async def test_process_pending_auctions(
        self, mock_redis_bus, bid_request, high_bid_response
    ):
        """Process all pending auctions meeting minimum bids."""
        exchange = RentSeekingExchange(bus=mock_redis_bus)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(high_bid_response, "msg-resp")

        deals = await exchange.process_pending_auctions(min_bids=1)

        assert len(deals) == 1
        assert deals[0].request_id == bid_request.request_id

    @pytest.mark.asyncio
    async def test_configurable_fee_range(self, mock_redis_bus, bid_request):
        """Exchange respects configured fee range."""
        # Test minimum (10%)
        config_min = FeeConfig(base_fee_pct=0.10)
        exchange_min = RentSeekingExchange(bus=mock_redis_bus, fee_config=config_min)
        assert exchange_min.fee_config.base_fee_pct == 0.10

        # Test maximum (20%)
        config_max = FeeConfig(base_fee_pct=0.20)
        exchange_max = RentSeekingExchange(bus=mock_redis_bus, fee_config=config_max)
        assert exchange_max.fee_config.base_fee_pct == 0.20

    def test_get_stats(self, mock_redis_bus):
        """Exchange reports statistics."""
        config = FeeConfig(base_fee_pct=0.15, min_fee_pct=0.10, max_fee_pct=0.20)
        exchange = RentSeekingExchange(
            bus=mock_redis_bus,
            fee_config=config,
            exchange_id="stats-test",
        )

        stats = exchange.get_stats()

        assert stats["exchange_id"] == "stats-test"
        assert stats["fee_config"]["base_pct"] == 15.0
        assert stats["fee_config"]["min_pct"] == 10.0
        assert stats["fee_config"]["max_pct"] == 20.0


class TestAuctionResult:
    """Tests for AuctionResult dataclass."""

    def test_bid_count(self, bid_request, high_bid_response, medium_bid_response):
        """bid_count returns correct count."""
        bids = [
            AuctionBid(
                response=high_bid_response,
                request=bid_request,
                effective_cpm=15.0,
                message_id="msg-1",
            ),
            AuctionBid(
                response=medium_bid_response,
                request=bid_request,
                effective_cpm=12.0,
                message_id="msg-2",
            ),
        ]

        result = AuctionResult(
            winner=bids[0],
            winning_price=12.0,
            original_price=15.0,
            all_bids=bids,
            auction_id="test",
        )

        assert result.bid_count == 2
        assert result.had_competition is True

    def test_no_competition(self, bid_request, high_bid_response):
        """Single bid means no competition."""
        bids = [
            AuctionBid(
                response=high_bid_response,
                request=bid_request,
                effective_cpm=15.0,
                message_id="msg-1",
            ),
        ]

        result = AuctionResult(
            winner=bids[0],
            winning_price=12.0,
            original_price=15.0,
            all_bids=bids,
            auction_id="test",
        )

        assert result.had_competition is False
