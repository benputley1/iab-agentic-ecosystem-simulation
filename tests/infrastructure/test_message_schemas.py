"""Tests for message schemas."""

import pytest
from datetime import datetime

from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    MessageType,
    STREAMS,
    CONSUMER_GROUPS,
)


class TestBidRequest:
    """Tests for BidRequest schema."""

    def test_create_with_defaults(self):
        """Test creating a bid request with minimal fields."""
        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=15.0,
        )

        assert request.buyer_id == "buyer-001"
        assert request.campaign_id == "camp-001"
        assert request.channel == "display"
        assert request.impressions_requested == 10000
        assert request.max_cpm == 15.0
        assert request.message_type == MessageType.BID_REQUEST
        assert request.request_id is not None
        assert request.timestamp is not None
        assert request.targeting == {}

    def test_create_with_targeting(self):
        """Test creating a bid request with targeting data."""
        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="video",
            impressions_requested=50000,
            max_cpm=25.0,
            targeting={
                "geo": ["US", "UK"],
                "segments": ["sports_enthusiasts", "tech_early_adopters"],
            },
        )

        assert request.targeting["geo"] == ["US", "UK"]
        assert len(request.targeting["segments"]) == 2

    def test_to_stream_data(self):
        """Test conversion to Redis Stream format."""
        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=15.0,
        )

        data = request.to_stream_data()

        assert data["buyer_id"] == "buyer-001"
        assert data["impressions_requested"] == "10000"  # String conversion
        assert data["max_cpm"] == "15.0"
        assert data["message_type"] == "bid_request"

    def test_roundtrip_serialization(self):
        """Test serialize and deserialize produces same data."""
        original = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=15.0,
            targeting={"geo": ["US"]},
        )

        stream_data = original.to_stream_data()
        restored = BidRequest.from_stream_data(stream_data)

        assert restored.buyer_id == original.buyer_id
        assert restored.campaign_id == original.campaign_id
        assert restored.impressions_requested == original.impressions_requested
        assert restored.max_cpm == original.max_cpm


class TestBidResponse:
    """Tests for BidResponse schema."""

    def test_create_with_defaults(self):
        """Test creating a bid response with minimal fields."""
        response = BidResponse(
            request_id="req-123",
            seller_id="seller-001",
            offered_cpm=12.0,
            available_impressions=8000,
        )

        assert response.request_id == "req-123"
        assert response.seller_id == "seller-001"
        assert response.offered_cpm == 12.0
        assert response.available_impressions == 8000
        assert response.deal_type == DealType.OPEN_AUCTION
        assert response.message_type == MessageType.BID_RESPONSE

    def test_create_with_deal_type(self):
        """Test creating a response with specific deal type."""
        response = BidResponse(
            request_id="req-123",
            seller_id="seller-001",
            offered_cpm=20.0,
            available_impressions=5000,
            deal_type=DealType.PROGRAMMATIC_GUARANTEED,
            deal_id="DEAL-ABC123",
        )

        assert response.deal_type == DealType.PROGRAMMATIC_GUARANTEED
        assert response.deal_id == "DEAL-ABC123"

    def test_to_stream_data(self):
        """Test conversion to Redis Stream format."""
        response = BidResponse(
            request_id="req-123",
            seller_id="seller-001",
            offered_cpm=12.0,
            available_impressions=8000,
        )

        data = response.to_stream_data()

        assert data["seller_id"] == "seller-001"
        assert data["offered_cpm"] == "12.0"
        assert data["available_impressions"] == "8000"
        assert data["deal_type"] == "OA"

    def test_roundtrip_serialization(self):
        """Test serialize and deserialize."""
        original = BidResponse(
            request_id="req-123",
            seller_id="seller-001",
            offered_cpm=12.0,
            available_impressions=8000,
            deal_type=DealType.PREFERRED_DEAL,
        )

        stream_data = original.to_stream_data()
        restored = BidResponse.from_stream_data(stream_data)

        assert restored.seller_id == original.seller_id
        assert restored.offered_cpm == original.offered_cpm
        assert restored.deal_type == original.deal_type


class TestDealConfirmation:
    """Tests for DealConfirmation schema."""

    def test_create_with_defaults(self):
        """Test creating a deal confirmation."""
        deal = DealConfirmation(
            request_id="req-123",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=15.0,
            total_cost=150.0,
            scenario="A",
        )

        assert deal.buyer_id == "buyer-001"
        assert deal.seller_id == "seller-001"
        assert deal.impressions == 10000
        assert deal.cpm == 15.0
        assert deal.total_cost == 150.0
        assert deal.exchange_fee == 0.0
        assert deal.scenario == "A"
        assert deal.deal_id.startswith("DEAL-")

    def test_create_with_exchange_fee(self):
        """Test creating a deal with exchange fee (Scenario A)."""
        deal = DealConfirmation(
            request_id="req-123",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=15.0,
            total_cost=150.0,
            exchange_fee=22.5,  # 15% fee
            scenario="A",
        )

        assert deal.exchange_fee == 22.5
        assert deal.seller_revenue == 127.5
        assert deal.fee_percentage == 15.0

    def test_seller_revenue_property(self):
        """Test seller revenue calculation."""
        deal = DealConfirmation(
            request_id="req-123",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=20.0,
            total_cost=200.0,
            exchange_fee=30.0,
            scenario="A",
        )

        assert deal.seller_revenue == 170.0

    def test_fee_percentage_property(self):
        """Test fee percentage calculation."""
        deal = DealConfirmation(
            request_id="req-123",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=20.0,
            total_cost=200.0,
            exchange_fee=40.0,
            scenario="A",
        )

        assert deal.fee_percentage == 20.0

    def test_from_deal_factory(self):
        """Test creating deal from request and response."""
        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=20.0,
        )

        response = BidResponse(
            request_id=request.request_id,
            seller_id="seller-001",
            offered_cpm=15.0,
            available_impressions=8000,
        )

        deal = DealConfirmation.from_deal(
            request=request,
            response=response,
            scenario="A",
            exchange_fee_pct=0.15,
        )

        assert deal.buyer_id == "buyer-001"
        assert deal.seller_id == "seller-001"
        assert deal.impressions == 8000  # min(requested, available)
        assert deal.cpm == 15.0
        assert deal.total_cost == 120.0  # (8000/1000) * 15
        assert deal.exchange_fee == 18.0  # 15% of 120
        assert deal.scenario == "A"

    def test_to_stream_data(self):
        """Test conversion to Redis Stream format."""
        deal = DealConfirmation(
            request_id="req-123",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=15.0,
            total_cost=150.0,
            exchange_fee=22.5,
            scenario="A",
        )

        data = deal.to_stream_data()

        assert data["buyer_id"] == "buyer-001"
        assert data["total_cost"] == "150.0"
        assert data["exchange_fee"] == "22.5"
        assert data["scenario"] == "A"

    def test_roundtrip_serialization(self):
        """Test serialize and deserialize."""
        original = DealConfirmation(
            request_id="req-123",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=15.0,
            total_cost=150.0,
            exchange_fee=22.5,
            scenario="A",
            ledger_entry_id="entry-123",
        )

        stream_data = original.to_stream_data()
        restored = DealConfirmation.from_stream_data(stream_data)

        assert restored.buyer_id == original.buyer_id
        assert restored.total_cost == original.total_cost
        assert restored.exchange_fee == original.exchange_fee
        assert restored.ledger_entry_id == original.ledger_entry_id


class TestStreamConstants:
    """Tests for stream and group constants."""

    def test_stream_names(self):
        """Test stream name constants."""
        assert "rtb:requests" in STREAMS.values()
        assert "rtb:responses" in STREAMS.values()
        assert "rtb:deals" in STREAMS.values()
        assert "rtb:events" in STREAMS.values()

    def test_consumer_groups(self):
        """Test consumer group constants."""
        assert "buyers" in CONSUMER_GROUPS
        assert "sellers" in CONSUMER_GROUPS
        assert "exchange" in CONSUMER_GROUPS
        assert "analytics" in CONSUMER_GROUPS
