"""Shared pytest fixtures and configuration."""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external services (Redis, etc.)"
    )


@pytest.fixture
def sample_bid_request():
    """Sample bid request for testing."""
    from src.infrastructure.message_schemas import BidRequest

    return BidRequest(
        buyer_id="test-buyer-001",
        campaign_id="test-camp-001",
        channel="display",
        impressions_requested=10000,
        max_cpm=15.0,
        targeting={
            "geo": ["US"],
            "segments": ["sports_enthusiasts"],
        },
    )


@pytest.fixture
def sample_bid_response(sample_bid_request):
    """Sample bid response for testing."""
    from src.infrastructure.message_schemas import BidResponse, DealType

    return BidResponse(
        request_id=sample_bid_request.request_id,
        seller_id="test-seller-001",
        offered_cpm=12.0,
        available_impressions=8000,
        deal_type=DealType.OPEN_AUCTION,
    )


@pytest.fixture
def sample_deal(sample_bid_request, sample_bid_response):
    """Sample deal confirmation for testing."""
    from src.infrastructure.message_schemas import DealConfirmation

    return DealConfirmation.from_deal(
        request=sample_bid_request,
        response=sample_bid_response,
        scenario="A",
        exchange_fee_pct=0.15,
    )
