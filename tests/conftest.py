"""Shared pytest fixtures and configuration."""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_campaign():
    """Sample campaign for testing."""
    return {
        "id": "test-camp-001",
        "buyer_id": "buyer-001",
        "name": "Test Campaign",
        "total_budget": 10000,
        "daily_budget": 333.33,
        "primary_kpi": "impressions",
        "target_impressions": 1000000,
        "target_cpm": 15.0,
        "channels": ["display"],
        "audience_segments": ["tech_early_adopters"],
        "scenario": "A",
    }


@pytest.fixture
def sample_publisher():
    """Sample publisher for testing."""
    return {
        "id": "pub-001",
        "name": "Test Publisher",
        "channels": ["display", "video"],
        "floor_cpm": 12.0,
        "daily_avails": 5000000,
        "audience_segments": ["tech_early_adopters", "sports_enthusiasts"],
    }


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external services (Redis, etc.)"
    )


def _redis_available():
    """Check if Redis is available."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6379))
        sock.close()
        return result == 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if Redis is not available."""
    if _redis_available():
        return
    skip_integration = pytest.mark.skip(reason="Redis not available")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


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
