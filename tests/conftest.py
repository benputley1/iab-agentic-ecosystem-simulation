"""Shared pytest fixtures for all tests."""

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


@pytest.fixture
def sample_bid_request():
    """Sample bid request for testing."""
    return {
        "id": "req-001",
        "buyer_id": "buyer-001",
        "campaign_id": "test-camp-001",
        "channel": "display",
        "impressions_requested": 10000,
        "max_cpm": 18.0,
        "targeting": {"audience": ["tech_early_adopters"]},
        "scenario": "A",
    }
