"""
Unit tests for Scenario A: Rent-Seeking Exchange.

These tests verify the scenario infrastructure without requiring Redis.
For full integration tests, see tests/integration/test_scenario_a.py.
"""

import pytest
from datetime import datetime, timedelta

from src.scenarios.base import ScenarioConfig, ScenarioMetrics
from src.scenarios.scenario_a import ScenarioA
from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
)
from src.agents.exchange.auction import SecondPriceAuction, AuctionBid
from src.agents.exchange.fees import FeeConfig


class TestScenarioMetrics:
    """Test scenario metrics collection."""

    def test_metrics_initial_state(self):
        """Metrics start at zero."""
        metrics = ScenarioMetrics(
            scenario_id="A",
            scenario_name="Test Scenario",
        )

        assert metrics.total_deals == 0
        assert metrics.total_impressions == 0
        assert metrics.total_buyer_spend == 0.0
        assert metrics.total_exchange_fees == 0.0
        assert metrics.intermediary_take_rate == 0.0

    def test_metrics_record_deal(self, sample_deal):
        """Metrics correctly record deal data."""
        metrics = ScenarioMetrics(
            scenario_id="A",
            scenario_name="Test Scenario",
        )

        metrics.record_deal(sample_deal)

        assert metrics.total_deals == 1
        assert metrics.total_impressions == sample_deal.impressions
        assert metrics.total_buyer_spend == sample_deal.total_cost
        assert metrics.total_exchange_fees == sample_deal.exchange_fee
        assert metrics.total_seller_revenue == sample_deal.seller_revenue

    def test_metrics_take_rate_calculation(self):
        """Intermediary take rate calculated correctly."""
        metrics = ScenarioMetrics(
            scenario_id="A",
            scenario_name="Test Scenario",
        )

        # Manually set values
        metrics.total_buyer_spend = 100.0
        metrics.total_exchange_fees = 15.0

        assert metrics.intermediary_take_rate == 15.0  # 15%

    def test_metrics_to_dict(self, sample_deal):
        """Metrics convert to dictionary correctly."""
        metrics = ScenarioMetrics(
            scenario_id="A",
            scenario_name="Test Scenario",
        )
        metrics.record_deal(sample_deal)

        result = metrics.to_dict()

        assert result["scenario_id"] == "A"
        assert result["total_deals"] == 1
        assert "intermediary_take_rate" in result


class TestScenarioConfig:
    """Test scenario configuration."""

    def test_default_config(self):
        """Default config has expected values."""
        config = ScenarioConfig()

        assert config.num_buyers == 5
        assert config.num_sellers == 5
        assert config.campaigns_per_buyer == 10
        assert config.simulation_days == 30
        assert config.exchange_fee_pct == 0.15

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = ScenarioConfig(
            num_buyers=3,
            num_sellers=3,
            exchange_fee_pct=0.20,
        )

        assert config.num_buyers == 3
        assert config.exchange_fee_pct == 0.20


class TestSecondPriceAuction:
    """Test second-price auction mechanics."""

    def test_empty_auction(self):
        """Auction with no bids returns no winner."""
        auction = SecondPriceAuction(auction_id="test-001")
        result = auction.run()

        assert result.winner is None
        assert result.bid_count == 0

    def test_single_bid_auction(self, sample_bid_request, sample_bid_response):
        """Single bid pays floor or 80% of bid."""
        auction = SecondPriceAuction(
            auction_id="test-001",
            floor_price=10.0,
        )

        bid = AuctionBid(
            response=sample_bid_response,
            request=sample_bid_request,
            effective_cpm=sample_bid_response.offered_cpm,
            message_id="msg-001",
        )
        auction.add_bid(bid)

        result = auction.run()

        assert result.winner is not None
        assert result.winner.response.seller_id == sample_bid_response.seller_id
        # Single bid: pays max(floor, 80% of bid)
        expected_price = max(10.0, sample_bid_response.offered_cpm * 0.8)
        assert result.winning_price == expected_price

    def test_multiple_bids_auction(self, sample_bid_request):
        """Multiple bids: winner pays second-highest price."""
        auction = SecondPriceAuction(
            auction_id="test-001",
            floor_price=5.0,
        )

        # Create three bids
        bids_data = [
            ("seller-001", 20.0),
            ("seller-002", 15.0),
            ("seller-003", 12.0),
        ]

        for seller_id, cpm in bids_data:
            response = BidResponse(
                request_id=sample_bid_request.request_id,
                seller_id=seller_id,
                offered_cpm=cpm,
                available_impressions=10000,
            )
            bid = AuctionBid(
                response=response,
                request=sample_bid_request,
                effective_cpm=cpm,
                message_id=f"msg-{seller_id}",
            )
            auction.add_bid(bid)

        result = auction.run()

        # Winner is highest bidder
        assert result.winner.response.seller_id == "seller-001"
        assert result.original_price == 20.0
        # Pays second-highest price
        assert result.winning_price == 15.0
        assert result.had_competition


class TestFeeExtraction:
    """Test exchange fee extraction."""

    def test_fee_config_default(self):
        """Default fee config is 15%."""
        config = FeeConfig()

        assert config.base_fee_pct == 0.15
        assert config.min_fee_pct == 0.10
        assert config.max_fee_pct == 0.20

    def test_fee_extraction_basic(self, sample_deal):
        """Fee extracted correctly from deal."""
        # sample_deal uses 15% fee
        expected_fee = sample_deal.total_cost * 0.15

        assert sample_deal.exchange_fee == pytest.approx(expected_fee, rel=0.01)
        assert sample_deal.fee_percentage == pytest.approx(15.0, rel=0.01)

    def test_seller_revenue_calculation(self, sample_deal):
        """Seller revenue is total cost minus fee."""
        expected_revenue = sample_deal.total_cost - sample_deal.exchange_fee

        assert sample_deal.seller_revenue == pytest.approx(expected_revenue)

    def test_fee_within_range(self):
        """Effective fee stays within configured range."""
        config = FeeConfig(
            base_fee_pct=0.15,
            min_fee_pct=0.10,
            max_fee_pct=0.20,
        )

        # Even with no preferential treatment, fee should be base
        fee = config.get_effective_fee()
        assert fee == 0.15

        # Fee should never go below min or above max
        assert 0.10 <= fee <= 0.20


class TestScenarioAInit:
    """Test Scenario A initialization."""

    def test_scenario_init_default(self):
        """Scenario initializes with defaults."""
        scenario = ScenarioA()

        assert scenario.scenario_id == "A"
        assert scenario.scenario_name == "Current State (Rent-Seeking Exchange)"
        assert scenario.fee_pct == 0.15

    def test_scenario_init_custom_fee(self):
        """Scenario accepts custom fee percentage."""
        scenario = ScenarioA(fee_pct=0.20)

        assert scenario.fee_pct == 0.20
        assert scenario.fee_config.base_fee_pct == 0.20

    def test_scenario_config_passed(self):
        """Scenario accepts config object."""
        config = ScenarioConfig(
            num_buyers=3,
            num_sellers=3,
            exchange_fee_pct=0.18,
        )
        scenario = ScenarioA(config=config)

        assert scenario.config.num_buyers == 3
        assert scenario.config.num_sellers == 3
