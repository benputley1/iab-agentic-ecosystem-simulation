"""Tests for buyer agent wrapper.

Tests the BuyerAgentWrapper, SimulationClient, and bidding strategies.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.buyer import (
    BuyerAgentWrapper,
    Campaign,
    BuyerState,
    BidStrategy,
    BidDecision,
    get_strategy,
    buyer_settings,
)
from src.agents.buyer.tools.sim_client import SimulationClient, InventoryItem, ClientResult
from src.agents.buyer.strategies import (
    TargetCPMStrategy,
    MaximizeReachStrategy,
    FloorPlusStrategy,
    PacingStrategy,
    BudgetAwareStrategy,
)
from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_campaign() -> Campaign:
    """Create a sample campaign for testing."""
    return Campaign(
        campaign_id="test-camp-001",
        name="Test Campaign",
        budget=10000.0,
        target_impressions=1000000,
        target_cpm=15.0,
        channel="display",
        targeting={"geo": "US", "demographic": "25-54"},
    )


@pytest.fixture
def sample_inventory() -> list[InventoryItem]:
    """Create sample inventory items."""
    return [
        InventoryItem(
            product_id="prod-001",
            seller_id="seller-001",
            name="Premium Display",
            channel="display",
            base_cpm=12.0,
            available_impressions=500000,
            targeting=["geo", "demographic"],
            floor_price=10.0,
            deal_type=DealType.OPEN_AUCTION,
        ),
        InventoryItem(
            product_id="prod-002",
            seller_id="seller-002",
            name="Standard Display",
            channel="display",
            base_cpm=8.0,
            available_impressions=1000000,
            targeting=["geo"],
            floor_price=6.0,
            deal_type=DealType.OPEN_AUCTION,
        ),
    ]


@pytest.fixture
def sample_bid_response() -> BidResponse:
    """Create a sample bid response."""
    return BidResponse(
        request_id="req-001",
        seller_id="seller-001",
        offered_cpm=12.0,
        available_impressions=500000,
        deal_type=DealType.OPEN_AUCTION,
    )


# -----------------------------------------------------------------------------
# Campaign Tests
# -----------------------------------------------------------------------------


class TestCampaign:
    """Tests for Campaign dataclass."""

    def test_campaign_creation(self, sample_campaign: Campaign):
        """Test campaign creation with default values."""
        assert sample_campaign.campaign_id == "test-camp-001"
        assert sample_campaign.budget == 10000.0
        assert sample_campaign.impressions_delivered == 0
        assert sample_campaign.spend == 0.0

    def test_remaining_budget(self, sample_campaign: Campaign):
        """Test remaining budget calculation."""
        sample_campaign.spend = 2500.0
        assert sample_campaign.remaining_budget == 7500.0

    def test_remaining_impressions(self, sample_campaign: Campaign):
        """Test remaining impressions calculation."""
        sample_campaign.impressions_delivered = 300000
        assert sample_campaign.remaining_impressions == 700000

    def test_is_active_with_budget_and_impressions(self, sample_campaign: Campaign):
        """Test campaign is active when budget and impressions remain."""
        assert sample_campaign.is_active is True

    def test_is_active_no_budget(self, sample_campaign: Campaign):
        """Test campaign inactive when budget depleted."""
        sample_campaign.spend = 10000.0
        assert sample_campaign.is_active is False

    def test_is_active_no_impressions(self, sample_campaign: Campaign):
        """Test campaign inactive when impressions delivered."""
        sample_campaign.impressions_delivered = 1000000
        assert sample_campaign.is_active is False


# -----------------------------------------------------------------------------
# Bidding Strategy Tests
# -----------------------------------------------------------------------------


class TestBiddingStrategies:
    """Tests for bidding strategies."""

    def test_target_cpm_accepts_below_target(self):
        """Target CPM strategy accepts offers below target."""
        strategy = TargetCPMStrategy()
        decision = strategy.calculate(
            target_cpm=15.0,
            offer_cpm=12.0,
            floor_price=10.0,
            budget_remaining=10000.0,
            impressions_remaining=1000000,
            pacing_factor=1.0,
        )
        assert decision.should_bid is True
        assert decision.bid_cpm == 12.0

    def test_target_cpm_rejects_above_target(self):
        """Target CPM strategy rejects offers above target."""
        strategy = TargetCPMStrategy()
        decision = strategy.calculate(
            target_cpm=15.0,
            offer_cpm=20.0,
            floor_price=10.0,
            budget_remaining=10000.0,
            impressions_remaining=1000000,
            pacing_factor=1.0,
        )
        assert decision.should_bid is False
        assert decision.bid_cpm == 15.0

    def test_maximize_reach_accepts_within_multiplier(self):
        """Maximize reach accepts offers within multiplier."""
        strategy = MaximizeReachStrategy(max_multiplier=1.5)
        decision = strategy.calculate(
            target_cpm=15.0,
            offer_cpm=20.0,  # 15 * 1.5 = 22.5
            floor_price=10.0,
            budget_remaining=10000.0,
            impressions_remaining=1000000,
            pacing_factor=1.0,
        )
        assert decision.should_bid is True
        assert decision.bid_cpm == 20.0

    def test_maximize_reach_rejects_above_multiplier(self):
        """Maximize reach rejects offers above multiplier."""
        strategy = MaximizeReachStrategy(max_multiplier=1.5)
        decision = strategy.calculate(
            target_cpm=15.0,
            offer_cpm=25.0,  # Above 15 * 1.5 = 22.5
            floor_price=10.0,
            budget_remaining=10000.0,
            impressions_remaining=1000000,
            pacing_factor=1.0,
        )
        assert decision.should_bid is False
        assert decision.bid_cpm == 22.5

    def test_floor_plus_bids_above_floor(self):
        """Floor plus strategy bids just above floor."""
        strategy = FloorPlusStrategy(floor_increment=0.05)
        decision = strategy.calculate(
            target_cpm=15.0,
            offer_cpm=10.0,
            floor_price=10.0,
            budget_remaining=10000.0,
            impressions_remaining=1000000,
            pacing_factor=1.0,
        )
        assert decision.should_bid is True
        assert decision.bid_cpm == 10.5  # 10 * 1.05

    def test_pacing_increases_bid_when_behind(self):
        """Pacing strategy increases target when behind pace."""
        strategy = PacingStrategy()
        decision = strategy.calculate(
            target_cpm=15.0,
            offer_cpm=18.0,
            floor_price=10.0,
            budget_remaining=10000.0,
            impressions_remaining=1000000,
            pacing_factor=1.3,  # 30% behind pace
        )
        # Adjusted target = 15 * 1.3 = 19.5
        assert decision.should_bid is True
        assert decision.bid_cpm == 18.0

    def test_budget_aware_caps_at_affordable(self):
        """Budget aware strategy caps bid at affordable CPM."""
        strategy = BudgetAwareStrategy()
        decision = strategy.calculate(
            target_cpm=15.0,
            offer_cpm=12.0,
            floor_price=10.0,
            budget_remaining=1000.0,  # Low budget
            impressions_remaining=100000,  # 1000/100000*1000 = $10 max CPM
            pacing_factor=1.0,
        )
        # Max affordable = 1000/100000*1000 = 10, target = 15, effective = 10
        assert decision.should_bid is False
        assert decision.bid_cpm == 10.0

    def test_get_strategy_returns_correct_type(self):
        """get_strategy returns correct strategy type."""
        strategy = get_strategy(BidStrategy.MAXIMIZE_REACH)
        assert isinstance(strategy, MaximizeReachStrategy)

    def test_get_strategy_default_target_cpm(self):
        """get_strategy defaults to target CPM for unknown."""
        strategy = get_strategy(BidStrategy.TARGET_CPM)
        assert isinstance(strategy, TargetCPMStrategy)


# -----------------------------------------------------------------------------
# BuyerState Tests
# -----------------------------------------------------------------------------


class TestBuyerState:
    """Tests for BuyerState dataclass."""

    def test_buyer_state_creation(self):
        """Test buyer state creation."""
        state = BuyerState(buyer_id="buyer-001")
        assert state.buyer_id == "buyer-001"
        assert state.campaigns == {}
        assert state.deals_made == []
        assert state.total_spend == 0.0
        assert state.total_impressions == 0

    def test_buyer_state_context_rot(self):
        """Test context rot tracking."""
        state = BuyerState(buyer_id="buyer-001")
        state.context_memory["key1"] = "value1"
        state.context_memory["key2"] = "value2"
        state.context_rot_events = 1
        assert len(state.context_memory) == 2
        assert state.context_rot_events == 1


# -----------------------------------------------------------------------------
# SimulationClient Tests (Unit - Mocked)
# -----------------------------------------------------------------------------


class TestSimulationClientUnit:
    """Unit tests for SimulationClient with mocked Redis."""

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Test client creation."""
        client = SimulationClient(
            buyer_id="buyer-001",
            scenario="A",
        )
        assert client.buyer_id == "buyer-001"
        assert client.scenario == "A"

    @pytest.mark.asyncio
    async def test_search_products_not_connected(self):
        """Test search products returns error when not connected."""
        client = SimulationClient(buyer_id="buyer-001")
        result = await client.search_products()
        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_request_deal_not_connected(self):
        """Test request deal returns error when not connected."""
        client = SimulationClient(buyer_id="buyer-001")
        result = await client.request_deal(
            seller_id="seller-001",
            campaign_id="camp-001",
            impressions=100000,
            max_cpm=15.0,
        )
        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_get_product_returns_mock(self):
        """Test get_product returns mock product data."""
        client = SimulationClient(buyer_id="buyer-001")
        # get_product doesn't require connection
        result = await client.get_product("PROD-seller-001-abc123")
        assert result.success is True
        assert result.data["id"] == "PROD-seller-001-abc123"
        assert "seller" in result.data["publisherId"]


# -----------------------------------------------------------------------------
# BuyerAgentWrapper Tests (Unit - Mocked)
# -----------------------------------------------------------------------------


class TestBuyerAgentWrapperUnit:
    """Unit tests for BuyerAgentWrapper with mocked dependencies."""

    def test_wrapper_creation(self):
        """Test wrapper creation."""
        wrapper = BuyerAgentWrapper(
            buyer_id="buyer-001",
            scenario="B",
            bid_strategy=BidStrategy.MAXIMIZE_REACH,
        )
        assert wrapper.buyer_id == "buyer-001"
        assert wrapper.scenario == "B"
        assert wrapper.bid_strategy == BidStrategy.MAXIMIZE_REACH

    def test_add_campaign(self, sample_campaign: Campaign):
        """Test adding a campaign."""
        wrapper = BuyerAgentWrapper(buyer_id="buyer-001")
        wrapper.add_campaign(sample_campaign)
        assert sample_campaign.campaign_id in wrapper.state.campaigns

    def test_get_active_campaigns(self, sample_campaign: Campaign):
        """Test getting active campaigns."""
        wrapper = BuyerAgentWrapper(buyer_id="buyer-001")
        wrapper.add_campaign(sample_campaign)
        active = wrapper.get_active_campaigns()
        assert len(active) == 1
        assert active[0].campaign_id == sample_campaign.campaign_id

    def test_get_active_campaigns_filters_inactive(self, sample_campaign: Campaign):
        """Test that inactive campaigns are filtered."""
        wrapper = BuyerAgentWrapper(buyer_id="buyer-001")
        sample_campaign.spend = sample_campaign.budget  # Deplete budget
        wrapper.add_campaign(sample_campaign)
        active = wrapper.get_active_campaigns()
        assert len(active) == 0

    def test_select_best_offer_target_cpm(self, sample_campaign: Campaign):
        """Test best offer selection with target CPM strategy."""
        wrapper = BuyerAgentWrapper(
            buyer_id="buyer-001",
            bid_strategy=BidStrategy.TARGET_CPM,
        )
        inventory = [
            {"seller_id": "s1", "cpm": 12.0, "available_impressions": 100000},
            {"seller_id": "s2", "cpm": 15.0, "available_impressions": 200000},  # Exactly at target
            {"seller_id": "s3", "cpm": 10.0, "available_impressions": 150000},
        ]
        # Target is 15.0, so cpm=15.0 is closest
        best = wrapper._select_best_offer(sample_campaign, inventory)
        assert best is not None
        assert best["seller_id"] == "s2"

    def test_select_best_offer_maximize_reach(self, sample_campaign: Campaign):
        """Test best offer selection with maximize reach strategy."""
        wrapper = BuyerAgentWrapper(
            buyer_id="buyer-001",
            bid_strategy=BidStrategy.MAXIMIZE_REACH,
        )
        inventory = [
            {"seller_id": "s1", "cpm": 12.0, "available_impressions": 100000},
            {"seller_id": "s2", "cpm": 15.0, "available_impressions": 200000},
            {"seller_id": "s3", "cpm": 10.0, "available_impressions": 150000},
        ]
        best = wrapper._select_best_offer(sample_campaign, inventory)
        assert best is not None
        assert best["seller_id"] == "s2"  # Highest impressions

    def test_select_best_offer_floor_plus(self, sample_campaign: Campaign):
        """Test best offer selection with floor plus strategy."""
        wrapper = BuyerAgentWrapper(
            buyer_id="buyer-001",
            bid_strategy=BidStrategy.FLOOR_PLUS,
        )
        inventory = [
            {"seller_id": "s1", "cpm": 12.0, "available_impressions": 100000},
            {"seller_id": "s2", "cpm": 15.0, "available_impressions": 200000},
            {"seller_id": "s3", "cpm": 10.0, "available_impressions": 150000},
        ]
        best = wrapper._select_best_offer(sample_campaign, inventory)
        assert best is not None
        assert best["seller_id"] == "s3"  # Lowest CPM

    def test_calculate_bid_target_cpm(self, sample_campaign: Campaign):
        """Test bid calculation with target CPM strategy."""
        wrapper = BuyerAgentWrapper(
            buyer_id="buyer-001",
            bid_strategy=BidStrategy.TARGET_CPM,
        )
        offer = {"cpm": 12.0, "floor_price": 10.0}
        bid = wrapper._calculate_bid(sample_campaign, offer)
        # Target CPM returns min(target, offer) = min(15, 12) = 12
        assert bid == 12.0

    def test_calculate_bid_floor_plus(self, sample_campaign: Campaign):
        """Test bid calculation with floor plus strategy."""
        wrapper = BuyerAgentWrapper(
            buyer_id="buyer-001",
            bid_strategy=BidStrategy.FLOOR_PLUS,
        )
        offer = {"cpm": 12.0, "floor_price": 10.0}
        bid = wrapper._calculate_bid(sample_campaign, offer)
        # Floor plus = floor * 1.05 = 10 * 1.05 = 10.5
        assert bid == 10.5

    def test_record_deal_updates_state(self, sample_campaign: Campaign):
        """Test that recording a deal updates state correctly."""
        wrapper = BuyerAgentWrapper(buyer_id="buyer-001")
        wrapper.add_campaign(sample_campaign)

        deal = DealConfirmation(
            request_id="req-001",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=100000,
            cpm=12.0,
            total_cost=1200.0,
            exchange_fee=0.0,
            scenario="B",
        )

        wrapper._record_deal(sample_campaign, deal)

        assert sample_campaign.impressions_delivered == 100000
        assert sample_campaign.spend == 1200.0
        assert len(wrapper.state.deals_made) == 1
        assert wrapper.state.total_spend == 1200.0
        assert wrapper.state.total_impressions == 100000


# -----------------------------------------------------------------------------
# Integration Tests (Require Redis)
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestBuyerAgentIntegration:
    """Integration tests for BuyerAgentWrapper.

    These tests require a running Redis instance.
    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_buyer_connect_disconnect(self):
        """Test buyer agent connection lifecycle."""
        wrapper = BuyerAgentWrapper(
            buyer_id="test-buyer",
            scenario="A",
            mock_llm=True,
        )
        try:
            await wrapper.connect()
            assert wrapper._client is not None
            assert wrapper._bus is not None
        finally:
            await wrapper.disconnect()
            assert wrapper._client is None

    @pytest.mark.asyncio
    async def test_buyer_context_manager(self):
        """Test buyer agent as async context manager."""
        async with BuyerAgentWrapper(
            buyer_id="test-buyer",
            scenario="B",
            mock_llm=True,
        ) as wrapper:
            await wrapper.connect()
            assert wrapper._client is not None
