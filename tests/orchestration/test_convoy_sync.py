"""Tests for convoy_sync module.

These tests use mocked Campaign objects to avoid pulling in the full
agent stack with CrewAI dependencies.
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Import only the convoy_sync module - avoid heavy dependencies
from src.orchestration.convoy_sync import (
    Convoy,
    ConvoyAgent,
    ConvoyRegistry,
    ConvoyState,
    ConvoyStatus,
    ConvoySyncManager,
    get_convoy_manager,
    reset_convoy_manager,
)
from src.infrastructure.message_schemas import DealConfirmation, DealType


@dataclass
class MockCampaign:
    """Mock campaign for testing without CrewAI dependency."""

    campaign_id: str
    name: str
    budget: float
    target_impressions: int
    target_cpm: float
    channel: str = "display"
    targeting: dict = field(default_factory=dict)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Runtime state
    impressions_delivered: int = 0
    spend: float = 0.0

    @property
    def remaining_budget(self) -> float:
        return self.budget - self.spend

    @property
    def remaining_impressions(self) -> int:
        return max(0, self.target_impressions - self.impressions_delivered)

    @property
    def is_active(self) -> bool:
        return self.remaining_budget > 0 and self.remaining_impressions > 0


@pytest.fixture
def campaign():
    """Create a test campaign."""
    return MockCampaign(
        campaign_id="test-camp-001",
        name="Test Campaign",
        budget=10000.0,
        target_impressions=1000000,
        target_cpm=15.0,
        channel="display",
        targeting={"geo": ["US"]},
    )


@pytest.fixture
def deal_confirmation():
    """Create a test deal confirmation."""
    return DealConfirmation(
        deal_id="deal-001",
        request_id="req-001",
        buyer_id="buyer-001",
        seller_id="seller-001",
        impressions=10000,
        cpm=12.0,
        total_cost=120.0,
        seller_revenue=102.0,
        exchange_fee=18.0,
        deal_type=DealType.OPEN_AUCTION,
        scenario="A",
    )


@pytest.fixture
def registry():
    """Create a fresh convoy registry."""
    return ConvoyRegistry()


@pytest.fixture
def sync_manager():
    """Create a fresh convoy sync manager."""
    reset_convoy_manager()
    return ConvoySyncManager()


class TestConvoyAgent:
    """Tests for ConvoyAgent."""

    def test_create_agent(self):
        """Test creating a convoy agent."""
        agent = ConvoyAgent(
            agent_id="buyer-001",
            agent_type="buyer",
            scenario="A",
        )

        assert agent.agent_id == "buyer-001"
        assert agent.agent_type == "buyer"
        assert agent.scenario == "A"
        assert agent.deals_participated == 0
        assert agent.last_activity is None

    def test_record_activity(self):
        """Test recording agent activity."""
        agent = ConvoyAgent(
            agent_id="buyer-001",
            agent_type="buyer",
            scenario="A",
        )

        assert agent.last_activity is None

        agent.record_activity()

        assert agent.last_activity is not None
        assert isinstance(agent.last_activity, datetime)


class TestConvoy:
    """Tests for Convoy."""

    def test_create_convoy(self, campaign):
        """Test creating a convoy."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        assert convoy.convoy_id == "convoy-001"
        assert convoy.campaign.campaign_id == "test-camp-001"
        assert convoy.scenario == "A"
        assert convoy.status == ConvoyStatus.PENDING
        assert len(convoy.agents) == 0
        assert len(convoy.deals) == 0

    def test_add_agent(self, campaign):
        """Test adding agents to a convoy."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        agent = convoy.add_agent("buyer-001", "buyer")

        assert agent.agent_id == "buyer-001"
        assert "buyer-001" in convoy.agents
        assert len(convoy.agents) == 1

    def test_remove_agent(self, campaign):
        """Test removing agents from a convoy."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        convoy.add_agent("buyer-001", "buyer")
        assert "buyer-001" in convoy.agents

        removed = convoy.remove_agent("buyer-001")

        assert removed is not None
        assert removed.agent_id == "buyer-001"
        assert "buyer-001" not in convoy.agents

    def test_record_deal(self, campaign, deal_confirmation):
        """Test recording a deal in a convoy."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        convoy.add_agent("buyer-001", "buyer")
        convoy.add_agent("seller-001", "seller")

        convoy.record_deal(deal_confirmation)

        assert len(convoy.deals) == 1
        assert convoy.total_spend == 120.0
        assert convoy.total_impressions == 10000
        assert convoy.total_iterations == 1

    def test_goal_progress(self, campaign, deal_confirmation):
        """Test goal progress calculation."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        # Initially 0 progress
        assert convoy.goal_progress == 0.0

        # Record a deal with 10000 impressions (1% of 1M target)
        convoy.record_deal(deal_confirmation)

        # Progress should be 1%
        assert convoy.goal_progress == pytest.approx(0.01, rel=0.01)

    def test_checkpoint(self, campaign):
        """Test creating a checkpoint."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        convoy.add_agent("buyer-001", "buyer")
        convoy.current_day = 5
        convoy.current_iteration = 10

        checkpoint = convoy.checkpoint()

        assert checkpoint.convoy_id == "convoy-001"
        assert checkpoint.day == 5
        assert checkpoint.iteration == 10
        assert "buyer-001" in checkpoint.agent_states
        assert len(convoy.checkpoints) == 1

    @pytest.mark.asyncio
    async def test_pause_resume(self, campaign):
        """Test pausing and resuming a convoy."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        # Initially not paused
        convoy.status = ConvoyStatus.ACTIVE

        convoy.pause()
        assert convoy.status == ConvoyStatus.PAUSED

        convoy.resume()
        assert convoy.status == ConvoyStatus.ACTIVE

    def test_to_dict(self, campaign):
        """Test serializing convoy to dict."""
        convoy = Convoy(
            convoy_id="convoy-001",
            campaign=campaign,
            scenario="A",
        )

        convoy.add_agent("buyer-001", "buyer")

        data = convoy.to_dict()

        assert data["convoy_id"] == "convoy-001"
        assert data["campaign_id"] == "test-camp-001"
        assert data["scenario"] == "A"
        assert "buyer-001" in data["agents"]


class TestConvoyRegistry:
    """Tests for ConvoyRegistry."""

    @pytest.mark.asyncio
    async def test_create_convoy(self, registry, campaign):
        """Test creating a convoy in the registry."""
        convoy = await registry.create_convoy(
            campaign=campaign,
            scenario="A",
        )

        assert convoy is not None
        assert convoy.campaign.campaign_id == "test-camp-001"
        assert registry.convoy_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_campaign_error(self, registry, campaign):
        """Test that duplicate campaigns raise an error."""
        await registry.create_convoy(campaign=campaign, scenario="A")

        with pytest.raises(ValueError, match="already mapped"):
            await registry.create_convoy(campaign=campaign, scenario="A")

    @pytest.mark.asyncio
    async def test_get_convoy(self, registry, campaign):
        """Test retrieving a convoy by ID."""
        created = await registry.create_convoy(
            campaign=campaign,
            scenario="A",
        )

        fetched = registry.get_convoy(created.convoy_id)

        assert fetched is not None
        assert fetched.convoy_id == created.convoy_id

    @pytest.mark.asyncio
    async def test_get_convoy_for_campaign(self, registry, campaign):
        """Test retrieving a convoy by campaign ID."""
        await registry.create_convoy(campaign=campaign, scenario="A")

        fetched = registry.get_convoy_for_campaign("test-camp-001")

        assert fetched is not None
        assert fetched.campaign.campaign_id == "test-camp-001"

    @pytest.mark.asyncio
    async def test_start_convoy(self, registry, campaign):
        """Test starting a convoy."""
        convoy = await registry.create_convoy(
            campaign=campaign,
            scenario="A",
        )

        assert convoy.status == ConvoyStatus.PENDING

        await registry.start_convoy(convoy.convoy_id)

        assert convoy.status == ConvoyStatus.ACTIVE
        assert convoy.started_at is not None

    @pytest.mark.asyncio
    async def test_complete_convoy(self, registry, campaign):
        """Test completing a convoy."""
        convoy = await registry.create_convoy(
            campaign=campaign,
            scenario="A",
        )
        await registry.start_convoy(convoy.convoy_id)

        await registry.complete_convoy(convoy.convoy_id)

        assert convoy.status == ConvoyStatus.COMPLETED
        assert convoy.completed_at is not None

    @pytest.mark.asyncio
    async def test_record_deal(self, registry, campaign, deal_confirmation):
        """Test recording a deal through the registry."""
        convoy = await registry.create_convoy(
            campaign=campaign,
            scenario="A",
        )

        registry.record_deal(convoy.convoy_id, deal_confirmation)

        assert len(convoy.deals) == 1
        assert convoy.total_spend == 120.0

    @pytest.mark.asyncio
    async def test_callback_on_complete(self, registry, campaign):
        """Test completion callback is fired."""
        completed_convoys = []

        def on_complete(convoy):
            completed_convoys.append(convoy)

        registry.on_convoy_complete(on_complete)

        convoy = await registry.create_convoy(
            campaign=campaign,
            scenario="A",
        )
        await registry.start_convoy(convoy.convoy_id)
        await registry.complete_convoy(convoy.convoy_id)

        assert len(completed_convoys) == 1
        assert completed_convoys[0].convoy_id == convoy.convoy_id

    @pytest.mark.asyncio
    async def test_get_metrics(self, registry, campaign):
        """Test getting registry metrics."""
        await registry.create_convoy(campaign=campaign, scenario="A")

        metrics = registry.get_metrics()

        assert metrics["total_convoys"] == 1
        assert metrics["active_convoys"] == 0
        assert metrics["completed_convoys"] == 0

    @pytest.mark.asyncio
    async def test_list_convoys(self, registry, campaign):
        """Test listing convoys with filters."""
        convoy = await registry.create_convoy(
            campaign=campaign,
            scenario="A",
        )
        await registry.start_convoy(convoy.convoy_id)

        # List all
        all_convoys = registry.list_convoys()
        assert len(all_convoys) == 1

        # Filter by status
        active = registry.list_convoys(status=ConvoyStatus.ACTIVE)
        assert len(active) == 1

        pending = registry.list_convoys(status=ConvoyStatus.PENDING)
        assert len(pending) == 0

        # Filter by scenario
        scenario_a = registry.list_convoys(scenario="A")
        assert len(scenario_a) == 1

        scenario_b = registry.list_convoys(scenario="B")
        assert len(scenario_b) == 0


class TestConvoySyncManager:
    """Tests for ConvoySyncManager."""

    @pytest.mark.asyncio
    async def test_sync_campaigns(self, sync_manager):
        """Test syncing campaigns to convoys."""
        campaigns = [
            MockCampaign(
                campaign_id=f"camp-{i}",
                name=f"Campaign {i}",
                budget=10000.0,
                target_impressions=1000000,
                target_cpm=15.0,
            )
            for i in range(3)
        ]

        convoys = await sync_manager.sync_campaigns(
            campaigns=campaigns,
            scenario="A",
        )

        assert len(convoys) == 3
        assert sync_manager.registry.convoy_count == 3

    @pytest.mark.asyncio
    async def test_start_scenario_convoys(self, sync_manager):
        """Test starting all convoys for a scenario."""
        campaigns = [
            MockCampaign(
                campaign_id=f"camp-{i}",
                name=f"Campaign {i}",
                budget=10000.0,
                target_impressions=1000000,
                target_cpm=15.0,
            )
            for i in range(2)
        ]

        await sync_manager.sync_campaigns(
            campaigns=campaigns,
            scenario="A",
        )

        await sync_manager.start_scenario_convoys("A")

        active = sync_manager.registry.list_convoys(status=ConvoyStatus.ACTIVE)
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_advance_day(self, sync_manager):
        """Test advancing simulation day."""
        campaign = MockCampaign(
            campaign_id="camp-001",
            name="Campaign 1",
            budget=10000.0,
            target_impressions=1000000,
            target_cpm=15.0,
        )

        convoys = await sync_manager.sync_campaigns(
            campaigns=[campaign],
            scenario="A",
        )
        await sync_manager.start_scenario_convoys("A")

        await sync_manager.advance_day("A", 5)

        convoy = convoys[0]
        assert convoy.current_day == 5
        assert convoy.current_iteration == 0

    @pytest.mark.asyncio
    async def test_complete_scenario(self, sync_manager):
        """Test completing all convoys for a scenario."""
        campaign = MockCampaign(
            campaign_id="camp-001",
            name="Campaign 1",
            budget=10000.0,
            target_impressions=1000000,
            target_cpm=15.0,
        )

        await sync_manager.sync_campaigns(
            campaigns=[campaign],
            scenario="A",
        )
        await sync_manager.start_scenario_convoys("A")

        metrics = await sync_manager.complete_scenario("A")

        assert metrics["scenario"] == "A"
        assert metrics["total_convoys"] == 1
        assert metrics["exhausted_convoys"] == 1  # Goal not met

    @pytest.mark.asyncio
    async def test_get_convoy_status(self, sync_manager):
        """Test getting convoy status by campaign or convoy ID."""
        campaign = MockCampaign(
            campaign_id="camp-001",
            name="Campaign 1",
            budget=10000.0,
            target_impressions=1000000,
            target_cpm=15.0,
        )

        convoys = await sync_manager.sync_campaigns(
            campaigns=[campaign],
            scenario="A",
        )

        # Get by campaign ID
        status = sync_manager.get_convoy_status(campaign_id="camp-001")
        assert status is not None
        assert status["campaign_id"] == "camp-001"

        # Get by convoy ID
        status = sync_manager.get_convoy_status(convoy_id=convoys[0].convoy_id)
        assert status is not None


class TestSingleton:
    """Tests for module-level singleton."""

    def test_get_convoy_manager(self):
        """Test getting the default convoy manager."""
        reset_convoy_manager()

        manager1 = get_convoy_manager()
        manager2 = get_convoy_manager()

        assert manager1 is manager2

    def test_reset_convoy_manager(self):
        """Test resetting the convoy manager."""
        manager1 = get_convoy_manager()
        reset_convoy_manager()
        manager2 = get_convoy_manager()

        assert manager1 is not manager2
