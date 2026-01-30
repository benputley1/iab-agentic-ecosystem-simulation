"""Tests for the L1 Portfolio Manager.

Tests cover:
- Budget allocation logic
- Channel selection
- L2 delegation
- Result aggregation
- Portfolio state management
"""

import json
import os
import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.buyer.l1_portfolio_manager import (
    PortfolioManager,
    create_portfolio_manager,
)
from src.agents.buyer.models import (
    Campaign,
    CampaignObjectives,
    CampaignStatus,
    AudienceSpec,
    BudgetAllocation,
    ChannelSelection,
    PortfolioState,
    SpecialistTask,
    SpecialistResult,
    Channel,
)
from src.agents.base.orchestrator import (
    CampaignState,
    StrategicDecision,
    SpecialistAssignment,
)
from src.agents.base.context import AgentContext, ContextPriority


@pytest.fixture
def sample_objectives():
    """Create sample campaign objectives."""
    return CampaignObjectives(
        reach_target=500000,
        frequency_cap=5,
        cpm_target=15.0,
        channel_mix={
            "display": 0.5,
            "video": 0.3,
            "ctv": 0.2,
        },
    )


@pytest.fixture
def sample_audience():
    """Create sample audience spec."""
    return AudienceSpec(
        segments=["tech_enthusiasts", "early_adopters"],
        demographics={"age": "25-44", "income": "high"},
        geo_targets=["US", "UK"],
        device_types=["desktop", "mobile"],
    )


@pytest.fixture
def sample_campaign(sample_objectives, sample_audience):
    """Create a sample campaign."""
    return Campaign(
        campaign_id="camp-001",
        name="Q1 Brand Awareness",
        advertiser="TechCorp",
        total_budget=50000.0,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 3, 31),
        objectives=sample_objectives,
        audience=sample_audience,
        status=CampaignStatus.ACTIVE,
        priority=1,
    )


@pytest.fixture
def sample_campaign_2(sample_audience):
    """Create a second sample campaign."""
    objectives = CampaignObjectives(
        reach_target=300000,
        frequency_cap=3,
        cpm_target=12.0,
        channel_mix={"display": 0.7, "mobile_app": 0.3},
    )
    return Campaign(
        campaign_id="camp-002",
        name="Product Launch",
        advertiser="TechCorp",
        total_budget=30000.0,
        start_date=date(2025, 2, 1),
        end_date=date(2025, 2, 28),
        objectives=objectives,
        audience=sample_audience,
        status=CampaignStatus.ACTIVE,
        priority=2,
    )


@pytest.fixture
def sample_campaign_state():
    """Create a sample CampaignState for base class operations."""
    return CampaignState(
        campaign_id="camp-001",
        name="Q1 Brand Awareness",
        budget_total=50000.0,
        objectives=["reach", "awareness"],
        channel_allocations={"display": 0.5, "video": 0.3, "ctv": 0.2},
        priority=1,
    )


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({
        "allocations": {"camp-001": {"display": 25000.0, "video": 15000.0}},
        "reasoning": "Test allocation",
        "total_allocated": 40000.0,
    })
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    return mock_response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    """Create a mock AsyncAnthropic client."""
    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
    return mock_client


@pytest.fixture
def portfolio_manager(mock_anthropic_client):
    """Create a Portfolio Manager with mock client."""
    manager = PortfolioManager(agent_id="test-pm-001")
    manager.client = mock_anthropic_client
    return manager


class TestPortfolioManagerBasics:
    """Test basic Portfolio Manager operations."""
    
    def test_creation(self):
        """Test basic manager creation."""
        manager = create_portfolio_manager(
            agent_id="test-001",
            scenario="B",
        )
        assert manager.agent_id == "test-001"
        assert manager.portfolio is not None
    
    @pytest.mark.asyncio
    async def test_add_l1_campaign(self, portfolio_manager, sample_campaign):
        """Test adding a Campaign to portfolio."""
        await portfolio_manager.initialize()
        portfolio_manager.add_l1_campaign(sample_campaign)
        
        assert sample_campaign.campaign_id in portfolio_manager.portfolio.campaigns
        assert portfolio_manager.portfolio.total_budget == 50000.0
    
    @pytest.mark.asyncio
    async def test_multiple_campaigns(
        self, portfolio_manager, sample_campaign, sample_campaign_2
    ):
        """Test managing multiple campaigns."""
        await portfolio_manager.initialize()
        portfolio_manager.add_l1_campaign(sample_campaign)
        portfolio_manager.add_l1_campaign(sample_campaign_2)
        
        assert len(portfolio_manager.portfolio.campaigns) == 2
        assert portfolio_manager.portfolio.total_budget == 80000.0


class TestBudgetAllocation:
    """Test budget allocation logic."""
    
    @pytest.mark.asyncio
    async def test_allocate_budget_llm_call(
        self, portfolio_manager, mock_anthropic_client, sample_campaign
    ):
        """Test that budget allocation calls LLM correctly."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "allocations": {
                "camp-001": {
                    "display": 25000.0,
                    "video": 15000.0,
                    "ctv": 10000.0,
                }
            },
            "reasoning": "Allocated based on channel mix objectives.",
            "total_allocated": 50000.0,
            "risk_factors": ["Market volatility"],
        })
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create.return_value = mock_response
        
        await portfolio_manager.initialize()
        portfolio_manager.add_l1_campaign(sample_campaign)
        allocation = await portfolio_manager.allocate_budget([sample_campaign])
        
        assert mock_anthropic_client.messages.create.called
        assert allocation.total_allocated == 50000.0
        assert "camp-001" in allocation.allocations
        assert allocation.allocations["camp-001"]["display"] == 25000.0
    
    @pytest.mark.asyncio
    async def test_allocate_budget_empty_campaigns(self, portfolio_manager):
        """Test allocation with no campaigns."""
        await portfolio_manager.initialize()
        allocation = await portfolio_manager.allocate_budget([])
        
        assert allocation.allocations == {}
        assert "No campaigns" in allocation.reasoning
    
    @pytest.mark.asyncio
    async def test_default_allocation_fallback(
        self, portfolio_manager, mock_anthropic_client, sample_campaign
    ):
        """Test fallback to default allocation on LLM failure."""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")
        
        await portfolio_manager.initialize()
        portfolio_manager.add_l1_campaign(sample_campaign)
        allocation = await portfolio_manager.allocate_budget([sample_campaign])
        
        assert "camp-001" in allocation.allocations
        assert "fallback" in allocation.reasoning.lower()


class TestChannelSelection:
    """Test channel selection logic."""
    
    @pytest.mark.asyncio
    async def test_select_channels_llm_call(
        self, portfolio_manager, mock_anthropic_client, sample_campaign
    ):
        """Test that channel selection calls LLM correctly."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "selected_channels": [
                {
                    "channel": "display",
                    "allocation_pct": 0.5,
                    "rationale": "High reach at target CPM",
                    "expected_reach": 250000,
                    "expected_cpm": 14.0,
                },
                {
                    "channel": "video",
                    "allocation_pct": 0.3,
                    "rationale": "Strong engagement",
                    "expected_reach": 150000,
                    "expected_cpm": 18.0,
                },
            ],
            "total_expected_reach": 400000,
            "blended_cpm": 15.5,
            "strategy_summary": "Balanced approach for brand awareness.",
        })
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create.return_value = mock_response
        
        await portfolio_manager.initialize()
        selections = await portfolio_manager.select_channels(sample_campaign)
        
        assert len(selections) == 2
        assert selections[0].channel == "display"
        assert selections[0].allocation_pct == 0.5
        assert selections[1].channel == "video"
    
    @pytest.mark.asyncio
    async def test_channel_selection_fallback(
        self, portfolio_manager, mock_anthropic_client, sample_campaign
    ):
        """Test fallback on channel selection failure."""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")
        
        await portfolio_manager.initialize()
        selections = await portfolio_manager.select_channels(sample_campaign)
        
        assert len(selections) == 1
        assert selections[0].channel == "display"
        assert "fallback" in selections[0].rationale.lower()


class TestStrategicDecision:
    """Test strategic decision making."""
    
    @pytest.mark.asyncio
    async def test_make_strategic_decision(
        self, portfolio_manager, mock_anthropic_client
    ):
        """Test making a strategic decision."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "decision_type": "budget_reallocation",
            "description": "Shift budget to higher performing channels",
            "rationale": "Display is outperforming video by 20%",
            "impact": {"display_increase": 10000, "video_decrease": 10000}
        })
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create.return_value = mock_response
        
        await portfolio_manager.initialize()
        
        context = AgentContext(
            source_level=1,
            source_agent_id="test-pm-001",
            task_description="Optimize budget allocation"
        )
        
        decision = await portfolio_manager.make_strategic_decision(
            context, "Should we reallocate budget?"
        )
        
        assert decision.decision_type == "budget_reallocation"
        assert "display_increase" in decision.impact


class TestSpecialistAssignments:
    """Test specialist assignment planning."""
    
    @pytest.mark.asyncio
    async def test_plan_specialist_assignments(
        self, portfolio_manager, sample_campaign_state
    ):
        """Test planning specialist assignments."""
        # Initialize first (without specialists)
        await portfolio_manager.initialize()
        
        # Register mock specialists after initialization
        for channel in ["display", "video", "ctv"]:
            mock_specialist = MagicMock()
            mock_specialist.agent_id = f"specialist-{channel}"
            mock_specialist.channel = channel
            mock_specialist.name = f"{channel.title()} Specialist"
            mock_specialist.initialize = AsyncMock()
            mock_specialist.cleanup = AsyncMock()
            portfolio_manager.register_specialist(mock_specialist)
        
        context = AgentContext(
            source_level=1,
            source_agent_id="test-pm-001",
            task_description="Execute campaign"
        )
        
        assignments = await portfolio_manager.plan_specialist_assignments(
            context, sample_campaign_state
        )
        
        # Should create assignments for display, video, ctv based on channel_allocations
        assert len(assignments) == 3
        channels = {a.channel for a in assignments}
        assert "display" in channels
        assert "video" in channels
        assert "ctv" in channels


class TestResultAggregation:
    """Test result aggregation logic."""
    
    @pytest.mark.asyncio
    async def test_aggregate_results(
        self, portfolio_manager, sample_campaign, sample_campaign_2
    ):
        """Test aggregating specialist results."""
        await portfolio_manager.initialize()
        portfolio_manager.add_l1_campaign(sample_campaign)
        portfolio_manager.add_l1_campaign(sample_campaign_2)
        
        results = [
            SpecialistResult(
                task_id="task-001",
                campaign_id="camp-001",
                channel="display",
                success=True,
                impressions_secured=50000,
                spend=750.0,
                deals=[{"deal_id": "d1"}, {"deal_id": "d2"}],
            ),
            SpecialistResult(
                task_id="task-002",
                campaign_id="camp-002",
                channel="display",
                success=True,
                impressions_secured=30000,
                spend=360.0,
                deals=[{"deal_id": "d3"}],
            ),
        ]
        
        portfolio = await portfolio_manager.aggregate_results(results)
        
        assert portfolio.total_spend == 1110.0
        assert portfolio.total_impressions == 80000
        
        camp1 = portfolio.campaigns["camp-001"]
        assert camp1.spend == 750.0
        assert camp1.impressions_delivered == 50000
        assert camp1.deals_made == 2
    
    @pytest.mark.asyncio
    async def test_aggregate_with_failures(
        self, portfolio_manager, sample_campaign
    ):
        """Test aggregation handles failed results."""
        await portfolio_manager.initialize()
        portfolio_manager.add_l1_campaign(sample_campaign)
        
        results = [
            SpecialistResult(
                task_id="task-001",
                campaign_id="camp-001",
                channel="display",
                success=True,
                impressions_secured=20000,
                spend=300.0,
            ),
            SpecialistResult(
                task_id="task-002",
                campaign_id="camp-001",
                channel="video",
                success=False,
                error="Publisher unavailable",
            ),
        ]
        
        portfolio = await portfolio_manager.aggregate_results(results)
        
        assert portfolio.total_spend == 300.0
        assert portfolio.total_impressions == 20000


class TestModels:
    """Test data model behavior."""
    
    def test_campaign_remaining_budget(self, sample_campaign):
        """Test remaining budget calculation."""
        sample_campaign.spend = 10000.0
        assert sample_campaign.remaining_budget == 40000.0
    
    def test_campaign_budget_utilization(self, sample_campaign):
        """Test budget utilization calculation."""
        sample_campaign.spend = 25000.0
        assert sample_campaign.budget_utilization == 0.5
    
    def test_campaign_daily_budget(self, sample_campaign):
        """Test daily budget calculation."""
        days = sample_campaign.campaign_duration_days
        expected_daily = 50000.0 / days
        assert abs(sample_campaign.daily_budget - expected_daily) < 0.01
    
    def test_campaign_to_dict(self, sample_campaign):
        """Test campaign serialization."""
        data = sample_campaign.to_dict()
        
        assert data["campaign_id"] == "camp-001"
        assert data["name"] == "Q1 Brand Awareness"
        assert data["total_budget"] == 50000.0
        assert "objectives" in data
        assert "audience" in data
    
    def test_budget_allocation_totals(self):
        """Test budget allocation helper methods."""
        allocation = BudgetAllocation(
            allocations={
                "camp-001": {"display": 5000.0, "video": 3000.0},
                "camp-002": {"display": 4000.0, "ctv": 2000.0},
            },
            total_allocated=14000.0,
        )
        
        assert allocation.get_campaign_total("camp-001") == 8000.0
        assert allocation.get_channel_total("display") == 9000.0
    
    def test_portfolio_state_recalculation(
        self, sample_campaign, sample_campaign_2
    ):
        """Test portfolio state auto-recalculates."""
        portfolio = PortfolioState(portfolio_id="test-portfolio")
        
        portfolio.add_campaign(sample_campaign)
        assert portfolio.total_budget == 50000.0
        
        portfolio.add_campaign(sample_campaign_2)
        assert portfolio.total_budget == 80000.0
        
        portfolio.remove_campaign("camp-001")
        assert portfolio.total_budget == 30000.0
    
    def test_objectives_validation(self):
        """Test objectives validation."""
        valid_objectives = CampaignObjectives(
            reach_target=100000,
            frequency_cap=5,
            cpm_target=15.0,
            channel_mix={"display": 0.6, "video": 0.4},
        )
        assert valid_objectives.validate()
        
        invalid_objectives = CampaignObjectives(
            reach_target=100000,
            frequency_cap=5,
            cpm_target=15.0,
            channel_mix={"display": 0.6, "video": 0.6},
        )
        assert not invalid_objectives.validate()
