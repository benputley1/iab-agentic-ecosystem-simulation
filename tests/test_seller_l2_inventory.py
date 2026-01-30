"""
Tests for L2 Seller Channel Inventory Agents.

Tests the 5 channel inventory specialists:
- DisplayInventoryAgent
- VideoInventoryAgent
- CTVInventoryAgent
- MobileAppInventoryAgent
- NativeInventoryAgent
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import L2 agents
from src.agents.seller.l2_display import DisplayInventoryAgent, IAB_DISPLAY_SIZES
from src.agents.seller.l2_video import VideoInventoryAgent, VIDEO_PLACEMENTS
from src.agents.seller.l2_ctv import CTVInventoryAgent, CTV_PLATFORMS
from src.agents.seller.l2_mobile import MobileAppInventoryAgent, MOBILE_FORMATS
from src.agents.seller.l2_native import NativeInventoryAgent, NATIVE_FORMATS

# Import base types
from src.agents.base.specialist import (
    AvailsRequest,
    AvailsResponse,
    PricingRequest,
    PricingResponse,
    Campaign,
    FitScore,
)
from src.agents.base.context import AgentContext, ContextPriority


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def display_agent():
    """Create a display inventory agent."""
    return DisplayInventoryAgent(agent_id="test-display-001", name="TestDisplay")


@pytest.fixture
def video_agent():
    """Create a video inventory agent."""
    return VideoInventoryAgent(agent_id="test-video-001", name="TestVideo")


@pytest.fixture
def ctv_agent():
    """Create a CTV inventory agent."""
    return CTVInventoryAgent(agent_id="test-ctv-001", name="TestCTV")


@pytest.fixture
def mobile_agent():
    """Create a mobile app inventory agent."""
    return MobileAppInventoryAgent(agent_id="test-mobile-001", name="TestMobile")


@pytest.fixture
def native_agent():
    """Create a native inventory agent."""
    return NativeInventoryAgent(agent_id="test-native-001", name="TestNative")


@pytest.fixture
def sample_avails_request():
    """Create a sample availability request."""
    return AvailsRequest(
        channel="display",
        impressions_needed=100_000,
        targeting={"sizes": ["300x250", "728x90"]},
    )


@pytest.fixture
def sample_pricing_request():
    """Create a sample pricing request."""
    return PricingRequest(
        channel="display",
        impressions=500_000,
        deal_type="pmp",
        buyer_tier="premium",
        targeting={"placement": "above_fold"},
    )


@pytest.fixture
def sample_campaign():
    """Create a sample campaign for fit evaluation."""
    return Campaign(
        campaign_id="campaign-001",
        name="Test Campaign",
        objective="Display brand awareness",
        budget=50_000.0,
        target_impressions=1_000_000,
        target_cpm=10.0,
    )


@pytest.fixture
def sample_context():
    """Create a sample agent context."""
    ctx = AgentContext(
        parent_agent_id="l1-orchestrator",
        task_description="Evaluate inventory",
    )
    ctx.add_item("buyer_id", "buyer-001", ContextPriority.HIGH)
    return ctx


# -----------------------------------------------------------------------------
# Display Agent Tests
# -----------------------------------------------------------------------------


class TestDisplayInventoryAgent:
    """Tests for DisplayInventoryAgent."""
    
    def test_initialization(self, display_agent):
        """Test agent initializes correctly."""
        assert display_agent.channel == "display"
        assert display_agent.agent_id == "test-display-001"
        assert len(display_agent.supported_sizes) > 0
    
    def test_default_pricing(self, display_agent):
        """Test display has correct default pricing."""
        assert display_agent.DEFAULT_FLOOR_CPM == 2.0
        assert display_agent.DEFAULT_BASE_CPM == 8.0
        assert display_agent.DEFAULT_MAX_CPM == 25.0
    
    def test_get_system_prompt(self, display_agent):
        """Test system prompt contains display-specific info."""
        prompt = display_agent.get_system_prompt()
        assert "display" in prompt.lower()
        assert "banner" in prompt.lower()
        assert "IAB" in prompt or "iab" in prompt.lower()
    
    def test_get_channel_capabilities(self, display_agent):
        """Test channel capabilities are returned."""
        capabilities = display_agent.get_channel_capabilities()
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert "banner_placement" in capabilities
    
    @pytest.mark.asyncio
    async def test_check_availability(self, display_agent, sample_avails_request):
        """Test availability check returns valid response."""
        response = await display_agent.check_availability(sample_avails_request)
        
        assert isinstance(response, AvailsResponse)
        assert response.impressions_available > 0
        assert isinstance(response.available, bool)
    
    @pytest.mark.asyncio
    async def test_check_availability_with_premium_sizes(self, display_agent):
        """Test availability with premium ad sizes."""
        request = AvailsRequest(
            channel="display",
            impressions_needed=50_000,
            targeting={
                "sizes": ["300x250", "970x250"],
                "placement": "above_fold",
            },
        )
        
        response = await display_agent.check_availability(request)
        
        assert response.details.get("quality_tier") == "premium"
    
    @pytest.mark.asyncio
    async def test_calculate_price(self, display_agent, sample_pricing_request):
        """Test pricing calculation."""
        response = await display_agent.calculate_price(sample_pricing_request)
        
        assert isinstance(response, PricingResponse)
        assert response.base_cpm > 0
        assert response.recommended_cpm >= response.floor_cpm
    
    @pytest.mark.asyncio
    async def test_calculate_price_volume_discount(self, display_agent):
        """Test volume discount is applied for large orders."""
        request = PricingRequest(
            channel="display",
            impressions=2_000_000,
            deal_type="open",
            targeting={},
        )
        
        response = await display_agent.calculate_price(request)
        
        assert response.discount_applied > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_fit(self, display_agent, sample_campaign):
        """Test campaign fit evaluation."""
        fit = await display_agent.evaluate_fit(sample_campaign)
        
        assert isinstance(fit, FitScore)
        assert 0.0 <= fit.score <= 1.0
        assert "channel_match" in fit.factors
    
    @pytest.mark.asyncio
    async def test_plan_execution_availability(self, display_agent, sample_context):
        """Test planning for availability check."""
        delegations = await display_agent.plan_execution(
            sample_context,
            "Check availability for 100K impressions"
        )
        
        assert len(delegations) > 0
        assert any("avails" in d.functional_agent_id for d in delegations)


# -----------------------------------------------------------------------------
# Video Agent Tests
# -----------------------------------------------------------------------------


class TestVideoInventoryAgent:
    """Tests for VideoInventoryAgent."""
    
    def test_initialization(self, video_agent):
        """Test agent initializes correctly."""
        assert video_agent.channel == "video"
        assert video_agent.DEFAULT_FLOOR_CPM == 8.0
        assert video_agent.DEFAULT_BASE_CPM == 18.0
    
    def test_get_channel_capabilities(self, video_agent):
        """Test video capabilities."""
        capabilities = video_agent.get_channel_capabilities()
        assert "pre_roll_placement" in capabilities
        assert "completion_rate_optimization" in capabilities
    
    @pytest.mark.asyncio
    async def test_check_availability_pre_roll(self, video_agent):
        """Test availability for pre-roll placement."""
        request = AvailsRequest(
            channel="video",
            impressions_needed=50_000,
            targeting={"placement": "pre_roll"},
        )
        
        response = await video_agent.check_availability(request)
        
        assert response.available is True
        assert "completion_rate" in response.notes.lower() or "completion" in str(response.details)
    
    @pytest.mark.asyncio
    async def test_calculate_price_non_skippable(self, video_agent):
        """Test pricing for non-skippable video."""
        request = PricingRequest(
            channel="video",
            impressions=100_000,
            deal_type="pg",
            targeting={
                "placement": "mid_roll",
                "duration": 30,
                "non_skippable": True,
            },
        )
        
        response = await video_agent.calculate_price(request)
        
        # Non-skippable should result in higher pricing
        assert response.recommended_cpm > response.floor_cpm
        assert "non_skippable" in response.notes.lower() or response.recommended_cpm > video_agent.DEFAULT_BASE_CPM
    
    @pytest.mark.asyncio
    async def test_evaluate_fit_video_campaign(self, video_agent):
        """Test fit evaluation for video campaign."""
        campaign = Campaign(
            campaign_id="video-campaign-001",
            name="Video Brand Campaign",
            objective="video brand awareness",
            budget=100_000.0,
            target_impressions=500_000,
            target_cpm=20.0,
        )
        
        fit = await video_agent.evaluate_fit(campaign)
        
        assert fit.factors["channel_match"] == 1.0


# -----------------------------------------------------------------------------
# CTV Agent Tests
# -----------------------------------------------------------------------------


class TestCTVInventoryAgent:
    """Tests for CTVInventoryAgent."""
    
    def test_initialization(self, ctv_agent):
        """Test agent initializes correctly."""
        assert ctv_agent.channel == "ctv"
        assert ctv_agent.DEFAULT_FLOOR_CPM == 15.0  # Premium pricing
        assert ctv_agent.DEFAULT_BASE_CPM == 30.0
    
    def test_get_channel_capabilities(self, ctv_agent):
        """Test CTV capabilities."""
        capabilities = ctv_agent.get_channel_capabilities()
        assert "household_targeting" in capabilities
        assert "streaming_app_inventory" in capabilities
    
    @pytest.mark.asyncio
    async def test_check_availability_household(self, ctv_agent):
        """Test availability with household targeting."""
        request = AvailsRequest(
            channel="ctv",
            impressions_needed=25_000,
            targeting={
                "platforms": ["roku", "fire_tv"],
                "household_targeting": True,
            },
        )
        
        response = await ctv_agent.check_availability(request)
        
        assert "household" in response.notes.lower()
    
    @pytest.mark.asyncio
    async def test_calculate_price_premium_content(self, ctv_agent):
        """Test pricing for premium CTV content."""
        request = PricingRequest(
            channel="ctv",
            impressions=100_000,
            deal_type="pmp",
            targeting={
                "content_type": "live_sports",
                "household_targeting": True,
            },
        )
        
        response = await ctv_agent.calculate_price(request)
        
        # Live sports should have premium pricing
        assert response.recommended_cpm > ctv_agent.DEFAULT_BASE_CPM
    
    @pytest.mark.asyncio
    async def test_evaluate_fit_low_budget(self, ctv_agent):
        """Test fit evaluation warns about low budget."""
        campaign = Campaign(
            campaign_id="ctv-campaign-001",
            name="Small CTV Campaign",
            objective="ctv brand awareness",
            budget=1_000.0,  # Low budget for CTV
            target_impressions=100_000,
            target_cpm=8.0,  # Below CTV floor
        )
        
        fit = await ctv_agent.evaluate_fit(campaign)
        
        # Should have low price fit
        assert fit.factors["price_alignment"] < 0.5


# -----------------------------------------------------------------------------
# Mobile Agent Tests
# -----------------------------------------------------------------------------


class TestMobileAppInventoryAgent:
    """Tests for MobileAppInventoryAgent."""
    
    def test_initialization(self, mobile_agent):
        """Test agent initializes correctly."""
        assert mobile_agent.channel == "mobile"
        assert mobile_agent.DEFAULT_FLOOR_CPM == 4.0
        assert "rewarded_video" in mobile_agent.supported_formats
    
    def test_get_channel_capabilities(self, mobile_agent):
        """Test mobile capabilities."""
        capabilities = mobile_agent.get_channel_capabilities()
        assert "rewarded_video" in capabilities
        assert "ios_targeting" in capabilities
    
    @pytest.mark.asyncio
    async def test_check_availability_rewarded(self, mobile_agent):
        """Test availability for rewarded video."""
        request = AvailsRequest(
            channel="mobile",
            impressions_needed=100_000,
            targeting={
                "format": "rewarded_video",
                "os": "ios",
            },
        )
        
        response = await mobile_agent.check_availability(request)
        
        assert "engagement" in response.notes.lower()
    
    @pytest.mark.asyncio
    async def test_calculate_price_ios_premium(self, mobile_agent):
        """Test iOS premium pricing."""
        request = PricingRequest(
            channel="mobile",
            impressions=200_000,
            deal_type="open",
            targeting={
                "format": "rewarded_video",
                "os": "ios",
            },
        )
        
        response = await mobile_agent.calculate_price(request)
        
        # iOS + rewarded_video should result in premium pricing
        assert response.recommended_cpm > mobile_agent.DEFAULT_BASE_CPM
    
    @pytest.mark.asyncio
    async def test_evaluate_fit_app_install(self, mobile_agent):
        """Test fit evaluation for app install campaign."""
        campaign = Campaign(
            campaign_id="mobile-campaign-001",
            name="App Install Campaign",
            objective="mobile app installs",
            budget=25_000.0,
            target_impressions=500_000,
            target_cpm=8.0,
        )
        
        fit = await mobile_agent.evaluate_fit(campaign)
        
        assert fit.factors["channel_match"] == 1.0


# -----------------------------------------------------------------------------
# Native Agent Tests
# -----------------------------------------------------------------------------


class TestNativeInventoryAgent:
    """Tests for NativeInventoryAgent."""
    
    def test_initialization(self, native_agent):
        """Test agent initializes correctly."""
        assert native_agent.channel == "native"
        assert native_agent.DEFAULT_FLOOR_CPM == 3.0
        assert "sponsored_content" in native_agent.supported_formats
    
    def test_get_channel_capabilities(self, native_agent):
        """Test native capabilities."""
        capabilities = native_agent.get_channel_capabilities()
        assert "content_recommendations" in capabilities
        assert "context_matching" in capabilities
    
    @pytest.mark.asyncio
    async def test_check_availability_context_match(self, native_agent):
        """Test availability with context matching."""
        request = AvailsRequest(
            channel="native",
            impressions_needed=20_000,
            targeting={
                "format": "branded_content",
                "context_match": "exact",
            },
        )
        
        response = await native_agent.check_availability(request)
        
        # Exact match should reduce available inventory
        assert response.details.get("context_match") == "exact"
    
    @pytest.mark.asyncio
    async def test_calculate_price_context_premium(self, native_agent):
        """Test exact context match pricing."""
        request = PricingRequest(
            channel="native",
            impressions=100_000,
            deal_type="pmp",
            targeting={
                "format": "sponsored_content",
                "context_match": "exact",
                "vertical": "lifestyle",
            },
        )
        
        response = await native_agent.calculate_price(request)
        
        # Context premium should result in higher pricing
        assert response.recommended_cpm > native_agent.DEFAULT_BASE_CPM
    
    @pytest.mark.asyncio
    async def test_evaluate_fit_brand_awareness(self, native_agent):
        """Test fit evaluation for brand awareness campaign."""
        campaign = Campaign(
            campaign_id="native-campaign-001",
            name="Brand Storytelling Campaign",
            objective="native content marketing",
            budget=75_000.0,
            target_impressions=500_000,
            target_cpm=12.0,
        )
        
        fit = await native_agent.evaluate_fit(campaign)
        
        # Native should have high brand safety
        assert fit.factors["brand_safety"] > 0.9


# -----------------------------------------------------------------------------
# Cross-Agent Tests
# -----------------------------------------------------------------------------


class TestL2AgentCommonBehavior:
    """Tests for common behavior across all L2 agents."""
    
    @pytest.mark.parametrize("agent_fixture,channel", [
        ("display_agent", "display"),
        ("video_agent", "video"),
        ("ctv_agent", "ctv"),
        ("mobile_agent", "mobile"),
        ("native_agent", "native"),
    ])
    def test_all_agents_have_required_methods(self, request, agent_fixture, channel):
        """Verify all agents implement the required interface."""
        agent = request.getfixturevalue(agent_fixture)
        
        # Check channel
        assert agent.channel == channel
        
        # Check required methods
        assert callable(agent.get_system_prompt)
        assert callable(agent.get_channel_capabilities)
        assert callable(agent.plan_execution)
        assert callable(agent.check_availability)
        assert callable(agent.calculate_price)
        assert callable(agent.evaluate_fit)
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_fixture", [
        "display_agent",
        "video_agent", 
        "ctv_agent",
        "mobile_agent",
        "native_agent",
    ])
    async def test_floor_price_enforcement(self, request, agent_fixture):
        """Test that floor price is always enforced."""
        agent = request.getfixturevalue(agent_fixture)
        
        pricing_request = PricingRequest(
            channel=agent.channel,
            impressions=1_000,
            deal_type="open",
            targeting={},
        )
        
        response = await agent.calculate_price(pricing_request)
        
        assert response.recommended_cpm >= agent.DEFAULT_FLOOR_CPM


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestL2AgentIntegration:
    """Integration tests for L2 agents."""
    
    @pytest.mark.asyncio
    async def test_premium_vs_standard_pricing(self, ctv_agent, mobile_agent):
        """Compare pricing between premium (CTV) and standard (mobile) channels."""
        impressions = 100_000
        
        ctv_request = PricingRequest(
            channel="ctv",
            impressions=impressions,
            deal_type="open",
            targeting={},
        )
        
        mobile_request = PricingRequest(
            channel="mobile",
            impressions=impressions,
            deal_type="open",
            targeting={},
        )
        
        ctv_price = await ctv_agent.calculate_price(ctv_request)
        mobile_price = await mobile_agent.calculate_price(mobile_request)
        
        # CTV should be more expensive
        assert ctv_price.floor_cpm > mobile_price.floor_cpm
        assert ctv_price.recommended_cpm > mobile_price.recommended_cpm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
