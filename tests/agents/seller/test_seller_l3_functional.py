"""
Tests for Seller L3 Functional Agents.

Tests cover:
- PricingAgent
- AvailsAgent
- ProposalReviewAgent
- UpsellAgent
- AudienceValidatorAgent
"""

import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.agents.seller.l3_pricing import PricingAgent, BuyerContext, Price
from src.agents.seller.l3_avails import AvailsAgent, DateRange, AvailsResult, Forecast, Allocation
from src.agents.seller.l3_proposal_review import (
    ProposalReviewAgent, Proposal, ProposalStatus, ReviewResult, CounterOffer
)
from src.agents.seller.l3_upsell import UpsellAgent, Deal, Opportunity, Bundle
from src.agents.seller.l3_audience_validator import (
    AudienceValidatorAgent, AudienceSpec, ValidationResult, Coverage
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    client.messages = MagicMock()
    return client


@pytest.fixture
def pricing_agent(mock_anthropic_client):
    """Create PricingAgent with mocked client."""
    return PricingAgent(anthropic_client=mock_anthropic_client)


@pytest.fixture
def avails_agent(mock_anthropic_client):
    """Create AvailsAgent with mocked client."""
    return AvailsAgent(anthropic_client=mock_anthropic_client)


@pytest.fixture
def proposal_agent(mock_anthropic_client):
    """Create ProposalReviewAgent with mocked client."""
    return ProposalReviewAgent(anthropic_client=mock_anthropic_client)


@pytest.fixture
def upsell_agent(mock_anthropic_client):
    """Create UpsellAgent with mocked client."""
    return UpsellAgent(anthropic_client=mock_anthropic_client)


@pytest.fixture
def audience_agent(mock_anthropic_client):
    """Create AudienceValidatorAgent with mocked client."""
    return AudienceValidatorAgent(anthropic_client=mock_anthropic_client)


@pytest.fixture
def sample_buyer_context():
    """Sample buyer context for pricing tests."""
    return BuyerContext(
        buyer_id="buyer-001",
        tier="premium",
        historical_spend=50000.0,
        relationship_months=12,
    )


@pytest.fixture
def sample_date_range():
    """Sample date range for avails tests."""
    today = date.today()
    return DateRange(
        start_date=today,
        end_date=today + timedelta(days=30),
    )


@pytest.fixture
def sample_proposal():
    """Sample proposal for review tests."""
    today = date.today()
    return Proposal(
        proposal_id="prop-001",
        buyer_id="buyer-001",
        product_id="prod-001",
        requested_impressions=500000,
        proposed_cpm=12.0,
        start_date=today.isoformat(),
        end_date=(today + timedelta(days=30)).isoformat(),
        deal_type="preferred_deal",
    )


@pytest.fixture
def sample_deal():
    """Sample deal for upsell tests."""
    today = date.today()
    return Deal(
        deal_id="deal-001",
        buyer_id="buyer-001",
        product_id="prod-001",
        channel="display",
        impressions=500000,
        cpm=15.0,
        start_date=today.isoformat(),
        end_date=(today + timedelta(days=30)).isoformat(),
        audience_segments=["tech_enthusiasts", "business_professionals"],
    )


@pytest.fixture
def sample_audience_spec():
    """Sample audience spec for validation tests."""
    return AudienceSpec(
        segments=["tech_enthusiasts", "business_professionals"],
        demographics={"age": ["25-34", "35-44"]},
        geo_targeting=["US", "UK"],
    )


# ============================================================================
# PricingAgent Tests
# ============================================================================

class TestPricingAgent:
    """Tests for PricingAgent."""
    
    def test_agent_initialization(self, pricing_agent):
        """Agent should initialize properly."""
        assert pricing_agent.name == "PricingAgent"
        assert "PriceCalculator" in pricing_agent._tools
        assert "FloorManager" in pricing_agent._tools
        assert "DiscountEngine" in pricing_agent._tools
    
    def test_system_prompt(self, pricing_agent):
        """Agent should have system prompt."""
        prompt = pricing_agent.get_system_prompt()
        assert "Pricing Agent" in prompt
        assert "PriceCalculator" in prompt
    
    @pytest.mark.asyncio
    async def test_calculate_price_basic(self, pricing_agent, sample_buyer_context):
        """Should calculate basic price."""
        price = await pricing_agent.calculate_price(
            product_id="prod-001",
            buyer_context=sample_buyer_context,
        )
        
        assert isinstance(price, Price)
        assert price.base_cpm > 0
        assert price.final_cpm > 0
        assert price.floor_cpm > 0
        assert price.currency == "USD"
    
    @pytest.mark.asyncio
    async def test_calculate_price_with_volume_discount(self, pricing_agent, sample_buyer_context):
        """Should apply volume discount for large orders."""
        price = await pricing_agent.calculate_price(
            product_id="prod-001",
            buyer_context=sample_buyer_context,
            requested_impressions=1000000,
        )
        
        assert price.discount_applied > 0
        assert price.final_cpm < price.base_cpm
    
    @pytest.mark.asyncio
    async def test_premium_tier_discount(self, pricing_agent):
        """Premium tier should get discount."""
        premium_buyer = BuyerContext(buyer_id="buyer-001", tier="premium")
        standard_buyer = BuyerContext(buyer_id="buyer-002", tier="standard")
        
        premium_price = await pricing_agent.calculate_price(
            product_id="prod-001",
            buyer_context=premium_buyer,
        )
        standard_price = await pricing_agent.calculate_price(
            product_id="prod-001",
            buyer_context=standard_buyer,
        )
        
        assert premium_price.discount_applied >= standard_price.discount_applied
    
    @pytest.mark.asyncio
    async def test_floor_price_enforced(self, pricing_agent):
        """Floor price should be enforced."""
        await pricing_agent.set_floor_price("prod-001", 20.0)
        
        buyer = BuyerContext(buyer_id="buyer-001", tier="premium")
        price = await pricing_agent.calculate_price(
            product_id="prod-001",
            buyer_context=buyer,
            requested_impressions=5000000,
        )
        
        assert price.final_cpm >= price.floor_cpm


# ============================================================================
# AvailsAgent Tests
# ============================================================================

class TestAvailsAgent:
    """Tests for AvailsAgent."""
    
    def test_agent_initialization(self, avails_agent):
        """Agent should initialize properly."""
        assert avails_agent.name == "AvailsAgent"
        assert "AvailsChecker" in avails_agent._tools
        assert "CapacityForecaster" in avails_agent._tools
    
    @pytest.mark.asyncio
    async def test_check_avails_basic(self, avails_agent, sample_date_range):
        """Should check basic availability."""
        result = await avails_agent.check_avails(
            product_id="prod-001",
            dates=sample_date_range,
        )
        
        assert isinstance(result, AvailsResult)
        assert result.product_id == "prod-001"
        assert result.available_impressions >= 0
    
    @pytest.mark.asyncio
    async def test_check_avails_with_request(self, avails_agent, sample_date_range):
        """Should check availability against requested impressions."""
        result = await avails_agent.check_avails(
            product_id="prod-001",
            dates=sample_date_range,
            impressions_requested=100000,
        )
        
        assert result.requested_impressions == 100000
        assert result.available is True or result.available_impressions < 100000
    
    @pytest.mark.asyncio
    async def test_forecast_capacity(self, avails_agent):
        """Should forecast capacity."""
        forecast = await avails_agent.forecast_capacity(
            product_id="prod-001",
            days=7,
        )
        
        assert isinstance(forecast, Forecast)
        assert forecast.forecast_days == 7
        assert len(forecast.daily_capacity) == 7
        assert forecast.total_capacity == sum(forecast.daily_capacity)
    
    @pytest.mark.asyncio
    async def test_allocate_inventory(self, avails_agent, sample_date_range):
        """Should allocate inventory."""
        allocation = await avails_agent.allocate_inventory(
            product_id="prod-001",
            deal_id="deal-001",
            impressions=100000,
            dates=sample_date_range,
        )
        
        assert allocation.allocation_id is not None
        assert allocation.impressions_allocated == 100000
        assert allocation.status == "confirmed"
    
    def test_date_range_days(self, sample_date_range):
        """DateRange should calculate days correctly."""
        assert sample_date_range.days == 31


# ============================================================================
# ProposalReviewAgent Tests
# ============================================================================

class TestProposalReviewAgent:
    """Tests for ProposalReviewAgent."""
    
    def test_agent_initialization(self, proposal_agent):
        """Agent should initialize properly."""
        assert proposal_agent.name == "ProposalReviewAgent"
        assert "ProposalGenerator" in proposal_agent._tools
        assert "CounterOfferBuilder" in proposal_agent._tools
        assert "DealIDGenerator" in proposal_agent._tools
    
    @pytest.mark.asyncio
    async def test_review_acceptable_proposal(self, proposal_agent, sample_proposal):
        """Should accept good proposal."""
        result = await proposal_agent.review_proposal(
            proposal=sample_proposal,
            floor_cpm=10.0,
            max_impressions=1000000,
        )
        
        assert isinstance(result, ReviewResult)
        assert result.status == ProposalStatus.ACCEPTABLE
        assert result.price_acceptable is True
        assert result.avails_acceptable is True
    
    @pytest.mark.asyncio
    async def test_review_low_price_proposal(self, proposal_agent, sample_proposal):
        """Should flag low price proposal."""
        result = await proposal_agent.review_proposal(
            proposal=sample_proposal,
            floor_cpm=15.0,  # Above proposed CPM of 12.0
            max_impressions=1000000,
        )
        
        assert result.price_acceptable is False
        assert result.status in [ProposalStatus.NEEDS_ADJUSTMENT, ProposalStatus.REJECTED]
        assert len(result.reasons) > 0
        assert "cpm" in result.suggested_adjustments
    
    @pytest.mark.asyncio
    async def test_review_high_volume_proposal(self, proposal_agent, sample_proposal):
        """Should flag high volume proposal."""
        result = await proposal_agent.review_proposal(
            proposal=sample_proposal,
            floor_cpm=10.0,
            max_impressions=100000,  # Below requested 500000
        )
        
        assert result.avails_acceptable is False
        assert result.status in [ProposalStatus.NEEDS_ADJUSTMENT, ProposalStatus.REJECTED]
        assert "impressions" in result.suggested_adjustments
    
    @pytest.mark.asyncio
    async def test_build_counter_offer(self, proposal_agent, sample_proposal):
        """Should build counter-offer."""
        counter = await proposal_agent.build_counter_offer(
            original=sample_proposal,
            constraints={
                "min_cpm": 15.0,
                "max_impressions": 300000,
            },
            seller_id="seller-001",
        )
        
        assert isinstance(counter, CounterOffer)
        assert counter.original_proposal_id == sample_proposal.proposal_id
        assert counter.offered_cpm >= 15.0
        assert counter.offered_impressions <= 300000
        assert len(counter.adjustments_made) > 0
    
    @pytest.mark.asyncio
    async def test_generate_deal_id(self, proposal_agent):
        """Should generate unique deal IDs."""
        id1 = await proposal_agent.generate_deal_id("buyer-001", "prod-001")
        id2 = await proposal_agent.generate_deal_id("buyer-001", "prod-001")
        
        assert id1.startswith("deal-")
        assert id2.startswith("deal-")
        assert id1 != id2


# ============================================================================
# UpsellAgent Tests
# ============================================================================

class TestUpsellAgent:
    """Tests for UpsellAgent."""
    
    def test_agent_initialization(self, upsell_agent):
        """Agent should initialize properly."""
        assert upsell_agent.name == "UpsellAgent"
        assert "CrossSellAnalyzer" in upsell_agent._tools
        assert "BundleBuilder" in upsell_agent._tools
    
    @pytest.mark.asyncio
    async def test_find_cross_sell(self, upsell_agent, sample_deal):
        """Should find cross-sell opportunities."""
        opportunities = await upsell_agent.find_cross_sell(sample_deal)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        opp = opportunities[0]
        assert isinstance(opp, Opportunity)
        assert opp.opportunity_type == "cross_sell"
        assert opp.estimated_value > 0
    
    @pytest.mark.asyncio
    async def test_cross_sell_recommendations_by_channel(self, upsell_agent):
        """Should recommend complementary channels."""
        display_deal = Deal(
            deal_id="deal-001",
            buyer_id="buyer-001",
            product_id="prod-001",
            channel="display",
            impressions=100000,
            cpm=15.0,
            start_date="2025-01-01",
            end_date="2025-01-31",
        )
        
        opps = await upsell_agent.find_cross_sell(display_deal)
        
        channels = [opp.recommended_product_name.lower() for opp in opps]
        assert any("video" in c for c in channels)
    
    @pytest.mark.asyncio
    async def test_create_bundle(self, upsell_agent):
        """Should create product bundle."""
        bundle = await upsell_agent.create_bundle(
            products=["prod-001", "prod-002", "prod-003"],
            bundle_name="Test Bundle",
        )
        
        assert isinstance(bundle, Bundle)
        assert bundle.name == "Test Bundle"
        assert len(bundle.products) == 3
        assert bundle.discount_pct > 0
        assert bundle.total_value > 0
        assert bundle.savings > 0
    
    @pytest.mark.asyncio
    async def test_bundle_discount_scales(self, upsell_agent):
        """Larger bundles should get bigger discounts."""
        small_bundle = await upsell_agent.create_bundle(
            products=["prod-001", "prod-002"],
        )
        large_bundle = await upsell_agent.create_bundle(
            products=["prod-001", "prod-002", "prod-003", "prod-004", "prod-005"],
        )
        
        assert large_bundle.discount_pct > small_bundle.discount_pct
    
    @pytest.mark.asyncio
    async def test_calculate_bundle_value(self, upsell_agent):
        """Should calculate bundle value."""
        value = await upsell_agent.calculate_bundle_value(
            products=["prod-001", "prod-002"],
        )
        
        assert "total_impressions" in value
        assert "individual_value" in value
        assert "bundle_value" in value


# ============================================================================
# AudienceValidatorAgent Tests
# ============================================================================

class TestAudienceValidatorAgent:
    """Tests for AudienceValidatorAgent."""
    
    def test_agent_initialization(self, audience_agent):
        """Agent should initialize properly."""
        assert audience_agent.name == "AudienceValidatorAgent"
        assert "AudienceValidation" in audience_agent._tools
        assert "CoverageCalculator" in audience_agent._tools
    
    @pytest.mark.asyncio
    async def test_validate_supported_audience(self, audience_agent, sample_audience_spec):
        """Should validate supported audience."""
        result = await audience_agent.validate_audience(sample_audience_spec)
        
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.achievable is True
        assert len(result.supported_segments) > 0
    
    @pytest.mark.asyncio
    async def test_validate_unsupported_audience(self, audience_agent):
        """Should flag unsupported segments."""
        spec = AudienceSpec(
            segments=["fictional_segment", "nonexistent_audience"],
        )
        
        result = await audience_agent.validate_audience(spec)
        
        assert len(result.unsupported_segments) > 0
        assert len(result.issues) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_coverage(self, audience_agent, sample_audience_spec):
        """Should calculate audience coverage."""
        coverage = await audience_agent.calculate_coverage(
            spec=sample_audience_spec,
            inventory="prod-001",
        )
        
        assert isinstance(coverage, Coverage)
        assert coverage.product_id == "prod-001"
        assert coverage.total_reach > 0
        assert 0 <= coverage.coverage_pct <= 100
    
    @pytest.mark.asyncio
    async def test_coverage_decreases_with_targeting(self, audience_agent):
        """More targeting should reduce coverage."""
        simple_spec = AudienceSpec(
            segments=["tech_enthusiasts"],
        )
        complex_spec = AudienceSpec(
            segments=["tech_enthusiasts", "business_professionals", "parents"],
            demographics={"age": ["25-34"], "income": ["high"]},
            geo_targeting=["US", "UK", "DE"],
        )
        
        simple_coverage = await audience_agent.calculate_coverage(simple_spec, "prod-001")
        complex_coverage = await audience_agent.calculate_coverage(complex_spec, "prod-001")
        
        assert simple_coverage.coverage_pct >= complex_coverage.coverage_pct
    
    @pytest.mark.asyncio
    async def test_check_capability(self, audience_agent, sample_audience_spec):
        """Should check targeting capability."""
        capability = await audience_agent.check_capability(sample_audience_spec)
        
        assert "fully_supported" in capability or "support_rate" in capability
    
    @pytest.mark.asyncio
    async def test_empty_audience_spec(self, audience_agent):
        """Empty spec should be valid."""
        empty_spec = AudienceSpec()
        
        assert empty_spec.is_empty is True
        
        result = await audience_agent.validate_audience(empty_spec)
        assert result.achievable is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestAgentIntegration:
    """Integration tests across agents."""
    
    @pytest.mark.asyncio
    async def test_pricing_and_avails_workflow(
        self, pricing_agent, avails_agent, sample_buyer_context, sample_date_range
    ):
        """Test pricing + avails workflow."""
        avails = await avails_agent.check_avails(
            product_id="prod-001",
            dates=sample_date_range,
            impressions_requested=100000,
        )
        
        if avails.available:
            price = await pricing_agent.calculate_price(
                product_id="prod-001",
                buyer_context=sample_buyer_context,
                requested_impressions=100000,
            )
            
            assert price.final_cpm > 0
    
    @pytest.mark.asyncio
    async def test_proposal_review_workflow(
        self, pricing_agent, avails_agent, proposal_agent, audience_agent,
        sample_proposal, sample_date_range
    ):
        """Test full proposal review workflow."""
        spec = AudienceSpec(segments=sample_proposal.targeting.get("segments", []))
        await audience_agent.validate_audience(spec)
        
        avails = await avails_agent.check_avails(
            product_id=sample_proposal.product_id,
            dates=sample_date_range,
            impressions_requested=sample_proposal.requested_impressions,
        )
        
        review = await proposal_agent.review_proposal(
            proposal=sample_proposal,
            floor_cpm=10.0,
            max_impressions=avails.available_impressions,
        )
        
        if review.status == ProposalStatus.NEEDS_ADJUSTMENT:
            counter = await proposal_agent.build_counter_offer(
                original=sample_proposal,
                constraints=review.suggested_adjustments,
            )
            assert counter is not None
