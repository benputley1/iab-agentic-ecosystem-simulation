"""Tests for the L1 Inventory Manager.

Tests the InventoryManager orchestrator agent including:
- Deal evaluation logic
- Yield optimization
- Cross-sell identification
- L2 delegation
"""

import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.agents.seller.l1_inventory_manager import (
    InventoryManager,
    create_inventory_manager,
    TIER_DISCOUNTS,
)
from src.agents.seller.models import (
    AudienceSpec,
    BuyerTier,
    ChannelType,
    CounterOffer,
    CrossSellOpportunity,
    Deal,
    DealAction,
    DealDecision,
    DealRequest,
    DealTypeEnum,
    InventoryPortfolio,
    Product,
    Task,
    TaskResult,
    YieldStrategy,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_product() -> Product:
    """Create a sample product for testing."""
    return Product(
        product_id="pub-001-display-001",
        name="Premium Display",
        description="High-quality display inventory",
        channel=ChannelType.DISPLAY,
        base_cpm=15.0,
        floor_cpm=10.0,
        daily_impressions=500000,
        supported_deal_types=[DealTypeEnum.PROGRAMMATIC_GUARANTEED, DealTypeEnum.PRIVATE_MARKETPLACE],
        audience_segments=["auto_intenders", "tech_enthusiasts"],
        content_categories=["technology", "news"],
    )


@pytest.fixture
def sample_portfolio(sample_product: Product) -> InventoryPortfolio:
    """Create a sample portfolio for testing."""
    video_product = Product(
        product_id="pub-001-video-001",
        name="Premium Video",
        description="High-quality video inventory",
        channel=ChannelType.VIDEO,
        base_cpm=25.0,
        floor_cpm=18.0,
        daily_impressions=200000,
        supported_deal_types=[DealTypeEnum.PRIVATE_MARKETPLACE],
    )
    
    ctv_product = Product(
        product_id="pub-001-ctv-001",
        name="CTV Premium",
        description="Connected TV inventory",
        channel=ChannelType.CTV,
        base_cpm=40.0,
        floor_cpm=30.0,
        daily_impressions=100000,
        supported_deal_types=[DealTypeEnum.PROGRAMMATIC_GUARANTEED],
    )
    
    return InventoryPortfolio(
        seller_id="pub-001",
        products=[sample_product, video_product, ctv_product],
        total_avails={
            "display": 500000,
            "video": 200000,
            "ctv": 100000,
        },
        floor_prices={
            "pub-001-display-001": 10.0,
            "pub-001-video-001": 18.0,
            "pub-001-ctv-001": 30.0,
        },
        fill_rate={
            "display": 0.75,
            "video": 0.60,
            "ctv": 0.85,
        },
        avg_cpm={
            "display": 12.0,
            "video": 22.0,
            "ctv": 38.0,
        },
        revenue_ytd=500000.0,
    )


@pytest.fixture
def sample_deal_request() -> DealRequest:
    """Create a sample deal request for testing."""
    return DealRequest(
        request_id="req-001",
        buyer_id="buyer-agency-001",
        buyer_tier=BuyerTier.AGENCY,
        product_id="pub-001-display-001",
        impressions=100000,
        max_cpm=12.0,
        deal_type=DealTypeEnum.PRIVATE_MARKETPLACE,
        flight_dates=(date.today(), date.today() + timedelta(days=30)),
        audience_spec=AudienceSpec(
            segments=["auto_intenders"],
            geo_targets=["US"],
        ),
        buyer_name="Test Agency",
        campaign_name="Q4 Campaign",
    )


@pytest.fixture
def sample_deal() -> Deal:
    """Create a sample active deal for testing."""
    return Deal(
        deal_id="deal-001",
        request_id="req-001",
        buyer_id="buyer-agency-001",
        buyer_tier=BuyerTier.AGENCY,
        product_id="pub-001-display-001",
        agreed_cpm=11.5,
        impressions=100000,
        deal_type=DealTypeEnum.PRIVATE_MARKETPLACE,
        flight_dates=(date.today(), date.today() + timedelta(days=30)),
        impressions_delivered=25000,
        revenue=287.5,
    )


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_content = MagicMock()
    mock_content.text = json.dumps({
        "action": "accept",
        "recommended_cpm": 11.5,
        "recommended_impressions": 100000,
        "reasoning": "Good deal from valuable agency partner at acceptable CPM",
        "confidence": 0.85,
        "counter_offer": None,
    })
    
    mock_usage = MagicMock()
    mock_usage.input_tokens = 500
    mock_usage.output_tokens = 100
    
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage = mock_usage
    
    return mock_response


# -----------------------------------------------------------------------------
# Model Tests
# -----------------------------------------------------------------------------


class TestModels:
    """Test data model functionality."""
    
    def test_deal_request_properties(self, sample_deal_request: DealRequest):
        """Test DealRequest computed properties."""
        assert sample_deal_request.duration_days == 31
        assert sample_deal_request.total_value == 1200.0  # 100k imps * $12 CPM / 1000
    
    def test_deal_properties(self, sample_deal: Deal):
        """Test Deal computed properties."""
        assert sample_deal.is_active is True
        assert sample_deal.remaining_impressions == 75000
        assert sample_deal.fill_rate == 0.25
    
    def test_product_is_premium(self, sample_product: Product):
        """Test Product premium detection."""
        # floor >= 15 or base >= 25 is premium
        # sample_product has floor=10, base=15, so NOT premium
        assert sample_product.is_premium is False
        
        # Create a premium product
        premium = Product(
            product_id="premium",
            name="Premium",
            description="",
            channel=ChannelType.CTV,
            base_cpm=30.0,
            floor_cpm=20.0,  # >= 15
            daily_impressions=100000,
        )
        assert premium.is_premium is True
        
        non_premium = Product(
            product_id="test",
            name="Standard",
            description="",
            channel=ChannelType.DISPLAY,
            base_cpm=8.0,
            floor_cpm=5.0,
            daily_impressions=1000000,
        )
        assert non_premium.is_premium is False
    
    def test_audience_spec_serialization(self):
        """Test AudienceSpec to/from dict."""
        spec = AudienceSpec(
            segments=["auto", "tech"],
            demographics={"age": "25-54"},
            geo_targets=["US", "UK"],
        )
        
        data = spec.to_dict()
        assert data["segments"] == ["auto", "tech"]
        assert data["demographics"]["age"] == "25-54"
        
        restored = AudienceSpec.from_dict(data)
        assert restored.segments == spec.segments
        assert restored.demographics == spec.demographics
    
    def test_portfolio_get_channel_products(self, sample_portfolio: InventoryPortfolio):
        """Test getting products by channel."""
        display_products = sample_portfolio.get_channel_products(ChannelType.DISPLAY)
        assert len(display_products) == 1
        assert display_products[0].product_id == "pub-001-display-001"
        
        video_products = sample_portfolio.get_channel_products(ChannelType.VIDEO)
        assert len(video_products) == 1
    
    def test_portfolio_total_impressions(self, sample_portfolio: InventoryPortfolio):
        """Test total daily impressions calculation."""
        total = sample_portfolio.get_total_daily_impressions()
        assert total == 800000  # 500k + 200k + 100k


# -----------------------------------------------------------------------------
# Inventory Manager Tests
# -----------------------------------------------------------------------------


class TestInventoryManagerInit:
    """Test InventoryManager initialization."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_init_with_portfolio(self, sample_portfolio: InventoryPortfolio):
        """Test initialization with a portfolio."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        assert manager.seller_id == "pub-001"
        assert manager.agent_id == "inventory-manager-pub-001"
        assert manager.portfolio == sample_portfolio
        assert manager.model == "claude-opus-4"
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_init_creates_empty_portfolio(self):
        """Test initialization creates empty portfolio if not provided."""
        manager = InventoryManager(seller_id="pub-002")
        
        assert manager.portfolio.seller_id == "pub-002"
        assert len(manager.portfolio.products) == 0
    
    def test_init_without_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                InventoryManager(seller_id="pub-001")


class TestDealEvaluation:
    """Test deal evaluation functionality."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_evaluate_deal_accept(
        self,
        sample_portfolio: InventoryPortfolio,
        sample_deal_request: DealRequest,
        mock_anthropic_response,
    ):
        """Test deal evaluation that accepts."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        with patch.object(manager.client.messages, "create", return_value=mock_anthropic_response):
            decision = await manager.evaluate_deal_request(sample_deal_request)
        
        assert decision.action == DealAction.ACCEPT
        assert decision.price == 11.5
        assert decision.impressions == 100000
        assert decision.confidence == 0.85
        assert "agency" in decision.reasoning.lower()
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_evaluate_deal_reject_unknown_product(
        self,
        sample_portfolio: InventoryPortfolio,
    ):
        """Test deal evaluation rejects unknown product."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        request = DealRequest(
            request_id="req-unknown",
            buyer_id="buyer-001",
            buyer_tier=BuyerTier.PUBLIC,
            product_id="unknown-product",
            impressions=50000,
            max_cpm=10.0,
            deal_type=DealTypeEnum.OPEN_AUCTION,
            flight_dates=(date.today(), date.today() + timedelta(days=7)),
            audience_spec=AudienceSpec(),
        )
        
        decision = await manager.evaluate_deal_request(request)
        
        assert decision.action == DealAction.REJECT
        assert "not found" in decision.reasoning.lower()
        assert decision.confidence == 1.0
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_evaluate_deal_counter_offer(
        self,
        sample_portfolio: InventoryPortfolio,
        sample_deal_request: DealRequest,
    ):
        """Test deal evaluation with counter offer."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        counter_response = MagicMock()
        counter_response.content = [MagicMock(text=json.dumps({
            "action": "counter",
            "recommended_cpm": 13.0,
            "recommended_impressions": 80000,
            "reasoning": "Offered CPM below floor, countering with acceptable terms",
            "confidence": 0.75,
            "counter_offer": {
                "suggested_cpm": 13.0,
                "suggested_impressions": 80000,
                "alternative_products": ["pub-001-video-001"],
                "reasoning": "Consider adding video for package deal",
            },
        }))]
        counter_response.usage = MagicMock(input_tokens=500, output_tokens=150)
        
        with patch.object(manager.client.messages, "create", return_value=counter_response):
            decision = await manager.evaluate_deal_request(sample_deal_request)
        
        assert decision.action == DealAction.COUNTER
        assert decision.counter_offer is not None
        assert decision.counter_offer.suggested_cpm == 13.0
        assert "pub-001-video-001" in decision.counter_offer.alternative_products


class TestYieldOptimization:
    """Test yield optimization functionality."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_optimize_yield(self, sample_portfolio: InventoryPortfolio):
        """Test yield optimization."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        yield_response = MagicMock()
        yield_response.content = [MagicMock(text=json.dumps({
            "floor_adjustments": {
                "pub-001-display-001": 1.05,
                "pub-001-video-001": 1.10,
            },
            "allocation_priorities": ["ctv", "video", "display"],
            "pacing_recommendations": {
                "pub-001-display-001": "steady",
                "pub-001-ctv-001": "aggressive",
            },
            "insights": [
                "CTV fill rate at 85% suggests room to raise floors",
                "Video underperforming - consider promotional pricing",
            ],
            "expected_revenue_lift": 5.0,
            "expected_fill_rate_change": -2.0,
            "confidence": 0.80,
        }))]
        yield_response.usage = MagicMock(input_tokens=800, output_tokens=200)
        
        with patch.object(manager.client.messages, "create", return_value=yield_response):
            strategy = await manager.optimize_yield()
        
        assert strategy.floor_adjustments["pub-001-display-001"] == 1.05
        assert strategy.allocation_priorities[0] == "ctv"
        assert "CTV fill rate" in strategy.insights[0]
        assert strategy.expected_revenue_lift == 5.0
        assert strategy.confidence == 0.80


class TestCrossSell:
    """Test cross-sell identification functionality."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_identify_cross_sell(
        self,
        sample_portfolio: InventoryPortfolio,
        sample_deal: Deal,
    ):
        """Test cross-sell opportunity identification."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        manager.add_deal(sample_deal)
        
        cross_sell_response = MagicMock()
        cross_sell_response.content = [MagicMock(text=json.dumps({
            "opportunities": [
                {
                    "recommended_product_id": "pub-001-video-001",
                    "recommended_channel": "video",
                    "suggested_impressions": 50000,
                    "suggested_cpm": 22.0,
                    "estimated_value": 1100.0,
                    "confidence": 0.75,
                    "reasoning": "Display buyer likely to benefit from video for brand campaign",
                },
                {
                    "recommended_product_id": "pub-001-ctv-001",
                    "recommended_channel": "ctv",
                    "suggested_impressions": 25000,
                    "suggested_cpm": 35.0,
                    "estimated_value": 875.0,
                    "confidence": 0.60,
                    "reasoning": "Premium CTV could extend reach to home viewers",
                },
            ],
            "top_recommendation": "pub-001-video-001",
            "approach_strategy": "Present as package deal with volume discount",
        }))]
        cross_sell_response.usage = MagicMock(input_tokens=600, output_tokens=250)
        
        with patch.object(manager.client.messages, "create", return_value=cross_sell_response):
            opportunities = await manager.identify_cross_sell(sample_deal)
        
        assert len(opportunities) == 2
        assert opportunities[0].recommended_product_id == "pub-001-video-001"
        assert opportunities[0].recommended_channel == ChannelType.VIDEO
        assert opportunities[0].estimated_value == 1100.0
        assert opportunities[0].confidence == 0.75


class TestDelegation:
    """Test L2 specialist delegation."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_delegate_to_channel_no_specialist(self, sample_portfolio: InventoryPortfolio):
        """Test delegation when no specialist is registered."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        task = Task(
            task_id="task-001",
            task_type="pricing",
            description="Calculate optimal floor price",
        )
        
        result = await manager.delegate_to_channel("display", task)
        
        assert result.success is False
        assert "No L2 specialist" in result.error
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_delegate_to_channel_with_specialist(self, sample_portfolio: InventoryPortfolio):
        """Test delegation to a registered specialist."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        # Create mock specialist
        mock_specialist = MagicMock()
        mock_specialist.process_task = AsyncMock(return_value={"floor_price": 12.0})
        
        manager.register_specialist("display", mock_specialist)
        
        task = Task(
            task_id="task-002",
            task_type="pricing",
            description="Calculate optimal floor price",
            data={"product_id": "pub-001-display-001"},
        )
        
        result = await manager.delegate_to_channel("display", task)
        
        assert result.success is True
        assert result.data["floor_price"] == 12.0
        mock_specialist.process_task.assert_called_once_with(task)


class TestMetrics:
    """Test metrics and tracking."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_get_metrics(self, sample_portfolio: InventoryPortfolio):
        """Test metrics retrieval."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        metrics = manager.get_metrics()
        
        assert metrics["agent_id"] == "inventory-manager-pub-001"
        assert metrics["agent_type"] == "inventory-manager"
        assert metrics["model"] == "claude-opus-4"
        assert metrics["total_decisions"] == 0
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_deal_tracking(self, sample_portfolio: InventoryPortfolio, sample_deal: Deal):
        """Test deal tracking functionality."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        manager.add_deal(sample_deal)
        
        deals = manager.get_active_deals()
        assert len(deals) == 1
        assert deals[0].deal_id == "deal-001"


class TestParsing:
    """Test LLM response parsing."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_parse_deal_decision_valid(self, sample_portfolio: InventoryPortfolio):
        """Test parsing valid deal decision response."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        response = json.dumps({
            "action": "accept",
            "recommended_cpm": 15.0,
            "recommended_impressions": 50000,
            "reasoning": "Good deal",
            "confidence": 0.9,
            "counter_offer": None,
        })
        
        decision = manager._parse_deal_decision("req-test", response)
        
        assert decision.action == DealAction.ACCEPT
        assert decision.price == 15.0
        assert decision.confidence == 0.9
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_parse_deal_decision_invalid(self, sample_portfolio: InventoryPortfolio):
        """Test parsing invalid response falls back to reject."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        response = "This is not valid JSON"
        
        decision = manager._parse_deal_decision("req-test", response)
        
        assert decision.action == DealAction.REJECT
        assert decision.confidence == 0.0
        assert "Failed to parse" in decision.reasoning
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_parse_yield_strategy_valid(self, sample_portfolio: InventoryPortfolio):
        """Test parsing valid yield strategy response."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        response = json.dumps({
            "floor_adjustments": {"product-1": 1.1},
            "allocation_priorities": ["ctv", "video"],
            "pacing_recommendations": {"product-1": "aggressive"},
            "insights": ["Increase floors"],
            "expected_revenue_lift": 8.0,
            "expected_fill_rate_change": -1.5,
            "confidence": 0.85,
        })
        
        strategy = manager._parse_yield_strategy(response)
        
        assert strategy.floor_adjustments["product-1"] == 1.1
        assert strategy.expected_revenue_lift == 8.0
        assert strategy.confidence == 0.85


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for full workflows."""
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_full_deal_workflow(
        self,
        sample_portfolio: InventoryPortfolio,
        sample_deal_request: DealRequest,
        mock_anthropic_response,
    ):
        """Test full deal evaluation workflow."""
        manager = InventoryManager(
            seller_id="pub-001",
            portfolio=sample_portfolio,
        )
        
        # Evaluate deal
        with patch.object(manager.client.messages, "create", return_value=mock_anthropic_response):
            decision = await manager.process_request(sample_deal_request)
        
        # Check decision recorded in context (use _context which returns the tracker)
        assert len(manager._context.decisions) == 1
        assert "deal_evaluation" in manager._context.decisions[0]["action"]
        
        # If accepted, add deal
        if decision.action == DealAction.ACCEPT:
            deal = Deal(
                deal_id="deal-new",
                request_id=sample_deal_request.request_id,
                buyer_id=sample_deal_request.buyer_id,
                buyer_tier=sample_deal_request.buyer_tier,
                product_id=sample_deal_request.product_id,
                agreed_cpm=decision.price,
                impressions=decision.impressions,
                deal_type=sample_deal_request.deal_type,
                flight_dates=sample_deal_request.flight_dates,
            )
            manager.add_deal(deal)
            
            assert len(manager.get_active_deals()) == 1
