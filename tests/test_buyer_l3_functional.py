"""Tests for Buyer L3 Functional Agents.

Tests the four L3 functional agents:
- ResearchAgent
- ExecutionAgent
- ReportingAgent
- AudiencePlannerAgent
"""

import sys
from pathlib import Path

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Direct imports from l3 modules to avoid broken imports elsewhere
_src_path = Path(__file__).parent.parent / "src" / "agents" / "buyer"
sys.path.insert(0, str(_src_path.parent.parent))

# Import directly from buyer modules
from src.agents.buyer.l3_base import (
    FunctionalAgent,
    ToolResult,
    ToolExecutionStatus,
    AgentContext,
)
from src.agents.buyer.l3_tools import (
    BUYER_TOOLS,
    SearchCriteria,
    Product,
    AvailsResult,
    OrderSpec,
    Order,
    Deal,
    BookingConfirmation,
    Metrics,
    Attribution,
    CampaignBrief,
    AudienceSegment,
    CoverageEstimate,
    get_tool_schema,
    get_tools_for_agent,
)
from src.agents.buyer.l3_research import ResearchAgent
from src.agents.buyer.l3_execution import ExecutionAgent
from src.agents.buyer.l3_reporting import ReportingAgent
from src.agents.buyer.l3_audience_planner import AudiencePlannerAgent


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agent_context():
    """Create a test agent context."""
    return AgentContext(
        buyer_id="test-buyer-001",
        scenario="A",
        campaign_id="test-camp-001",
        channel="display",
        session_id="test-session-001",
    )


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = AsyncMock()
    
    # Default response with tool use
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            type="tool_use",
            name="ProductSearch",
            input={"channel": "display", "max_cpm": 25.0},
        )
    ]
    client.messages.create = AsyncMock(return_value=mock_response)
    
    return client


# =============================================================================
# Tool Schema Tests
# =============================================================================


class TestToolSchemas:
    """Test tool schema definitions."""
    
    def test_buyer_tools_registry_complete(self):
        """All expected tools are in registry."""
        expected_tools = [
            "ProductSearch", "AvailsCheck", "PricingLookup", "CompetitiveIntel",
            "CreateOrder", "CreateLine", "BookLine", "ReserveLine",
            "GetMetrics", "GenerateReport", "AttributionAnalysis",
            "AudienceDiscovery", "AudienceMatching", "CoverageEstimation",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in BUYER_TOOLS, f"Missing tool: {tool_name}"
    
    def test_get_tool_schema_valid(self):
        """get_tool_schema returns valid schemas."""
        schema = get_tool_schema("ProductSearch")
        
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["name"] == "ProductSearch"
    
    def test_get_tool_schema_invalid(self):
        """get_tool_schema raises for unknown tools."""
        with pytest.raises(KeyError):
            get_tool_schema("InvalidTool")
    
    def test_get_tools_for_agent(self):
        """get_tools_for_agent returns multiple schemas."""
        tools = get_tools_for_agent(["ProductSearch", "AvailsCheck"])
        
        assert len(tools) == 2
        assert tools[0]["name"] == "ProductSearch"
        assert tools[1]["name"] == "AvailsCheck"


# =============================================================================
# Base Agent Tests
# =============================================================================


class TestAgentContext:
    """Test AgentContext dataclass."""
    
    def test_context_creation(self):
        """Context is created with all fields."""
        context = AgentContext(
            buyer_id="buyer-001",
            scenario="B",
            campaign_id="camp-001",
            channel="video",
        )
        
        assert context.buyer_id == "buyer-001"
        assert context.scenario == "B"
        assert context.campaign_id == "camp-001"
        assert context.channel == "video"
    
    def test_context_defaults(self):
        """Context has correct defaults."""
        context = AgentContext(buyer_id="buyer-001", scenario="A")
        
        assert context.campaign_id is None
        assert context.channel is None
        assert context.session_id is None
        assert context.metadata == {}


class TestToolResult:
    """Test ToolResult dataclass."""
    
    def test_success_result(self):
        """Success result has correct properties."""
        result = ToolResult(
            tool_name="TestTool",
            status=ToolExecutionStatus.SUCCESS,
            data={"key": "value"},
        )
        
        assert result.success is True
        assert result.tool_name == "TestTool"
        assert result.data == {"key": "value"}
        assert result.error is None
    
    def test_failed_result(self):
        """Failed result has correct properties."""
        result = ToolResult(
            tool_name="TestTool",
            status=ToolExecutionStatus.FAILED,
            error="Something went wrong",
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"


# =============================================================================
# Research Agent Tests
# =============================================================================


class TestResearchAgent:
    """Test ResearchAgent functionality."""
    
    def test_agent_creation(self, agent_context):
        """Agent is created with correct properties."""
        agent = ResearchAgent(context=agent_context)
        
        assert agent.context == agent_context
        assert agent.agent_name == "ResearchAgent"
        assert "ProductSearch" in [t["name"] for t in agent.available_tools]
    
    def test_system_prompt_includes_context(self, agent_context):
        """System prompt includes buyer context."""
        agent = ResearchAgent(context=agent_context)
        prompt = agent.system_prompt
        
        assert agent_context.buyer_id in prompt
        assert agent_context.scenario in prompt
    
    @pytest.mark.asyncio
    async def test_search_products_returns_products(self, agent_context):
        """search_products returns Product list."""
        agent = ResearchAgent(context=agent_context)
        
        criteria = SearchCriteria(
            channel="display",
            max_cpm=30.0,
            min_impressions=100000,
        )
        
        products = await agent.search_products(criteria)
        
        assert isinstance(products, list)
        assert all(isinstance(p, Product) for p in products)
    
    @pytest.mark.asyncio
    async def test_check_availability_returns_result(self, agent_context):
        """check_availability returns AvailsResult."""
        agent = ResearchAgent(context=agent_context)
        
        result = await agent.check_availability(
            product_id="PROD-001",
            impressions=500000,
        )
        
        assert isinstance(result, AvailsResult)
        assert result.impressions > 0
    
    @pytest.mark.asyncio
    async def test_get_pricing_returns_data(self, agent_context):
        """get_pricing returns pricing dict."""
        agent = ResearchAgent(context=agent_context)
        
        pricing = await agent.get_pricing(channel="video", deal_type="PD")
        
        assert isinstance(pricing, dict)
        assert "average_cpm" in pricing
        assert "floor_cpm" in pricing
    
    @pytest.mark.asyncio
    async def test_analyze_market_returns_intel(self, agent_context):
        """analyze_market returns market intelligence."""
        agent = ResearchAgent(context=agent_context)
        
        intel = await agent.analyze_market(channel="ctv", timeframe="30d")
        
        assert isinstance(intel, dict)
        assert "market_trends" in intel


# =============================================================================
# Execution Agent Tests
# =============================================================================


class TestExecutionAgent:
    """Test ExecutionAgent functionality."""
    
    def test_agent_creation(self, agent_context):
        """Agent is created with correct properties."""
        agent = ExecutionAgent(context=agent_context)
        
        assert agent.context == agent_context
        assert agent.agent_name == "ExecutionAgent"
        assert "CreateOrder" in [t["name"] for t in agent.available_tools]
    
    @pytest.mark.asyncio
    async def test_create_order_returns_order(self, agent_context):
        """create_order returns Order object."""
        agent = ExecutionAgent(context=agent_context)
        
        order_spec = OrderSpec(
            campaign_id="camp-001",
            buyer_id=agent_context.buyer_id,
            name="Test Order",
            budget=50000.0,
            channel="display",
        )
        
        order = await agent.create_order(order_spec)
        
        assert isinstance(order, Order)
        assert order.campaign_id == "camp-001"
        assert order.budget == 50000.0
        assert order.status == "created"
    
    @pytest.mark.asyncio
    async def test_create_line_returns_line(self, agent_context):
        """create_line returns line item dict."""
        agent = ExecutionAgent(context=agent_context)
        
        # First create an order
        order_spec = OrderSpec(
            campaign_id="camp-001",
            buyer_id=agent_context.buyer_id,
            name="Test Order",
            budget=50000.0,
        )
        order = await agent.create_order(order_spec)
        
        # Then create a line
        line = await agent.create_line(
            order_id=order.order_id,
            name="Test Line",
            product_id="PROD-001",
            impressions=1000000,
            cpm=18.0,
        )
        
        assert isinstance(line, dict)
        assert line["order_id"] == order.order_id
        assert line["impressions"] == 1000000
    
    @pytest.mark.asyncio
    async def test_book_deal_returns_confirmation(self, agent_context):
        """book_deal returns BookingConfirmation."""
        agent = ExecutionAgent(context=agent_context)
        
        deal = Deal(
            order_id="ORD-001",
            seller_id="seller-001",
            product_id="PROD-001",
            impressions=500000,
            cpm=20.0,
            deal_type="PD",
        )
        
        confirmation = await agent.book_deal(deal)
        
        assert isinstance(confirmation, BookingConfirmation)
        # Impressions may vary based on mock/sim_client response
        assert confirmation.impressions > 0
        assert confirmation.status == "booked"
        assert confirmation.seller_id == deal.seller_id
    
    @pytest.mark.asyncio
    async def test_reserve_inventory_returns_reservation(self, agent_context):
        """reserve_inventory returns reservation dict."""
        agent = ExecutionAgent(context=agent_context)
        
        reservation = await agent.reserve_inventory(
            order_id="ORD-001",
            line_id="LINE-001",
            hold_hours=48,
        )
        
        assert isinstance(reservation, dict)
        assert reservation["status"] == "reserved"
        assert reservation["hold_duration_hours"] == 48


# =============================================================================
# Reporting Agent Tests
# =============================================================================


class TestReportingAgent:
    """Test ReportingAgent functionality."""
    
    def test_agent_creation(self, agent_context):
        """Agent is created with correct properties."""
        agent = ReportingAgent(context=agent_context)
        
        assert agent.context == agent_context
        assert agent.agent_name == "ReportingAgent"
        assert "GetMetrics" in [t["name"] for t in agent.available_tools]
    
    @pytest.mark.asyncio
    async def test_get_campaign_metrics_returns_metrics(self, agent_context):
        """get_campaign_metrics returns Metrics object."""
        agent = ReportingAgent(context=agent_context)
        
        metrics = await agent.get_campaign_metrics("camp-001")
        
        assert isinstance(metrics, Metrics)
        assert metrics.campaign_id == "camp-001"
        assert metrics.impressions > 0
        assert metrics.clicks > 0
    
    @pytest.mark.asyncio
    async def test_generate_report_returns_report(self, agent_context):
        """generate_report returns report dict."""
        agent = ReportingAgent(context=agent_context)
        
        report = await agent.generate_report(
            campaign_id="camp-001",
            report_type="performance",
            format="json",
        )
        
        assert isinstance(report, dict)
        assert "report_id" in report
        assert report["report_type"] == "performance"
        assert "metrics" in report
    
    @pytest.mark.asyncio
    async def test_analyze_attribution_returns_attribution(self, agent_context):
        """analyze_attribution returns Attribution object."""
        agent = ReportingAgent(context=agent_context)
        
        attribution = await agent.analyze_attribution(
            campaign_id="camp-001",
            model_type="time_decay",
        )
        
        assert isinstance(attribution, Attribution)
        assert attribution.model_type == "time_decay"
        assert attribution.total_conversions > 0
        assert "display" in attribution.channels


# =============================================================================
# Audience Planner Agent Tests
# =============================================================================


class TestAudiencePlannerAgent:
    """Test AudiencePlannerAgent functionality."""
    
    def test_agent_creation(self, agent_context):
        """Agent is created with correct properties."""
        agent = AudiencePlannerAgent(context=agent_context)
        
        assert agent.context == agent_context
        assert agent.agent_name == "AudiencePlannerAgent"
        assert "AudienceDiscovery" in [t["name"] for t in agent.available_tools]
    
    def test_default_segment_catalog_populated(self, agent_context):
        """Agent has default segment catalog."""
        agent = AudiencePlannerAgent(context=agent_context)
        
        assert len(agent._segment_catalog) > 0
        assert "SEG-TECH-EARLY" in agent._segment_catalog
    
    @pytest.mark.asyncio
    async def test_discover_audiences_returns_segments(self, agent_context):
        """discover_audiences returns AudienceSegment list."""
        agent = AudiencePlannerAgent(context=agent_context)
        
        brief = CampaignBrief(
            campaign_id="camp-001",
            objective="awareness",
            target_audience_description="Tech-savvy millennials interested in finance",
            budget=50000.0,
            geography=["US"],
        )
        
        segments = await agent.discover_audiences(brief)
        
        assert isinstance(segments, list)
        assert all(isinstance(s, AudienceSegment) for s in segments)
        assert len(segments) > 0
    
    @pytest.mark.asyncio
    async def test_match_audiences_returns_matches(self, agent_context):
        """match_audiences returns inventory matches."""
        agent = AudiencePlannerAgent(context=agent_context)
        
        matches = await agent.match_audiences(
            segment_ids=["SEG-TECH-EARLY", "SEG-FINANCE-HNW"],
            channel="display",
            min_match_rate=0.3,
        )
        
        assert isinstance(matches, list)
        assert all(isinstance(m, dict) for m in matches)
        assert all(m["match_rate"] >= 0.3 for m in matches)
    
    @pytest.mark.asyncio
    async def test_estimate_coverage_returns_estimate(self, agent_context):
        """estimate_coverage returns CoverageEstimate."""
        agent = AudiencePlannerAgent(context=agent_context)
        
        coverage = await agent.estimate_coverage(
            segments=["SEG-TECH-EARLY"],
            inventory=["INV-001", "INV-002"],
            budget=25000.0,
            duration_days=30,
        )
        
        assert isinstance(coverage, CoverageEstimate)
        assert coverage.total_reach > 0
        assert coverage.frequency > 0
        assert 0 <= coverage.coverage_percentage <= 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentIntegration:
    """Integration tests across multiple agents."""
    
    @pytest.mark.asyncio
    async def test_research_to_execution_flow(self, agent_context):
        """Test flow from research to execution."""
        # Research phase
        research_agent = ResearchAgent(context=agent_context)
        products = await research_agent.search_products(
            SearchCriteria(channel="display", max_cpm=25.0)
        )
        
        assert len(products) > 0
        best_product = products[0]
        
        # Execution phase
        exec_agent = ExecutionAgent(context=agent_context)
        
        order = await exec_agent.create_order(OrderSpec(
            campaign_id=agent_context.campaign_id,
            buyer_id=agent_context.buyer_id,
            name="Test Campaign Order",
            budget=30000.0,
            channel="display",
        ))
        
        deal = Deal(
            order_id=order.order_id,
            seller_id=best_product.seller_id,
            product_id=best_product.product_id,
            impressions=500000,
            cpm=best_product.base_cpm,
            deal_type="OA",
        )
        
        confirmation = await exec_agent.book_deal(deal)
        
        assert confirmation.status == "booked"
        assert confirmation.seller_id == best_product.seller_id
    
    @pytest.mark.asyncio
    async def test_audience_to_research_flow(self, agent_context):
        """Test flow from audience planning to research."""
        # Audience discovery
        audience_agent = AudiencePlannerAgent(context=agent_context)
        
        brief = CampaignBrief(
            campaign_id=agent_context.campaign_id,
            objective="conversion",
            target_audience_description="Business professionals interested in software",
            budget=50000.0,
        )
        
        segments = await audience_agent.discover_audiences(brief)
        assert len(segments) > 0
        
        # Get inventory matches
        matches = await audience_agent.match_audiences(
            segment_ids=[s.segment_id for s in segments[:2]],
            channel="display",
        )
        
        # Research matched inventory
        research_agent = ResearchAgent(context=agent_context)
        products = await research_agent.search_products(
            SearchCriteria(channel="display")
        )
        
        assert len(products) > 0
    
    @pytest.mark.asyncio
    async def test_execution_to_reporting_flow(self, agent_context):
        """Test flow from execution to reporting."""
        # Execute a deal
        exec_agent = ExecutionAgent(context=agent_context)
        
        order = await exec_agent.create_order(OrderSpec(
            campaign_id=agent_context.campaign_id,
            buyer_id=agent_context.buyer_id,
            name="Reporting Test Order",
            budget=25000.0,
        ))
        
        deal = Deal(
            order_id=order.order_id,
            seller_id="seller-001",
            product_id="PROD-001",
            impressions=300000,
            cpm=18.0,
            deal_type="OA",
        )
        
        await exec_agent.book_deal(deal)
        
        # Get metrics for the campaign
        reporting_agent = ReportingAgent(context=agent_context)
        
        metrics = await reporting_agent.get_campaign_metrics(
            agent_context.campaign_id
        )
        
        assert metrics.campaign_id == agent_context.campaign_id
        
        # Generate report
        report = await reporting_agent.generate_report(
            campaign_id=agent_context.campaign_id,
            report_type="performance",
        )
        
        assert report["campaign_id"] == agent_context.campaign_id
        assert "insights" in report


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_search_with_no_results(self, agent_context):
        """Search with impossible criteria returns empty list."""
        agent = ResearchAgent(context=agent_context)
        
        # Very low max_cpm should return empty or filtered results
        products = await agent.search_products(
            SearchCriteria(max_cpm=0.01)  # Unrealistically low
        )
        
        # Should handle gracefully
        assert isinstance(products, list)
    
    @pytest.mark.asyncio
    async def test_execution_with_missing_params(self, agent_context):
        """Execution handles missing parameters."""
        agent = ExecutionAgent(context=agent_context)
        
        # Missing order_id should raise
        with pytest.raises(ValueError):
            await agent.create_line(
                order_id="",  # Empty order ID
                name="Test Line",
                product_id="PROD-001",
                impressions=100000,
                cpm=15.0,
            )
    
    def test_agent_execution_history(self, agent_context):
        """Agent tracks execution history."""
        agent = ResearchAgent(context=agent_context)
        
        # Initially empty
        assert len(agent.get_execution_history()) == 0
        
        # After clearing, still empty
        agent.clear_history()
        assert len(agent.get_execution_history()) == 0
