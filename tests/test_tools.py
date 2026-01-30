"""Tests for MCP tool integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools import Tool, ToolResult, ToolSpec, ToolRegistry, MCPClient
from src.tools.base import ToolStatus
from src.tools.mcp_client import MCPClientConfig, MCPError
from src.tools.buyer import (
    ProductSearchTool,
    AvailsCheckTool,
    CreateOrderTool,
    DiscoverInventoryTool,
    AudienceDiscoveryTool,
    ALL_BUYER_TOOLS,
)
from src.tools.seller import (
    PriceCalculatorTool,
    FloorManagerTool,
    ProposalGeneratorTool,
    ListAdUnitsTool,
    AudienceValidationTool,
    ALL_SELLER_TOOLS,
)


# ============================================================================
# Tool Base Class Tests
# ============================================================================

class TestToolResult:
    """Tests for ToolResult."""
    
    def test_ok_result(self):
        """Test successful result creation."""
        result = ToolResult.ok({"key": "value"}, extra="metadata")
        
        assert result.success is True
        assert result.status == ToolStatus.SUCCESS
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.metadata["extra"] == "metadata"
    
    def test_fail_result(self):
        """Test failed result creation."""
        result = ToolResult.fail("Something went wrong", code=500)
        
        assert result.success is False
        assert result.status == ToolStatus.ERROR
        assert result.error == "Something went wrong"
        assert result.data is None
        assert result.metadata["code"] == 500
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = ToolResult.ok({"test": 123})
        d = result.to_dict()
        
        assert d["status"] == "success"
        assert d["data"] == {"test": 123}
        assert d["error"] is None


class TestToolSpec:
    """Tests for ToolSpec."""
    
    def test_to_dict(self):
        """Test spec conversion to dict."""
        spec = ToolSpec(
            name="TestTool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            }
        )
        
        d = spec.to_dict()
        assert d["name"] == "TestTool"
        assert d["description"] == "A test tool"
        assert "properties" in d["input_schema"]


class ConcreteTestTool(Tool):
    """Concrete implementation for testing."""
    
    name = "TestTool"
    description = "A tool for testing"
    parameters = {
        "required_arg": {"type": "string", "required": True},
        "optional_arg": {"type": "integer"},
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult.ok({"received": kwargs})


class TestToolBase:
    """Tests for Tool base class."""
    
    def test_tool_initialization(self):
        """Test tool can be instantiated."""
        tool = ConcreteTestTool()
        assert tool.name == "TestTool"
        assert tool.description == "A tool for testing"
    
    def test_get_spec(self):
        """Test getting tool specification."""
        tool = ConcreteTestTool()
        spec = tool.get_spec()
        
        assert isinstance(spec, ToolSpec)
        assert spec.name == "TestTool"
        assert "required_arg" in spec.input_schema["properties"]
    
    def test_validate_args_missing_required(self):
        """Test validation catches missing required args."""
        tool = ConcreteTestTool()
        errors = tool.validate_args(optional_arg=123)
        
        assert len(errors) == 1
        assert "required_arg" in errors[0]
    
    def test_validate_args_invalid_type(self):
        """Test validation catches type errors."""
        tool = ConcreteTestTool()
        errors = tool.validate_args(required_arg="ok", optional_arg="not an int")
        
        assert any("type" in e for e in errors)
    
    def test_validate_args_valid(self):
        """Test validation passes for valid args."""
        tool = ConcreteTestTool()
        errors = tool.validate_args(required_arg="value", optional_arg=42)
        
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test tool execution."""
        tool = ConcreteTestTool()
        result = await tool.execute(required_arg="test", optional_arg=5)
        
        assert result.success
        assert result.data["received"]["required_arg"] == "test"


# ============================================================================
# Registry Tests
# ============================================================================

class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        
        registry.register(tool)
        
        assert "TestTool" in registry
        assert len(registry) == 1
    
    def test_register_duplicate_raises(self):
        """Test registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool1 = ConcreteTestTool()
        tool2 = ConcreteTestTool()
        
        registry.register(tool1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)
    
    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register(tool)
        
        retrieved = registry.get("TestTool")
        assert retrieved is tool
        
        missing = registry.get("NonExistent")
        assert missing is None
    
    def test_get_or_raise(self):
        """Test get_or_raise raises for missing tool."""
        registry = ToolRegistry()
        
        with pytest.raises(KeyError, match="not found"):
            registry.get_or_raise("NonExistent")
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register(tool)
        
        removed = registry.unregister("TestTool")
        
        assert removed is tool
        assert "TestTool" not in registry
    
    def test_list_tools(self):
        """Test listing all tools as specs."""
        registry = ToolRegistry()
        registry.register(ProductSearchTool())
        registry.register(PriceCalculatorTool())
        
        specs = registry.list_tools()
        
        assert len(specs) == 2
        assert all(isinstance(s, ToolSpec) for s in specs)
    
    def test_register_many(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        tools = [ConcreteTestTool()]  # Can't add duplicates, just test with one
        
        # Create unique tools
        class Tool1(ConcreteTestTool):
            name = "Tool1"
        
        class Tool2(ConcreteTestTool):
            name = "Tool2"
        
        registry.register_many([Tool1(), Tool2()])
        
        assert len(registry) == 2
    
    def test_to_llm_context(self):
        """Test exporting for LLM context."""
        registry = ToolRegistry()
        registry.register(ConcreteTestTool())
        
        context = registry.to_llm_context()
        
        assert len(context) == 1
        assert context[0]["name"] == "TestTool"
        assert "input_schema" in context[0]
    
    def test_clear(self):
        """Test clearing registry."""
        registry = ToolRegistry()
        registry.register(ConcreteTestTool())
        
        registry.clear()
        
        assert len(registry) == 0
    
    def test_iteration(self):
        """Test iterating over registry."""
        registry = ToolRegistry()
        registry.register(ConcreteTestTool())
        
        tools = list(registry)
        assert len(tools) == 1


# ============================================================================
# MCP Client Tests
# ============================================================================

class TestMCPClient:
    """Tests for MCPClient."""
    
    def test_default_config(self):
        """Test client uses default config."""
        client = MCPClient()
        
        assert client.server_url == "https://agentic-direct-server-hwgrypmndq-uk.a.run.app"
    
    def test_custom_config(self):
        """Test client with custom config."""
        config = MCPClientConfig(
            server_url="https://custom.server.com",
            timeout=60.0,
        )
        client = MCPClient(config)
        
        assert client.server_url == "https://custom.server.com"
        assert client.config.timeout == 60.0
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        client = MCPClient()
        
        await client.connect()
        assert client._client is not None
        
        await client.disconnect()
        assert client._client is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with MCPClient() as client:
            assert client._client is not None
    
    @pytest.mark.asyncio
    async def test_list_tools_mock(self):
        """Test listing tools with mocked response."""
        client = MCPClient()
        
        mock_response = {
            "tools": [
                {"name": "Tool1", "description": "First tool"},
                {"name": "Tool2", "description": "Second tool"},
            ]
        }
        
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            tools = await client.list_tools()
            
            assert len(tools) == 2
            assert tools[0]["name"] == "Tool1"
            mock_request.assert_called_once_with("GET", "/tools")
    
    @pytest.mark.asyncio
    async def test_call_tool_mock(self):
        """Test calling a tool with mocked response."""
        client = MCPClient()
        
        mock_response = {
            "result": {"products": [{"id": "prod1", "name": "Test Product"}]}
        }
        
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.call_tool("ProductSearch", {"channel": "display"})
            
            assert result["result"]["products"][0]["id"] == "prod1"
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test execute_tool returns ToolResult on success."""
        client = MCPClient()
        
        mock_response = {"result": {"data": "test"}}
        
        with patch.object(client, 'call_tool', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            result = await client.execute_tool("TestTool", {})
            
            assert result.success
            assert result.data == {"data": "test"}
    
    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """Test execute_tool returns ToolResult on error."""
        client = MCPClient()
        
        with patch.object(client, 'call_tool', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = MCPError("Server error", code=500)
            
            result = await client.execute_tool("TestTool", {})
            
            assert not result.success
            assert result.status == ToolStatus.ERROR
            assert "Server error" in result.error
    
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        client = MCPClient()
        client._tools_cache = [{"name": "cached"}]
        
        client.invalidate_cache()
        
        assert client._tools_cache is None


# ============================================================================
# Buyer Tool Tests
# ============================================================================

class TestBuyerTools:
    """Tests for buyer-side tools."""
    
    def test_all_buyer_tools_count(self):
        """Test correct number of buyer tools."""
        assert len(ALL_BUYER_TOOLS) == 14
    
    @pytest.mark.asyncio
    async def test_product_search_tool(self):
        """Test ProductSearchTool execution."""
        tool = ProductSearchTool()
        
        result = await tool.execute(channel="display", max_cpm=10.0)
        
        assert result.success
        assert "products" in result.data
        assert result.data["query"]["channel"] == "display"
    
    @pytest.mark.asyncio
    async def test_product_search_validation(self):
        """Test ProductSearchTool validation."""
        tool = ProductSearchTool()
        
        # Missing required channel
        result = await tool.execute(max_cpm=10.0)
        
        assert not result.success
        assert "channel" in result.error
    
    @pytest.mark.asyncio
    async def test_avails_check_tool(self):
        """Test AvailsCheckTool execution."""
        tool = AvailsCheckTool()
        
        result = await tool.execute(
            product_id="prod123",
            start_date="2025-02-01",
            end_date="2025-02-28",
        )
        
        assert result.success
        assert "available" in result.data
    
    @pytest.mark.asyncio
    async def test_create_order_tool(self):
        """Test CreateOrderTool execution."""
        tool = CreateOrderTool()
        
        result = await tool.execute(
            advertiser_id="adv123",
            name="Test Campaign",
            budget=10000.0,
        )
        
        assert result.success
        assert result.data["status"] == "draft"
    
    @pytest.mark.asyncio
    async def test_discover_inventory_tool(self):
        """Test DiscoverInventoryTool execution."""
        tool = DiscoverInventoryTool()
        
        result = await tool.execute(channel="video", min_viewability=0.7)
        
        assert result.success
        assert "inventory" in result.data
    
    @pytest.mark.asyncio
    async def test_audience_discovery_tool(self):
        """Test AudienceDiscoveryTool execution."""
        tool = AudienceDiscoveryTool()
        
        result = await tool.execute(category="demographics", min_reach=100000)
        
        assert result.success
        assert "segments" in result.data


# ============================================================================
# Seller Tool Tests
# ============================================================================

class TestSellerTools:
    """Tests for seller-side tools."""
    
    def test_all_seller_tools_count(self):
        """Test correct number of seller tools."""
        assert len(ALL_SELLER_TOOLS) == 17
    
    @pytest.mark.asyncio
    async def test_price_calculator_tool(self):
        """Test PriceCalculatorTool execution."""
        tool = PriceCalculatorTool()
        
        result = await tool.execute(
            inventory_id="inv123",
            volume=1000000,
            deal_type="pmp",
        )
        
        assert result.success
        assert "recommended_cpm" in result.data
    
    @pytest.mark.asyncio
    async def test_floor_manager_tool(self):
        """Test FloorManagerTool execution."""
        tool = FloorManagerTool()
        
        result = await tool.execute(
            action="set",
            inventory_id="inv123",
            floor_cpm=5.0,
        )
        
        assert result.success
        assert result.data["applied"] is True
    
    @pytest.mark.asyncio
    async def test_proposal_generator_tool(self):
        """Test ProposalGeneratorTool execution."""
        tool = ProposalGeneratorTool()
        
        result = await tool.execute(
            buyer_id="buyer123",
            inventory_ids=["inv1", "inv2"],
            deal_type="preferred",
        )
        
        assert result.success
        assert result.data["buyer_id"] == "buyer123"
    
    @pytest.mark.asyncio
    async def test_list_ad_units_tool(self):
        """Test ListAdUnitsTool execution."""
        tool = ListAdUnitsTool()
        
        result = await tool.execute(status="active", include_sizes=True)
        
        assert result.success
        assert "ad_units" in result.data
    
    @pytest.mark.asyncio
    async def test_audience_validation_tool(self):
        """Test AudienceValidationTool execution."""
        tool = AudienceValidationTool()
        
        result = await tool.execute(
            segment_ids=["seg1", "seg2"],
            check_reach=True,
            min_reach_threshold=50000,
        )
        
        assert result.success
        assert "valid_segments" in result.data


# ============================================================================
# Integration Tests
# ============================================================================

class TestToolIntegration:
    """Integration tests for tool system."""
    
    def test_full_registry_setup(self):
        """Test setting up registry with all tools."""
        registry = ToolRegistry()
        
        # Register all buyer tools
        for tool_class in ALL_BUYER_TOOLS:
            registry.register(tool_class())
        
        # Register all seller tools
        for tool_class in ALL_SELLER_TOOLS:
            registry.register(tool_class())
        
        # Should have 31 tools (14 buyer + 17 seller)
        assert len(registry) == 31
    
    def test_tool_names_unique(self):
        """Test all tool names are unique."""
        all_tools = ALL_BUYER_TOOLS + ALL_SELLER_TOOLS
        names = [t().name for t in all_tools]
        
        assert len(names) == len(set(names)), "Duplicate tool names found"
    
    def test_all_tools_have_required_attributes(self):
        """Test all tools have required attributes."""
        all_tools = ALL_BUYER_TOOLS + ALL_SELLER_TOOLS
        
        for tool_class in all_tools:
            tool = tool_class()
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'parameters')
            assert tool.name, f"{tool_class.__name__} has empty name"
            assert tool.description, f"{tool_class.__name__} has empty description"
    
    @pytest.mark.asyncio
    async def test_registry_with_mcp_client(self):
        """Test registry integration with MCP client."""
        registry = ToolRegistry()
        client = MCPClient()
        
        registry.set_mcp_client(client)
        registry.register(ProductSearchTool())
        
        assert registry._mcp_client is client
        assert registry.has("ProductSearch")
