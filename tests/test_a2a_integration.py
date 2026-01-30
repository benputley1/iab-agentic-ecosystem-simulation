"""Integration tests for A2A seller server and buyer client.

Tests A2A v0.3.0 compliance including:
- Agent card discovery endpoint
- JSON-RPC 2.0 message handling
- Various message intents (discovery, pricing, availability, proposal, deal)
- A2A client connection and methods
- Full buyer-seller flow via A2A

These tests require the A2A server (a2a_server.py) and client (a2a_client.py)
to be implemented as per IMPLEMENTATION_PLAN_A2A_SERVER.md
"""

import pytest
import asyncio
import uuid
from typing import Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Try to import the A2A modules - they may not exist yet
try:
    from src.agents.seller.a2a_server import create_app, get_agent_card
    A2A_SERVER_AVAILABLE = True
except ImportError:
    A2A_SERVER_AVAILABLE = False
    create_app = None
    get_agent_card = None

try:
    from src.agents.seller.iab_adapter import IABSellerAdapter
    IAB_ADAPTER_AVAILABLE = True
except ImportError:
    IAB_ADAPTER_AVAILABLE = False
    IABSellerAdapter = None

try:
    from src.agents.buyer.a2a_client import A2AClient, A2AResponse
    A2A_CLIENT_AVAILABLE = True
except ImportError:
    A2A_CLIENT_AVAILABLE = False
    A2AClient = None
    A2AResponse = None

# Skip all tests if A2A modules are not available
pytestmark = pytest.mark.skipif(
    not A2A_SERVER_AVAILABLE,
    reason="A2A server module not yet implemented"
)


@pytest.fixture
async def seller_adapter():
    """Create a mock seller adapter for testing."""
    if not IAB_ADAPTER_AVAILABLE:
        pytest.skip("IABSellerAdapter not available")
    
    adapter = IABSellerAdapter("test-seller", mock_llm=True)
    # Mock the connect to avoid Redis requirement
    adapter._bus = Mock()
    adapter._bus.connect = AsyncMock()
    adapter._bus.disconnect = AsyncMock()
    await adapter.connect()
    yield adapter
    await adapter.disconnect()


@pytest.fixture
def a2a_app(seller_adapter):
    """Create A2A FastAPI app with test adapter."""
    if not A2A_SERVER_AVAILABLE:
        pytest.skip("A2A server module not yet implemented")
    return create_app(seller_adapter)


@pytest.fixture
def test_client(a2a_app):
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    return TestClient(a2a_app)


class TestAgentCard:
    """Tests for A2A agent card endpoint."""
    
    def test_agent_card_endpoint(self, test_client):
        """Test agent card is accessible at well-known URL."""
        response = test_client.get("/a2a/seller/.well-known/agent-card.json")
        assert response.status_code == 200
        
        card = response.json()
        assert card["protocolVersion"] == "0.3.0"
        assert card["name"] == "alkimi-seller-agent"
        assert len(card["skills"]) >= 5
    
    def test_agent_card_legacy_endpoint(self, test_client):
        """Test legacy agent card endpoint for compatibility."""
        response = test_client.get("/a2a/seller/card")
        assert response.status_code == 200
        
        card = response.json()
        assert card["protocolVersion"] == "0.3.0"
    
    def test_agent_card_has_required_fields(self, test_client):
        """Test agent card contains all A2A v0.3.0 required fields."""
        card = test_client.get("/a2a/seller/.well-known/agent-card.json").json()
        
        required_fields = [
            "name",
            "description", 
            "protocolVersion",
            "version",
            "url",
            "skills",
            "capabilities",
        ]
        for field in required_fields:
            assert field in card, f"Missing required field: {field}"
    
    def test_agent_card_skills_structure(self, test_client):
        """Test skills have proper structure."""
        card = test_client.get("/a2a/seller/.well-known/agent-card.json").json()
        
        for skill in card["skills"]:
            assert "id" in skill
            assert "name" in skill
            assert "description" in skill
            assert "inputModes" in skill
            assert "outputModes" in skill
    
    def test_agent_card_skills_cover_iab_flows(self, test_client):
        """Test that skills cover IAB agentic-direct flows."""
        card = test_client.get("/a2a/seller/.well-known/agent-card.json").json()
        
        skill_ids = {s["id"] for s in card["skills"]}
        expected_skills = {
            "product-discovery",
            "pricing",
            "availability",
            "proposal",
            "deal-generation",
        }
        assert expected_skills.issubset(skill_ids), f"Missing skills: {expected_skills - skill_ids}"


class TestJsonRpc:
    """Tests for JSON-RPC 2.0 endpoint."""
    
    def test_message_send_discovery(self, test_client):
        """Test message/send for product discovery."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": "List available products"}],
                },
            },
            "id": "test-discovery-1",
        })
        
        assert response.status_code == 200
        result = response.json()
        assert "result" in result
        assert result["result"]["status"]["state"] == "completed"
        assert "taskId" in result["result"]
    
    def test_message_send_pricing(self, test_client):
        """Test message/send for pricing request."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "What are your prices?"}],
                },
            },
            "id": "test-pricing-1",
        })
        
        result = response.json()
        assert "result" in result
        
        # Check for data part with pricing info
        parts = result["result"]["parts"]
        data_parts = [p for p in parts if p.get("kind") == "data"]
        assert len(data_parts) > 0
        assert "pricing" in data_parts[0]["data"]
    
    def test_message_send_availability(self, test_client):
        """Test message/send for availability check."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Check availability for display-premium"}],
                },
            },
            "id": "test-avail-1",
        })
        
        result = response.json()
        assert "result" in result
        assert result["result"]["status"]["state"] == "completed"
    
    def test_message_send_proposal(self, test_client):
        """Test message/send for proposal submission."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "I want to propose a deal for 1M impressions at $12 CPM"}],
                },
            },
            "id": "test-proposal-1",
        })
        
        result = response.json()
        assert "result" in result
        
        # Check proposal response has proposal_id
        data_parts = [p for p in result["result"]["parts"] if p.get("kind") == "data"]
        if data_parts:
            assert "proposal_id" in data_parts[0]["data"] or "status" in data_parts[0]["data"]
    
    def test_message_send_deal_generation(self, test_client):
        """Test message/send for deal ID generation."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Generate deal ID for proposal PROP-12345"}],
                },
            },
            "id": "test-deal-1",
        })
        
        result = response.json()
        assert "result" in result
        
        # Check for deal_id in response
        data_parts = [p for p in result["result"]["parts"] if p.get("kind") == "data"]
        if data_parts:
            assert "deal_id" in data_parts[0]["data"]
    
    def test_message_send_context_continuity(self, test_client):
        """Test that contextId enables conversation continuity."""
        context_id = f"ctx-{uuid.uuid4().hex[:8]}"
        
        # First message
        response1 = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "contextId": context_id,
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "List products"}],
                },
            },
            "id": "test-ctx-1",
        })
        
        result1 = response1.json()
        returned_ctx = result1["result"].get("contextId", "")
        
        # Second message with same context
        response2 = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "contextId": returned_ctx or context_id,
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "What about pricing?"}],
                },
            },
            "id": "test-ctx-2",
        })
        
        assert response2.status_code == 200
    
    def test_message_send_missing_text_error(self, test_client):
        """Test error handling for messages without text."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [],  # No text part
                },
            },
            "id": "test-error-1",
        })
        
        result = response.json()
        assert "error" in result
        assert result["error"]["code"] == -32602
    
    def test_unknown_method_error(self, test_client):
        """Test error handling for unknown JSON-RPC methods."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "params": {},
            "id": "test-unknown-1",
        })
        
        result = response.json()
        assert "error" in result
        assert result["error"]["code"] == -32601
    
    def test_tasks_get(self, test_client):
        """Test tasks/get method returns task status."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {
                "taskId": "task-12345",
            },
            "id": "test-tasks-1",
        })
        
        result = response.json()
        assert "result" in result
        assert result["result"]["taskId"] == "task-12345"
        assert result["result"]["status"]["state"] == "completed"
    
    def test_default_endpoint_alias(self, test_client):
        """Test POST to /a2a/seller/ also works as JSON-RPC endpoint."""
        response = test_client.post("/a2a/seller/", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "List products"}],
                },
            },
            "id": "test-alias-1",
        })
        
        assert response.status_code == 200
        assert "result" in response.json()


@pytest.mark.skipif(not A2A_CLIENT_AVAILABLE, reason="A2A client module not available")
class TestA2AClient:
    """Tests for A2A client module."""
    
    @pytest.mark.asyncio
    async def test_client_connection(self, a2a_app):
        """Test client can connect and fetch agent card."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.agent_card_url = "/a2a/seller/.well-known/agent-card.json"
            
            card = await client.get_agent_card()
            assert card["protocolVersion"] == "0.3.0"
    
    @pytest.mark.asyncio
    async def test_list_products(self, a2a_app):
        """Test client list_products convenience method."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            response = await client.list_products()
            assert response.success
            assert "products" in response.data
    
    @pytest.mark.asyncio
    async def test_get_pricing(self, a2a_app):
        """Test client get_pricing method."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            response = await client.get_pricing()
            assert response.success
            assert "pricing" in response.data
    
    @pytest.mark.asyncio
    async def test_check_availability(self, a2a_app):
        """Test client check_availability method."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            response = await client.check_availability("display-premium", 1000000)
            assert response.success
    
    @pytest.mark.asyncio
    async def test_submit_proposal(self, a2a_app):
        """Test client submit_proposal method."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            response = await client.submit_proposal(
                "display-premium",
                1000000,
                12.0,
                "2026-02-01",
                "2026-02-28",
            )
            assert response.success
    
    @pytest.mark.asyncio
    async def test_generate_deal_id(self, a2a_app):
        """Test client generate_deal_id method."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            response = await client.generate_deal_id("PROP-12345")
            assert response.success
            assert "deal_id" in response.data
    
    @pytest.mark.asyncio
    async def test_response_parsing(self, a2a_app):
        """Test A2AResponse correctly parses server response."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            response = await client.send_message("List products")
            
            assert isinstance(response, A2AResponse)
            assert response.task_id != ""
            assert response.text != ""
            assert response.raw != {}


@pytest.mark.skipif(not A2A_CLIENT_AVAILABLE, reason="A2A client module not available")
class TestFullFlow:
    """End-to-end tests for buyer-seller flow via A2A."""
    
    @pytest.mark.asyncio
    async def test_discovery_to_deal_flow(self, a2a_app):
        """Test full buyer-seller flow: discovery → pricing → availability → proposal → deal."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            client.agent_card_url = "/a2a/seller/.well-known/agent-card.json"
            
            # 1. Discover products
            products = await client.list_products()
            assert products.success, f"Discovery failed: {products.error}"
            assert "products" in products.data
            product_list = products.data["products"]
            assert len(product_list) > 0, "No products available"
            product_id = product_list[0]["product_id"]
            
            # 2. Get pricing
            pricing = await client.get_pricing(product_id)
            assert pricing.success, f"Pricing failed: {pricing.error}"
            assert "pricing" in pricing.data
            
            # 3. Check availability
            avail = await client.check_availability(product_id, 1000000)
            assert avail.success, f"Availability check failed: {avail.error}"
            
            # 4. Submit proposal
            proposal = await client.submit_proposal(
                product_id,
                1000000,
                10.0,
                "2026-02-01",
                "2026-02-28",
            )
            assert proposal.success, f"Proposal failed: {proposal.error}"
            assert "proposal_id" in proposal.data or "status" in proposal.data
            proposal_id = proposal.data.get("proposal_id", "PROP-test")
            
            # 5. Generate deal
            deal = await client.generate_deal_id(proposal_id)
            assert deal.success, f"Deal generation failed: {deal.error}"
            assert "deal_id" in deal.data
            
            # Verify deal_id format
            deal_id = deal.data["deal_id"]
            assert deal_id.startswith("DEAL-"), f"Invalid deal_id format: {deal_id}"
    
    @pytest.mark.asyncio
    async def test_multiple_products_flow(self, a2a_app):
        """Test handling multiple products in discovery."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            # Discovery should return multiple products
            response = await client.send_message("What inventory do you have available?")
            assert response.success
            
            products = response.data.get("products", [])
            # Should have multiple product types
            assert len(products) >= 1
    
    @pytest.mark.asyncio
    async def test_context_maintained_across_requests(self, a2a_app):
        """Test conversation context is maintained."""
        import httpx
        
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            client.jsonrpc_url = "/a2a/seller/jsonrpc"
            
            # First request
            r1 = await client.send_message("List products")
            ctx1 = r1.context_id
            
            # Second request should use same context
            r2 = await client.send_message("What about pricing?")
            ctx2 = r2.context_id
            
            # Context should be maintained (client tracks it)
            assert client._context_id != ""


class TestIntentRouting:
    """Tests for intent detection and routing in A2A server."""
    
    def test_discovery_intent_keywords(self, test_client):
        """Test discovery intent is triggered by various keywords."""
        discovery_phrases = [
            "List all available products",
            "What inventory do you have?",
            "Show me your products",
            "What's available?",
        ]
        
        for phrase in discovery_phrases:
            response = test_client.post("/a2a/seller/jsonrpc", json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": phrase}],
                    },
                },
                "id": f"test-{uuid.uuid4().hex[:8]}",
            })
            
            result = response.json()
            assert "result" in result, f"Failed for phrase: {phrase}"
            # Discovery should return products data
            data_parts = [p for p in result["result"]["parts"] if p.get("kind") == "data"]
            if data_parts:
                assert "products" in data_parts[0]["data"], f"No products for: {phrase}"
    
    def test_pricing_intent_keywords(self, test_client):
        """Test pricing intent is triggered by various keywords."""
        pricing_phrases = [
            "What are your prices?",
            "Tell me about CPM rates",
            "What's the cost?",
            "Pricing information please",
        ]
        
        for phrase in pricing_phrases:
            response = test_client.post("/a2a/seller/jsonrpc", json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": phrase}],
                    },
                },
                "id": f"test-{uuid.uuid4().hex[:8]}",
            })
            
            result = response.json()
            assert "result" in result, f"Failed for phrase: {phrase}"
    
    def test_fallback_response(self, test_client):
        """Test fallback response for unrecognized intents."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "What's the weather like?"}],
                },
            },
            "id": "test-fallback-1",
        })
        
        result = response.json()
        assert "result" in result
        # Should get a helpful fallback message
        text_parts = [p for p in result["result"]["parts"] if p.get("kind") == "text"]
        assert len(text_parts) > 0
        # Fallback should mention available skills
        text = text_parts[0]["text"].lower()
        assert "help" in text or "discovery" in text or "pricing" in text


class TestErrorHandling:
    """Tests for error handling in A2A server."""
    
    def test_invalid_json_rpc_version(self, test_client):
        """Test handling of invalid JSON-RPC version."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "1.0",  # Invalid version
            "method": "message/send",
            "params": {},
            "id": "test-1",
        })
        
        # Should still process (many servers are lenient about version)
        assert response.status_code == 200
    
    def test_missing_method(self, test_client):
        """Test handling of missing method field."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "params": {},
            "id": "test-1",
        })
        
        result = response.json()
        assert "error" in result
    
    def test_malformed_message(self, test_client):
        """Test handling of malformed message structure."""
        response = test_client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": "not a dict",  # Should be a dict
            },
            "id": "test-1",
        })
        
        # Server should handle gracefully
        assert response.status_code == 200


# Marker for integration tests that require running servers
@pytest.mark.integration
class TestLiveServer:
    """Integration tests requiring a running A2A server.
    
    These are skipped by default. Run with: pytest -m integration
    """
    
    @pytest.mark.asyncio
    async def test_live_server_connection(self):
        """Test connection to a live A2A server."""
        if not A2A_CLIENT_AVAILABLE:
            pytest.skip("A2A client not available")
        
        import httpx
        
        client = A2AClient("http://localhost:8001")
        try:
            await client.connect()
            card = await client.get_agent_card()
            assert card["protocolVersion"] == "0.3.0"
        finally:
            await client.disconnect()
