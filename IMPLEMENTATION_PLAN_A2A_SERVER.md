# Implementation Plan: A2A Server Layer for Scenario B

**Objective:** Add A2A v0.3.0 server endpoints to the Python seller agent so Scenario B uses the actual IAB protocol (HTTP JSON-RPC) instead of Redis pub/sub.

**Effort Estimate:** 3-4 days → Can parallelize to ~1 day with multiple agents

---

## Architecture Overview

```
BEFORE (Current):
┌─────────────┐     Redis pub/sub     ┌─────────────┐
│ Buyer Agent │ ◄──────────────────► │ Seller Agent │
└─────────────┘                       └─────────────┘

AFTER (IAB Compliant):
┌─────────────┐     HTTP JSON-RPC     ┌─────────────┐
│ Buyer Agent │ ──────────────────►  │ Seller A2A  │
│             │                       │   Server    │
│ A2A Client  │ ◄──────────────────  │ (FastAPI)   │
└─────────────┘                       └─────────────┘
                                            │
                                            ▼
                                      ┌─────────────┐
                                      │ IABSeller   │
                                      │  Adapter    │
                                      └─────────────┘
```

---

## Task Breakdown

### Task 1: A2A Server for Seller (NEW FILE)
**File:** `src/agents/seller/a2a_server.py`
**Parallelizable:** Yes
**Estimate:** 2-3 hours

Create FastAPI server implementing A2A v0.3.0:

```python
"""
A2A v0.3.0 Server for Seller Agent.

Endpoints:
- GET  /a2a/seller/.well-known/agent-card.json  (Agent discovery)
- POST /a2a/seller/jsonrpc                       (JSON-RPC 2.0)
- POST /a2a/seller/                              (Default endpoint)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import uuid
from datetime import datetime

from .iab_adapter import IABSellerAdapter
from .models import Product

app = FastAPI(
    title="Alkimi Seller A2A Server",
    description="A2A v0.3.0 compliant seller agent for IAB agentic-direct",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global adapter instance (initialized on startup)
_adapter: IABSellerAdapter = None
_base_url: str = "http://localhost:8001"


def get_agent_card() -> dict:
    """Generate A2A v0.3.0 compliant agent card."""
    return {
        "name": "alkimi-seller-agent",
        "description": "Alkimi Exchange seller agent for programmatic direct deals. "
                      "Handles inventory discovery, pricing, proposals, and deal generation.",
        "protocolVersion": "0.3.0",
        "version": "1.0.0",
        "url": f"{_base_url}/a2a/seller",
        "skills": [
            {
                "id": "product-discovery",
                "name": "Product Discovery",
                "description": "Search and list available advertising inventory",
                "tags": ["advertising", "inventory", "search"],
                "examples": [
                    "What inventory do you have available?",
                    "List all CTV products",
                    "Show me premium display inventory",
                ],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["application/json"],
            },
            {
                "id": "pricing",
                "name": "Get Pricing",
                "description": "Get CPM pricing for products with tiered discounts",
                "tags": ["advertising", "pricing", "cpm"],
                "examples": [
                    "What's the CPM for display-premium?",
                    "Get pricing for 1M impressions of CTV",
                    "What discounts are available for agencies?",
                ],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["application/json"],
            },
            {
                "id": "availability",
                "name": "Check Availability",
                "description": "Check inventory availability for date ranges",
                "tags": ["advertising", "inventory", "availability"],
                "examples": [
                    "Is display-premium available for 500K impressions in February?",
                    "Check CTV availability for Q1",
                ],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["application/json"],
            },
            {
                "id": "proposal",
                "name": "Submit Proposal",
                "description": "Submit a deal proposal for review",
                "tags": ["advertising", "deal", "proposal"],
                "examples": [
                    "I want to propose a deal for 1M impressions at $12 CPM",
                    "Submit proposal for display-premium, 500K imps, Feb 1-28",
                ],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
            },
            {
                "id": "deal-generation",
                "name": "Generate Deal ID",
                "description": "Generate Deal ID for DSP activation",
                "tags": ["advertising", "deal", "dsp"],
                "examples": [
                    "Generate deal ID for accepted proposal PROP-123",
                    "Create deal for TTD activation",
                ],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
            },
        ],
        "capabilities": {
            "pushNotifications": False,
            "streaming": False,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["application/json"],
        "securitySchemes": {
            "none": {
                "type": "none",
                "description": "No authentication required for simulation",
            }
        },
        "security": [{"none": []}],
        "additionalInterfaces": [
            {
                "transport": "http",
                "url": f"{_base_url}/a2a/seller/jsonrpc",
            },
        ],
    }


@app.get("/a2a/seller/.well-known/agent-card.json")
async def agent_card():
    """A2A agent card discovery endpoint."""
    return get_agent_card()


@app.get("/a2a/seller/card")
async def agent_card_legacy():
    """Legacy agent card endpoint for compatibility."""
    return get_agent_card()


@app.post("/a2a/seller/jsonrpc")
@app.post("/a2a/seller/")
async def jsonrpc_handler(request: Request):
    """
    JSON-RPC 2.0 endpoint for A2A messages.
    
    Supported methods:
    - message/send: Process natural language request
    - tasks/get: Get task status (sync mode - always completed)
    """
    body = await request.json()
    
    jsonrpc = body.get("jsonrpc", "2.0")
    method = body.get("method", "")
    params = body.get("params", {})
    request_id = body.get("id", str(uuid.uuid4()))
    
    if method == "message/send":
        return await handle_message_send(params, request_id)
    elif method == "tasks/get":
        return await handle_tasks_get(params, request_id)
    else:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
            "id": request_id,
        }


async def handle_message_send(params: dict, request_id: str) -> dict:
    """Handle message/send JSON-RPC method."""
    message = params.get("message", {})
    parts = message.get("parts", [])
    
    # Extract text from message parts
    text = ""
    for part in parts:
        if part.get("kind") == "text":
            text = part.get("text", "")
            break
    
    if not text:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "No text in message"},
            "id": request_id,
        }
    
    # Process the message through our adapter
    result = await process_a2a_message(text)
    
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    
    return {
        "jsonrpc": "2.0",
        "result": {
            "taskId": task_id,
            "contextId": params.get("contextId", f"ctx-{uuid.uuid4().hex[:8]}"),
            "status": {"state": "completed", "timestamp": datetime.utcnow().isoformat()},
            "parts": [
                {"kind": "text", "text": result.get("text", "")},
                {"kind": "data", "data": result.get("data", {})},
            ],
        },
        "id": request_id,
    }


async def handle_tasks_get(params: dict, request_id: str) -> dict:
    """Handle tasks/get JSON-RPC method (sync mode - always completed)."""
    task_id = params.get("taskId", "")
    
    return {
        "jsonrpc": "2.0",
        "result": {
            "taskId": task_id,
            "status": {"state": "completed"},
        },
        "id": request_id,
    }


async def process_a2a_message(text: str) -> dict:
    """
    Process natural language A2A message and route to appropriate handler.
    
    This is the AI-powered tool selection layer. For simulation, we use
    keyword matching. For production, this would use an LLM.
    """
    text_lower = text.lower()
    
    # Route based on intent (simple keyword matching for simulation)
    if any(kw in text_lower for kw in ["list", "inventory", "products", "available", "what do you have"]):
        return await handle_discovery(text)
    elif any(kw in text_lower for kw in ["price", "pricing", "cpm", "cost", "rate"]):
        return await handle_pricing(text)
    elif any(kw in text_lower for kw in ["available", "availability", "check", "can you"]):
        return await handle_availability(text)
    elif any(kw in text_lower for kw in ["propose", "proposal", "deal", "buy", "book"]):
        return await handle_proposal(text)
    elif any(kw in text_lower for kw in ["generate", "deal id", "activate", "dsp"]):
        return await handle_deal_generation(text)
    else:
        return {
            "text": "I can help with: inventory discovery, pricing, availability checks, "
                   "proposals, and deal generation. What would you like to know?",
            "data": {"skills": [s["id"] for s in get_agent_card()["skills"]]},
        }


async def handle_discovery(text: str) -> dict:
    """Handle inventory discovery requests."""
    if _adapter is None:
        return {"text": "Adapter not initialized", "data": {}}
    
    products = list(_adapter.products.values())
    
    return {
        "text": f"I have {len(products)} products available:\n" + 
               "\n".join([f"- {p.name} ({p.product_id}): {p.inventory_type}, base CPM ${p.base_cpm}" 
                         for p in products]),
        "data": {
            "products": [
                {
                    "product_id": p.product_id,
                    "name": p.name,
                    "inventory_type": p.inventory_type,
                    "base_cpm": p.base_cpm,
                    "floor_cpm": p.floor_cpm,
                }
                for p in products
            ]
        },
    }


async def handle_pricing(text: str) -> dict:
    """Handle pricing requests."""
    if _adapter is None:
        return {"text": "Adapter not initialized", "data": {}}
    
    # For simulation, return pricing for all products
    products = list(_adapter.products.values())
    
    pricing_info = []
    for p in products:
        pricing_info.append({
            "product_id": p.product_id,
            "base_cpm": p.base_cpm,
            "floor_cpm": p.floor_cpm,
            "tiers": {
                "public": p.base_cpm,
                "seat": round(p.base_cpm * 0.95, 2),
                "agency": round(p.base_cpm * 0.90, 2),
                "advertiser": round(p.base_cpm * 0.85, 2),
            },
        })
    
    return {
        "text": "Pricing (CPM) by tier:\n" +
               "\n".join([f"- {p['product_id']}: Public ${p['base_cpm']}, "
                         f"Agency ${p['tiers']['agency']}, "
                         f"Advertiser ${p['tiers']['advertiser']}"
                         for p in pricing_info]),
        "data": {"pricing": pricing_info},
    }


async def handle_availability(text: str) -> dict:
    """Handle availability check requests."""
    if _adapter is None:
        return {"text": "Adapter not initialized", "data": {}}
    
    # For simulation, report availability for all products
    products = list(_adapter.products.values())
    
    availability = []
    for p in products:
        avail, max_imps = _adapter._inventory.check_availability(p.product_id, 1000000)
        availability.append({
            "product_id": p.product_id,
            "available": avail,
            "max_impressions": max_imps,
        })
    
    return {
        "text": "Availability:\n" +
               "\n".join([f"- {a['product_id']}: {'✓' if a['available'] else '✗'} "
                         f"(max {a['max_impressions']:,} imps)"
                         for a in availability]),
        "data": {"availability": availability},
    }


async def handle_proposal(text: str) -> dict:
    """Handle proposal submission requests."""
    # For simulation, generate a mock proposal response
    proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
    
    return {
        "text": f"Proposal {proposal_id} received. Reviewing...\n"
               "Status: ACCEPTED (simulation mode)",
        "data": {
            "proposal_id": proposal_id,
            "status": "accepted",
            "recommendation": "accept",
        },
    }


async def handle_deal_generation(text: str) -> dict:
    """Handle deal ID generation requests."""
    deal_id = f"DEAL-{uuid.uuid4().hex[:12].upper()}"
    
    return {
        "text": f"Deal generated: {deal_id}\n"
               "Ready for DSP activation (TTD, DV360, Amazon).",
        "data": {
            "deal_id": deal_id,
            "deal_type": "preferred_deal",
            "activation_instructions": {
                "ttd": f"Add Deal ID {deal_id} in The Trade Desk",
                "dv360": f"Add Deal ID {deal_id} in DV360 Marketplace",
                "amazon": f"Add Deal ID {deal_id} in Amazon DSP",
            },
        },
    }


def create_app(adapter: IABSellerAdapter, base_url: str = "http://localhost:8001") -> FastAPI:
    """Create A2A server with initialized adapter."""
    global _adapter, _base_url
    _adapter = adapter
    _base_url = base_url
    return app


def run_server(adapter: IABSellerAdapter, host: str = "0.0.0.0", port: int = 8001):
    """Run the A2A server."""
    import uvicorn
    create_app(adapter, f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # For testing standalone
    import asyncio
    
    async def main():
        adapter = IABSellerAdapter("test-seller", mock_llm=True)
        await adapter.connect()
        run_server(adapter)
    
    asyncio.run(main())
```

**Deliverables:**
- [ ] `a2a_server.py` with full A2A v0.3.0 compliance
- [ ] Agent card generation with all skills
- [ ] JSON-RPC 2.0 message handling
- [ ] Intent routing to existing adapter methods

---

### Task 2: A2A Client for Buyer (NEW FILE)
**File:** `src/agents/buyer/a2a_client.py`
**Parallelizable:** Yes (with Task 1)
**Estimate:** 1-2 hours

Create async HTTP client for buyer to call seller's A2A endpoints:

```python
"""
A2A Client for Buyer Agent.

Connects to seller's A2A server via HTTP JSON-RPC 2.0.
"""

import httpx
import uuid
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class A2AResponse:
    """Parsed A2A response."""
    task_id: str = ""
    context_id: str = ""
    text: str = ""
    data: dict = field(default_factory=dict)
    success: bool = True
    error: str = ""
    raw: dict = field(default_factory=dict)


class A2AClient:
    """
    A2A v0.3.0 client for connecting to seller agents.
    
    Usage:
        async with A2AClient("http://localhost:8001") as client:
            # Discover agent
            card = await client.get_agent_card()
            
            # Send message
            response = await client.send_message("What inventory do you have?")
            print(response.text)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        agent_role: str = "seller",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_role = agent_role
        self.agent_url = f"{self.base_url}/a2a/{agent_role}"
        self.jsonrpc_url = f"{self.agent_url}/jsonrpc"
        self.agent_card_url = f"{self.agent_url}/.well-known/agent-card.json"
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._context_id: str = ""
        self._agent_card: Optional[dict] = None
    
    async def __aenter__(self) -> "A2AClient":
        await self.connect()
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.disconnect()
    
    async def connect(self) -> None:
        """Connect and fetch agent card."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        try:
            self._agent_card = await self.get_agent_card()
        except Exception:
            self._agent_card = None
    
    async def disconnect(self) -> None:
        """Disconnect from server."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def get_agent_card(self) -> dict:
        """Fetch agent card for capability discovery."""
        response = await self._client.get(self.agent_card_url)
        response.raise_for_status()
        return response.json()
    
    @property
    def agent_card(self) -> Optional[dict]:
        """Get cached agent card."""
        return self._agent_card
    
    async def send_message(
        self,
        message: str,
        context_id: Optional[str] = None,
    ) -> A2AResponse:
        """
        Send natural language message to agent.
        
        Args:
            message: Natural language request
            context_id: Optional context for conversation continuity
        
        Returns:
            A2AResponse with text and structured data
        """
        ctx = context_id or self._context_id
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                },
            },
            "id": str(uuid.uuid4()),
        }
        
        if ctx:
            payload["params"]["contextId"] = ctx
        
        response = await self._client.post(
            self.jsonrpc_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        result = response.json()
        
        return self._parse_response(result)
    
    def _parse_response(self, result: dict) -> A2AResponse:
        """Parse JSON-RPC response into A2AResponse."""
        response = A2AResponse(raw=result)
        
        if "error" in result:
            response.success = False
            response.error = result["error"].get("message", "Unknown error")
            return response
        
        res = result.get("result", {})
        response.task_id = res.get("taskId", "")
        response.context_id = res.get("contextId", "")
        
        # Update session context
        if response.context_id:
            self._context_id = response.context_id
        
        # Extract parts
        for part in res.get("parts", []):
            kind = part.get("kind", "")
            if kind == "text":
                response.text = part.get("text", "")
            elif kind == "data":
                response.data = part.get("data", {})
        
        return response
    
    # Convenience methods matching IAB agentic-direct patterns
    
    async def list_products(self) -> A2AResponse:
        """List available products."""
        return await self.send_message("List all available products")
    
    async def get_pricing(self, product_id: Optional[str] = None) -> A2AResponse:
        """Get pricing information."""
        if product_id:
            return await self.send_message(f"What is the pricing for {product_id}?")
        return await self.send_message("What are your prices?")
    
    async def check_availability(
        self,
        product_id: str,
        impressions: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> A2AResponse:
        """Check product availability."""
        msg = f"Is {product_id} available for {impressions:,} impressions"
        if start_date and end_date:
            msg += f" from {start_date} to {end_date}"
        return await self.send_message(msg + "?")
    
    async def submit_proposal(
        self,
        product_id: str,
        impressions: int,
        cpm: float,
        start_date: str,
        end_date: str,
    ) -> A2AResponse:
        """Submit a deal proposal."""
        return await self.send_message(
            f"I want to propose a deal for {product_id}: "
            f"{impressions:,} impressions at ${cpm} CPM from {start_date} to {end_date}"
        )
    
    async def generate_deal_id(
        self,
        proposal_id: str,
        dsp_platform: Optional[str] = None,
    ) -> A2AResponse:
        """Generate deal ID for DSP activation."""
        msg = f"Generate deal ID for proposal {proposal_id}"
        if dsp_platform:
            msg += f" for {dsp_platform}"
        return await self.send_message(msg)
```

**Deliverables:**
- [ ] `a2a_client.py` with async HTTP client
- [ ] Agent card discovery
- [ ] JSON-RPC 2.0 message sending
- [ ] Response parsing
- [ ] Convenience methods matching IAB patterns

---

### Task 3: Update Scenario B to Use A2A
**File:** `src/scenarios/scenario_b.py`
**Parallelizable:** No (depends on Tasks 1 & 2)
**Estimate:** 2-3 hours

Modify `ScenarioB` class to use HTTP A2A instead of Redis:

```python
# Key changes needed in scenario_b.py:

# 1. Add imports
from agents.seller.a2a_server import create_app, run_server
from agents.buyer.a2a_client import A2AClient

# 2. Add A2A server management
class ScenarioB(BaseScenario):
    def __init__(self, ...):
        # ... existing init ...
        
        # A2A infrastructure
        self._seller_servers: dict[str, Process] = {}  # seller_id -> server process
        self._buyer_clients: dict[str, A2AClient] = {}  # buyer_id -> A2A client
        self._seller_ports: dict[str, int] = {}  # seller_id -> port
        self._base_port = 8100  # Seller servers start at 8100
    
    async def _start_seller_a2a_servers(self):
        """Start A2A servers for all sellers."""
        import multiprocessing
        
        for i, (seller_id, adapter) in enumerate(self._seller_adapters.items()):
            port = self._base_port + i
            self._seller_ports[seller_id] = port
            
            # Start server in subprocess
            process = multiprocessing.Process(
                target=run_server,
                args=(adapter, "0.0.0.0", port),
            )
            process.start()
            self._seller_servers[seller_id] = process
            
            logger.info("a2a_server.started", seller_id=seller_id, port=port)
    
    async def _create_buyer_clients(self):
        """Create A2A clients for buyers to connect to sellers."""
        for buyer_id in self._buyer_ids:
            # Each buyer gets clients to all sellers
            self._buyer_clients[buyer_id] = {}
            for seller_id, port in self._seller_ports.items():
                client = A2AClient(f"http://localhost:{port}")
                await client.connect()
                self._buyer_clients[buyer_id][seller_id] = client
    
    # 3. Replace Redis-based negotiation with A2A
    async def _negotiate_deal_a2a(
        self,
        buyer_id: str,
        seller_id: str,
        request: BidRequest,
    ) -> Optional[BidResponse]:
        """Negotiate deal via A2A protocol."""
        client = self._buyer_clients[buyer_id][seller_id]
        
        # Discovery
        products_response = await client.list_products()
        if not products_response.success:
            return None
        
        # Find matching product
        products = products_response.data.get("products", [])
        matching = [p for p in products if self._matches_request(p, request)]
        if not matching:
            return None
        
        product = matching[0]
        
        # Get pricing
        pricing_response = await client.get_pricing(product["product_id"])
        
        # Check availability
        avail_response = await client.check_availability(
            product["product_id"],
            request.impressions_requested,
            request.start_date,
            request.end_date,
        )
        
        if not avail_response.data.get("availability", [{}])[0].get("available"):
            return None
        
        # Submit proposal
        proposal_response = await client.submit_proposal(
            product["product_id"],
            request.impressions_requested,
            request.max_cpm,
            request.start_date,
            request.end_date,
        )
        
        if proposal_response.data.get("status") != "accepted":
            return None
        
        # Generate deal
        deal_response = await client.generate_deal_id(
            proposal_response.data.get("proposal_id", ""),
        )
        
        # Convert to BidResponse
        return BidResponse(
            request_id=request.request_id,
            seller_id=seller_id,
            offered_cpm=pricing_response.data.get("pricing", [{}])[0].get("base_cpm", 0),
            available_impressions=avail_response.data.get("availability", [{}])[0].get("max_impressions", 0),
            deal_type=DealType.PREFERRED_DEAL,
            deal_id=deal_response.data.get("deal_id", ""),
            inventory_details={"product_id": product["product_id"]},
        )
```

**Deliverables:**
- [ ] A2A server lifecycle management (start/stop)
- [ ] Buyer client pool management
- [ ] Replace `_negotiate_via_redis()` with `_negotiate_deal_a2a()`
- [ ] Maintain context rot simulation (applies to client state)
- [ ] Maintain hallucination injection (applies to message interpretation)

---

### Task 4: Update Scenario B Entry Points
**File:** `src/scenarios/scenario_b.py` (continued)
**Parallelizable:** No (depends on Task 3)
**Estimate:** 1 hour

Update `run()`, `setup()`, `teardown()` methods:

```python
async def setup(self):
    """Initialize A2A infrastructure."""
    await super().setup()
    
    # Start seller A2A servers
    await self._start_seller_a2a_servers()
    
    # Wait for servers to be ready
    await asyncio.sleep(2)
    
    # Create buyer clients
    await self._create_buyer_clients()
    
    logger.info("scenario_b.a2a_setup_complete", 
                sellers=len(self._seller_servers),
                buyers=len(self._buyer_clients))

async def teardown(self):
    """Cleanup A2A infrastructure."""
    # Close buyer clients
    for buyer_id, clients in self._buyer_clients.items():
        for client in clients.values():
            await client.disconnect()
    
    # Stop seller servers
    for seller_id, process in self._seller_servers.items():
        process.terminate()
        process.join(timeout=5)
    
    await super().teardown()
```

**Deliverables:**
- [ ] Server startup in `setup()`
- [ ] Client connection in `setup()`
- [ ] Graceful shutdown in `teardown()`
- [ ] Health checks before simulation starts

---

### Task 5: Integration Tests
**File:** `tests/test_a2a_integration.py`
**Parallelizable:** Yes (with Tasks 1 & 2)
**Estimate:** 1-2 hours

```python
"""Integration tests for A2A seller server and buyer client."""

import pytest
import asyncio
from agents.seller.a2a_server import create_app
from agents.seller.iab_adapter import IABSellerAdapter
from agents.buyer.a2a_client import A2AClient
from fastapi.testclient import TestClient
import httpx


@pytest.fixture
async def seller_adapter():
    adapter = IABSellerAdapter("test-seller", mock_llm=True)
    await adapter.connect()
    yield adapter
    await adapter.disconnect()


@pytest.fixture
def a2a_app(seller_adapter):
    return create_app(seller_adapter)


class TestAgentCard:
    def test_agent_card_endpoint(self, a2a_app):
        client = TestClient(a2a_app)
        response = client.get("/a2a/seller/.well-known/agent-card.json")
        assert response.status_code == 200
        
        card = response.json()
        assert card["protocolVersion"] == "0.3.0"
        assert card["name"] == "alkimi-seller-agent"
        assert len(card["skills"]) >= 5
    
    def test_agent_card_has_required_fields(self, a2a_app):
        client = TestClient(a2a_app)
        card = client.get("/a2a/seller/.well-known/agent-card.json").json()
        
        required = ["name", "protocolVersion", "version", "url", "skills", "capabilities"]
        for field in required:
            assert field in card, f"Missing required field: {field}"


class TestJsonRpc:
    def test_message_send(self, a2a_app):
        client = TestClient(a2a_app)
        
        response = client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "List available products"}],
                },
            },
            "id": "test-1",
        })
        
        assert response.status_code == 200
        result = response.json()
        assert "result" in result
        assert result["result"]["status"]["state"] == "completed"
    
    def test_pricing_request(self, a2a_app):
        client = TestClient(a2a_app)
        
        response = client.post("/a2a/seller/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "What are your prices?"}],
                },
            },
            "id": "test-2",
        })
        
        result = response.json()
        assert "pricing" in result["result"]["parts"][1]["data"]


class TestA2AClient:
    @pytest.mark.asyncio
    async def test_client_connection(self, a2a_app):
        # Use httpx test client
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            
            card = await client.get_agent_card()
            assert card["protocolVersion"] == "0.3.0"
    
    @pytest.mark.asyncio
    async def test_list_products(self, a2a_app):
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            
            response = await client.list_products()
            assert response.success
            assert "products" in response.data


class TestFullFlow:
    @pytest.mark.asyncio
    async def test_discovery_to_deal(self, a2a_app):
        """Test full buyer-seller flow via A2A."""
        async with httpx.AsyncClient(app=a2a_app, base_url="http://test") as http:
            client = A2AClient("http://test")
            client._client = http
            
            # 1. Discover products
            products = await client.list_products()
            assert products.success
            product_id = products.data["products"][0]["product_id"]
            
            # 2. Get pricing
            pricing = await client.get_pricing(product_id)
            assert pricing.success
            
            # 3. Check availability
            avail = await client.check_availability(product_id, 1000000)
            assert avail.success
            
            # 4. Submit proposal
            proposal = await client.submit_proposal(
                product_id, 1000000, 10.0, "2026-02-01", "2026-02-28"
            )
            assert proposal.success
            
            # 5. Generate deal
            deal = await client.generate_deal_id(proposal.data["proposal_id"])
            assert deal.success
            assert "deal_id" in deal.data
```

**Deliverables:**
- [ ] Agent card tests
- [ ] JSON-RPC endpoint tests
- [ ] Client tests
- [ ] Full flow integration test

---

### Task 6: CLI Updates
**File:** `src/cli.py`
**Parallelizable:** Yes
**Estimate:** 30 minutes

Add A2A-specific CLI options:

```python
@click.option(
    "--a2a-mode/--no-a2a-mode",
    default=True,
    help="Use A2A protocol for Scenario B (default: enabled)",
)
@click.option(
    "--a2a-base-port",
    default=8100,
    help="Base port for seller A2A servers (default: 8100)",
)
```

**Deliverables:**
- [ ] `--a2a-mode` flag (default on)
- [ ] `--a2a-base-port` option
- [ ] Update help text

---

## Execution Plan for Parallel Development

```
┌──────────────────────────────────────────────────────────────────┐
│                        PARALLEL PHASE 1                          │
│                        (Can run simultaneously)                   │
├──────────────────────┬──────────────────────┬────────────────────┤
│   Agent 1 (Polecat)  │   Agent 2 (Polecat)  │  Agent 3 (Polecat) │
│                      │                      │                    │
│   Task 1: A2A Server │   Task 2: A2A Client │  Task 5: Tests     │
│   (a2a_server.py)    │   (a2a_client.py)    │  (test_a2a_*.py)   │
│                      │                      │                    │
│   ~2-3 hours         │   ~1-2 hours         │  ~1-2 hours        │
└──────────────────────┴──────────────────────┴────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        SEQUENTIAL PHASE 2                         │
│                        (Depends on Phase 1)                       │
├──────────────────────────────────────────────────────────────────┤
│                        Main Agent (NJ)                           │
│                                                                  │
│   Task 3: Update scenario_b.py negotiation logic                 │
│   Task 4: Update setup/teardown lifecycle                        │
│   Task 6: CLI updates                                            │
│                                                                  │
│   ~3-4 hours                                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        VALIDATION PHASE                          │
├──────────────────────────────────────────────────────────────────┤
│   • Run integration tests                                        │
│   • Run Scenario B with --a2a-mode                              │
│   • Compare metrics to Redis-based run                          │
│   • Verify agent cards match IAB spec                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Summary

| File | Action | Lines Est. |
|------|--------|------------|
| `src/agents/seller/a2a_server.py` | CREATE | ~400 |
| `src/agents/buyer/a2a_client.py` | CREATE | ~200 |
| `src/scenarios/scenario_b.py` | MODIFY | ~150 added |
| `src/cli.py` | MODIFY | ~20 added |
| `tests/test_a2a_integration.py` | CREATE | ~200 |
| **Total** | | **~970 lines** |

---

## Validation Checklist

- [ ] Agent card at `/.well-known/agent-card.json` matches A2A v0.3.0 spec
- [ ] JSON-RPC 2.0 `message/send` works with natural language
- [ ] JSON-RPC 2.0 `tasks/get` returns completed status
- [ ] Buyer can discover seller via agent card
- [ ] Full flow: discovery → pricing → availability → proposal → deal
- [ ] Context rot still applies (to client-side state)
- [ ] Hallucination injection still works (on message interpretation)
- [ ] Metrics collection unchanged
- [ ] 30-day simulation completes successfully
- [ ] Results comparable to Redis-based run (same discrepancy rates)

---

## Dependencies

```toml
# Add to pyproject.toml [project.dependencies]
httpx = ">=0.25.0"
uvicorn = ">=0.24.0"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Port conflicts | Use dynamic port allocation, configurable base port |
| Server startup race | Add health check loop before clients connect |
| Process cleanup failure | Use atexit handlers + SIGTERM handling |
| Performance regression | Async HTTP should be comparable to Redis |

---

## Success Criteria

1. **Functional:** Scenario B runs end-to-end with A2A protocol
2. **Compatible:** Agent cards pass IAB agentic-direct validation
3. **Equivalent:** Same reconciliation metrics as Redis version
4. **Testable:** All integration tests pass
5. **Documented:** Clear README updates for A2A mode

---

*Plan created: 2026-01-30*
*Target completion: 1 day with parallel agents*
