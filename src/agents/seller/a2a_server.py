"""
A2A v0.3.0 Server for Seller Agent.

Endpoints:
- GET  /a2a/seller/.well-known/agent-card.json  (Agent discovery)
- POST /a2a/seller/jsonrpc                       (JSON-RPC 2.0)
- POST /a2a/seller/                              (Default endpoint)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional
import uuid
from datetime import datetime

from .iab_adapter import IABSellerAdapter
from .inventory import Product

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
_adapter: Optional[IABSellerAdapter] = None
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
    # Note: More specific intents must be checked before general ones
    
    # Deal generation - check first as it's most specific
    if any(kw in text_lower for kw in ["generate deal", "deal id", "activate", "dsp"]):
        return await handle_deal_generation(text)
    # Discovery
    elif any(kw in text_lower for kw in ["list", "inventory", "products", "what do you have"]):
        return await handle_discovery(text)
    # Pricing
    elif any(kw in text_lower for kw in ["price", "pricing", "cpm", "cost", "rate"]):
        return await handle_pricing(text)
    # Availability - check keywords more carefully to avoid overlap with discovery
    elif any(kw in text_lower for kw in ["availability", "check available", "is available", "can you deliver"]):
        return await handle_availability(text)
    # Proposal - general deal intent
    elif any(kw in text_lower for kw in ["propose", "proposal", "deal", "buy", "book"]):
        return await handle_proposal(text)
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
