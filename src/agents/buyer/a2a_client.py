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
