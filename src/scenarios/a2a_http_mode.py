"""
A2A HTTP Mode for Scenario B.

This module provides HTTP-based A2A communication following the IAB agentic-direct
specification. It replaces Redis pub/sub with direct HTTP JSON-RPC 2.0 calls
between buyer and seller agents.

Usage:
    from scenarios.a2a_http_mode import A2AHTTPManager

    manager = A2AHTTPManager(base_port=8100)
    await manager.start_seller_servers(seller_adapters)
    await manager.create_buyer_clients(seller_ports)
    
    # Use clients
    response = await manager.negotiate_deal(buyer_id, seller_id, request)
    
    await manager.shutdown()
"""

import asyncio
import multiprocessing
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import structlog

from agents.seller.iab_adapter import IABSellerAdapter
from agents.buyer.a2a_client import A2AClient, A2AResponse
from infrastructure.message_schemas import BidRequest, BidResponse, DealType

logger = structlog.get_logger()


@dataclass
class A2AServerInfo:
    """Information about a running A2A server."""
    seller_id: str
    port: int
    process: Optional[multiprocessing.Process] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    healthy: bool = False


@dataclass
class A2ANegotiationResult:
    """Result of an A2A negotiation."""
    success: bool
    deal_id: Optional[str] = None
    product_id: Optional[str] = None
    cpm: float = 0.0
    impressions: int = 0
    error: Optional[str] = None
    steps: list[str] = field(default_factory=list)


class A2AHTTPManager:
    """
    Manages A2A HTTP servers and clients for Scenario B.
    
    Handles:
    - Starting seller A2A servers (one per seller)
    - Creating buyer A2A clients (one per buyer-seller pair)
    - Coordinating negotiations via HTTP JSON-RPC
    - Health checking and lifecycle management
    """
    
    def __init__(
        self,
        base_port: int = 8100,
        health_check_timeout: float = 30.0,
        request_timeout: float = 30.0,
    ):
        self.base_port = base_port
        self.health_check_timeout = health_check_timeout
        self.request_timeout = request_timeout
        
        # Server tracking
        self._servers: dict[str, A2AServerInfo] = {}  # seller_id -> server info
        
        # Client pools: buyer_id -> {seller_id -> client}
        self._clients: dict[str, dict[str, A2AClient]] = {}
        
        # Metrics
        self._negotiations_attempted = 0
        self._negotiations_succeeded = 0
        self._total_latency_ms = 0.0
    
    async def start_seller_servers(
        self,
        seller_adapters: dict[str, IABSellerAdapter],
    ) -> dict[str, int]:
        """
        Start A2A servers for all sellers.
        
        Args:
            seller_adapters: Dict of seller_id -> IABSellerAdapter
            
        Returns:
            Dict of seller_id -> port
        """
        ports = {}
        
        for i, (seller_id, adapter) in enumerate(seller_adapters.items()):
            port = self.base_port + i
            
            # Start server in subprocess
            process = multiprocessing.Process(
                target=_run_seller_server,
                args=(seller_id, adapter.seller_id, port),
                daemon=True,
            )
            process.start()
            
            self._servers[seller_id] = A2AServerInfo(
                seller_id=seller_id,
                port=port,
                process=process,
            )
            ports[seller_id] = port
            
            logger.info(
                "a2a_http.server_started",
                seller_id=seller_id,
                port=port,
                pid=process.pid,
            )
        
        # Wait for servers to be ready
        await self._wait_for_servers()
        
        return ports
    
    async def _wait_for_servers(self) -> None:
        """Wait for all servers to be healthy."""
        import httpx
        
        start_time = time.time()
        
        while time.time() - start_time < self.health_check_timeout:
            all_healthy = True
            
            for seller_id, info in self._servers.items():
                if info.healthy:
                    continue
                    
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        url = f"http://localhost:{info.port}/a2a/seller/.well-known/agent-card.json"
                        response = await client.get(url)
                        if response.status_code == 200:
                            info.healthy = True
                            logger.info(
                                "a2a_http.server_healthy",
                                seller_id=seller_id,
                                port=info.port,
                            )
                except Exception:
                    all_healthy = False
            
            if all_healthy:
                logger.info(
                    "a2a_http.all_servers_healthy",
                    count=len(self._servers),
                )
                return
            
            await asyncio.sleep(0.5)
        
        # Report unhealthy servers
        unhealthy = [s for s, i in self._servers.items() if not i.healthy]
        if unhealthy:
            logger.warning(
                "a2a_http.servers_unhealthy",
                unhealthy=unhealthy,
            )
    
    async def create_buyer_clients(
        self,
        buyer_ids: list[str],
    ) -> None:
        """
        Create A2A clients for all buyers.
        
        Each buyer gets a client to each seller.
        
        Args:
            buyer_ids: List of buyer IDs
        """
        for buyer_id in buyer_ids:
            self._clients[buyer_id] = {}
            
            for seller_id, info in self._servers.items():
                client = A2AClient(
                    base_url=f"http://localhost:{info.port}",
                    agent_role="seller",
                    timeout=self.request_timeout,
                )
                await client.connect()
                self._clients[buyer_id][seller_id] = client
                
            logger.info(
                "a2a_http.buyer_clients_created",
                buyer_id=buyer_id,
                seller_count=len(self._servers),
            )
    
    async def negotiate_deal(
        self,
        buyer_id: str,
        seller_id: str,
        request: BidRequest,
    ) -> A2ANegotiationResult:
        """
        Negotiate a deal via A2A HTTP protocol.
        
        Follows IAB agentic-direct flow:
        1. Discovery - list products
        2. Pricing - get CPMs
        3. Availability - check inventory
        4. Proposal - submit deal request
        5. Deal generation - get Deal ID
        
        Args:
            buyer_id: Buyer agent ID
            seller_id: Seller agent ID
            request: Bid request with campaign details
            
        Returns:
            A2ANegotiationResult with deal info or error
        """
        self._negotiations_attempted += 1
        start_time = time.time()
        steps = []
        
        client = self._clients.get(buyer_id, {}).get(seller_id)
        if not client:
            return A2ANegotiationResult(
                success=False,
                error=f"No client for buyer={buyer_id}, seller={seller_id}",
            )
        
        try:
            # Step 1: Discovery
            steps.append("discovery")
            products_response = await client.list_products()
            if not products_response.success:
                return A2ANegotiationResult(
                    success=False,
                    error=f"Discovery failed: {products_response.error}",
                    steps=steps,
                )
            
            products = products_response.data.get("products", [])
            if not products:
                return A2ANegotiationResult(
                    success=False,
                    error="No products available",
                    steps=steps,
                )
            
            # Find matching product by channel
            matching = [
                p for p in products
                if p.get("inventory_type", "").lower() == request.channel.lower()
            ]
            if not matching:
                # Fall back to first product
                matching = products[:1]
            
            product = matching[0]
            product_id = product["product_id"]
            
            # Step 2: Pricing
            steps.append("pricing")
            pricing_response = await client.get_pricing(product_id)
            if not pricing_response.success:
                return A2ANegotiationResult(
                    success=False,
                    error=f"Pricing failed: {pricing_response.error}",
                    steps=steps,
                )
            
            pricing = pricing_response.data.get("pricing", [{}])[0]
            offered_cpm = pricing.get("base_cpm", 0)
            
            # Check if within budget
            if offered_cpm > request.max_cpm * 1.1:  # 10% tolerance
                return A2ANegotiationResult(
                    success=False,
                    error=f"Price too high: ${offered_cpm} > ${request.max_cpm}",
                    steps=steps,
                )
            
            # Step 3: Availability
            steps.append("availability")
            avail_response = await client.check_availability(
                product_id,
                request.impressions_requested,
            )
            
            availability = avail_response.data.get("availability", [{}])[0]
            if not availability.get("available", False):
                max_imps = availability.get("max_impressions", 0)
                return A2ANegotiationResult(
                    success=False,
                    error=f"Insufficient inventory: max {max_imps:,}",
                    steps=steps,
                )
            
            # Step 4: Proposal
            steps.append("proposal")
            proposal_response = await client.submit_proposal(
                product_id=product_id,
                impressions=request.impressions_requested,
                cpm=min(offered_cpm, request.max_cpm),
                start_date=request.start_date or "2026-02-01",
                end_date=request.end_date or "2026-02-28",
            )
            
            proposal_status = proposal_response.data.get("status", "")
            if proposal_status not in ("accepted", "accept"):
                return A2ANegotiationResult(
                    success=False,
                    error=f"Proposal {proposal_status}",
                    steps=steps,
                )
            
            proposal_id = proposal_response.data.get("proposal_id", "")
            
            # Step 5: Deal generation
            steps.append("deal_generation")
            deal_response = await client.generate_deal_id(proposal_id)
            
            deal_id = deal_response.data.get("deal_id", "")
            if not deal_id:
                return A2ANegotiationResult(
                    success=False,
                    error="No deal ID generated",
                    steps=steps,
                )
            
            # Success!
            elapsed_ms = (time.time() - start_time) * 1000
            self._negotiations_succeeded += 1
            self._total_latency_ms += elapsed_ms
            
            logger.info(
                "a2a_http.negotiation_success",
                buyer_id=buyer_id,
                seller_id=seller_id,
                deal_id=deal_id,
                product_id=product_id,
                cpm=offered_cpm,
                impressions=request.impressions_requested,
                latency_ms=round(elapsed_ms, 2),
            )
            
            return A2ANegotiationResult(
                success=True,
                deal_id=deal_id,
                product_id=product_id,
                cpm=offered_cpm,
                impressions=request.impressions_requested,
                steps=steps,
            )
            
        except Exception as e:
            logger.error(
                "a2a_http.negotiation_error",
                buyer_id=buyer_id,
                seller_id=seller_id,
                error=str(e),
                steps=steps,
            )
            return A2ANegotiationResult(
                success=False,
                error=str(e),
                steps=steps,
            )
    
    async def shutdown(self) -> None:
        """Shutdown all servers and clients."""
        # Close clients
        for buyer_id, clients in self._clients.items():
            for seller_id, client in clients.items():
                try:
                    await client.disconnect()
                except Exception:
                    pass
        self._clients.clear()
        
        # Stop servers
        for seller_id, info in self._servers.items():
            if info.process and info.process.is_alive():
                info.process.terminate()
                info.process.join(timeout=5)
                if info.process.is_alive():
                    info.process.kill()
        self._servers.clear()
        
        logger.info(
            "a2a_http.shutdown",
            negotiations_attempted=self._negotiations_attempted,
            negotiations_succeeded=self._negotiations_succeeded,
            avg_latency_ms=round(
                self._total_latency_ms / max(1, self._negotiations_succeeded), 2
            ),
        )
    
    @property
    def stats(self) -> dict:
        """Get negotiation statistics."""
        return {
            "attempted": self._negotiations_attempted,
            "succeeded": self._negotiations_succeeded,
            "success_rate": (
                self._negotiations_succeeded / max(1, self._negotiations_attempted)
            ),
            "avg_latency_ms": (
                self._total_latency_ms / max(1, self._negotiations_succeeded)
            ),
        }


def _run_seller_server(seller_id: str, adapter_seller_id: str, port: int) -> None:
    """
    Run seller A2A server in subprocess.
    
    This is called in a separate process to avoid asyncio event loop issues.
    """
    import asyncio
    
    async def _main():
        from agents.seller.iab_adapter import IABSellerAdapter
        from agents.seller.a2a_server import create_app
        import uvicorn
        
        # Create fresh adapter in this process
        adapter = IABSellerAdapter(adapter_seller_id, mock_llm=True)
        await adapter.connect()
        
        # Create and run app
        app = create_app(adapter, f"http://localhost:{port}")
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    asyncio.run(_main())
