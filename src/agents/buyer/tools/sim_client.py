"""Simulation client for buyer agents.

Replaces the IAB buyer-agent's UnifiedClient/OpenDirectClient with
Redis Streams-based communication for the RTB simulation.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from infrastructure.redis_bus import RedisBus
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    CONSUMER_GROUPS,
)


@dataclass
class ClientResult:
    """Result wrapper for client operations."""

    success: bool
    data: Any = None
    error: Optional[str] = None


@dataclass
class InventoryItem:
    """Simulated inventory item from a seller."""

    product_id: str
    seller_id: str
    name: str
    channel: str
    base_cpm: float
    available_impressions: int
    targeting: list[str]
    floor_price: float
    deal_type: DealType = DealType.OPEN_AUCTION


class SimulationClient:
    """Client for buyer-seller communication via Redis Streams.

    This client replaces the IAB buyer-agent's OpenDirect/MCP/A2A clients
    with Redis Streams-based messaging for the simulation environment.
    """

    def __init__(
        self,
        buyer_id: str,
        redis_bus: Optional[RedisBus] = None,
        scenario: str = "A",
    ):
        """Initialize simulation client.

        Args:
            buyer_id: Unique identifier for this buyer
            redis_bus: Optional pre-configured Redis bus
            scenario: Simulation scenario ("A", "B", or "C")
        """
        self.buyer_id = buyer_id
        self._bus = redis_bus
        self._owned_bus = False
        self.scenario = scenario
        self._pending_requests: dict[str, BidRequest] = {}
        self._received_responses: dict[str, list[BidResponse]] = {}

    async def connect(self) -> "SimulationClient":
        """Connect to Redis Streams.

        Returns:
            Self for context manager chaining
        """
        if self._bus is None:
            from infrastructure.redis_bus import create_redis_bus

            self._bus = await create_redis_bus(
                consumer_id=f"buyer-{self.buyer_id}"
            )
            self._owned_bus = True
        return self

    async def disconnect(self) -> None:
        """Disconnect from Redis Streams."""
        if self._bus and self._owned_bus:
            await self._bus.disconnect()
            self._bus = None

    async def __aenter__(self) -> "SimulationClient":
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Product Discovery (replaces OpenDirect product search)
    # -------------------------------------------------------------------------

    async def search_products(
        self,
        query: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> ClientResult:
        """Search for available inventory from sellers.

        In the simulation, this publishes a bid request and waits for
        seller responses to discover available inventory.

        Args:
            query: Natural language query (used for logging)
            filters: Search filters (channel, maxPrice, etc.)

        Returns:
            ClientResult with list of InventoryItem
        """
        if not self._bus:
            return ClientResult(success=False, error="Not connected")

        filters = filters or {}

        # Create a discovery request
        request = BidRequest(
            buyer_id=self.buyer_id,
            campaign_id=f"discovery-{uuid.uuid4().hex[:8]}",
            channel=filters.get("channel", "display"),
            impressions_requested=filters.get("minImpressions", 100000),
            max_cpm=filters.get("maxPrice", 50.0),
            targeting=filters.get("targeting", {}),
        )

        # Publish request
        await self._bus.publish_bid_request(request)
        self._pending_requests[request.request_id] = request

        # Wait for responses (with timeout)
        responses = await self._wait_for_responses(
            request.request_id,
            timeout_seconds=5.0,
        )

        # Convert responses to inventory items
        items = [
            InventoryItem(
                product_id=f"PROD-{resp.seller_id}-{resp.response_id[:8]}",
                seller_id=resp.seller_id,
                name=f"Inventory from {resp.seller_id}",
                channel=request.channel,
                base_cpm=resp.offered_cpm,
                available_impressions=resp.available_impressions,
                targeting=list(request.targeting.keys()),
                floor_price=resp.offered_cpm * 0.8,
                deal_type=resp.deal_type,
            )
            for resp in responses
        ]

        return ClientResult(success=True, data=items)

    async def list_products(self) -> ClientResult:
        """List all available products (no filtering).

        Returns:
            ClientResult with list of InventoryItem
        """
        return await self.search_products()

    async def get_product(self, product_id: str) -> ClientResult:
        """Get details for a specific product.

        Args:
            product_id: Product identifier

        Returns:
            ClientResult with product details dict
        """
        # In simulation, we may not have persistent product catalog
        # Return a mock product based on the ID pattern
        parts = product_id.split("-")
        seller_id = parts[1] if len(parts) > 1 else "unknown"

        return ClientResult(
            success=True,
            data={
                "id": product_id,
                "name": f"Inventory from {seller_id}",
                "publisherId": seller_id,
                "basePrice": 20.0,  # Default CPM
                "channel": "display",
                "availableImpressions": 1000000,
                "targeting": ["geo", "demographic"],
            },
        )

    # -------------------------------------------------------------------------
    # Deal Negotiation (replaces OpenDirect order/line management)
    # -------------------------------------------------------------------------

    async def request_deal(
        self,
        seller_id: str,
        campaign_id: str,
        impressions: int,
        max_cpm: float,
        channel: str = "display",
        deal_type: DealType = DealType.OPEN_AUCTION,
        targeting: Optional[dict] = None,
    ) -> ClientResult:
        """Request a deal from a specific seller.

        Args:
            seller_id: Target seller ID
            campaign_id: Campaign identifier
            impressions: Requested impression volume
            max_cpm: Maximum CPM willing to pay
            channel: Ad channel
            deal_type: Type of deal to request
            targeting: Targeting parameters

        Returns:
            ClientResult with DealConfirmation if accepted
        """
        if not self._bus:
            return ClientResult(success=False, error="Not connected")

        # Create bid request
        request = BidRequest(
            buyer_id=self.buyer_id,
            campaign_id=campaign_id,
            channel=channel,
            impressions_requested=impressions,
            max_cpm=max_cpm,
            targeting=targeting or {},
        )

        # Route to specific seller (Scenario B/C) or broadcast (Scenario A)
        if self.scenario in ("B", "C"):
            await self._bus.route_to_seller(request, seller_id)
        else:
            await self._bus.publish_bid_request(request)

        self._pending_requests[request.request_id] = request

        # Wait for response
        responses = await self._wait_for_responses(
            request.request_id,
            timeout_seconds=10.0,
            expected_seller=seller_id,
        )

        if not responses:
            return ClientResult(
                success=False,
                error=f"No response from seller {seller_id}",
            )

        # Find best response (by CPM)
        best_response = min(responses, key=lambda r: r.offered_cpm)

        # Check if within our max CPM
        if best_response.offered_cpm > max_cpm:
            return ClientResult(
                success=False,
                error=f"Best offer {best_response.offered_cpm} exceeds max {max_cpm}",
            )

        # Create deal confirmation
        exchange_fee_pct = 0.15 if self.scenario == "A" else 0.0
        deal = DealConfirmation.from_deal(
            request=request,
            response=best_response,
            scenario=self.scenario,
            exchange_fee_pct=exchange_fee_pct,
        )

        # Publish deal
        await self._bus.publish_deal(deal)

        return ClientResult(success=True, data=deal)

    async def check_avails(
        self,
        seller_id: str,
        channel: str,
        impressions: int,
    ) -> ClientResult:
        """Check availability with a seller.

        Args:
            seller_id: Target seller ID
            channel: Ad channel
            impressions: Requested impression volume

        Returns:
            ClientResult with availability info
        """
        if not self._bus:
            return ClientResult(success=False, error="Not connected")

        # Create a quick discovery request
        request = BidRequest(
            buyer_id=self.buyer_id,
            campaign_id=f"avails-{uuid.uuid4().hex[:8]}",
            channel=channel,
            impressions_requested=impressions,
            max_cpm=100.0,  # High max to get all offers
            targeting={},
        )

        if self.scenario in ("B", "C"):
            await self._bus.route_to_seller(request, seller_id)
        else:
            await self._bus.publish_bid_request(request)

        responses = await self._wait_for_responses(
            request.request_id,
            timeout_seconds=3.0,
            expected_seller=seller_id,
        )

        if not responses:
            return ClientResult(
                success=True,
                data={
                    "available": False,
                    "impressions": 0,
                    "cpm": None,
                },
            )

        best = max(responses, key=lambda r: r.available_impressions)
        return ClientResult(
            success=True,
            data={
                "available": True,
                "impressions": best.available_impressions,
                "cpm": best.offered_cpm,
                "deal_type": best.deal_type.value,
            },
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    async def _wait_for_responses(
        self,
        request_id: str,
        timeout_seconds: float = 5.0,
        expected_seller: Optional[str] = None,
    ) -> list[BidResponse]:
        """Wait for bid responses to a request.

        Args:
            request_id: Request ID to match
            timeout_seconds: Maximum wait time
            expected_seller: If set, only accept from this seller

        Returns:
            List of matching BidResponse
        """
        if not self._bus:
            return []

        responses = []
        start_time = asyncio.get_event_loop().time()
        group = CONSUMER_GROUPS["buyers"]

        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            # Read responses
            raw_responses = await self._bus.read_bid_responses(
                group=group,
                count=10,
                block_ms=int((timeout_seconds * 1000) / 10),
            )

            for msg_id, response in raw_responses:
                if response.request_id == request_id:
                    if expected_seller is None or response.seller_id == expected_seller:
                        responses.append(response)

                # Acknowledge all messages we read
                await self._bus.ack_bid_responses(group, msg_id)

            # If we got responses, return (don't wait for more)
            if responses:
                break

        return responses
