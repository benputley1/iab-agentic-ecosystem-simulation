"""CrewAI tools for buyer agents in RTB simulation.

These tools wrap the SimulationClient to provide CrewAI-compatible
interfaces for inventory discovery and deal management.
"""

import asyncio
from typing import Any, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .sim_client import SimulationClient, InventoryItem
from ....infrastructure.message_schemas import DealType


# -----------------------------------------------------------------------------
# Input Schemas
# -----------------------------------------------------------------------------


class DiscoverInventoryInput(BaseModel):
    """Input schema for inventory discovery tool."""

    query: Optional[str] = Field(
        default=None,
        description="Natural language query for inventory (e.g., 'CTV inventory under $25 CPM')",
    )
    channel: Optional[str] = Field(
        default=None,
        description="Channel filter (e.g., 'ctv', 'display', 'video', 'mobile')",
    )
    max_cpm: Optional[float] = Field(
        default=None,
        description="Maximum CPM price filter",
        ge=0,
    )
    min_impressions: Optional[int] = Field(
        default=None,
        description="Minimum available impressions filter",
        ge=0,
    )


class RequestDealInput(BaseModel):
    """Input schema for deal request tool."""

    seller_id: str = Field(
        ...,
        description="Seller ID to request deal from",
    )
    campaign_id: str = Field(
        ...,
        description="Campaign ID for this deal",
    )
    impressions: int = Field(
        ...,
        description="Number of impressions to request",
        ge=1,
    )
    max_cpm: float = Field(
        ...,
        description="Maximum CPM willing to pay",
        ge=0,
    )
    channel: str = Field(
        default="display",
        description="Ad channel (display, video, ctv, native)",
    )
    deal_type: str = Field(
        default="OA",
        description="Deal type: OA (Open Auction), PD (Preferred Deal), PG (Programmatic Guaranteed)",
    )


class CheckAvailsInput(BaseModel):
    """Input schema for availability check tool."""

    seller_id: str = Field(
        ...,
        description="Seller ID to check availability with",
    )
    channel: str = Field(
        default="display",
        description="Ad channel to check",
    )
    impressions: int = Field(
        ...,
        description="Requested impression volume",
        ge=1,
    )


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class SimDiscoverInventoryTool(BaseTool):
    """Discover available advertising inventory from sellers.

    This tool searches for inventory across sellers in the simulation,
    using Redis Streams for A2A communication.
    """

    name: str = "discover_inventory"
    description: str = """Discover available advertising inventory from sellers.
Search for inventory matching your campaign requirements.

Args:
    query: Natural language query (e.g., 'CTV inventory under $25 CPM')
    channel: Channel filter ('ctv', 'display', 'video', 'mobile')
    max_cpm: Maximum CPM price
    min_impressions: Minimum available impressions

Returns:
    List of available inventory with pricing and targeting info."""

    args_schema: type[BaseModel] = DiscoverInventoryInput
    _client: SimulationClient

    def __init__(self, client: SimulationClient, **kwargs: Any):
        """Initialize with simulation client.

        Args:
            client: SimulationClient for seller communication
        """
        super().__init__(**kwargs)
        self._client = client

    def _run(
        self,
        query: Optional[str] = None,
        channel: Optional[str] = None,
        max_cpm: Optional[float] = None,
        min_impressions: Optional[int] = None,
    ) -> str:
        """Synchronous wrapper for async discovery."""
        return asyncio.run(
            self._arun(
                query=query,
                channel=channel,
                max_cpm=max_cpm,
                min_impressions=min_impressions,
            )
        )

    async def _arun(
        self,
        query: Optional[str] = None,
        channel: Optional[str] = None,
        max_cpm: Optional[float] = None,
        min_impressions: Optional[int] = None,
    ) -> str:
        """Discover inventory with filters."""
        try:
            filters = {}
            if channel:
                filters["channel"] = channel
            if max_cpm is not None:
                filters["maxPrice"] = max_cpm
            if min_impressions is not None:
                filters["minImpressions"] = min_impressions

            result = await self._client.search_products(
                query=query,
                filters=filters if filters else None,
            )

            if not result.success:
                return f"Error discovering inventory: {result.error}"

            return self._format_results(result.data, query)

        except Exception as e:
            return f"Error discovering inventory: {e}"

    def _format_results(
        self,
        items: list[InventoryItem],
        query: Optional[str],
    ) -> str:
        """Format discovery results."""
        if not items:
            return "No inventory found matching your criteria."

        output_lines = [
            "Inventory Discovery Results",
            f"Scenario: {self._client.scenario}",
            "-" * 50,
            "",
        ]

        for i, item in enumerate(items, 1):
            output_lines.extend([
                f"{i}. {item.name}",
                f"   Product ID: {item.product_id}",
                f"   Seller: {item.seller_id}",
                f"   Channel: {item.channel}",
                f"   CPM: ${item.base_cpm:.2f}",
                f"   Floor: ${item.floor_price:.2f}",
                f"   Available: {item.available_impressions:,}",
                f"   Deal Type: {item.deal_type.value}",
                f"   Targeting: {', '.join(item.targeting) if item.targeting else 'Standard'}",
                "",
            ])

        output_lines.append("-" * 50)
        output_lines.append(f"Total products found: {len(items)}")

        return "\n".join(output_lines)


class SimRequestDealTool(BaseTool):
    """Request a deal from a seller for programmatic activation.

    This tool creates deals through the RTB simulation, using
    Redis Streams for buyer-seller communication.
    """

    name: str = "request_deal"
    description: str = """Request a deal from a seller for programmatic activation.
Negotiate and book inventory with a specific seller.

Args:
    seller_id: Seller ID to request deal from
    campaign_id: Campaign ID for this deal
    impressions: Number of impressions to request
    max_cpm: Maximum CPM willing to pay
    channel: Ad channel (display, video, ctv, native)
    deal_type: Deal type (OA, PD, PG)

Returns:
    Deal confirmation with ID and pricing, or error if rejected."""

    args_schema: type[BaseModel] = RequestDealInput
    _client: SimulationClient

    def __init__(self, client: SimulationClient, **kwargs: Any):
        """Initialize with simulation client.

        Args:
            client: SimulationClient for seller communication
        """
        super().__init__(**kwargs)
        self._client = client

    def _run(
        self,
        seller_id: str,
        campaign_id: str,
        impressions: int,
        max_cpm: float,
        channel: str = "display",
        deal_type: str = "OA",
    ) -> str:
        """Synchronous wrapper for async deal request."""
        return asyncio.run(
            self._arun(
                seller_id=seller_id,
                campaign_id=campaign_id,
                impressions=impressions,
                max_cpm=max_cpm,
                channel=channel,
                deal_type=deal_type,
            )
        )

    async def _arun(
        self,
        seller_id: str,
        campaign_id: str,
        impressions: int,
        max_cpm: float,
        channel: str = "display",
        deal_type: str = "OA",
    ) -> str:
        """Request a deal from the seller."""
        try:
            # Parse deal type
            try:
                deal_type_enum = DealType(deal_type.upper())
            except ValueError:
                deal_type_enum = DealType.OPEN_AUCTION

            result = await self._client.request_deal(
                seller_id=seller_id,
                campaign_id=campaign_id,
                impressions=impressions,
                max_cpm=max_cpm,
                channel=channel,
                deal_type=deal_type_enum,
            )

            if not result.success:
                return f"Deal request failed: {result.error}"

            return self._format_deal(result.data)

        except Exception as e:
            return f"Error requesting deal: {e}"

    def _format_deal(self, deal) -> str:
        """Format deal confirmation for output."""
        output_lines = [
            "=" * 60,
            "DEAL CONFIRMED",
            "=" * 60,
            "",
            f"Deal ID: {deal.deal_id}",
            "",
            "Deal Details",
            "-" * 30,
            f"Buyer: {deal.buyer_id}",
            f"Seller: {deal.seller_id}",
            f"Scenario: {deal.scenario}",
            f"Impressions: {deal.impressions:,}",
            "",
            "Pricing",
            "-" * 30,
            f"CPM: ${deal.cpm:.2f}",
            f"Total Cost: ${deal.total_cost:.2f}",
        ]

        if deal.exchange_fee > 0:
            output_lines.extend([
                f"Exchange Fee: ${deal.exchange_fee:.2f} ({deal.fee_percentage:.1f}%)",
                f"Seller Revenue: ${deal.seller_revenue:.2f}",
            ])

        if deal.ledger_entry_id:
            output_lines.extend([
                "",
                "Ledger",
                "-" * 30,
                f"Entry ID: {deal.ledger_entry_id}",
            ])
            if deal.blob_id:
                output_lines.append(f"Blob ID: {deal.blob_id}")

        output_lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(output_lines)


class SimCheckAvailsTool(BaseTool):
    """Check inventory availability with a seller.

    This tool queries a seller to check if they have available
    inventory matching the request.
    """

    name: str = "check_avails"
    description: str = """Check inventory availability with a seller.
Query a seller to see if they have inventory available.

Args:
    seller_id: Seller ID to check availability with
    channel: Ad channel to check (display, video, ctv, native)
    impressions: Requested impression volume

Returns:
    Availability status with impressions and pricing if available."""

    args_schema: type[BaseModel] = CheckAvailsInput
    _client: SimulationClient

    def __init__(self, client: SimulationClient, **kwargs: Any):
        """Initialize with simulation client.

        Args:
            client: SimulationClient for seller communication
        """
        super().__init__(**kwargs)
        self._client = client

    def _run(
        self,
        seller_id: str,
        channel: str = "display",
        impressions: int = 100000,
    ) -> str:
        """Synchronous wrapper for async avails check."""
        return asyncio.run(
            self._arun(
                seller_id=seller_id,
                channel=channel,
                impressions=impressions,
            )
        )

    async def _arun(
        self,
        seller_id: str,
        channel: str = "display",
        impressions: int = 100000,
    ) -> str:
        """Check availability with seller."""
        try:
            result = await self._client.check_avails(
                seller_id=seller_id,
                channel=channel,
                impressions=impressions,
            )

            if not result.success:
                return f"Error checking availability: {result.error}"

            data = result.data
            if not data.get("available"):
                return f"No availability from {seller_id} for {channel} channel."

            return (
                f"Availability from {seller_id}:\n"
                f"  Channel: {channel}\n"
                f"  Available: {data['impressions']:,} impressions\n"
                f"  CPM: ${data['cpm']:.2f}\n"
                f"  Deal Type: {data['deal_type']}"
            )

        except Exception as e:
            return f"Error checking availability: {e}"
