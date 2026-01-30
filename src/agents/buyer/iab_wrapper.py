"""
IAB Buyer Agent Wrapper - Full integration with IAB buyer-agent package.

This wrapper uses the IAB Tech Lab buyer-agent's UnifiedClient to:
- Use MCP protocol for structured operations (list products, create orders)
- Use A2A protocol for natural language requests
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
import structlog

# Add vendor path for IAB repos
VENDOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "vendor", "iab", "buyer-agent", "src"
)
if VENDOR_PATH not in sys.path:
    sys.path.insert(0, VENDOR_PATH)

from infrastructure.redis_bus import RedisBus
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
)

logger = structlog.get_logger()


@dataclass
class Campaign:
    """Campaign configuration for the buyer agent."""
    campaign_id: str
    name: str
    budget: float
    target_impressions: int
    target_cpm: float
    channel: str = "display"
    targeting: dict = field(default_factory=dict)
    impressions_delivered: int = 0
    spend: float = 0.0

    @property
    def remaining_budget(self) -> float:
        return self.budget - self.spend

    @property
    def remaining_impressions(self) -> int:
        return max(0, self.target_impressions - self.impressions_delivered)

    @property
    def is_active(self) -> bool:
        return self.remaining_budget > 0 and self.remaining_impressions > 0


@dataclass
class BuyerState:
    """State tracking for a buyer agent."""
    buyer_id: str
    campaigns: dict[str, Campaign] = field(default_factory=dict)
    deals_made: list[DealConfirmation] = field(default_factory=list)
    total_spend: float = 0.0
    total_impressions: int = 0
    llm_calls: int = 0
    llm_cost: float = 0.0


class IABBuyerWrapper:
    """
    Wrapper using IAB buyer-agent's UnifiedClient for real LLM-powered decisions.
    
    Supports two protocols:
    - MCP: Direct tool calls (faster, deterministic)
    - A2A: Natural language requests (flexible, AI-interpreted)
    """

    IAB_SERVER_URL = "https://agentic-direct-server-hwgrypmndq-uk.a.run.app"

    def __init__(
        self,
        buyer_id: str,
        scenario: str = "A",
        mock_llm: bool = False,
        redis_bus: Optional[RedisBus] = None,
    ):
        self.buyer_id = buyer_id
        self.scenario = scenario
        self.mock_llm = mock_llm or os.getenv("RTB_MOCK_LLM", "").lower() == "true"
        
        self._bus = redis_bus
        self._owned_bus = False
        
        # State tracking
        self.state = BuyerState(buyer_id=buyer_id)
        
        # IAB client (lazy loaded)
        self._unified_client = None
        self._a2a_client = None
        self._connected = False

        logger.info(
            "iab_buyer_wrapper.init",
            buyer_id=buyer_id,
            scenario=scenario,
            mock_llm=self.mock_llm,
        )

    async def connect(self) -> "IABBuyerWrapper":
        """Connect to IAB server and initialize clients."""
        if self._connected:
            return self
            
        # Connect Redis bus if needed
        if self._bus is None:
            self._bus = RedisBus()
            await self._bus.connect(consumer_id=f"buyer-{self.buyer_id}")
            self._owned_bus = True
        
        if not self.mock_llm:
            await self._init_iab_clients()
        
        self._connected = True
        logger.info(
            "iab_buyer_wrapper.connected",
            buyer_id=self.buyer_id,
            iab_connected=self._unified_client is not None,
        )
        return self

    async def disconnect(self) -> None:
        """Disconnect all clients."""
        if self._unified_client:
            try:
                await self._unified_client.close()
            except Exception:
                pass
            self._unified_client = None
            
        if self._a2a_client:
            try:
                await self._a2a_client.close()
            except Exception:
                pass
            self._a2a_client = None
            
        if self._bus and self._owned_bus:
            await self._bus.disconnect()
            self._bus = None
            
        self._connected = False
        logger.info(
            "iab_buyer_wrapper.disconnected",
            buyer_id=self.buyer_id,
            llm_calls=self.state.llm_calls,
            llm_cost=f"${self.state.llm_cost:.4f}",
        )

    async def __aenter__(self) -> "IABBuyerWrapper":
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def _init_iab_clients(self) -> None:
        """Initialize IAB buyer-agent clients."""
        try:
            from ad_buyer.clients.unified_client import UnifiedClient, Protocol
            from ad_buyer.clients.a2a_client import A2AClient
            
            # Initialize unified client with MCP protocol
            self._unified_client = UnifiedClient(
                base_url=self.IAB_SERVER_URL,
                protocol=Protocol.MCP,
                a2a_agent_type="buyer",
            )
            await self._unified_client.connect()
            
            # Also connect A2A for natural language
            await self._unified_client.connect(Protocol.A2A)
            
            logger.info(
                "iab_buyer_wrapper.iab_clients_connected",
                mcp_tools=len(self._unified_client.tools) if self._unified_client._mcp_client else 0,
            )
            
        except Exception as e:
            logger.warning(
                "iab_buyer_wrapper.iab_init_failed",
                error=str(e),
            )
            self._unified_client = None

    def add_campaign(self, campaign: Campaign) -> None:
        """Add a campaign to this buyer."""
        self.state.campaigns[campaign.campaign_id] = campaign

    def get_active_campaigns(self) -> list[Campaign]:
        """Get all active campaigns."""
        return [c for c in self.state.campaigns.values() if c.is_active]

    async def discover_inventory(
        self,
        campaign: Campaign,
    ) -> list[dict]:
        """Discover available inventory using IAB protocols.
        
        Uses MCP for structured queries or A2A for natural language.
        """
        if self.mock_llm or self._unified_client is None:
            return self._mock_discover_inventory(campaign)
        
        try:
            from ad_buyer.clients.unified_client import Protocol
            
            # Use MCP for structured discovery
            result = await self._unified_client.discover_inventory(
                channel=campaign.channel,
                max_cpm=campaign.target_cpm * 1.5,
                min_impressions=min(10000, campaign.remaining_impressions),
            )
            
            self.state.llm_calls += 1
            # Estimate cost based on MCP call complexity
            self.state.llm_cost += 0.001
            
            if result.success and result.data:
                logger.info(
                    "iab_buyer_wrapper.inventory_discovered",
                    buyer_id=self.buyer_id,
                    products=len(result.data) if isinstance(result.data, list) else 1,
                    protocol=result.protocol.value,
                )
                
                # Normalize response to list of dicts
                if isinstance(result.data, list):
                    return result.data
                elif isinstance(result.data, dict):
                    return result.data.get("products", [result.data])
                    
            return []
            
        except Exception as e:
            logger.error("iab_buyer_wrapper.discover_error", error=str(e))
            return self._mock_discover_inventory(campaign)

    def _mock_discover_inventory(self, campaign: Campaign) -> list[dict]:
        """Mock inventory discovery for testing."""
        return [
            {
                "product_id": f"prod-{campaign.channel}-001",
                "seller_id": "seller-001",
                "name": f"Premium {campaign.channel.title()} Inventory",
                "channel": campaign.channel,
                "base_cpm": campaign.target_cpm * 0.9,
                "available_impressions": 1000000,
                "floor_price": campaign.target_cpm * 0.7,
            }
        ]

    async def request_deal_via_a2a(
        self,
        campaign: Campaign,
        product: dict,
    ) -> Optional[dict]:
        """Request a deal using A2A natural language protocol.
        
        This demonstrates the natural language capabilities of A2A
        where the AI interprets the request and executes appropriate actions.
        """
        if self.mock_llm or self._unified_client is None:
            return self._mock_request_deal(campaign, product)
        
        try:
            # Construct natural language request
            nl_request = (
                f"I'd like to book {campaign.remaining_impressions:,} impressions "
                f"of {product.get('name', 'inventory')} "
                f"for campaign '{campaign.name}' "
                f"at around ${campaign.target_cpm:.2f} CPM. "
                f"This is for {campaign.channel} channel. "
                f"My budget is ${campaign.remaining_budget:.2f}."
            )
            
            result = await self._unified_client.send_natural_language(nl_request)
            
            self.state.llm_calls += 1
            # A2A calls involve LLM interpretation - higher cost
            self.state.llm_cost += 0.01
            
            if result.success:
                logger.info(
                    "iab_buyer_wrapper.a2a_deal_response",
                    buyer_id=self.buyer_id,
                    response_preview=str(result.data)[:200] if result.data else "None",
                )
                return result.data
                
            return None
            
        except Exception as e:
            logger.error("iab_buyer_wrapper.a2a_error", error=str(e))
            return self._mock_request_deal(campaign, product)

    async def request_deal_via_mcp(
        self,
        campaign: Campaign,
        product: dict,
    ) -> Optional[dict]:
        """Request a deal using MCP structured protocol.
        
        MCP provides direct tool calls without AI interpretation.
        """
        if self.mock_llm or self._unified_client is None:
            return self._mock_request_deal(campaign, product)
        
        try:
            result = await self._unified_client.request_deal(
                product_id=product.get("product_id", product.get("id")),
                deal_type="PD",  # Preferred Deal
                impressions=campaign.remaining_impressions,
                flight_start=datetime.utcnow().strftime("%Y-%m-%d"),
                flight_end=(datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d"),
                target_cpm=campaign.target_cpm,
            )
            
            self.state.llm_calls += 1
            # MCP calls are cheaper - no AI interpretation
            self.state.llm_cost += 0.0005
            
            if result.success:
                logger.info(
                    "iab_buyer_wrapper.mcp_deal_response",
                    buyer_id=self.buyer_id,
                    deal_id=result.data.get("deal_id") if result.data else None,
                )
                return result.data
                
            return None
            
        except Exception as e:
            logger.error("iab_buyer_wrapper.mcp_error", error=str(e))
            return self._mock_request_deal(campaign, product)

    def _mock_request_deal(self, campaign: Campaign, product: dict) -> dict:
        """Mock deal request for testing."""
        import uuid
        return {
            "deal_id": f"DEAL-{uuid.uuid4().hex[:8].upper()}",
            "product_id": product.get("product_id"),
            "price": campaign.target_cpm * 0.95,
            "impressions": min(campaign.remaining_impressions, 100000),
            "status": "accepted",
        }

    async def run_bidding_cycle(
        self,
        use_a2a: bool = False,
    ) -> list[DealConfirmation]:
        """Run a bidding cycle for all active campaigns.
        
        Args:
            use_a2a: If True, use A2A natural language protocol. Otherwise use MCP.
            
        Returns:
            List of deals made
        """
        deals = []
        active_campaigns = self.get_active_campaigns()
        
        for campaign in active_campaigns:
            try:
                # Discover inventory
                inventory = await self.discover_inventory(campaign)
                
                if not inventory:
                    continue
                
                # Select best product
                product = self._select_best_product(campaign, inventory)
                if not product:
                    continue
                
                # Request deal using chosen protocol
                if use_a2a:
                    deal_data = await self.request_deal_via_a2a(campaign, product)
                else:
                    deal_data = await self.request_deal_via_mcp(campaign, product)
                
                if deal_data and deal_data.get("deal_id"):
                    # Record the deal
                    confirmation = self._create_deal_confirmation(campaign, deal_data)
                    deals.append(confirmation)
                    self._record_deal(campaign, confirmation)
                    
            except Exception as e:
                logger.error(
                    "iab_buyer_wrapper.bidding_error",
                    campaign_id=campaign.campaign_id,
                    error=str(e),
                )
                
        return deals

    def _select_best_product(
        self,
        campaign: Campaign,
        inventory: list[dict],
    ) -> Optional[dict]:
        """Select the best product from inventory."""
        max_cpm = campaign.target_cpm * 1.5
        valid = [p for p in inventory if p.get("base_cpm", 0) <= max_cpm]
        
        if not valid:
            return None
        
        # Sort by CPM (closest to target)
        valid.sort(key=lambda x: abs(x.get("base_cpm", 0) - campaign.target_cpm))
        return valid[0]

    def _create_deal_confirmation(
        self,
        campaign: Campaign,
        deal_data: dict,
    ) -> DealConfirmation:
        """Create a deal confirmation from deal response."""
        impressions = deal_data.get("impressions", campaign.remaining_impressions)
        price = deal_data.get("price", campaign.target_cpm)
        
        return DealConfirmation(
            deal_id=deal_data.get("deal_id"),
            request_id=f"req-{campaign.campaign_id}",
            buyer_id=self.buyer_id,
            seller_id=deal_data.get("seller_id", "unknown"),
            cpm=price,
            impressions=impressions,
            total_cost=(impressions / 1000) * price,
            deal_type=DealType.PREFERRED_DEAL,
            terms={
                "product_id": deal_data.get("product_id"),
                "flight_start": deal_data.get("flight_start"),
                "flight_end": deal_data.get("flight_end"),
            },
        )

    def _record_deal(self, campaign: Campaign, deal: DealConfirmation) -> None:
        """Record a deal in state."""
        campaign.impressions_delivered += deal.impressions
        campaign.spend += deal.total_cost
        
        self.state.deals_made.append(deal)
        self.state.total_spend += deal.total_cost
        self.state.total_impressions += deal.impressions

    @property
    def llm_stats(self) -> dict:
        """Get LLM usage statistics."""
        return {
            "calls": self.state.llm_calls,
            "cost": self.state.llm_cost,
        }
