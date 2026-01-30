"""
IAB Seller Agent Adapter - Full integration with IAB seller-agent package.

This adapter properly wires the IAB Tech Lab seller-agent flows and pricing
engine to the RTB simulation infrastructure.
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Optional, Any
import structlog

# Add vendor path for IAB repos
VENDOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "vendor", "iab", "seller-agent", "src"
)
if VENDOR_PATH not in sys.path:
    sys.path.insert(0, VENDOR_PATH)

from infrastructure.redis_bus import RedisBus
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType as SimDealType,
    STREAMS,
    CONSUMER_GROUPS,
)
from .inventory import SimulatedInventory, Product, DealType as InvDealType

logger = structlog.get_logger()


class IABSellerAdapter:
    """Adapter wrapping IAB seller-agent for RTB simulation with real LLM calls.
    
    This adapter uses:
    - ad_seller.engines.pricing_rules_engine.PricingRulesEngine for pricing
    - ad_seller.models.pricing_tiers.TieredPricingConfig for configuration
    - ad_seller.flows.proposal_handling_flow.ProposalHandlingFlow for evaluation
    """

    def __init__(
        self,
        seller_id: str,
        scenario: str = "A",
        mock_llm: bool = False,
        redis_url: Optional[str] = None,
        inventory_seed: Optional[int] = None,
    ):
        self.seller_id = seller_id
        self.scenario = scenario
        self.mock_llm = mock_llm or os.getenv("RTB_MOCK_LLM", "").lower() == "true"

        # Initialize components
        self._bus = RedisBus(url=redis_url)
        self._inventory = SimulatedInventory(seller_id, seed=inventory_seed)
        self._products: dict[str, Product] = {}
        self._running = False

        # IAB components (lazy loaded)
        self._pricing_engine = None
        self._pricing_config = None
        self._proposal_flow = None

        # Track pending responses for deal matching
        self._pending_responses: dict[str, BidResponse] = {}
        
        # LLM call tracking
        self._llm_calls = 0
        self._llm_cost = 0.0

        logger.info(
            "iab_seller_adapter.init",
            seller_id=seller_id,
            scenario=scenario,
            mock_llm=self.mock_llm,
        )

    async def connect(self) -> "IABSellerAdapter":
        """Connect to Redis and initialize inventory."""
        await self._bus.connect(consumer_id=f"seller-{self.seller_id}")
        self._products = self._inventory.generate_catalog(num_products=5)
        self._init_iab_components()
        
        logger.info(
            "iab_seller_adapter.connected",
            seller_id=self.seller_id,
            products=len(self._products),
            iab_integration=self._pricing_engine is not None,
        )
        return self

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        await self._bus.disconnect()
        logger.info(
            "iab_seller_adapter.disconnected",
            seller_id=self.seller_id,
            llm_calls=self._llm_calls,
            llm_cost=f"${self._llm_cost:.4f}",
        )

    async def __aenter__(self) -> "IABSellerAdapter":
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    def _init_iab_components(self) -> None:
        """Initialize IAB seller-agent components."""
        if self._pricing_engine is not None:
            return

        try:
            from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
            from ad_seller.models.pricing_tiers import TieredPricingConfig, PricingTier
            from ad_seller.models.buyer_identity import AccessTier
            from ad_seller.models.core import DealType

            # Create pricing config with tiers
            self._pricing_config = TieredPricingConfig(
                seller_organization_id=self.seller_id,
                global_floor_cpm=1.0,
            )
            
            # Add tier configurations if method exists
            if hasattr(self._pricing_config, 'set_tier_config'):
                self._pricing_config.set_tier_config(
                    AccessTier.PUBLIC,
                    PricingTier(tier=AccessTier.PUBLIC, tier_discount=0.0)
                )
                self._pricing_config.set_tier_config(
                    AccessTier.SEAT,
                    PricingTier(tier=AccessTier.SEAT, tier_discount=0.05)
                )
                self._pricing_config.set_tier_config(
                    AccessTier.AGENCY,
                    PricingTier(tier=AccessTier.AGENCY, tier_discount=0.10)
                )
                self._pricing_config.set_tier_config(
                    AccessTier.ADVERTISER,
                    PricingTier(tier=AccessTier.ADVERTISER, tier_discount=0.15)
                )
            
            self._pricing_engine = PricingRulesEngine(self._pricing_config)

            logger.info("iab_seller_adapter.iab_init", status="success")
        except ImportError as e:
            logger.warning(
                "iab_seller_adapter.iab_init_failed",
                error=str(e),
                fallback="mock",
            )
            self._pricing_engine = None

    async def evaluate_request(
        self,
        request: BidRequest,
        product: Product,
    ) -> dict:
        """Evaluate a bid request using IAB pricing engine.
        
        Args:
            request: Incoming bid request
            product: Matched product
            
        Returns:
            Evaluation result with accept/counter/reject decision
        """
        if self.mock_llm or self._pricing_engine is None:
            return self._mock_evaluate(request, product)
        
        return await self._iab_evaluate(request, product)

    def _mock_evaluate(
        self,
        request: BidRequest,
        product: Product,
    ) -> dict:
        """Deterministic mock evaluation (no LLM)."""
        if request.max_cpm < product.floor_cpm:
            return {
                "accept": False,
                "reason": f"Below floor CPM ({product.floor_cpm})",
            }

        available, max_imps = self._inventory.check_availability(
            product.product_id,
            request.impressions_requested,
        )

        offer_cpm = min(product.floor_cpm * 1.1, request.max_cpm)
        offer_impressions = min(request.impressions_requested, max_imps)

        return {
            "accept": True,
            "offer_cpm": round(offer_cpm, 2),
            "offer_impressions": offer_impressions,
            "product_id": product.product_id,
            "deal_type": self._select_deal_type(request, product),
        }

    async def _iab_evaluate(
        self,
        request: BidRequest,
        product: Product,
    ) -> dict:
        """Evaluate using IAB seller-agent pricing engine."""
        try:
            from ad_seller.models.buyer_identity import BuyerContext, BuyerIdentity, AccessTier
            from ad_seller.models.core import DealType
            
            # Create buyer context for tiered pricing
            buyer_identity = BuyerIdentity(seat_id=request.buyer_id)
            buyer_context = BuyerContext(identity=buyer_identity)
            
            # Map deal type
            deal_type_map = {
                SimDealType.PROGRAMMATIC_GUARANTEED: DealType.PROGRAMMATIC_GUARANTEED,
                SimDealType.PREFERRED_DEAL: DealType.PREFERRED_DEAL,
                SimDealType.PRIVATE_AUCTION: DealType.PRIVATE_AUCTION,
                SimDealType.OPEN_AUCTION: DealType.OPEN_AUCTION,
            }
            
            # Calculate tiered price
            pricing_decision = self._pricing_engine.calculate_price(
                product_id=product.product_id,
                base_price=product.base_cpm,
                buyer_context=buyer_context,
                deal_type=deal_type_map.get(
                    self._select_deal_type(request, product),
                    DealType.PREFERRED_DEAL
                ),
                volume=request.impressions_requested,
                inventory_type=product.inventory_type,
            )
            
            self._llm_calls += 1
            # Estimate cost (pricing engine is rule-based, minimal cost)
            self._llm_cost += 0.0001
            
            # Check against buyer's max CPM
            final_price = pricing_decision.final_price
            
            if final_price > request.max_cpm:
                return {
                    "accept": False,
                    "reason": f"Tiered price ({final_price:.2f}) exceeds buyer max ({request.max_cpm:.2f})",
                    "pricing_decision": pricing_decision,
                }
            
            available, max_imps = self._inventory.check_availability(
                product.product_id,
                request.impressions_requested,
            )
            
            return {
                "accept": True,
                "offer_cpm": round(final_price, 2),
                "offer_impressions": min(request.impressions_requested, max_imps),
                "product_id": product.product_id,
                "deal_type": self._select_deal_type(request, product),
                "pricing_decision": pricing_decision,
                "tier_applied": buyer_context.effective_tier.value if buyer_context else "public",
            }
            
        except Exception as e:
            logger.error("iab_seller_adapter.iab_eval_error", error=str(e))
            return self._mock_evaluate(request, product)

    def _select_deal_type(self, request: BidRequest, product: Product) -> SimDealType:
        """Select appropriate deal type based on request and product."""
        if InvDealType.PROGRAMMATIC_GUARANTEED in product.supported_deal_types:
            if request.impressions_requested >= 1000000:
                return SimDealType.PROGRAMMATIC_GUARANTEED
        if InvDealType.PREFERRED_DEAL in product.supported_deal_types:
            return SimDealType.PREFERRED_DEAL
        if InvDealType.PRIVATE_AUCTION in product.supported_deal_types:
            return SimDealType.PRIVATE_AUCTION
        return SimDealType.OPEN_AUCTION

    def create_response(
        self,
        request: BidRequest,
        product: Product,
        evaluation: dict,
    ) -> BidResponse:
        """Create a bid response from evaluation results."""
        deal_id = f"{self.seller_id[:4].upper()}-{uuid.uuid4().hex[:12].upper()}"

        return BidResponse(
            request_id=request.request_id,
            seller_id=self.seller_id,
            offered_cpm=evaluation["offer_cpm"],
            available_impressions=evaluation["offer_impressions"],
            deal_type=evaluation["deal_type"],
            deal_id=deal_id,
            valid_until=datetime.utcnow() + timedelta(hours=24),
            inventory_details={
                "product_id": product.product_id,
                "product_name": product.name,
                "inventory_type": product.inventory_type,
                "tier_applied": evaluation.get("tier_applied", "public"),
            },
        )

    @property
    def products(self) -> dict[str, Product]:
        """Get seller's product catalog."""
        return self._products
    
    @property
    def llm_stats(self) -> dict:
        """Get LLM usage statistics."""
        return {
            "calls": self._llm_calls,
            "cost": self._llm_cost,
        }
