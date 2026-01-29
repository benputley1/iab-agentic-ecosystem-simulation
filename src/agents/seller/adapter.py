"""
Seller Agent Adapter - Wraps IAB seller-agent flows for RTB simulation.

This adapter:
1. Listens to Redis Streams for incoming BidRequests
2. Translates to IAB seller-agent proposal format
3. Runs ProposalHandlingFlow for evaluation
4. Generates BidResponses back to Redis

Supports three simulation scenarios:
- Scenario A: Exchange-mediated (fees extracted)
- Scenario B: Direct A2A (no exchange, context rot risk)
- Scenario C: Blockchain-ledger (immutable records)
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Optional
import structlog

# Add vendor path for IAB repos
VENDOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "vendor", "iab", "seller-agent", "src"
)
if VENDOR_PATH not in sys.path:
    sys.path.insert(0, VENDOR_PATH)

from ...infrastructure.redis_bus import RedisBus
from ...infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType as SimDealType,
    STREAMS,
    CONSUMER_GROUPS,
)
from .inventory import SimulatedInventory, Product, DealType as InvDealType

logger = structlog.get_logger()


class SellerAgentAdapter:
    """Adapter wrapping IAB seller-agent for RTB simulation.

    Connects IAB seller-agent flows to the simulation's Redis-based
    message bus, translating between formats and adding simulation-
    specific behaviors.

    Usage:
        async with SellerAgentAdapter(seller_id="pub-001") as adapter:
            await adapter.run()
    """

    def __init__(
        self,
        seller_id: str,
        scenario: str = "A",
        mock_llm: bool = False,
        redis_url: Optional[str] = None,
        inventory_seed: Optional[int] = None,
    ):
        """Initialize the seller adapter.

        Args:
            seller_id: Unique identifier for this seller
            scenario: Simulation scenario (A, B, or C)
            mock_llm: Use deterministic mock instead of LLM calls
            redis_url: Redis connection URL
            inventory_seed: Seed for reproducible inventory
        """
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
        self._proposal_flow = None
        self._deal_flow = None

        # Track pending responses for deal matching
        self._pending_responses: dict[str, BidResponse] = {}

        logger.info(
            "seller_adapter.init",
            seller_id=seller_id,
            scenario=scenario,
            mock_llm=self.mock_llm,
        )

    async def connect(self) -> "SellerAgentAdapter":
        """Connect to Redis and initialize inventory."""
        await self._bus.connect(consumer_id=f"seller-{self.seller_id}")

        # Generate inventory
        self._products = self._inventory.generate_catalog(num_products=5)
        logger.info(
            "seller_adapter.connected",
            seller_id=self.seller_id,
            products=len(self._products),
        )

        return self

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        await self._bus.disconnect()
        logger.info("seller_adapter.disconnected", seller_id=self.seller_id)

    async def __aenter__(self) -> "SellerAgentAdapter":
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    def _init_iab_components(self) -> None:
        """Lazy-initialize IAB seller-agent components."""
        if self._pricing_engine is not None:
            return

        try:
            from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
            from ad_seller.models.pricing_tiers import TieredPricingConfig
            from ad_seller.models.buyer_identity import AccessTier

            # Create pricing config
            pricing_config = TieredPricingConfig(
                seller_organization_id=self.seller_id,
                global_floor_cpm=1.0,
            )
            self._pricing_engine = PricingRulesEngine(pricing_config)

            logger.info("seller_adapter.iab_init", status="success")
        except ImportError as e:
            logger.warning(
                "seller_adapter.iab_init_failed",
                error=str(e),
                fallback="mock",
            )
            self._pricing_engine = None

    async def run(self, max_iterations: Optional[int] = None) -> None:
        """Run the seller agent event loop.

        Listens for bid requests and generates responses.

        Args:
            max_iterations: Stop after N iterations (for testing)
        """
        self._running = True
        iterations = 0

        # Ensure consumer group exists
        await self._bus.ensure_consumer_group(
            STREAMS["bid_requests"],
            CONSUMER_GROUPS["sellers"],
        )

        logger.info("seller_adapter.running", seller_id=self.seller_id)

        while self._running:
            if max_iterations and iterations >= max_iterations:
                break

            try:
                # Read from bid_requests stream
                messages = await self._bus.read_bid_requests(
                    group=CONSUMER_GROUPS["sellers"],
                    count=10,
                    block_ms=1000,
                )

                for msg_id, request in messages:
                    try:
                        await self._handle_bid_request(request, msg_id)
                    except Exception as e:
                        logger.error(
                            "seller_adapter.request_error",
                            error=str(e),
                            msg_id=msg_id,
                        )

                    # Acknowledge message
                    await self._bus.ack_bid_requests(
                        CONSUMER_GROUPS["sellers"],
                        msg_id,
                    )

                iterations += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("seller_adapter.loop_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("seller_adapter.stopped", seller_id=self.seller_id)

    async def _handle_bid_request(self, request: BidRequest, msg_id: str) -> None:
        """Handle an incoming bid request.

        Translates to IAB format, evaluates, and responds.

        Args:
            request: Incoming bid request
            msg_id: Redis message ID
        """
        logger.info(
            "seller_adapter.request_received",
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            channel=request.channel,
            impressions=request.impressions_requested,
            max_cpm=request.max_cpm,
        )

        # Find matching product for this channel
        products = self._inventory.get_products_for_channel(request.channel)
        if not products:
            logger.info(
                "seller_adapter.no_inventory",
                channel=request.channel,
                seller_id=self.seller_id,
            )
            return

        # Select best matching product
        product = self._select_product(products, request)

        # Evaluate the request
        if self.mock_llm:
            # Use deterministic mock evaluation
            evaluation = self._mock_evaluate(request, product)
        else:
            # Use IAB flows
            evaluation = await self._iab_evaluate(request, product)

        if not evaluation["accept"]:
            logger.info(
                "seller_adapter.request_rejected",
                request_id=request.request_id,
                reason=evaluation.get("reason", "unknown"),
            )
            return

        # Generate bid response
        response = self._create_response(request, product, evaluation)

        # Publish response
        await self._bus.publish_bid_response(response)

        # Track for potential deal
        self._pending_responses[request.request_id] = response

        logger.info(
            "seller_adapter.response_sent",
            request_id=request.request_id,
            response_id=response.response_id,
            offered_cpm=response.offered_cpm,
            available_impressions=response.available_impressions,
        )

    def _select_product(
        self,
        products: list[Product],
        request: BidRequest,
    ) -> Product:
        """Select the best product for a request.

        Matches based on:
        - Price compatibility (buyer max_cpm vs floor)
        - Inventory availability
        - Audience segment overlap

        Args:
            products: Available products
            request: Bid request

        Returns:
            Best matching product
        """
        best_product = None
        best_score = -1

        for product in products:
            score = 0

            # Price compatibility
            if request.max_cpm >= product.floor_cpm:
                score += 50
                # Bonus for headroom
                headroom = (request.max_cpm - product.floor_cpm) / product.floor_cpm
                score += min(headroom * 10, 20)

            # Availability check
            available, max_imps = self._inventory.check_availability(
                product.product_id,
                request.impressions_requested,
            )
            if available:
                score += 30

            # Audience overlap
            if request.targeting:
                segments = request.targeting.get("segments", [])
                overlap = len(set(segments) & set(product.audience_targeting))
                score += overlap * 5

            if score > best_score:
                best_score = score
                best_product = product

        return best_product or products[0]

    def _mock_evaluate(
        self,
        request: BidRequest,
        product: Product,
    ) -> dict:
        """Deterministic mock evaluation (no LLM).

        Uses simple rules for testing:
        - Accept if buyer's max_cpm >= product's floor
        - Offer at floor + 10% headroom
        - Provide available inventory

        Args:
            request: Bid request
            product: Selected product

        Returns:
            Evaluation result
        """
        # Check price acceptability
        if request.max_cpm < product.floor_cpm:
            return {
                "accept": False,
                "reason": f"Below floor CPM ({product.floor_cpm})",
            }

        # Check availability
        available, max_imps = self._inventory.check_availability(
            product.product_id,
            request.impressions_requested,
        )

        # Calculate offer CPM (floor + 10% headroom, capped at buyer max)
        offer_cpm = min(
            product.floor_cpm * 1.1,
            request.max_cpm,
        )

        # Available impressions
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
        """Evaluate using IAB seller-agent flows.

        Translates bid request to proposal format and runs
        the ProposalHandlingFlow.

        Args:
            request: Bid request
            product: Selected product

        Returns:
            Evaluation result
        """
        self._init_iab_components()

        # If IAB components failed to load, fall back to mock
        if self._pricing_engine is None:
            return self._mock_evaluate(request, product)

        try:
            from ad_seller.flows.proposal_handling_flow import ProposalHandlingFlow
            from ad_seller.models.buyer_identity import BuyerContext, BuyerIdentity

            # Translate to proposal format
            proposal_data = {
                "product_id": product.product_id,
                "impressions": request.impressions_requested,
                "price": request.max_cpm,
                "start_date": datetime.utcnow().isoformat(),
                "end_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "deal_type": "preferred_deal",
                "buyer_id": request.buyer_id,
                "campaign_id": request.campaign_id,
                "targeting": request.targeting,
            }

            # Create buyer context
            buyer_identity = BuyerIdentity(
                seat_id=request.buyer_id,
            )
            buyer_context = BuyerContext(identity=buyer_identity)

            # Create product dict for flow
            products_dict = {
                product.product_id: product,
            }

            # Run proposal handling flow
            flow = ProposalHandlingFlow()
            result = flow.handle_proposal(
                proposal_id=request.request_id,
                proposal_data=proposal_data,
                buyer_context=buyer_context,
                products=products_dict,
            )

            # Translate result
            if result["recommendation"] == "accept":
                eval_data = result.get("evaluation", {})
                return {
                    "accept": True,
                    "offer_cpm": eval_data.get("recommended_price", product.floor_cpm * 1.1),
                    "offer_impressions": min(
                        request.impressions_requested,
                        eval_data.get("available_impressions", 1000000),
                    ),
                    "product_id": product.product_id,
                    "deal_type": self._select_deal_type(request, product),
                }
            elif result["recommendation"] == "counter":
                counter = result.get("counter_terms", {})
                return {
                    "accept": True,
                    "offer_cpm": counter.get("proposed_price", product.base_cpm),
                    "offer_impressions": counter.get("max_impressions", request.impressions_requested),
                    "product_id": product.product_id,
                    "deal_type": self._select_deal_type(request, product),
                }
            else:
                return {
                    "accept": False,
                    "reason": result.get("errors", ["Proposal rejected"])[0] if result.get("errors") else "Proposal rejected",
                }

        except Exception as e:
            logger.error("seller_adapter.iab_eval_error", error=str(e))
            return self._mock_evaluate(request, product)

    def _select_deal_type(self, request: BidRequest, product: Product) -> SimDealType:
        """Select appropriate deal type based on request and product.

        Args:
            request: Bid request
            product: Selected product

        Returns:
            Deal type for response
        """
        # Map inventory deal types to simulation deal types
        if InvDealType.PROGRAMMATIC_GUARANTEED in product.supported_deal_types:
            if request.impressions_requested >= 1000000:
                return SimDealType.PROGRAMMATIC_GUARANTEED
        if InvDealType.PREFERRED_DEAL in product.supported_deal_types:
            return SimDealType.PREFERRED_DEAL
        if InvDealType.PRIVATE_AUCTION in product.supported_deal_types:
            return SimDealType.PRIVATE_AUCTION
        return SimDealType.OPEN_AUCTION

    def _create_response(
        self,
        request: BidRequest,
        product: Product,
        evaluation: dict,
    ) -> BidResponse:
        """Create a bid response from evaluation results.

        Args:
            request: Original request
            product: Selected product
            evaluation: Evaluation results

        Returns:
            Bid response
        """
        # Pre-generate deal ID for tracking
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
            },
        )

    @property
    def products(self) -> dict[str, Product]:
        """Get seller's product catalog."""
        return self._products


async def run_seller_agent(
    seller_id: str,
    scenario: str = "A",
    mock_llm: bool = False,
    redis_url: Optional[str] = None,
) -> None:
    """Run a seller agent as a standalone process.

    Args:
        seller_id: Seller identifier
        scenario: Simulation scenario
        mock_llm: Use mock LLM
        redis_url: Redis URL
    """
    async with SellerAgentAdapter(
        seller_id=seller_id,
        scenario=scenario,
        mock_llm=mock_llm,
        redis_url=redis_url,
    ) as adapter:
        await adapter.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a seller agent")
    parser.add_argument("--seller-id", required=True, help="Seller identifier")
    parser.add_argument("--scenario", default="A", help="Simulation scenario (A/B/C)")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM")
    parser.add_argument("--redis-url", help="Redis URL")

    args = parser.parse_args()

    asyncio.run(
        run_seller_agent(
            seller_id=args.seller_id,
            scenario=args.scenario,
            mock_llm=args.mock_llm,
            redis_url=args.redis_url,
        )
    )
