"""
Scenario B: IAB Pure A2A Direct Buyer-Seller Communication.

Implements the IAB agentic-rtb-framework spec for direct agent-to-agent
advertising transactions WITHOUT an exchange intermediary.

Key characteristics:
- Direct buyer↔seller communication (no exchange)
- No persistent state across agent restarts
- Context rot accumulates over simulation days (NO RECOVERY)
- No single source of truth for disputes
- Hallucination risk from stale embeddings
- 0% fees (no intermediary)

CRITICAL DIFFERENCE FROM SCENARIO A:
- Scenario A: Agents have context rot BUT exchange provides ~60% recovery
- Scenario B: Agents have context rot with NO recovery mechanism
- Both have same base decay rate, but B's errors compound unchecked

This scenario demonstrates the challenges of pure A2A systems:
1. Context loss leads to repeated negotiations
2. Agents may "hallucinate" based on corrupted/stale data
3. No ground truth verification for claims
4. Performance degrades over 30-day campaigns
5. Errors compound without exchange verification layer

The Critical Gap (Intentional):
This scenario explicitly does NOT persist state or provide verification
to highlight the value proposition of Scenario C (ledger-backed verification).
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import defaultdict
import structlog

from .base import BaseScenario, ScenarioConfig, ScenarioMetrics
from .context_rot import (
    ContextRotSimulator,
    ContextRotConfig,
    AgentMemory,
    RecoverySource,
    SCENARIO_B_ROT_CONFIG,
)
from infrastructure.redis_bus import RedisBus
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    CONSUMER_GROUPS,
)
from agents.ucp.hallucination import (
    HallucinationManager,
    HallucinationInjector,
)
from metrics.collector import MetricCollector

logger = structlog.get_logger()


# Note: Using shared ContextRotConfig, AgentMemory, and ContextRotSimulator
# from context_rot.py module. Scenario B uses SCENARIO_B_ROT_CONFIG which
# has recovery_source=NONE and recovery_accuracy=0.0 (no recovery).
#
# The key difference from Scenario A:
# - Scenario A: Same context rot but exchange provides ~60% recovery
# - Scenario B: Same context rot with 0% recovery (errors compound)


class ScenarioB(BaseScenario):
    """
    Scenario B: IAB Pure A2A Direct Communication.

    Implements direct buyer-seller transactions without an exchange,
    demonstrating the challenges of stateless A2A systems.

    Flow:
    ```
    Buyer Agent                                            Seller Agent
        │                                                       │
        │──────────── A2A discovery query ─────────────────────►│
        │◄─────────── product listings (MCP) ──────────────────│
        │──────────── pricing request (identity + volume) ─────►│
        │◄─────────── tiered price response ───────────────────│
        │──────────── proposal submission (MCP create_order) ──►│
        │◄─────────── accept/counter/reject ───────────────────│
        │──────────── deal confirmation ───────────────────────►│
        │◄─────────── deal ID for DSP ─────────────────────────│
    ```
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        redis_bus: Optional[RedisBus] = None,
        metric_collector: Optional[MetricCollector] = None,
        context_rot_config: Optional[ContextRotConfig] = None,
        seed: Optional[int] = None,
        # A2A HTTP Mode (IAB agentic-direct compliance)
        a2a_http_mode: bool = False,
        a2a_base_port: int = 8100,
        seller_adapters: Optional[dict] = None,
    ):
        config = config or self._default_config()
        super().__init__(
            scenario_id="B",
            scenario_name="IAB Pure A2A",
            config=config,
            redis_bus=redis_bus,
            metric_collector=metric_collector,
        )

        # Context rot simulation - using shared module with NO recovery
        # SCENARIO_B_ROT_CONFIG has recovery_source=NONE, recovery_accuracy=0.0
        self._context_rot = ContextRotSimulator(
            context_rot_config or SCENARIO_B_ROT_CONFIG,
            seed=seed,
        )

        # Hallucination injection (unique to Scenario B)
        self._hallucination_mgr = HallucinationManager(
            scenario="B",
            injection_rate=self.config.hallucination_rate,
        )

        # Agent memories (volatile, in-memory only)
        self._buyer_memories: dict[str, AgentMemory] = {}
        self._seller_memories: dict[str, AgentMemory] = {}

        # Active negotiations tracking
        self._active_negotiations: dict[str, dict] = {}

        # Track connected state
        self._connected = False

        # Random for simulation decisions
        self._random = random.Random(seed)
        
        # A2A HTTP Mode (IAB agentic-direct compliance)
        # When enabled, uses HTTP JSON-RPC 2.0 instead of Redis pub/sub
        self._a2a_http_mode = a2a_http_mode
        self._a2a_base_port = a2a_base_port
        self._a2a_manager = None
        self._seller_adapters = seller_adapters or {}
        
        if a2a_http_mode:
            logger.info(
                "scenario_b.a2a_http_mode_enabled",
                base_port=a2a_base_port,
            )

    @classmethod
    def _default_config(cls) -> ScenarioConfig:
        return ScenarioConfig(
            scenario_code="B",
            name="IAB Pure A2A",
            description="Direct buyer↔seller per IAB spec, no exchange intermediary, context rot simulation",
            exchange_fee_pct=0.0,  # No exchange fees in Scenario B
            context_decay_rate=0.02,
            hallucination_rate=0.05,
        )

    @property
    def scenario_code(self) -> str:
        return "B"

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def setup(self) -> None:
        """Set up scenario resources."""
        await self.connect()
        
        # Initialize A2A HTTP mode if enabled
        if self._a2a_http_mode and self._seller_adapters:
            from .a2a_http_mode import A2AHTTPManager
            
            self._a2a_manager = A2AHTTPManager(
                base_port=self._a2a_base_port,
            )
            
            # Start seller A2A servers
            await self._a2a_manager.start_seller_servers(self._seller_adapters)
            
            # Create buyer clients for all known buyers
            buyer_ids = list(self._buyer_memories.keys()) or ["default-buyer"]
            await self._a2a_manager.create_buyer_clients(buyer_ids)
            
            logger.info(
                "scenario_b.a2a_setup_complete",
                sellers=len(self._seller_adapters),
            )

    async def teardown(self) -> None:
        """Clean up scenario resources."""
        # Shutdown A2A HTTP mode
        if self._a2a_manager:
            await self._a2a_manager.shutdown()
            self._a2a_manager = None
        
        # Clear volatile memories
        self._buyer_memories.clear()
        self._seller_memories.clear()
        self._active_negotiations.clear()
        logger.info("scenario_b.teardown")

    async def connect(self) -> "ScenarioB":
        """Connect and initialize scenario-specific components."""
        if self._connected:
            return self

        # Connect to Redis bus
        await self.connect_bus()

        # Connect to ground truth repository (optional, for verification)
        try:
            await self.connect_ground_truth()
            # Wire ground truth to hallucination manager
            if self._ground_truth_repo:
                self._hallucination_mgr.set_db_connection(
                    self._ground_truth_repo._pool
                )
        except Exception as e:
            logger.warning(
                "scenario_b.ground_truth_unavailable",
                error=str(e),
            )

        # Set up hallucination manager metrics
        if self._metrics_collector:
            self._hallucination_mgr.set_metric_collector(self._metrics_collector)

        self._connected = True

        logger.info(
            "scenario_b.connected",
            context_decay_rate=self.config.context_decay_rate,
            hallucination_rate=self.config.hallucination_rate,
            has_ground_truth=self._ground_truth_repo is not None,
        )
        return self

    async def disconnect(self) -> None:
        """Disconnect from all services."""
        await self.disconnect_bus()
        await self.disconnect_ground_truth()
        self._connected = False
        logger.info("scenario_b.disconnected")

    # -------------------------------------------------------------------------
    # Memory Management
    # -------------------------------------------------------------------------

    def get_or_create_buyer_memory(self, buyer_id: str) -> AgentMemory:
        """Get or create memory for a buyer agent."""
        if buyer_id not in self._buyer_memories:
            self._buyer_memories[buyer_id] = AgentMemory(
                agent_id=buyer_id,
                agent_type="buyer",
            )
        return self._buyer_memories[buyer_id]

    def get_or_create_seller_memory(self, seller_id: str) -> AgentMemory:
        """Get or create memory for a seller agent."""
        if seller_id not in self._seller_memories:
            self._seller_memories[seller_id] = AgentMemory(
                agent_id=seller_id,
                agent_type="seller",
            )
        return self._seller_memories[seller_id]

    # -------------------------------------------------------------------------
    # A2A Communication (No Exchange)
    # -------------------------------------------------------------------------

    async def process_bid_request(
        self,
        request: BidRequest,
    ) -> list[BidResponse]:
        """
        Process bid request via direct A2A communication.

        In Scenario B, buyer communicates directly with sellers:
        1. Buyer broadcasts discovery query
        2. Each seller evaluates and responds directly
        3. No intermediary filtering or fee extraction

        Args:
            request: Buyer's bid request

        Returns:
            List of bid responses from interested sellers
        """
        buyer_memory = self.get_or_create_buyer_memory(request.buyer_id)
        buyer_memory.pending_requests[request.request_id] = request

        # Track this negotiation
        self._active_negotiations[request.request_id] = {
            "request": request,
            "responses": [],
            "started": datetime.utcnow(),
            "buyer_id": request.buyer_id,
        }

        # Publish to sellers stream (direct, no exchange mediation)
        if self._bus:
            await self._bus.publish_bid_request(request)

        logger.info(
            "scenario_b.bid_request",
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            channel=request.channel,
            max_cpm=request.max_cpm,
            impressions=request.impressions_requested,
        )

        # Record decision (may be based on corrupted data - no ground truth in B)
        await self.record_decision(
            verified=False,
            agent_id=request.buyer_id,
            agent_type="buyer",
            decision_type="bid_request",
            decision_input={
                "campaign_id": request.campaign_id,
                "channel": request.channel,
                "max_cpm": request.max_cpm,
                "impressions": request.impressions_requested,
            },
            decision_output={"request_id": request.request_id},
        )

        # Note: Responses will be collected asynchronously
        return []

    async def process_bid_request_a2a_http(
        self,
        request: BidRequest,
        seller_ids: Optional[list[str]] = None,
    ) -> Optional[DealConfirmation]:
        """
        Process bid request via A2A HTTP protocol (IAB agentic-direct).
        
        This method uses HTTP JSON-RPC 2.0 instead of Redis pub/sub,
        following the IAB agentic-direct specification.
        
        Args:
            request: Buyer's bid request
            seller_ids: Optional list of sellers to contact (defaults to all)
            
        Returns:
            DealConfirmation if successful, None otherwise
        """
        if not self._a2a_manager:
            logger.error("scenario_b.a2a_http_not_initialized")
            return None
        
        buyer_memory = self.get_or_create_buyer_memory(request.buyer_id)
        buyer_memory.pending_requests[request.request_id] = request
        
        # Determine which sellers to contact
        target_sellers = seller_ids or list(self._seller_adapters.keys())
        
        logger.info(
            "scenario_b.a2a_http_request",
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            channel=request.channel,
            max_cpm=request.max_cpm,
            sellers=len(target_sellers),
        )
        
        # Try each seller until we get a deal
        for seller_id in target_sellers:
            result = await self._a2a_manager.negotiate_deal(
                buyer_id=request.buyer_id,
                seller_id=seller_id,
                request=request,
            )
            
            if result.success:
                # Apply hallucination to price (context rot simulation)
                final_cpm = self._hallucination_mgr.process_price_data(
                    real_price=result.cpm,
                    agent_id=request.buyer_id,
                    agent_type="buyer",
                    publisher_id=seller_id,
                    simulation_day=self.current_day,
                )
                
                # Create deal confirmation
                deal = DealConfirmation(
                    deal_id=result.deal_id,
                    request_id=request.request_id,
                    buyer_id=request.buyer_id,
                    seller_id=seller_id,
                    campaign_id=request.campaign_id,
                    channel=request.channel,
                    cpm=final_cpm,
                    impressions=result.impressions,
                    total_cost=(result.impressions / 1000) * final_cpm,
                    seller_revenue=(result.impressions / 1000) * final_cpm,  # No fees
                    exchange_fee=0.0,  # Scenario B: no exchange
                    scenario="B",
                    deal_type=DealType.PREFERRED_DEAL,
                )
                
                # Update memories
                seller_memory = self.get_or_create_seller_memory(seller_id)
                buyer_memory.deal_history[deal.deal_id] = deal
                seller_memory.deal_history[deal.deal_id] = deal
                
                # Record metrics
                self.record_deal(deal)
                
                # Record decision
                await self.record_decision(
                    verified=False,  # No ground truth in Scenario B
                    agent_id=request.buyer_id,
                    agent_type="buyer",
                    decision_type="deal_accepted_a2a_http",
                    decision_input={
                        "request_id": request.request_id,
                        "seller_id": seller_id,
                        "offered_cpm": result.cpm,
                        "negotiation_steps": result.steps,
                    },
                    decision_output={
                        "deal_id": result.deal_id,
                        "final_cpm": final_cpm,
                    },
                )
                
                logger.info(
                    "scenario_b.a2a_http_deal_created",
                    deal_id=deal.deal_id,
                    buyer_id=deal.buyer_id,
                    seller_id=deal.seller_id,
                    cpm=deal.cpm,
                    impressions=deal.impressions,
                    steps=result.steps,
                )
                
                return deal
        
        # No deal found
        logger.info(
            "scenario_b.a2a_http_no_deal",
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            sellers_tried=len(target_sellers),
        )
        return None

    async def process_bid_response(
        self,
        request: BidRequest,
        response: BidResponse,
    ) -> Optional[DealConfirmation]:
        """
        Process seller's direct response to buyer.

        In Scenario B:
        - No auction mechanics (first acceptable offer wins)
        - Buyer directly evaluates seller's offer
        - May be based on hallucinated/corrupted data

        Args:
            request: Original bid request
            response: Seller's response

        Returns:
            DealConfirmation if buyer accepts, None otherwise
        """
        buyer_memory = self.get_or_create_buyer_memory(request.buyer_id)
        seller_memory = self.get_or_create_seller_memory(response.seller_id)

        # Apply potential hallucination to price data
        offered_cpm = self._hallucination_mgr.process_price_data(
            real_price=response.offered_cpm,
            agent_id=request.buyer_id,
            agent_type="buyer",
            publisher_id=response.seller_id,
            simulation_day=self.current_day,
        )

        # Check if offer is acceptable (simple threshold check)
        # In real system, this would use LLM decision-making
        is_acceptable = offered_cpm <= request.max_cpm * 1.1  # 10% tolerance

        # Record decision (not verified - Scenario B has no ground truth accessible)
        await self.record_decision(
            verified=False,
            agent_id=request.buyer_id,
            agent_type="buyer",
            decision_type="accept" if is_acceptable else "reject",
            decision_input={
                "request_id": request.request_id,
                "offered_cpm": offered_cpm,
                "max_cpm": request.max_cpm,
                "seller_id": response.seller_id,
            },
            decision_output={"accepted": is_acceptable},
        )

        if is_acceptable:
            # Create deal directly (no exchange fees)
            deal = await self.create_deal(request, response)

            # Update memories
            buyer_memory.deal_history[deal.deal_id] = deal
            seller_memory.deal_history[deal.deal_id] = deal

            # Update reputation tracking
            buyer_memory.partner_reputation[response.seller_id] = (
                buyer_memory.partner_reputation.get(response.seller_id, 0.5) * 0.9 + 0.1
            )

            logger.info(
                "scenario_b.deal_accepted",
                request_id=request.request_id,
                deal_id=deal.deal_id,
                buyer_id=deal.buyer_id,
                seller_id=deal.seller_id,
                cpm=deal.cpm,
                impressions=deal.impressions,
            )

            return deal

        logger.info(
            "scenario_b.offer_rejected",
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            seller_id=response.seller_id,
            offered_cpm=offered_cpm,
            max_cpm=request.max_cpm,
        )
        return None

    async def create_deal(
        self,
        request: BidRequest,
        response: BidResponse,
    ) -> DealConfirmation:
        """
        Create deal confirmation for Scenario B.

        Key differences from Scenario A:
        - No exchange fee (0%)
        - No auction (direct negotiation)
        - No persistent record (in-memory only)

        Args:
            request: Buyer's request
            response: Seller's accepted response

        Returns:
            DealConfirmation with zero exchange fees
        """
        # Create deal with NO exchange fee
        deal = DealConfirmation.from_deal(
            request=request,
            response=response,
            scenario="B",
            exchange_fee_pct=0.0,  # No intermediary fees
        )

        # Publish deal event
        if self._bus:
            await self._bus.publish_deal(deal)

        # Record metrics
        self.record_deal(deal)

        logger.info(
            "scenario_b.deal_created",
            deal_id=deal.deal_id,
            buyer_spend=deal.total_cost,
            seller_revenue=deal.seller_revenue,
            exchange_fee=deal.exchange_fee,  # Should be 0
        )

        return deal

    # -------------------------------------------------------------------------
    # Context Rot Simulation
    # -------------------------------------------------------------------------

    async def apply_daily_context_rot(self) -> dict[str, int]:
        """
        Apply context rot to all agent memories at end of day.
        
        In Scenario B, there is NO recovery mechanism (recovery_source=NONE),
        so all context rot results in permanent loss. This is the key difference
        from Scenario A where exchange provides ~60% recovery.

        Returns:
            Dict mapping agent_id to keys_lost count (net loss, after any recovery)
        """
        results = {}

        # Apply to buyers
        for buyer_id, memory in self._buyer_memories.items():
            # Check for restart first
            restart_event = self._context_rot.check_restart(memory, self.current_day)
            if restart_event:
                # In Scenario B: no recovery, so net_loss = keys_lost
                net_loss = restart_event.keys_lost  # recovery is 0 in Scenario B
                results[buyer_id] = net_loss
                await self.record_context_rot(
                    agent_id=buyer_id,
                    keys_lost=restart_event.keys_lost,
                    is_decay=False,
                    agent_type="buyer",
                    recovery_attempted=False,  # Scenario B has no recovery mechanism
                    recovery_successful=False,
                    recovery_accuracy=0.0,
                    recovery_source="none",
                )
            else:
                # Apply gradual decay
                decay_event = self._context_rot.apply_daily_decay(
                    memory, self.current_day
                )
                if decay_event.keys_lost > 0:
                    net_loss = decay_event.keys_lost  # No recovery in Scenario B
                    results[buyer_id] = net_loss
                    await self.record_context_rot(
                        agent_id=buyer_id,
                        keys_lost=decay_event.keys_lost,
                        is_decay=True,
                        agent_type="buyer",
                        keys_lost_names=decay_event.keys_lost_names,
                        recovery_attempted=False,
                        recovery_successful=False,
                        recovery_accuracy=0.0,
                        recovery_source="none",
                    )

        # Apply to sellers
        for seller_id, memory in self._seller_memories.items():
            restart_event = self._context_rot.check_restart(memory, self.current_day)
            if restart_event:
                net_loss = restart_event.keys_lost
                results[seller_id] = net_loss
                await self.record_context_rot(
                    agent_id=seller_id,
                    keys_lost=restart_event.keys_lost,
                    is_decay=False,
                    agent_type="seller",
                    recovery_attempted=False,
                    recovery_successful=False,
                    recovery_accuracy=0.0,
                    recovery_source="none",
                )
            else:
                decay_event = self._context_rot.apply_daily_decay(
                    memory, self.current_day
                )
                if decay_event.keys_lost > 0:
                    net_loss = decay_event.keys_lost
                    results[seller_id] = net_loss
                    await self.record_context_rot(
                        agent_id=seller_id,
                        keys_lost=decay_event.keys_lost,
                        is_decay=True,
                        agent_type="seller",
                        keys_lost_names=decay_event.keys_lost_names,
                        recovery_attempted=False,
                        recovery_successful=False,
                        recovery_accuracy=0.0,
                        recovery_source="none",
                    )

        return results

    # -------------------------------------------------------------------------
    # Daily Simulation
    # -------------------------------------------------------------------------

    async def run_day(
        self,
        day: int,
        buyers: Optional[list[Any]] = None,
        sellers: Optional[list[Any]] = None,
    ) -> list[DealConfirmation]:
        """
        Execute one simulation day for Scenario B.

        Day flow:
        1. Buyers send bid requests directly to sellers
        2. Sellers respond with offers
        3. Buyers accept/reject offers directly
        4. Deals recorded in volatile memory
        5. Context rot applied at end of day

        Args:
            day: Simulation day (1-30)
            buyers: Optional list of buyer agent wrappers
            sellers: Optional list of seller adapters

        Returns:
            List of deals created
        """
        self.current_day = day
        day_start = datetime.utcnow()

        day_metrics = {
            "day": day,
            "scenario": "B",
            "deals_made": 0,
            "total_spend": 0.0,
            "total_impressions": 0,
            "context_rot_events": 0,
            "keys_lost": 0,
            "hallucinations_injected": 0,
        }

        # If no buyers/sellers provided, simulate internally
        num_buyers = len(buyers) if buyers else self.config.num_buyers
        num_sellers = len(sellers) if sellers else self.config.num_sellers

        logger.info(
            "scenario_b.day_start",
            day=day,
            buyers=num_buyers,
            sellers=num_sellers,
        )

        deals_created = []

        # If no external buyers, run simulated deals
        if not buyers:
            # Simulate deals for the day
            for i in range(num_buyers):
                buyer_id = f"buyer-{i+1:03d}"
                for j in range(num_sellers):
                    seller_id = f"seller-{j+1:03d}"
                    # 30% chance of a deal per buyer-seller pair
                    if self._random.random() < 0.3:
                        impressions = self._random.randint(50000, 500000)
                        cpm = self._random.uniform(5.0, 25.0)
                        
                        # Create simulated request and response
                        request = BidRequest(
                            request_id=f"req-{self._random.randint(10000,99999)}-{day}",
                            buyer_id=buyer_id,
                            campaign_id=f"camp-{buyer_id}-{j+1:03d}",
                            channel=self._random.choice(["display", "video", "ctv"]),
                            impressions_requested=impressions,
                            max_cpm=cpm * 1.2,  # Buyer willing to pay up to 20% more
                            scenario="B",
                        )
                        
                        response = BidResponse(
                            response_id=f"resp-{self._random.randint(10000,99999)}-{day}",
                            request_id=request.request_id,
                            seller_id=seller_id,
                            offered_cpm=cpm,
                            available_impressions=impressions,
                            deal_type=DealType.PRIVATE_AUCTION,
                            scenario="B",
                        )
                        
                        deal = await self.create_deal(request, response)
                        if deal:
                            deals_created.append(deal)
                            day_metrics["deals_made"] += 1
                            day_metrics["total_spend"] += deal.total_cost
                            day_metrics["total_impressions"] += deal.impressions
        else:
            # Run bidding cycles for each provided buyer
            for buyer in buyers:
                try:
                    # Run buyer's bidding cycle
                    deals = await buyer.run_bidding_cycle(max_iterations=5)

                    for deal in deals:
                        deals_created.append(deal)
                        day_metrics["deals_made"] += 1
                        day_metrics["total_spend"] += deal.total_cost
                        day_metrics["total_impressions"] += deal.impressions

                except Exception as e:
                    logger.error(
                        "scenario_b.buyer_error",
                        buyer_id=buyer.buyer_id,
                        error=str(e),
                    )

        # Apply end-of-day context rot
        rot_results = await self.apply_daily_context_rot()
        day_metrics["context_rot_events"] = len(rot_results)
        day_metrics["keys_lost"] = sum(rot_results.values())

        # Track hallucination injections
        day_metrics["hallucinations_injected"] = len(
            self._hallucination_mgr.injector.get_injections()
        )

        # Record daily metrics
        if self._metrics_collector:
            self._metrics_collector.record_daily_metrics(
                scenario="B",
                simulation_day=day,
                goal_attainment=self._calculate_daily_goal_attainment(buyers or []),
                context_losses=day_metrics["context_rot_events"],
                recovery_accuracy=self._calculate_recovery_accuracy(),
                active_campaigns=self._count_active_campaigns(buyers or []),
                total_spend=day_metrics["total_spend"],
            )

        logger.info(
            "scenario_b.day_complete",
            day=day,
            deals=day_metrics["deals_made"],
            spend=day_metrics["total_spend"],
            context_rot_events=day_metrics["context_rot_events"],
            duration_ms=(datetime.utcnow() - day_start).total_seconds() * 1000,
        )

        # Update scenario metrics
        self.metrics.total_deals += day_metrics["deals_made"]
        self.metrics.total_buyer_spend += day_metrics["total_spend"]
        self.metrics.total_seller_revenue += day_metrics["total_spend"]  # No exchange fee in B
        self.metrics.total_impressions += day_metrics["total_impressions"]
        self.metrics.context_rot_events += day_metrics["context_rot_events"]
        self.metrics.keys_lost_total += day_metrics["keys_lost"]

        return deals_created

    def _calculate_daily_goal_attainment(self, buyers: list[Any]) -> float:
        """Calculate average goal attainment across buyers."""
        if not buyers:
            return 0.0

        attainments = []
        for buyer in buyers:
            for campaign in buyer.state.campaigns.values():
                if campaign.target_impressions > 0:
                    attainment = min(
                        1.0,
                        campaign.impressions_delivered / campaign.target_impressions,
                    )
                    attainments.append(attainment)

        return sum(attainments) / len(attainments) if attainments else 0.0

    def _calculate_recovery_accuracy(self) -> float:
        """Calculate context recovery accuracy."""
        if self.metrics.recovery_attempts == 0:
            return 1.0  # No attempts = no failures
        return self.metrics.recovery_successes / self.metrics.recovery_attempts

    def _count_active_campaigns(self, buyers: list[Any]) -> int:
        """Count campaigns still active."""
        count = 0
        for buyer in buyers:
            count += len(buyer.get_active_campaigns())
        return count

    # -------------------------------------------------------------------------
    # Simulation Utilities
    # -------------------------------------------------------------------------

    async def run_single_deal(
        self,
        buyer_id: str,
        seller_id: str,
        impressions: int,
        cpm: float,
    ) -> dict:
        """
        Execute a single deal for testing.

        Args:
            buyer_id: Buyer identifier
            seller_id: Seller identifier
            impressions: Number of impressions
            cpm: Agreed CPM

        Returns:
            Deal result dict
        """
        # Create mock request
        request = BidRequest(
            buyer_id=buyer_id,
            campaign_id=f"test-{buyer_id}",
            channel="display",
            impressions_requested=impressions,
            max_cpm=cpm * 1.2,
        )

        # Create mock response
        response = BidResponse(
            request_id=request.request_id,
            seller_id=seller_id,
            offered_cpm=cpm,
            available_impressions=impressions,
            deal_type=DealType.PREFERRED_DEAL,
        )

        # Create deal
        deal = await self.create_deal(request, response)

        return {
            "deal_id": deal.deal_id,
            "buyer_spend": deal.total_cost,
            "seller_revenue": deal.seller_revenue,
            "exchange_fee": deal.exchange_fee,
            "scenario": "B",
        }

    def get_memory_summary(self) -> dict:
        """Get summary of all agent memories."""
        return {
            "buyers": {
                agent_id: memory.to_dict()
                for agent_id, memory in self._buyer_memories.items()
            },
            "sellers": {
                agent_id: memory.to_dict()
                for agent_id, memory in self._seller_memories.items()
            },
            "total_rot_events": sum(
                m.rot_events for m in self._buyer_memories.values()
            ) + sum(m.rot_events for m in self._seller_memories.values()),
            "total_keys_lost": sum(
                m.keys_lost_total for m in self._buyer_memories.values()
            ) + sum(m.keys_lost_total for m in self._seller_memories.values()),
        }

    def get_hallucination_summary(self) -> dict:
        """Get summary of hallucination injection/detection."""
        return self._hallucination_mgr.get_summary()


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


async def run_scenario_b_test(
    days: int = 1,
    buyers: int = 1,
    sellers: int = 1,
    mock_llm: bool = True,
    skip_redis: bool = False,
) -> dict:
    """
    Run a test of Scenario B.

    Args:
        days: Number of simulation days
        buyers: Number of buyer agents
        sellers: Number of seller agents
        mock_llm: Use mock LLM (no API calls)
        skip_redis: Skip Redis connection (for offline testing)

    Returns:
        Scenario metrics
    """
    from unittest.mock import AsyncMock

    config = ScenarioConfig(
        scenario_code="B",
        name="IAB Pure A2A Test",
        description="Test run of Scenario B",
        num_buyers=buyers,
        num_sellers=sellers,
        simulation_days=days,
        mock_llm=mock_llm,
    )

    # Create mock Redis if requested
    mock_bus = None
    if skip_redis:
        mock_bus = AsyncMock()
        mock_bus.publish_bid_request = AsyncMock(return_value="msg-001")
        mock_bus.publish_bid_response = AsyncMock(return_value="msg-002")
        mock_bus.publish_deal = AsyncMock(return_value="msg-003")
        mock_bus.connect = AsyncMock()
        mock_bus.disconnect = AsyncMock()

    scenario = ScenarioB(config, redis_bus=mock_bus)
    scenario._connected = True if skip_redis else False
    scenario._bus = mock_bus

    if not skip_redis:
        await scenario.connect()

    try:
        # For testing, simulate a simple deal
        result = await scenario.run_single_deal(
            buyer_id="test-buyer-001",
            seller_id="test-seller-001",
            impressions=100000,
            cpm=15.0,
        )

        return {
            "test_deal": result,
            "metrics": scenario.get_summary(),
            "memory": scenario.get_memory_summary(),
            "hallucinations": scenario.get_hallucination_summary(),
        }
    finally:
        if not skip_redis:
            await scenario.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Scenario B simulation")
    parser.add_argument("--days", type=int, default=1, help="Simulation days")
    parser.add_argument("--buyers", type=int, default=1, help="Number of buyers")
    parser.add_argument("--sellers", type=int, default=1, help="Number of sellers")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM")
    parser.add_argument("--skip-redis", action="store_true", help="Skip Redis (use mock)")

    args = parser.parse_args()

    result = asyncio.run(
        run_scenario_b_test(
            days=args.days,
            buyers=args.buyers,
            sellers=args.sellers,
            mock_llm=args.mock_llm,
            skip_redis=args.skip_redis,
        )
    )

    import json
    print(json.dumps(result, indent=2, default=str))
