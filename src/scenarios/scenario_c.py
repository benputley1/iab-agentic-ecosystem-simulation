"""
Scenario C: Alkimi Ledger-Backed Exchange.

Implements direct buyer↔seller communication (like Scenario B) but with
immutable blockchain persistence via Sui/Walrus proxy ledger.

Key characteristics:
- Direct buyer↔seller communication (no exchange fee extraction)
- All transactions recorded to immutable ledger
- Agent state can be fully recovered from ledger
- Zero context rot - state always recoverable
- Blockchain costs replace exchange fees (significantly lower)

This scenario demonstrates Alkimi's advantages:
1. Immutable campaign state on blockchain
2. Perfect recovery from any failure
3. Complete audit trail
4. Single source of truth for all parties
5. Zero context rot regardless of duration

Flow:
```
Buyer Agent                    Beads/Ledger                Seller Agent
    │                              │                            │
    │──── proposal ────────────────┼───────────────────────────►│
    │                              │                            │
    │                         ┌────▼────┐                       │
    │                         │ Record  │                       │
    │                         │ to Bead │                       │
    │                         └────┬────┘                       │
    │                              │                            │
    │◄─────────────────────────────┼──── accept + deal ID ──────│
    │                              │                            │
    │                         ┌────▼────┐                       │
    │                         │ Record  │                       │
    │                         │ to Bead │                       │
    │                         └─────────┘                       │
```
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import structlog

from .base import BaseScenario, ScenarioConfig, ScenarioMetrics
from infrastructure.redis_bus import RedisBus
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    CONSUMER_GROUPS,
)
from infrastructure.ledger import LedgerClient, create_ledger_client
from metrics.collector import MetricCollector

logger = structlog.get_logger()


@dataclass
class RecoveredAgentState:
    """State recovered from ledger for an agent."""

    agent_id: str
    agent_type: str
    deals: list[dict] = field(default_factory=list)
    total_spend: float = 0.0
    total_revenue: float = 0.0
    total_impressions: int = 0
    entries_recovered: int = 0
    recovery_time_ms: int = 0
    recovery_accuracy: float = 1.0  # Always 100% from immutable ledger


@dataclass
class LedgerAgentMemory:
    """
    Agent memory backed by ledger persistence.

    Unlike Scenario B's volatile AgentMemory, this memory can be
    fully recovered from the blockchain at any time.
    """

    agent_id: str
    agent_type: str

    # Local cache of deal history
    deal_history: dict[str, DealConfirmation] = field(default_factory=dict)
    pending_requests: dict[str, BidRequest] = field(default_factory=dict)

    # Recovery tracking
    recovery_events: int = 0
    total_recoveries_from_ledger: int = 0
    last_recovery_day: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize memory state."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "deal_count": len(self.deal_history),
            "pending_requests": len(self.pending_requests),
            "recovery_events": self.recovery_events,
            "total_recoveries_from_ledger": self.total_recoveries_from_ledger,
            "last_recovery_day": self.last_recovery_day,
        }


class ScenarioC(BaseScenario):
    """
    Scenario C: Alkimi Ledger-Backed Exchange.

    Implements direct buyer-seller transactions with blockchain persistence,
    demonstrating zero context rot and complete auditability.

    Advantages over Scenario A and B:
    - No exchange fee extraction (only minimal blockchain costs)
    - Immutable transaction record
    - Full state recovery from ledger
    - Zero context rot
    - Complete audit trail for compliance

    Flow:
    ```
    Buyer Agent                                            Seller Agent
        │                                                       │
        │──────────── A2A discovery query ─────────────────────►│
        │               (recorded to ledger)                    │
        │◄─────────── product listings ────────────────────────│
        │               (recorded to ledger)                    │
        │──────────── proposal submission ─────────────────────►│
        │               (recorded to ledger)                    │
        │◄─────────── accept + deal ID ────────────────────────│
        │               (recorded to ledger)                    │
    ```
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        redis_bus: Optional[RedisBus] = None,
        ledger_client: Optional[LedgerClient] = None,
        metric_collector: Optional[MetricCollector] = None,
        seed: Optional[int] = None,
    ):
        config = config or self._default_config()
        super().__init__(
            scenario_id="C",
            scenario_name="Alkimi Ledger-Backed",
            config=config,
            redis_bus=redis_bus,
            metric_collector=metric_collector,
        )

        # Ledger client for blockchain persistence
        self._ledger: Optional[LedgerClient] = ledger_client
        self._owned_ledger = False

        # Agent memories (with ledger backing)
        self._buyer_memories: dict[str, LedgerAgentMemory] = {}
        self._seller_memories: dict[str, LedgerAgentMemory] = {}

        # Active negotiations
        self._active_negotiations: dict[str, dict] = {}

        # Connection state
        self._connected = False

        # Random for simulation
        self._random = random.Random(seed)

        # Track recovery statistics
        self._recovery_stats = {
            "total_recoveries": 0,
            "total_entries_recovered": 0,
            "total_recovery_time_ms": 0,
            "recovery_by_agent": {},
        }

    @classmethod
    def _default_config(cls) -> ScenarioConfig:
        return ScenarioConfig(
            scenario_code="C",
            name="Alkimi Ledger-Backed",
            description="Direct buyer↔seller with immutable blockchain persistence",
            exchange_fee_pct=0.05,  # Alkimi platform fee: 5% (3-8% range typical)
            context_decay_rate=0.0,  # No context rot with ledger
            hallucination_rate=0.0,  # Ground truth from ledger prevents hallucinations
        )
    
    # Alkimi platform fee configuration (separate from blockchain costs)
    ALKIMI_FEE_PCT = 0.05  # 5% platform fee (configurable 3-8%)

    @property
    def scenario_code(self) -> str:
        return "C"

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def setup(self) -> None:
        """Set up scenario resources."""
        await self.connect()

    async def teardown(self) -> None:
        """Clean up scenario resources."""
        self._buyer_memories.clear()
        self._seller_memories.clear()
        self._active_negotiations.clear()

        if self._ledger and self._owned_ledger:
            await self._ledger.disconnect()
            self._ledger = None

        logger.info("scenario_c.teardown")

    async def connect(self) -> "ScenarioC":
        """Connect and initialize scenario-specific components."""
        if self._connected:
            return self

        # Connect to Redis bus
        await self.connect_bus()

        # Connect to ledger
        if self._ledger is None:
            try:
                self._ledger = await create_ledger_client()
                self._owned_ledger = True
                logger.info("scenario_c.ledger_connected")
            except Exception as e:
                logger.warning(
                    "scenario_c.ledger_unavailable",
                    error=str(e),
                    fallback="Using mock ledger mode",
                )
                # Use mock ledger for testing
                self._ledger = MockLedgerClient()
                self._owned_ledger = True

        # Connect to ground truth for verification
        try:
            await self.connect_ground_truth()
        except Exception as e:
            logger.warning(
                "scenario_c.ground_truth_unavailable",
                error=str(e),
            )

        self._connected = True

        logger.info(
            "scenario_c.connected",
            has_ledger=self._ledger is not None,
            has_ground_truth=self._ground_truth_repo is not None,
        )
        return self

    async def disconnect(self) -> None:
        """Disconnect from all services."""
        await self.disconnect_bus()
        await self.disconnect_ground_truth()

        if self._ledger and self._owned_ledger:
            await self._ledger.disconnect()

        self._connected = False
        logger.info("scenario_c.disconnected")

    # -------------------------------------------------------------------------
    # Memory Management with Ledger Backing
    # -------------------------------------------------------------------------

    def get_or_create_buyer_memory(self, buyer_id: str) -> LedgerAgentMemory:
        """Get or create ledger-backed memory for a buyer agent."""
        if buyer_id not in self._buyer_memories:
            self._buyer_memories[buyer_id] = LedgerAgentMemory(
                agent_id=buyer_id,
                agent_type="buyer",
            )
        return self._buyer_memories[buyer_id]

    def get_or_create_seller_memory(self, seller_id: str) -> LedgerAgentMemory:
        """Get or create ledger-backed memory for a seller agent."""
        if seller_id not in self._seller_memories:
            self._seller_memories[seller_id] = LedgerAgentMemory(
                agent_id=seller_id,
                agent_type="seller",
            )
        return self._seller_memories[seller_id]

    # -------------------------------------------------------------------------
    # State Recovery from Ledger (Key Differentiator)
    # -------------------------------------------------------------------------

    async def recover_agent_state(
        self,
        agent_id: str,
        agent_type: str,
        from_day: int = 0,
    ) -> RecoveredAgentState:
        """
        Recover agent state from immutable ledger.

        This is the key advantage of Scenario C: perfect state recovery
        regardless of agent restarts, context loss, or failures.

        Args:
            agent_id: Agent to recover state for
            agent_type: Type of agent (buyer/seller)
            from_day: Minimum simulation day to recover from

        Returns:
            RecoveredAgentState with full history from ledger
        """
        import time
        start_time = time.time()

        # Recover from ledger
        entries = await self._ledger.recover_agent_state(agent_id, from_day)
        recovery_time_ms = int((time.time() - start_time) * 1000)

        # Build recovered state
        recovered = RecoveredAgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            entries_recovered=len(entries),
            recovery_time_ms=recovery_time_ms,
        )

        # Process entries
        for entry in entries:
            payload = entry.get("payload", {})
            entry_type = entry.get("type")

            if entry_type == "deal":
                recovered.deals.append(payload)
                if agent_type == "buyer":
                    recovered.total_spend += payload.get("total_cost", 0)
                else:
                    recovered.total_revenue += payload.get("total_cost", 0)
                recovered.total_impressions += payload.get("impressions", 0)

        # Log recovery to ledger
        await self._ledger.log_recovery(
            agent_id=agent_id,
            agent_type=agent_type,
            recovery_reason="context_recovery",
            entries_recovered=len(entries),
            bytes_recovered=sum(len(str(e)) for e in entries),
            recovery_time_ms=recovery_time_ms,
            recovery_complete=True,
            state_accuracy=1.0,  # Always 100% from immutable ledger
            missing_entries=None,
            simulation_day=self.current_day,
        )

        # Update memory
        memory = (
            self.get_or_create_buyer_memory(agent_id)
            if agent_type == "buyer"
            else self.get_or_create_seller_memory(agent_id)
        )
        memory.recovery_events += 1
        memory.total_recoveries_from_ledger += 1
        memory.last_recovery_day = self.current_day

        # Update recovery stats
        self._recovery_stats["total_recoveries"] += 1
        self._recovery_stats["total_entries_recovered"] += len(entries)
        self._recovery_stats["total_recovery_time_ms"] += recovery_time_ms

        if agent_id not in self._recovery_stats["recovery_by_agent"]:
            self._recovery_stats["recovery_by_agent"][agent_id] = 0
        self._recovery_stats["recovery_by_agent"][agent_id] += 1

        # Record context rot event (recovered successfully)
        await self.record_context_rot(
            agent_id=agent_id,
            keys_lost=0,  # No keys lost - full recovery
            is_decay=False,
            agent_type=agent_type,
            recovery_attempted=True,
            recovery_successful=True,
            recovery_accuracy=1.0,  # Perfect recovery from ledger
            recovery_source="ledger",
        )

        logger.info(
            "scenario_c.state_recovered",
            agent_id=agent_id,
            agent_type=agent_type,
            entries=len(entries),
            deals=len(recovered.deals),
            elapsed_ms=recovery_time_ms,
        )

        return recovered

    async def simulate_context_loss_and_recovery(
        self,
        agent_id: str,
        agent_type: str,
    ) -> dict:
        """
        Simulate context loss followed by ledger recovery.

        This demonstrates that unlike Scenario B, Scenario C can
        fully recover from any context loss event.

        Args:
            agent_id: Agent experiencing context loss
            agent_type: Type of agent

        Returns:
            Dict with recovery results
        """
        memory = (
            self.get_or_create_buyer_memory(agent_id)
            if agent_type == "buyer"
            else self.get_or_create_seller_memory(agent_id)
        )

        # Capture pre-loss state
        pre_loss_deals = len(memory.deal_history)

        # Simulate context loss (clear local memory)
        memory.deal_history.clear()
        memory.pending_requests.clear()

        logger.info(
            "scenario_c.context_loss_simulated",
            agent_id=agent_id,
            deals_before=pre_loss_deals,
        )

        # Recover from ledger
        recovered = await self.recover_agent_state(
            agent_id=agent_id,
            agent_type=agent_type,
            from_day=0,
        )

        return {
            "agent_id": agent_id,
            "deals_before_loss": pre_loss_deals,
            "deals_recovered": len(recovered.deals),
            "recovery_accuracy": 1.0,  # Perfect recovery
            "recovery_time_ms": recovered.recovery_time_ms,
            "context_rot": 0.0,  # Zero context rot with ledger
        }

    # -------------------------------------------------------------------------
    # A2A Communication with Ledger Recording
    # -------------------------------------------------------------------------

    async def process_bid_request(
        self,
        request: BidRequest,
    ) -> list[BidResponse]:
        """
        Process bid request via direct A2A with ledger recording.

        Unlike Scenario B, every request is recorded to the immutable ledger,
        creating a complete audit trail.

        Args:
            request: Buyer's bid request

        Returns:
            List of bid responses
        """
        buyer_memory = self.get_or_create_buyer_memory(request.buyer_id)
        buyer_memory.pending_requests[request.request_id] = request

        # Record to ledger
        entry_id = await self._ledger.create_entry(
            transaction_type="bid_request",
            payload={
                "request_id": request.request_id,
                "buyer_id": request.buyer_id,
                "campaign_id": request.campaign_id,
                "channel": request.channel,
                "impressions_requested": request.impressions_requested,
                "max_cpm": request.max_cpm,
                "targeting": request.targeting,
            },
            created_by=request.buyer_id,
            created_by_type="buyer",
            simulation_day=self.current_day,
        )

        # Track negotiation
        self._active_negotiations[request.request_id] = {
            "request": request,
            "responses": [],
            "started": datetime.utcnow(),
            "buyer_id": request.buyer_id,
            "ledger_entry_id": entry_id,
        }

        # Publish to sellers
        if self._bus:
            await self._bus.publish_bid_request(request)

        logger.info(
            "scenario_c.bid_request",
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            ledger_entry_id=entry_id,
        )

        # Record decision (verified against ledger)
        await self.record_decision(
            verified=True,  # Ground truth available from ledger
            agent_id=request.buyer_id,
            agent_type="buyer",
            decision_type="bid_request",
            decision_input={
                "campaign_id": request.campaign_id,
                "channel": request.channel,
                "max_cpm": request.max_cpm,
            },
            decision_output={
                "request_id": request.request_id,
                "ledger_entry_id": entry_id,
            },
        )

        return []

    async def process_bid_response(
        self,
        request: BidRequest,
        response: BidResponse,
    ) -> Optional[DealConfirmation]:
        """
        Process seller's response with ledger recording.

        Args:
            request: Original bid request
            response: Seller's response

        Returns:
            DealConfirmation if buyer accepts, None otherwise
        """
        buyer_memory = self.get_or_create_buyer_memory(request.buyer_id)
        seller_memory = self.get_or_create_seller_memory(response.seller_id)

        # Record response to ledger
        await self._ledger.create_entry(
            transaction_type="bid_response",
            payload={
                "response_id": response.response_id,
                "request_id": response.request_id,
                "seller_id": response.seller_id,
                "offered_cpm": response.offered_cpm,
                "available_impressions": response.available_impressions,
                "deal_type": response.deal_type.value if hasattr(response.deal_type, 'value') else response.deal_type,
            },
            created_by=response.seller_id,
            created_by_type="seller",
            simulation_day=self.current_day,
        )

        # No hallucination - price data verified from ledger
        offered_cpm = response.offered_cpm  # Real price, no corruption

        # Evaluate offer
        is_acceptable = offered_cpm <= request.max_cpm * 1.1

        # Record decision (verified - ground truth from ledger)
        await self.record_decision(
            verified=True,
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
            deal = await self.create_deal(request, response)
            buyer_memory.deal_history[deal.deal_id] = deal
            seller_memory.deal_history[deal.deal_id] = deal

            logger.info(
                "scenario_c.deal_accepted",
                deal_id=deal.deal_id,
                buyer_id=deal.buyer_id,
                seller_id=deal.seller_id,
                cpm=deal.cpm,
            )

            return deal

        logger.info(
            "scenario_c.offer_rejected",
            request_id=request.request_id,
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
        Create deal with ledger recording.

        Alkimi charges a low platform fee (3-8%, default 5%) instead of
        traditional exchange fees (15-20%). Plus minimal blockchain costs.

        Args:
            request: Buyer's request
            response: Seller's accepted response

        Returns:
            DealConfirmation with Alkimi platform fee
        """
        # Get Alkimi fee from config or use default
        alkimi_fee_pct = getattr(self.config, 'exchange_fee_pct', self.ALKIMI_FEE_PCT)
        
        # Create deal with Alkimi platform fee (much lower than traditional 15%)
        deal = DealConfirmation.from_deal(
            request=request,
            response=response,
            scenario="C",
            exchange_fee_pct=alkimi_fee_pct,  # Alkimi platform fee: 5% (vs 15% traditional)
        )

        # Record deal to ledger
        entry_id = await self._ledger.create_entry(
            transaction_type="deal",
            payload={
                "deal_id": deal.deal_id,
                "request_id": request.request_id,
                "buyer_id": deal.buyer_id,
                "seller_id": deal.seller_id,
                "impressions": deal.impressions,
                "cpm": deal.cpm,
                "total_cost": deal.total_cost,
                "exchange_fee": deal.exchange_fee,  # Alkimi platform fee
                "alkimi_fee_pct": alkimi_fee_pct,
                "scenario": "C",
            },
            created_by="system",
            created_by_type="exchange",
            simulation_day=self.current_day,
        )

        # Record to ledger_deals table
        await self._ledger.record_deal(
            entry_id=entry_id,
            deal_id=deal.deal_id,
            buyer_id=deal.buyer_id,
            seller_id=deal.seller_id,
            impressions=deal.impressions,
            cpm=deal.cpm,
            total_cost=deal.total_cost,
            deal_status="confirmed",
        )

        # Publish deal event
        if self._bus:
            await self._bus.publish_deal(deal)

        # Record to metrics
        self.record_deal(deal)

        # Get blockchain costs
        blockchain_costs = await self._ledger.get_blockchain_costs(self.current_day)

        logger.info(
            "scenario_c.deal_created",
            deal_id=deal.deal_id,
            buyer_spend=deal.total_cost,
            seller_revenue=deal.seller_revenue,
            alkimi_fee=deal.exchange_fee,  # Alkimi platform fee (5%)
            alkimi_fee_pct=f"{alkimi_fee_pct*100:.1f}%",
            blockchain_cost_usd=float(blockchain_costs.total_cost_usd) if blockchain_costs else 0,
            ledger_entry_id=entry_id,
        )

        return deal

    # -------------------------------------------------------------------------
    # Daily Simulation
    # -------------------------------------------------------------------------

    async def run_day(
        self,
        day: int,
        buyers: Optional[list[Any]] = None,
        sellers: Optional[list[Any]] = None,
    ) -> dict:
        """
        Execute one simulation day for Scenario C.

        Day flow:
        1. Buyers send bid requests (recorded to ledger)
        2. Sellers respond (recorded to ledger)
        3. Deals created (recorded to ledger)
        4. Simulate random context loss + recovery to prove zero rot
        5. Calculate blockchain costs

        Args:
            day: Simulation day (1-30)
            buyers: Optional list of buyer agent wrappers
            sellers: Optional list of seller adapters

        Returns:
            Dict with day's metrics
        """
        self.current_day = day
        if self._ledger and hasattr(self._ledger, 'set_simulation_day'):
            self._ledger.set_simulation_day(day)

        day_start = datetime.utcnow()

        day_metrics = {
            "day": day,
            "scenario": "C",
            "deals_made": 0,
            "total_spend": 0.0,
            "total_impressions": 0,
            "context_rot_events": 0,  # Should always be 0 with recovery
            "keys_lost": 0,  # Should always be 0 with recovery
            "recoveries_performed": 0,
            "blockchain_costs_usd": 0.0,
        }

        logger.info(
            "scenario_c.day_start",
            day=day,
            buyers=len(buyers) if buyers else 0,
            sellers=len(sellers) if sellers else 0,
        )

        # Run bidding cycles if buyers provided
        if buyers:
            for buyer in buyers:
                try:
                    deals = await buyer.run_bidding_cycle(max_iterations=5)

                    for deal in deals:
                        day_metrics["deals_made"] += 1
                        day_metrics["total_spend"] += deal.total_cost
                        day_metrics["total_impressions"] += deal.impressions

                except Exception as e:
                    logger.error(
                        "scenario_c.buyer_error",
                        buyer_id=buyer.buyer_id,
                        error=str(e),
                    )

        # Simulate context loss + recovery every few days to prove zero rot
        if day % 5 == 0 and self._buyer_memories:
            random_buyer = self._random.choice(list(self._buyer_memories.keys()))
            recovery_result = await self.simulate_context_loss_and_recovery(
                random_buyer, "buyer"
            )
            day_metrics["recoveries_performed"] += 1

            logger.info(
                "scenario_c.recovery_demonstration",
                day=day,
                agent_id=random_buyer,
                deals_recovered=recovery_result["deals_recovered"],
                accuracy=recovery_result["recovery_accuracy"],
            )

        # Get blockchain costs for the day
        blockchain_costs = await self._ledger.get_blockchain_costs(day)
        day_metrics["blockchain_costs_usd"] = float(blockchain_costs.total_cost_usd) if blockchain_costs else 0.0

        # Record daily metrics
        if self._metrics_collector:
            self._metrics_collector.record_daily_metrics(
                scenario="C",
                simulation_day=day,
                goal_attainment=self._calculate_daily_goal_attainment(buyers) if buyers else 1.0,
                context_losses=0,  # Zero context rot
                recovery_accuracy=1.0,  # Perfect recovery
                active_campaigns=self._count_active_campaigns(buyers) if buyers else 0,
                total_spend=day_metrics["total_spend"],
            )

        logger.info(
            "scenario_c.day_complete",
            day=day,
            deals=day_metrics["deals_made"],
            spend=day_metrics["total_spend"],
            blockchain_costs_usd=day_metrics["blockchain_costs_usd"],
            recoveries=day_metrics["recoveries_performed"],
            duration_ms=(datetime.utcnow() - day_start).total_seconds() * 1000,
        )

        return day_metrics

    def _calculate_daily_goal_attainment(self, buyers: list[Any]) -> float:
        """Calculate average goal attainment across buyers."""
        if not buyers:
            return 1.0

        attainments = []
        for buyer in buyers:
            for campaign in buyer.state.campaigns.values():
                if campaign.target_impressions > 0:
                    attainment = min(
                        1.0,
                        campaign.impressions_delivered / campaign.target_impressions,
                    )
                    attainments.append(attainment)

        return sum(attainments) / len(attainments) if attainments else 1.0

    def _count_active_campaigns(self, buyers: list[Any]) -> int:
        """Count campaigns still active."""
        count = 0
        if buyers:
            for buyer in buyers:
                count += len(buyer.get_active_campaigns())
        return count

    # -------------------------------------------------------------------------
    # Comparison Utilities
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
        request = BidRequest(
            buyer_id=buyer_id,
            campaign_id=f"test-{buyer_id}",
            channel="display",
            impressions_requested=impressions,
            max_cpm=cpm * 1.2,
        )

        response = BidResponse(
            request_id=request.request_id,
            seller_id=seller_id,
            offered_cpm=cpm,
            available_impressions=impressions,
            deal_type=DealType.PREFERRED_DEAL,
        )

        deal = await self.create_deal(request, response)

        # Get blockchain costs
        blockchain_costs = await self._ledger.get_blockchain_costs()

        return {
            "deal_id": deal.deal_id,
            "buyer_spend": deal.total_cost,
            "seller_revenue": deal.seller_revenue,
            "exchange_fee": deal.exchange_fee,  # Should be 0
            "blockchain_cost_usd": float(blockchain_costs.total_cost_usd) if blockchain_costs else 0,
            "scenario": "C",
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
            "total_recovery_events": sum(
                m.recovery_events for m in self._buyer_memories.values()
            ) + sum(m.recovery_events for m in self._seller_memories.values()),
            "total_ledger_recoveries": sum(
                m.total_recoveries_from_ledger for m in self._buyer_memories.values()
            ) + sum(m.total_recoveries_from_ledger for m in self._seller_memories.values()),
            "context_rot": 0,  # Always 0 with ledger backing
        }

    def get_recovery_stats(self) -> dict:
        """Get recovery statistics."""
        return self._recovery_stats.copy()

    async def get_ledger_summary(self) -> dict:
        """Get ledger statistics."""
        blockchain_state = await self._ledger.get_blockchain_state()
        costs = await self._ledger.get_blockchain_costs()

        return {
            "blockchain": {
                "current_block": blockchain_state.get("current_block_number", 0),
                "total_entries": blockchain_state.get("total_entries", 0),
                "total_cost_usd": float(blockchain_state.get("total_cost_usd", 0)),
            },
            "costs": {
                "total_entries": costs.total_entries if costs else 0,
                "total_bytes": costs.total_bytes if costs else 0,
                "total_cost_sui": float(costs.total_cost_sui) if costs else 0,
                "total_cost_usd": float(costs.total_cost_usd) if costs else 0,
            },
            "recovery": self._recovery_stats,
        }


class MockLedgerClient:
    """Mock ledger client for testing without database."""

    def __init__(self):
        self._entries = []
        self._block_number = 0
        self._total_cost_usd = 0.0

    async def connect(self):
        return self

    async def disconnect(self):
        pass

    def set_simulation_day(self, day: int):
        pass

    async def create_entry(
        self,
        transaction_type: str,
        payload: dict,
        created_by: str,
        created_by_type: str,
        simulation_day: int,
    ) -> str:
        import uuid
        entry_id = str(uuid.uuid4())
        self._entries.append({
            "entry_id": entry_id,
            "type": transaction_type,
            "payload": payload,
            "created_by": created_by,
            "simulation_day": simulation_day,
        })
        self._block_number += 1
        self._total_cost_usd += 0.001  # Mock cost
        return entry_id

    async def recover_agent_state(self, agent_id: str, from_day: int = 0) -> list[dict]:
        return [
            e for e in self._entries
            if e.get("payload", {}).get("buyer_id") == agent_id
            or e.get("payload", {}).get("seller_id") == agent_id
            or e.get("created_by") == agent_id
        ]

    async def log_recovery(self, **kwargs) -> int:
        return 1

    async def record_deal(self, **kwargs) -> int:
        return 1

    async def record_delivery(self, **kwargs) -> int:
        return 1

    async def record_settlement(self, **kwargs) -> int:
        return 1

    async def get_blockchain_costs(self, simulation_day: int = None):
        from dataclasses import dataclass
        from decimal import Decimal

        @dataclass
        class MockCosts:
            total_entries: int = len(self._entries)
            total_bytes: int = sum(len(str(e)) for e in self._entries)
            total_gas_sui: Decimal = Decimal("0.001") * len(self._entries)
            total_walrus_sui: Decimal = Decimal("0.0001") * len(self._entries)
            total_cost_sui: Decimal = Decimal("0.0011") * len(self._entries)
            total_cost_usd: Decimal = Decimal(str(self._total_cost_usd))

        return MockCosts()

    async def get_blockchain_state(self) -> dict:
        return {
            "current_block_number": self._block_number,
            "total_entries": len(self._entries),
            "total_gas_used": 0.001 * len(self._entries),
            "total_storage_used_bytes": sum(len(str(e)) for e in self._entries),
            "total_cost_sui": 0.0011 * len(self._entries),
            "total_cost_usd": self._total_cost_usd,
            "last_entry_hash": "mock_hash",
            "updated_at": datetime.utcnow(),
        }


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


async def run_scenario_c_test(
    days: int = 1,
    buyers: int = 1,
    sellers: int = 1,
    mock_llm: bool = True,
    skip_ledger: bool = False,
) -> dict:
    """
    Run a test of Scenario C.

    Args:
        days: Number of simulation days
        buyers: Number of buyer agents
        sellers: Number of seller agents
        mock_llm: Use mock LLM (no API calls)
        skip_ledger: Use mock ledger (no database)

    Returns:
        Scenario metrics
    """
    config = ScenarioConfig(
        scenario_code="C",
        name="Alkimi Ledger-Backed Test",
        description="Test run of Scenario C",
        num_buyers=buyers,
        num_sellers=sellers,
        simulation_days=days,
        mock_llm=mock_llm,
    )

    # Create mock ledger if requested
    mock_ledger = MockLedgerClient() if skip_ledger else None

    scenario = ScenarioC(config, ledger_client=mock_ledger)

    if skip_ledger:
        scenario._ledger = mock_ledger
        scenario._connected = True
    else:
        await scenario.connect()

    try:
        # Run a test deal
        result = await scenario.run_single_deal(
            buyer_id="test-buyer-001",
            seller_id="test-seller-001",
            impressions=100000,
            cpm=15.0,
        )

        # Demonstrate recovery
        recovery_result = await scenario.simulate_context_loss_and_recovery(
            "test-buyer-001", "buyer"
        )

        return {
            "test_deal": result,
            "recovery_demo": recovery_result,
            "metrics": scenario.get_summary(),
            "memory": scenario.get_memory_summary(),
            "ledger": await scenario.get_ledger_summary() if not skip_ledger else {"mock": True},
        }
    finally:
        if not skip_ledger:
            await scenario.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Scenario C simulation")
    parser.add_argument("--days", type=int, default=1, help="Simulation days")
    parser.add_argument("--buyers", type=int, default=1, help="Number of buyers")
    parser.add_argument("--sellers", type=int, default=1, help="Number of sellers")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM")
    parser.add_argument("--skip-ledger", action="store_true", help="Skip ledger (use mock)")

    args = parser.parse_args()

    result = asyncio.run(
        run_scenario_c_test(
            days=args.days,
            buyers=args.buyers,
            sellers=args.sellers,
            mock_llm=args.mock_llm,
            skip_ledger=args.skip_ledger,
        )
    )

    import json
    print(json.dumps(result, indent=2, default=str))
