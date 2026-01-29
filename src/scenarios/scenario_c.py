"""
Scenario C: Alkimi ledger-backed exchange.

Direct buyer-seller transactions with blockchain persistence instead of
exchange fees. All transactions are recorded to an immutable ledger
(simulating Sui/Walrus).

Key differences from Scenario A:
- No intermediary fee extraction
- All transactions recorded to ledger with blockchain costs
- Agent state can be recovered from ledger
- Costs are blockchain gas/storage, not exchange fees
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import structlog

from ..infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    CONSUMER_GROUPS,
)
from ..infrastructure.redis_bus import RedisBus
from ..infrastructure.ledger import LedgerClient

logger = structlog.get_logger()


@dataclass
class PendingNegotiation:
    """Tracks an ongoing buyer-seller negotiation."""
    request: BidRequest
    responses: list[BidResponse] = field(default_factory=list)
    request_entry_id: Optional[str] = None  # Ledger entry for request
    response_entry_ids: list[str] = field(default_factory=list)


class AlkimiLedger:
    """
    Scenario C: Alkimi ledger-backed exchange.

    Facilitates direct buyer-seller transactions with:
    - No exchange fee (0% intermediary cost)
    - Blockchain costs for transaction recording
    - Immutable audit trail
    - State recovery capability

    Blockchain costs are estimated based on Sui gas and Walrus storage.
    """

    def __init__(
        self,
        bus: RedisBus,
        ledger: LedgerClient,
        exchange_id: str = "alkimi-001",
    ):
        """
        Initialize Alkimi ledger exchange.

        Args:
            bus: Redis bus for message routing
            ledger: Ledger client for persistence
            exchange_id: Unique identifier for this exchange
        """
        self.bus = bus
        self.ledger = ledger
        self.exchange_id = exchange_id

        # Track pending negotiations by request_id
        self._negotiations: dict[str, PendingNegotiation] = {}
        self._response_message_ids: dict[str, list[str]] = defaultdict(list)

        # Track current simulation day for ledger entries
        self._simulation_day: int = 1

    def set_simulation_day(self, day: int) -> None:
        """Set the current simulation day for ledger entries."""
        self._simulation_day = day

    async def handle_bid_request(
        self,
        request: BidRequest,
        message_id: str,
    ) -> str:
        """
        Process incoming bid request from buyer.

        Records to ledger and forwards to sellers.

        Args:
            request: Buyer's bid request
            message_id: Redis message ID

        Returns:
            Ledger entry ID for the request
        """
        # Record request to ledger
        entry_id = await self.ledger.create_entry(
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
            simulation_day=self._simulation_day,
        )

        # Track the negotiation
        self._negotiations[request.request_id] = PendingNegotiation(
            request=request,
            request_entry_id=entry_id,
        )

        logger.info(
            "alkimi.request_received",
            exchange_id=self.exchange_id,
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            campaign_id=request.campaign_id,
            max_cpm=request.max_cpm,
            ledger_entry_id=entry_id,
        )

        # Forward to main bid_requests stream for sellers
        await self.bus.publish_bid_request(request)

        # Acknowledge the original request
        await self.bus.ack_bid_requests(
            CONSUMER_GROUPS["exchange"],
            message_id,
        )

        return entry_id

    async def handle_bid_response(
        self,
        response: BidResponse,
        message_id: str,
    ) -> Optional[str]:
        """
        Process incoming bid response from seller.

        Records to ledger and collects for deal processing.

        Args:
            response: Seller's bid response
            message_id: Redis message ID

        Returns:
            Ledger entry ID for the response, or None if orphaned
        """
        negotiation = self._negotiations.get(response.request_id)
        if not negotiation:
            logger.warning(
                "alkimi.orphan_response",
                exchange_id=self.exchange_id,
                response_id=response.response_id,
                request_id=response.request_id,
            )
            return None

        # Record response to ledger
        entry_id = await self.ledger.create_entry(
            transaction_type="bid_response",
            payload={
                "response_id": response.response_id,
                "request_id": response.request_id,
                "seller_id": response.seller_id,
                "offered_cpm": response.offered_cpm,
                "available_impressions": response.available_impressions,
                "deal_type": response.deal_type.value,
            },
            created_by=response.seller_id,
            created_by_type="seller",
            simulation_day=self._simulation_day,
        )

        # Collect response
        negotiation.responses.append(response)
        negotiation.response_entry_ids.append(entry_id)
        self._response_message_ids[response.request_id].append(message_id)

        logger.info(
            "alkimi.response_collected",
            exchange_id=self.exchange_id,
            request_id=response.request_id,
            seller_id=response.seller_id,
            offered_cpm=response.offered_cpm,
            ledger_entry_id=entry_id,
            collected_count=len(negotiation.responses),
        )

        return entry_id

    async def finalize_deal(
        self,
        request_id: str,
    ) -> Optional[DealConfirmation]:
        """
        Finalize a deal from collected responses.

        Selects best offer (lowest CPM within budget) and creates deal.
        No exchange fee is extracted - only blockchain costs apply.

        Args:
            request_id: Request to finalize deal for

        Returns:
            DealConfirmation if deal created, None otherwise
        """
        negotiation = self._negotiations.get(request_id)
        if not negotiation or not negotiation.responses:
            logger.warning(
                "alkimi.deal_skipped",
                exchange_id=self.exchange_id,
                request_id=request_id,
                has_negotiation=negotiation is not None,
                response_count=len(negotiation.responses) if negotiation else 0,
            )
            return None

        request = negotiation.request

        # Find best offer (lowest CPM within buyer's budget)
        valid_responses = [
            r for r in negotiation.responses
            if r.offered_cpm <= request.max_cpm
        ]

        if not valid_responses:
            logger.info(
                "alkimi.no_valid_offers",
                exchange_id=self.exchange_id,
                request_id=request_id,
                buyer_max_cpm=request.max_cpm,
                best_offered=min(r.offered_cpm for r in negotiation.responses),
            )
            self._cleanup_negotiation(request_id)
            return None

        # Select lowest CPM offer
        best_response = min(valid_responses, key=lambda r: r.offered_cpm)

        # Create deal with NO exchange fee (Scenario C key difference)
        impressions = min(
            request.impressions_requested,
            best_response.available_impressions,
        )
        total_cost = (impressions / 1000) * best_response.offered_cpm

        deal = DealConfirmation(
            request_id=request_id,
            buyer_id=request.buyer_id,
            seller_id=best_response.seller_id,
            impressions=impressions,
            cpm=best_response.offered_cpm,
            total_cost=total_cost,
            exchange_fee=0.0,  # No exchange fee in Scenario C
            scenario="C",
            deal_id=best_response.deal_id,
        )

        # Record deal to ledger
        deal_entry_id = await self.ledger.create_entry(
            transaction_type="deal",
            payload={
                "deal_id": deal.deal_id,
                "request_id": request_id,
                "buyer_id": deal.buyer_id,
                "seller_id": deal.seller_id,
                "impressions": deal.impressions,
                "cpm": deal.cpm,
                "total_cost": deal.total_cost,
                "scenario": "C",
            },
            created_by=self.exchange_id,
            created_by_type="exchange",
            simulation_day=self._simulation_day,
        )

        # Update deal with ledger references
        deal.ledger_entry_id = deal_entry_id

        # Record to ledger_deals table
        await self.ledger.record_deal(
            entry_id=deal_entry_id,
            deal_id=deal.deal_id,
            buyer_id=deal.buyer_id,
            seller_id=deal.seller_id,
            impressions=deal.impressions,
            cpm=deal.cpm,
            total_cost=deal.total_cost,
            deal_status="confirmed",
        )

        # Publish deal to stream
        await self.bus.publish_deal(deal)

        # Route response back to buyer (at seller's price - no markup)
        buyer_response = BidResponse(
            response_id=best_response.response_id,
            request_id=request_id,
            seller_id=best_response.seller_id,
            offered_cpm=best_response.offered_cpm,  # No markup
            available_impressions=best_response.available_impressions,
            deal_type=best_response.deal_type,
            deal_id=deal.deal_id,
        )
        await self.bus.route_to_buyer(buyer_response, request.buyer_id)

        logger.info(
            "alkimi.deal_created",
            exchange_id=self.exchange_id,
            deal_id=deal.deal_id,
            buyer_id=deal.buyer_id,
            seller_id=deal.seller_id,
            impressions=deal.impressions,
            cpm=deal.cpm,
            total_cost=deal.total_cost,
            exchange_fee=0.0,
            ledger_entry_id=deal_entry_id,
        )

        # Acknowledge all collected responses
        if self._response_message_ids[request_id]:
            await self.bus.ack_bid_responses(
                CONSUMER_GROUPS["exchange"],
                *self._response_message_ids[request_id],
            )

        # Clean up
        self._cleanup_negotiation(request_id)

        return deal

    async def record_delivery(
        self,
        deal_id: str,
        batch_number: int,
        impressions_in_batch: int,
        cumulative_impressions: int,
        created_by: str,
    ) -> str:
        """
        Record an impression delivery batch to the ledger.

        Args:
            deal_id: Deal being delivered
            batch_number: Sequential batch number
            impressions_in_batch: Impressions in this batch
            cumulative_impressions: Total delivered so far
            created_by: Agent recording the delivery

        Returns:
            Ledger entry ID
        """
        import hashlib

        # Create delivery proof hash
        proof_data = f"{deal_id}:{batch_number}:{impressions_in_batch}:{cumulative_impressions}"
        proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()

        # Record to ledger
        entry_id = await self.ledger.create_entry(
            transaction_type="delivery",
            payload={
                "deal_id": deal_id,
                "batch_number": batch_number,
                "impressions_in_batch": impressions_in_batch,
                "cumulative_impressions": cumulative_impressions,
                "proof_hash": proof_hash,
            },
            created_by=created_by,
            created_by_type="system",
            simulation_day=self._simulation_day,
        )

        # Record to ledger_deliveries table
        await self.ledger.record_delivery(
            entry_id=entry_id,
            deal_id=deal_id,
            batch_number=batch_number,
            impressions_in_batch=impressions_in_batch,
            cumulative_impressions=cumulative_impressions,
            delivery_proof_hash=proof_hash,
        )

        return entry_id

    async def record_settlement(
        self,
        deal_id: str,
        buyer_spend: float,
        seller_revenue: float,
    ) -> str:
        """
        Record a financial settlement to the ledger.

        Args:
            deal_id: Deal being settled
            buyer_spend: Amount buyer paid
            seller_revenue: Amount seller receives (before blockchain costs)

        Returns:
            Ledger entry ID
        """
        # Get blockchain costs for this deal
        cost_per_1k = await self.ledger.get_cost_per_1k_impressions(deal_id)
        blockchain_costs = cost_per_1k if cost_per_1k else 0.0

        # Record to ledger
        entry_id = await self.ledger.create_entry(
            transaction_type="settlement",
            payload={
                "deal_id": deal_id,
                "buyer_spend": buyer_spend,
                "seller_revenue": seller_revenue,
                "blockchain_costs": blockchain_costs,
                "net_to_seller": seller_revenue - blockchain_costs,
            },
            created_by=self.exchange_id,
            created_by_type="exchange",
            simulation_day=self._simulation_day,
        )

        # Record to ledger_settlements table
        await self.ledger.record_settlement(
            entry_id=entry_id,
            deal_id=deal_id,
            buyer_spend=buyer_spend,
            seller_revenue=seller_revenue,
            blockchain_costs=blockchain_costs,
        )

        return entry_id

    def _cleanup_negotiation(self, request_id: str) -> None:
        """Remove negotiation from tracking after completion."""
        self._negotiations.pop(request_id, None)
        self._response_message_ids.pop(request_id, None)

    async def process_pending_deals(
        self,
        min_responses: int = 1,
    ) -> list[DealConfirmation]:
        """
        Process all pending negotiations that have enough responses.

        Args:
            min_responses: Minimum responses required to finalize

        Returns:
            List of created deals
        """
        deals = []

        for request_id in list(self._negotiations.keys()):
            negotiation = self._negotiations[request_id]
            if len(negotiation.responses) >= min_responses:
                deal = await self.finalize_deal(request_id)
                if deal:
                    deals.append(deal)

        return deals

    async def get_stats(self) -> dict:
        """Get exchange statistics including blockchain costs."""
        blockchain_state = await self.ledger.get_blockchain_state()
        costs = await self.ledger.get_blockchain_costs()

        return {
            "exchange_id": self.exchange_id,
            "scenario": "C",
            "pending_negotiations": len(self._negotiations),
            "collected_responses": sum(
                len(n.responses) for n in self._negotiations.values()
            ),
            "exchange_fee_pct": 0.0,  # No exchange fee
            "blockchain": {
                "current_block": blockchain_state["current_block_number"],
                "total_entries": blockchain_state["total_entries"],
                "total_cost_usd": float(blockchain_state["total_cost_usd"]),
            },
            "costs": {
                "total_entries": costs.total_entries,
                "total_bytes": costs.total_bytes,
                "total_cost_sui": float(costs.total_cost_sui),
                "total_cost_usd": float(costs.total_cost_usd),
            },
        }


# -------------------------------------------------------------------------
# State Recovery
# -------------------------------------------------------------------------

class AgentStateRecovery:
    """
    Recovers agent state from the ledger.

    Used when an agent loses context (restart, context rot, checkpoint miss).
    """

    def __init__(self, ledger: LedgerClient):
        """
        Initialize state recovery handler.

        Args:
            ledger: Ledger client for reading state
        """
        self.ledger = ledger

    async def recover_buyer_state(
        self,
        buyer_id: str,
        from_day: int = 0,
        simulation_day: int = 1,
    ) -> dict:
        """
        Recover buyer agent state from ledger.

        Args:
            buyer_id: Buyer agent ID
            from_day: Minimum simulation day to recover from
            simulation_day: Current simulation day (for logging)

        Returns:
            Dict containing recovered state
        """
        import time
        start_time = time.time()

        entries = await self.ledger.recover_agent_state(buyer_id, from_day)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Build state from entries
        state = {
            "agent_id": buyer_id,
            "agent_type": "buyer",
            "bid_requests": [],
            "deals": [],
            "total_spend": 0.0,
            "total_impressions": 0,
        }

        total_bytes = 0
        for entry in entries:
            payload = entry.get("payload", {})
            entry_type = entry.get("type")
            total_bytes += len(str(payload))

            if entry_type == "bid_request":
                state["bid_requests"].append(payload)
            elif entry_type == "deal":
                state["deals"].append(payload)
                state["total_spend"] += payload.get("total_cost", 0)
                state["total_impressions"] += payload.get("impressions", 0)

        # Log recovery
        await self.ledger.log_recovery(
            agent_id=buyer_id,
            agent_type="buyer",
            recovery_reason="context_recovery",
            entries_recovered=len(entries),
            bytes_recovered=total_bytes,
            recovery_time_ms=elapsed_ms,
            recovery_complete=True,
            state_accuracy=1.0,  # Perfect recovery from immutable ledger
            missing_entries=None,
            simulation_day=simulation_day,
        )

        logger.info(
            "state_recovery.buyer_recovered",
            buyer_id=buyer_id,
            entries=len(entries),
            deals=len(state["deals"]),
            total_spend=state["total_spend"],
            elapsed_ms=elapsed_ms,
        )

        return state

    async def recover_seller_state(
        self,
        seller_id: str,
        from_day: int = 0,
        simulation_day: int = 1,
    ) -> dict:
        """
        Recover seller agent state from ledger.

        Args:
            seller_id: Seller agent ID
            from_day: Minimum simulation day to recover from
            simulation_day: Current simulation day (for logging)

        Returns:
            Dict containing recovered state
        """
        import time
        start_time = time.time()

        entries = await self.ledger.recover_agent_state(seller_id, from_day)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Build state from entries
        state = {
            "agent_id": seller_id,
            "agent_type": "seller",
            "bid_responses": [],
            "deals": [],
            "deliveries": [],
            "total_revenue": 0.0,
            "total_impressions_sold": 0,
        }

        total_bytes = 0
        for entry in entries:
            payload = entry.get("payload", {})
            entry_type = entry.get("type")
            total_bytes += len(str(payload))

            if entry_type == "bid_response":
                state["bid_responses"].append(payload)
            elif entry_type == "deal":
                state["deals"].append(payload)
                state["total_revenue"] += payload.get("total_cost", 0)
                state["total_impressions_sold"] += payload.get("impressions", 0)
            elif entry_type == "delivery":
                state["deliveries"].append(payload)

        # Log recovery
        await self.ledger.log_recovery(
            agent_id=seller_id,
            agent_type="seller",
            recovery_reason="context_recovery",
            entries_recovered=len(entries),
            bytes_recovered=total_bytes,
            recovery_time_ms=elapsed_ms,
            recovery_complete=True,
            state_accuracy=1.0,
            missing_entries=None,
            simulation_day=simulation_day,
        )

        logger.info(
            "state_recovery.seller_recovered",
            seller_id=seller_id,
            entries=len(entries),
            deals=len(state["deals"]),
            total_revenue=state["total_revenue"],
            elapsed_ms=elapsed_ms,
        )

        return state
