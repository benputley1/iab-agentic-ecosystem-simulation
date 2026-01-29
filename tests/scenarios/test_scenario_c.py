"""Tests for Scenario C - Alkimi ledger-backed exchange."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from datetime import datetime

from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
    CONSUMER_GROUPS,
)
from src.scenarios.scenario_c import (
    AlkimiLedger,
    AgentStateRecovery,
    PendingNegotiation,
)
from src.infrastructure.ledger import (
    LedgerClient,
    LedgerEntry,
    BlockchainCosts,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def bid_request():
    """Sample bid request."""
    return BidRequest(
        request_id="req-001",
        buyer_id="buyer-001",
        campaign_id="camp-001",
        channel="display",
        impressions_requested=10000,
        max_cpm=20.0,
        targeting={"geo": ["US"]},
        floor_price=5.0,
    )


@pytest.fixture
def high_bid_response(bid_request):
    """High bid response (within budget)."""
    return BidResponse(
        response_id="resp-high",
        request_id=bid_request.request_id,
        seller_id="seller-premium",
        offered_cpm=15.0,
        available_impressions=10000,
        deal_type=DealType.OPEN_AUCTION,
        deal_id="DEAL-TEST001",
    )


@pytest.fixture
def low_bid_response(bid_request):
    """Low bid response (best value within budget)."""
    return BidResponse(
        response_id="resp-low",
        request_id=bid_request.request_id,
        seller_id="seller-budget",
        offered_cpm=8.0,
        available_impressions=15000,
        deal_type=DealType.OPEN_AUCTION,
        deal_id="DEAL-TEST002",
    )


@pytest.fixture
def over_budget_response(bid_request):
    """Response over buyer's max CPM."""
    return BidResponse(
        response_id="resp-over",
        request_id=bid_request.request_id,
        seller_id="seller-expensive",
        offered_cpm=25.0,  # Over max_cpm of 20.0
        available_impressions=10000,
        deal_type=DealType.OPEN_AUCTION,
    )


@pytest.fixture
def mock_redis_bus():
    """Mock Redis bus for testing."""
    bus = MagicMock()
    bus.publish_bid_request = AsyncMock(return_value="msg-001")
    bus.publish_bid_response = AsyncMock(return_value="msg-002")
    bus.publish_deal = AsyncMock(return_value="msg-003")
    bus.route_to_buyer = AsyncMock(return_value="msg-004")
    bus.ack_bid_requests = AsyncMock(return_value=1)
    bus.ack_bid_responses = AsyncMock(return_value=1)
    return bus


@pytest.fixture
def mock_ledger():
    """Mock ledger client for testing."""
    ledger = MagicMock(spec=LedgerClient)

    # Default return values
    ledger.create_entry = AsyncMock(return_value="ledger-test-001")
    ledger.record_deal = AsyncMock(return_value=1)
    ledger.record_delivery = AsyncMock(return_value=1)
    ledger.record_settlement = AsyncMock(return_value=1)
    ledger.log_recovery = AsyncMock(return_value=1)
    ledger.get_cost_per_1k_impressions = AsyncMock(return_value=0.05)
    ledger.get_blockchain_state = AsyncMock(return_value={
        "current_block_number": 100,
        "total_entries": 50,
        "total_gas_used": 0.5,
        "total_storage_used_bytes": 50000,
        "total_cost_sui": 0.75,
        "total_cost_usd": 1.125,
        "last_entry_hash": "abc123",
        "updated_at": datetime.utcnow(),
    })
    ledger.get_blockchain_costs = AsyncMock(return_value=BlockchainCosts(
        total_entries=50,
        total_bytes=50000,
        total_gas_sui=Decimal("0.5"),
        total_walrus_sui=Decimal("0.25"),
        total_cost_sui=Decimal("0.75"),
        total_cost_usd=Decimal("1.125"),
    ))
    ledger.recover_agent_state = AsyncMock(return_value=[])

    return ledger


# -------------------------------------------------------------------------
# AlkimiLedger Tests
# -------------------------------------------------------------------------


class TestAlkimiLedger:
    """Tests for Alkimi ledger-backed exchange."""

    @pytest.mark.asyncio
    async def test_handle_bid_request(
        self, mock_redis_bus, mock_ledger, bid_request
    ):
        """Exchange records request to ledger and forwards it."""
        exchange = AlkimiLedger(
            bus=mock_redis_bus,
            ledger=mock_ledger,
            exchange_id="alkimi-test",
        )

        entry_id = await exchange.handle_bid_request(bid_request, "msg-original")

        # Verify ledger entry created
        mock_ledger.create_entry.assert_called_once()
        call_args = mock_ledger.create_entry.call_args
        assert call_args.kwargs["transaction_type"] == "bid_request"
        assert call_args.kwargs["created_by"] == "buyer-001"
        assert call_args.kwargs["created_by_type"] == "buyer"

        # Verify request published
        mock_redis_bus.publish_bid_request.assert_called_once_with(bid_request)

        # Verify acknowledgment
        mock_redis_bus.ack_bid_requests.assert_called_once()

        # Verify tracking
        assert bid_request.request_id in exchange._negotiations

        # Verify return value
        assert entry_id == "ledger-test-001"

    @pytest.mark.asyncio
    async def test_handle_bid_response(
        self, mock_redis_bus, mock_ledger, bid_request, high_bid_response
    ):
        """Exchange records response to ledger and collects it."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        # First register the request
        await exchange.handle_bid_request(bid_request, "msg-req")
        mock_ledger.create_entry.reset_mock()

        # Then handle response
        entry_id = await exchange.handle_bid_response(high_bid_response, "msg-resp")

        # Verify ledger entry created
        mock_ledger.create_entry.assert_called_once()
        call_args = mock_ledger.create_entry.call_args
        assert call_args.kwargs["transaction_type"] == "bid_response"
        assert call_args.kwargs["created_by"] == "seller-premium"
        assert call_args.kwargs["created_by_type"] == "seller"

        # Verify collected
        negotiation = exchange._negotiations[bid_request.request_id]
        assert len(negotiation.responses) == 1
        assert negotiation.responses[0].seller_id == "seller-premium"

    @pytest.mark.asyncio
    async def test_orphan_response_ignored(
        self, mock_redis_bus, mock_ledger, high_bid_response
    ):
        """Response without matching request is ignored."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        # Handle response without prior request
        entry_id = await exchange.handle_bid_response(high_bid_response, "msg-orphan")

        # Should not create ledger entry
        mock_ledger.create_entry.assert_not_called()

        # Should return None
        assert entry_id is None

    @pytest.mark.asyncio
    async def test_finalize_deal_no_exchange_fee(
        self, mock_redis_bus, mock_ledger, bid_request, high_bid_response, low_bid_response
    ):
        """Alkimi finalizes deals with NO exchange fee (key Scenario C difference)."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        # Setup negotiation
        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(high_bid_response, "msg-high")
        await exchange.handle_bid_response(low_bid_response, "msg-low")

        # Finalize deal
        deal = await exchange.finalize_deal(bid_request.request_id)

        assert deal is not None
        assert deal.scenario == "C"
        assert deal.exchange_fee == 0.0  # KEY: No exchange fee
        assert deal.seller_id == "seller-budget"  # Lowest CPM selected
        assert deal.cpm == 8.0  # No markup - seller's actual price

    @pytest.mark.asyncio
    async def test_finalize_deal_selects_lowest_cpm(
        self, mock_redis_bus, mock_ledger, bid_request, high_bid_response, low_bid_response
    ):
        """Alkimi selects lowest valid CPM offer (best value for buyer)."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(high_bid_response, "msg-high")  # 15.0
        await exchange.handle_bid_response(low_bid_response, "msg-low")    # 8.0

        deal = await exchange.finalize_deal(bid_request.request_id)

        assert deal.seller_id == "seller-budget"
        assert deal.cpm == 8.0  # Lowest valid offer

    @pytest.mark.asyncio
    async def test_finalize_deal_filters_over_budget(
        self, mock_redis_bus, mock_ledger, bid_request, high_bid_response, over_budget_response
    ):
        """Alkimi filters out responses over buyer's max CPM."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(high_bid_response, "msg-high")        # 15.0 (ok)
        await exchange.handle_bid_response(over_budget_response, "msg-over")  # 25.0 (over)

        deal = await exchange.finalize_deal(bid_request.request_id)

        assert deal is not None
        assert deal.seller_id == "seller-premium"  # Only valid option
        assert deal.cpm == 15.0

    @pytest.mark.asyncio
    async def test_finalize_deal_no_valid_offers(
        self, mock_redis_bus, mock_ledger, bid_request, over_budget_response
    ):
        """Alkimi returns None when no offers are within budget."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(over_budget_response, "msg-over")  # 25.0 (over)

        deal = await exchange.finalize_deal(bid_request.request_id)

        assert deal is None

    @pytest.mark.asyncio
    async def test_finalize_deal_records_to_ledger(
        self, mock_redis_bus, mock_ledger, bid_request, low_bid_response
    ):
        """Alkimi records deal to ledger with proper references."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(low_bid_response, "msg-low")

        mock_ledger.create_entry.reset_mock()

        deal = await exchange.finalize_deal(bid_request.request_id)

        # Verify deal ledger entry created
        assert mock_ledger.create_entry.call_count == 1
        call_args = mock_ledger.create_entry.call_args
        assert call_args.kwargs["transaction_type"] == "deal"

        # Verify deal recorded to ledger_deals
        mock_ledger.record_deal.assert_called_once()

        # Verify deal has ledger reference
        assert deal.ledger_entry_id == "ledger-test-001"

    @pytest.mark.asyncio
    async def test_finalize_deal_routes_without_markup(
        self, mock_redis_bus, mock_ledger, bid_request, low_bid_response
    ):
        """Alkimi routes response to buyer without price markup."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(low_bid_response, "msg-low")

        await exchange.finalize_deal(bid_request.request_id)

        # Verify buyer gets SAME price as seller offered (no markup)
        mock_redis_bus.route_to_buyer.assert_called_once()
        routed_response = mock_redis_bus.route_to_buyer.call_args[0][0]
        assert routed_response.offered_cpm == 8.0  # Same as seller's offer

    @pytest.mark.asyncio
    async def test_cleanup_after_deal(
        self, mock_redis_bus, mock_ledger, bid_request, low_bid_response
    ):
        """Alkimi cleans up tracking after deal completion."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(low_bid_response, "msg-resp")

        # Before deal
        assert bid_request.request_id in exchange._negotiations

        await exchange.finalize_deal(bid_request.request_id)

        # After deal - cleaned up
        assert bid_request.request_id not in exchange._negotiations

    @pytest.mark.asyncio
    async def test_record_delivery(
        self, mock_redis_bus, mock_ledger
    ):
        """Alkimi records delivery batches to ledger."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        entry_id = await exchange.record_delivery(
            deal_id="DEAL-TEST001",
            batch_number=1,
            impressions_in_batch=1000,
            cumulative_impressions=1000,
            created_by="seller-001",
        )

        # Verify ledger entry created
        mock_ledger.create_entry.assert_called_once()
        call_args = mock_ledger.create_entry.call_args
        assert call_args.kwargs["transaction_type"] == "delivery"

        # Verify recorded to ledger_deliveries
        mock_ledger.record_delivery.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_settlement(
        self, mock_redis_bus, mock_ledger
    ):
        """Alkimi records settlements to ledger with blockchain costs."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        entry_id = await exchange.record_settlement(
            deal_id="DEAL-TEST001",
            buyer_spend=100.0,
            seller_revenue=100.0,
        )

        # Verify ledger entry created
        mock_ledger.create_entry.assert_called_once()
        call_args = mock_ledger.create_entry.call_args
        assert call_args.kwargs["transaction_type"] == "settlement"

        # Verify recorded to ledger_settlements
        mock_ledger.record_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_pending_deals(
        self, mock_redis_bus, mock_ledger, bid_request, low_bid_response
    ):
        """Process all pending negotiations meeting minimum responses."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        await exchange.handle_bid_request(bid_request, "msg-req")
        await exchange.handle_bid_response(low_bid_response, "msg-resp")

        deals = await exchange.process_pending_deals(min_responses=1)

        assert len(deals) == 1
        assert deals[0].request_id == bid_request.request_id

    @pytest.mark.asyncio
    async def test_get_stats_includes_blockchain_costs(
        self, mock_redis_bus, mock_ledger
    ):
        """Stats include blockchain costs (vs exchange fees in Scenario A)."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        stats = await exchange.get_stats()

        assert stats["scenario"] == "C"
        assert stats["exchange_fee_pct"] == 0.0
        assert "blockchain" in stats
        assert "costs" in stats
        assert stats["costs"]["total_cost_usd"] == 1.125

    def test_set_simulation_day(self, mock_redis_bus, mock_ledger):
        """Can set simulation day for ledger entries."""
        exchange = AlkimiLedger(bus=mock_redis_bus, ledger=mock_ledger)

        exchange.set_simulation_day(15)
        assert exchange._simulation_day == 15


# -------------------------------------------------------------------------
# AgentStateRecovery Tests
# -------------------------------------------------------------------------


class TestAgentStateRecovery:
    """Tests for agent state recovery from ledger."""

    @pytest.mark.asyncio
    async def test_recover_buyer_state(self, mock_ledger):
        """Recovers buyer state from ledger entries."""
        mock_ledger.recover_agent_state.return_value = [
            {
                "entry_id": "ledger-001",
                "type": "bid_request",
                "payload": {
                    "request_id": "req-001",
                    "buyer_id": "buyer-001",
                    "impressions_requested": 10000,
                },
                "day": 1,
            },
            {
                "entry_id": "ledger-002",
                "type": "deal",
                "payload": {
                    "deal_id": "DEAL-001",
                    "impressions": 8000,
                    "total_cost": 64.0,
                },
                "day": 1,
            },
        ]

        recovery = AgentStateRecovery(mock_ledger)
        state = await recovery.recover_buyer_state("buyer-001", from_day=0)

        assert state["agent_id"] == "buyer-001"
        assert state["agent_type"] == "buyer"
        assert len(state["bid_requests"]) == 1
        assert len(state["deals"]) == 1
        assert state["total_spend"] == 64.0
        assert state["total_impressions"] == 8000

        # Verify recovery logged
        mock_ledger.log_recovery.assert_called_once()
        call_args = mock_ledger.log_recovery.call_args
        assert call_args.kwargs["agent_id"] == "buyer-001"
        assert call_args.kwargs["recovery_complete"] is True
        assert call_args.kwargs["state_accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_recover_seller_state(self, mock_ledger):
        """Recovers seller state from ledger entries."""
        mock_ledger.recover_agent_state.return_value = [
            {
                "entry_id": "ledger-001",
                "type": "bid_response",
                "payload": {
                    "response_id": "resp-001",
                    "seller_id": "seller-001",
                    "offered_cpm": 10.0,
                },
                "day": 1,
            },
            {
                "entry_id": "ledger-002",
                "type": "deal",
                "payload": {
                    "deal_id": "DEAL-001",
                    "impressions": 5000,
                    "total_cost": 50.0,
                },
                "day": 1,
            },
            {
                "entry_id": "ledger-003",
                "type": "delivery",
                "payload": {
                    "deal_id": "DEAL-001",
                    "batch_number": 1,
                    "impressions_in_batch": 1000,
                },
                "day": 2,
            },
        ]

        recovery = AgentStateRecovery(mock_ledger)
        state = await recovery.recover_seller_state("seller-001", from_day=0)

        assert state["agent_id"] == "seller-001"
        assert state["agent_type"] == "seller"
        assert len(state["bid_responses"]) == 1
        assert len(state["deals"]) == 1
        assert len(state["deliveries"]) == 1
        assert state["total_revenue"] == 50.0
        assert state["total_impressions_sold"] == 5000

    @pytest.mark.asyncio
    async def test_recover_empty_state(self, mock_ledger):
        """Recovery with no entries returns empty state."""
        mock_ledger.recover_agent_state.return_value = []

        recovery = AgentStateRecovery(mock_ledger)
        state = await recovery.recover_buyer_state("buyer-new")

        assert state["agent_id"] == "buyer-new"
        assert len(state["bid_requests"]) == 0
        assert len(state["deals"]) == 0
        assert state["total_spend"] == 0.0

    @pytest.mark.asyncio
    async def test_recover_from_specific_day(self, mock_ledger):
        """Recovery can start from a specific simulation day."""
        recovery = AgentStateRecovery(mock_ledger)
        await recovery.recover_buyer_state("buyer-001", from_day=15)

        mock_ledger.recover_agent_state.assert_called_with("buyer-001", 15)


# -------------------------------------------------------------------------
# Integration Tests (require database)
# -------------------------------------------------------------------------


@pytest.mark.integration
class TestAlkimiLedgerIntegration:
    """Integration tests requiring PostgreSQL."""

    @pytest.mark.asyncio
    async def test_full_deal_flow(self):
        """Full deal flow with real ledger client."""
        # This test requires running PostgreSQL
        # Skip if not available
        pytest.skip("Integration test - requires PostgreSQL")
