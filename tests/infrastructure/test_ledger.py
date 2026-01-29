"""Tests for ledger client infrastructure."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from datetime import datetime

from src.infrastructure.ledger import (
    LedgerClient,
    LedgerEntry,
    RecoveryResult,
    BlockchainCosts,
    create_ledger_client,
)


# -------------------------------------------------------------------------
# Unit Tests (mocked database)
# -------------------------------------------------------------------------


class TestLedgerClient:
    """Tests for LedgerClient class."""

    def test_init_default_dsn(self):
        """Client uses default DSN from environment."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://test:test@localhost/test"}):
            client = LedgerClient()
            assert "localhost" in client.dsn

    def test_init_custom_dsn(self):
        """Client accepts custom DSN."""
        client = LedgerClient(dsn="postgresql://custom:custom@custom/custom")
        assert "custom" in client.dsn

    def test_pool_not_connected_raises(self):
        """Accessing pool before connect raises RuntimeError."""
        client = LedgerClient()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.pool

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self):
        """Connect creates asyncpg pool."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_pool = MagicMock()
            mock_create.return_value = mock_pool

            client = LedgerClient(dsn="postgresql://test:test@localhost/test")
            result = await client.connect()

            assert result is client
            assert client._pool is mock_pool
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_closes_pool(self):
        """Disconnect closes the pool."""
        mock_pool = AsyncMock()

        client = LedgerClient()
        client._pool = mock_pool

        await client.disconnect()

        mock_pool.close.assert_called_once()
        assert client._pool is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Client works as async context manager."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_pool = AsyncMock()
            mock_create.return_value = mock_pool

            async with LedgerClient(dsn="postgresql://test:test@localhost/test") as client:
                assert client._pool is mock_pool

            mock_pool.close.assert_called_once()


class TestLedgerClientOperations:
    """Tests for LedgerClient operations (mocked)."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock pool with connection."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        pool.acquire.return_value.__aexit__.return_value = None
        return pool, conn

    @pytest.fixture
    def client_with_pool(self, mock_pool):
        """Create client with mocked pool."""
        pool, conn = mock_pool
        client = LedgerClient()
        client._pool = pool
        return client, conn

    @pytest.mark.asyncio
    async def test_create_entry(self, client_with_pool):
        """create_entry calls PostgreSQL function correctly."""
        client, conn = client_with_pool
        conn.fetchval.return_value = "ledger-test-123"

        entry_id = await client.create_entry(
            transaction_type="bid_request",
            payload={"buyer_id": "buyer-001"},
            created_by="buyer-001",
            created_by_type="buyer",
            simulation_day=1,
        )

        assert entry_id == "ledger-test-123"
        conn.fetchval.assert_called_once()

        # Verify SQL function called
        call_args = conn.fetchval.call_args[0]
        assert "create_ledger_entry" in call_args[0]
        assert call_args[1] == "bid_request"
        assert "buyer_id" in call_args[2]  # JSON payload

    @pytest.mark.asyncio
    async def test_get_entry(self, client_with_pool):
        """get_entry fetches entry by ID."""
        client, conn = client_with_pool
        conn.fetchrow.return_value = {
            "entry_id": "ledger-001",
            "blob_id": "walrus-001",
            "transaction_type": "deal",
            "payload": {"deal_id": "DEAL-001"},
            "payload_size_bytes": 100,
            "created_by": "buyer-001",
            "created_by_type": "buyer",
            "entry_hash": "abc" * 21 + "a",  # 64 chars
            "previous_hash": None,
            "block_number": 1,
            "estimated_sui_gas": Decimal("0.001"),
            "estimated_walrus_cost": Decimal("0.0005"),
            "total_cost_sui": Decimal("0.0015"),
            "total_cost_usd": Decimal("0.00225"),
            "simulation_day": 1,
            "created_at": datetime.utcnow(),
        }

        entry = await client.get_entry("ledger-001")

        assert entry is not None
        assert entry.entry_id == "ledger-001"
        assert entry.transaction_type == "deal"

    @pytest.mark.asyncio
    async def test_get_entry_not_found(self, client_with_pool):
        """get_entry returns None for missing entry."""
        client, conn = client_with_pool
        conn.fetchrow.return_value = None

        entry = await client.get_entry("nonexistent")

        assert entry is None

    @pytest.mark.asyncio
    async def test_recover_agent_state(self, client_with_pool):
        """recover_agent_state calls PostgreSQL function."""
        client, conn = client_with_pool
        conn.fetchval.return_value = [
            {"entry_id": "ledger-001", "type": "deal", "payload": {}}
        ]

        entries = await client.recover_agent_state("buyer-001", from_day=5)

        assert len(entries) == 1
        conn.fetchval.assert_called_once()

        # Verify SQL function called
        call_args = conn.fetchval.call_args[0]
        assert "recover_agent_state" in call_args[0]
        assert call_args[1] == "buyer-001"
        assert call_args[2] == 5

    @pytest.mark.asyncio
    async def test_log_recovery(self, client_with_pool):
        """log_recovery inserts recovery log entry."""
        client, conn = client_with_pool
        conn.fetchval.return_value = 42

        log_id = await client.log_recovery(
            agent_id="buyer-001",
            agent_type="buyer",
            recovery_reason="context_recovery",
            entries_recovered=10,
            bytes_recovered=5000,
            recovery_time_ms=50,
            recovery_complete=True,
            state_accuracy=0.99,
            missing_entries=None,
            simulation_day=15,
        )

        assert log_id == 42
        conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_deal(self, client_with_pool):
        """record_deal inserts deal record."""
        client, conn = client_with_pool
        conn.fetchval.return_value = 1

        record_id = await client.record_deal(
            entry_id="ledger-001",
            deal_id="DEAL-001",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=10.0,
            total_cost=100.0,
            deal_status="confirmed",
        )

        assert record_id == 1
        conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_delivery(self, client_with_pool):
        """record_delivery inserts delivery record."""
        client, conn = client_with_pool
        conn.fetchval.return_value = 1

        record_id = await client.record_delivery(
            entry_id="ledger-001",
            deal_id="DEAL-001",
            batch_number=1,
            impressions_in_batch=1000,
            cumulative_impressions=1000,
        )

        assert record_id == 1

    @pytest.mark.asyncio
    async def test_record_settlement(self, client_with_pool):
        """record_settlement inserts settlement record."""
        client, conn = client_with_pool
        conn.fetchval.return_value = 1

        record_id = await client.record_settlement(
            entry_id="ledger-001",
            deal_id="DEAL-001",
            buyer_spend=100.0,
            seller_revenue=100.0,
            blockchain_costs=0.05,
        )

        assert record_id == 1

    @pytest.mark.asyncio
    async def test_get_blockchain_costs(self, client_with_pool):
        """get_blockchain_costs returns aggregated costs."""
        client, conn = client_with_pool
        conn.fetchrow.return_value = {
            "total_entries": 100,
            "total_bytes": 100000,
            "total_gas_sui": Decimal("1.0"),
            "total_walrus_sui": Decimal("0.5"),
            "total_cost_sui": Decimal("1.5"),
            "total_cost_usd": Decimal("2.25"),
        }

        costs = await client.get_blockchain_costs()

        assert costs.total_entries == 100
        assert costs.total_bytes == 100000
        assert costs.total_cost_sui == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_get_blockchain_costs_by_day(self, client_with_pool):
        """get_blockchain_costs can filter by simulation day."""
        client, conn = client_with_pool
        conn.fetchrow.return_value = {
            "total_entries": 10,
            "total_bytes": 10000,
            "total_gas_sui": Decimal("0.1"),
            "total_walrus_sui": Decimal("0.05"),
            "total_cost_sui": Decimal("0.15"),
            "total_cost_usd": Decimal("0.225"),
        }

        costs = await client.get_blockchain_costs(simulation_day=5)

        assert costs.total_entries == 10

        # Verify day filter applied
        call_args = conn.fetchrow.call_args[0]
        assert "simulation_day = $1" in call_args[0]

    @pytest.mark.asyncio
    async def test_get_blockchain_state(self, client_with_pool):
        """get_blockchain_state returns current state."""
        client, conn = client_with_pool
        now = datetime.utcnow()
        conn.fetchrow.return_value = {
            "current_block_number": 500,
            "total_entries": 450,
            "total_gas_used": Decimal("5.0"),
            "total_storage_used_bytes": 500000,
            "total_cost_sui": Decimal("7.5"),
            "total_cost_usd": Decimal("11.25"),
            "last_entry_hash": "abc123",
            "updated_at": now,
        }

        state = await client.get_blockchain_state()

        assert state["current_block_number"] == 500
        assert state["total_entries"] == 450
        assert state["total_cost_usd"] == 11.25


# -------------------------------------------------------------------------
# Data Class Tests
# -------------------------------------------------------------------------


class TestLedgerEntry:
    """Tests for LedgerEntry dataclass."""

    def test_ledger_entry_creation(self):
        """LedgerEntry can be created with all fields."""
        entry = LedgerEntry(
            entry_id="ledger-001",
            blob_id="walrus-001",
            transaction_type="deal",
            payload={"deal_id": "DEAL-001"},
            payload_size_bytes=100,
            created_by="buyer-001",
            created_by_type="buyer",
            entry_hash="a" * 64,
            previous_hash=None,
            block_number=1,
            estimated_sui_gas=Decimal("0.001"),
            estimated_walrus_cost=Decimal("0.0005"),
            total_cost_sui=Decimal("0.0015"),
            total_cost_usd=Decimal("0.00225"),
            simulation_day=1,
            created_at=datetime.utcnow(),
        )

        assert entry.entry_id == "ledger-001"
        assert entry.transaction_type == "deal"


class TestBlockchainCosts:
    """Tests for BlockchainCosts dataclass."""

    def test_blockchain_costs_creation(self):
        """BlockchainCosts can be created."""
        costs = BlockchainCosts(
            total_entries=100,
            total_bytes=50000,
            total_gas_sui=Decimal("0.5"),
            total_walrus_sui=Decimal("0.25"),
            total_cost_sui=Decimal("0.75"),
            total_cost_usd=Decimal("1.125"),
        )

        assert costs.total_entries == 100
        assert costs.total_cost_usd == Decimal("1.125")


# -------------------------------------------------------------------------
# Factory Function Tests
# -------------------------------------------------------------------------


class TestCreateLedgerClient:
    """Tests for create_ledger_client factory."""

    @pytest.mark.asyncio
    async def test_create_ledger_client(self):
        """Factory creates connected client."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_pool = MagicMock()
            mock_create.return_value = mock_pool

            client = await create_ledger_client(
                dsn="postgresql://test:test@localhost/test"
            )

            assert client._pool is mock_pool
            mock_create.assert_called_once()


# -------------------------------------------------------------------------
# Integration Tests (require database)
# -------------------------------------------------------------------------


@pytest.mark.integration
class TestLedgerClientIntegration:
    """Integration tests requiring PostgreSQL."""

    @pytest.mark.asyncio
    async def test_create_and_get_entry(self):
        """Create entry and retrieve it."""
        # Skip if PostgreSQL not available
        pytest.skip("Integration test - requires PostgreSQL")

    @pytest.mark.asyncio
    async def test_hash_chain_integrity(self):
        """Verify hash chain links entries correctly."""
        pytest.skip("Integration test - requires PostgreSQL")

    @pytest.mark.asyncio
    async def test_cost_calculation(self):
        """Verify gas and storage costs calculated correctly."""
        pytest.skip("Integration test - requires PostgreSQL")
