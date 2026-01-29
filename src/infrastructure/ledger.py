"""
Ledger client for Scenario C - Alkimi ledger-backed persistence.

Provides async PostgreSQL wrapper for the Sui/Walrus proxy ledger,
including entry creation, state recovery, and cost tracking.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
import structlog
import asyncpg

logger = structlog.get_logger()


@dataclass
class LedgerEntry:
    """A single ledger entry (immutable record)."""
    entry_id: str
    blob_id: str
    transaction_type: str
    payload: dict
    payload_size_bytes: int
    created_by: str
    created_by_type: str
    entry_hash: str
    previous_hash: Optional[str]
    block_number: int
    estimated_sui_gas: Decimal
    estimated_walrus_cost: Decimal
    total_cost_sui: Decimal
    total_cost_usd: Decimal
    simulation_day: int
    created_at: datetime


@dataclass
class RecoveryResult:
    """Result of an agent state recovery."""
    agent_id: str
    entries: list[dict]
    total_entries: int
    total_bytes: int
    recovery_time_ms: int
    success: bool
    missing_entries: list[str]


@dataclass
class BlockchainCosts:
    """Aggregated blockchain costs."""
    total_entries: int
    total_bytes: int
    total_gas_sui: Decimal
    total_walrus_sui: Decimal
    total_cost_sui: Decimal
    total_cost_usd: Decimal


class LedgerClient:
    """
    Async client for Scenario C ledger operations.

    Wraps PostgreSQL ledger functions for:
    - Creating immutable ledger entries
    - Recovering agent state from ledger
    - Tracking blockchain costs
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        """
        Initialize ledger client.

        Args:
            dsn: PostgreSQL connection string (default: from DATABASE_URL env)
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
        """
        self.dsn = dsn or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/rtb_sim"
        )
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: Optional[asyncpg.Pool] = None

    @property
    def pool(self) -> asyncpg.Pool:
        """Get connection pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError("LedgerClient not connected. Call connect() first.")
        return self._pool

    async def connect(self) -> "LedgerClient":
        """
        Connect to PostgreSQL and create connection pool.

        Returns:
            Self for chaining
        """
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_connections,
            max_size=self.max_connections,
        )
        logger.info("ledger_client.connected", dsn=self.dsn.split("@")[-1])
        return self

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("ledger_client.disconnected")

    async def __aenter__(self) -> "LedgerClient":
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Ledger Entry Operations
    # -------------------------------------------------------------------------

    async def create_entry(
        self,
        transaction_type: str,
        payload: dict,
        created_by: str,
        created_by_type: str,
        simulation_day: int,
    ) -> str:
        """
        Create a new ledger entry with automatic cost calculation.

        Uses the PostgreSQL create_ledger_entry() function which handles:
        - Hash calculation
        - Chain linking
        - Gas estimation
        - Storage cost calculation

        Args:
            transaction_type: Type of transaction (bid_request, bid_response, deal, etc.)
            payload: Transaction data to store
            created_by: Agent ID creating this entry
            created_by_type: Agent type (buyer, seller, exchange, system)
            simulation_day: Current simulation day

        Returns:
            entry_id of the created entry
        """
        async with self.pool.acquire() as conn:
            entry_id = await conn.fetchval(
                "SELECT create_ledger_entry($1, $2::jsonb, $3, $4, $5)",
                transaction_type,
                json.dumps(payload),
                created_by,
                created_by_type,
                simulation_day,
            )

        logger.info(
            "ledger_client.entry_created",
            entry_id=entry_id,
            transaction_type=transaction_type,
            created_by=created_by,
            simulation_day=simulation_day,
        )

        return entry_id

    async def get_entry(self, entry_id: str) -> Optional[LedgerEntry]:
        """
        Get a ledger entry by ID.

        Args:
            entry_id: Entry ID to fetch

        Returns:
            LedgerEntry if found, None otherwise
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM ledger_entries WHERE entry_id = $1
                """,
                entry_id,
            )

        if not row:
            return None

        return LedgerEntry(
            entry_id=row["entry_id"],
            blob_id=row["blob_id"],
            transaction_type=row["transaction_type"],
            payload=row["payload"],
            payload_size_bytes=row["payload_size_bytes"],
            created_by=row["created_by"],
            created_by_type=row["created_by_type"],
            entry_hash=row["entry_hash"],
            previous_hash=row["previous_hash"],
            block_number=row["block_number"],
            estimated_sui_gas=row["estimated_sui_gas"],
            estimated_walrus_cost=row["estimated_walrus_cost"],
            total_cost_sui=row["total_cost_sui"],
            total_cost_usd=row["total_cost_usd"],
            simulation_day=row["simulation_day"],
            created_at=row["created_at"],
        )

    async def get_entries_by_day(
        self,
        simulation_day: int,
        transaction_type: Optional[str] = None,
    ) -> list[LedgerEntry]:
        """
        Get all ledger entries for a simulation day.

        Args:
            simulation_day: Day to query
            transaction_type: Optional filter by transaction type

        Returns:
            List of LedgerEntry objects
        """
        async with self.pool.acquire() as conn:
            if transaction_type:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ledger_entries
                    WHERE simulation_day = $1 AND transaction_type = $2
                    ORDER BY block_number
                    """,
                    simulation_day,
                    transaction_type,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ledger_entries
                    WHERE simulation_day = $1
                    ORDER BY block_number
                    """,
                    simulation_day,
                )

        return [
            LedgerEntry(
                entry_id=row["entry_id"],
                blob_id=row["blob_id"],
                transaction_type=row["transaction_type"],
                payload=row["payload"],
                payload_size_bytes=row["payload_size_bytes"],
                created_by=row["created_by"],
                created_by_type=row["created_by_type"],
                entry_hash=row["entry_hash"],
                previous_hash=row["previous_hash"],
                block_number=row["block_number"],
                estimated_sui_gas=row["estimated_sui_gas"],
                estimated_walrus_cost=row["estimated_walrus_cost"],
                total_cost_sui=row["total_cost_sui"],
                total_cost_usd=row["total_cost_usd"],
                simulation_day=row["simulation_day"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -------------------------------------------------------------------------
    # State Recovery
    # -------------------------------------------------------------------------

    async def recover_agent_state(
        self,
        agent_id: str,
        from_day: int = 0,
    ) -> list[dict]:
        """
        Recover agent state from ledger entries.

        Uses the PostgreSQL recover_agent_state() function.

        Args:
            agent_id: Agent ID to recover state for
            from_day: Minimum simulation day (default: 0 = all)

        Returns:
            List of entry dictionaries with payload data
        """
        import time
        start_time = time.time()

        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT recover_agent_state($1, $2)",
                agent_id,
                from_day,
            )

        entries = result if result else []
        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "ledger_client.state_recovered",
            agent_id=agent_id,
            from_day=from_day,
            entries_recovered=len(entries),
            elapsed_ms=elapsed_ms,
        )

        return entries

    async def log_recovery(
        self,
        agent_id: str,
        agent_type: str,
        recovery_reason: str,
        entries_recovered: int,
        bytes_recovered: int,
        recovery_time_ms: int,
        recovery_complete: bool,
        state_accuracy: Optional[float],
        missing_entries: Optional[list[str]],
        simulation_day: int,
    ) -> int:
        """
        Log a state recovery event.

        Args:
            agent_id: Agent that recovered
            agent_type: Type of agent
            recovery_reason: Why recovery was needed
            entries_recovered: Number of entries recovered
            bytes_recovered: Total bytes recovered
            recovery_time_ms: Time taken in milliseconds
            recovery_complete: Whether recovery was successful
            state_accuracy: Accuracy vs pre-loss state (0.0-1.0)
            missing_entries: Any entries that couldn't be recovered
            simulation_day: Current simulation day

        Returns:
            ID of the log entry
        """
        async with self.pool.acquire() as conn:
            log_id = await conn.fetchval(
                """
                INSERT INTO ledger_recovery_log (
                    agent_id, agent_type, recovery_reason,
                    entries_recovered, bytes_recovered, recovery_complete,
                    state_accuracy, missing_entries, recovery_time_ms,
                    simulation_day
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """,
                agent_id,
                agent_type,
                recovery_reason,
                entries_recovered,
                bytes_recovered,
                recovery_complete,
                state_accuracy,
                missing_entries,
                recovery_time_ms,
                simulation_day,
            )

        logger.info(
            "ledger_client.recovery_logged",
            log_id=log_id,
            agent_id=agent_id,
            recovery_reason=recovery_reason,
            success=recovery_complete,
        )

        return log_id

    # -------------------------------------------------------------------------
    # Deal Recording
    # -------------------------------------------------------------------------

    async def record_deal(
        self,
        entry_id: str,
        deal_id: str,
        buyer_id: str,
        seller_id: str,
        impressions: int,
        cpm: float,
        total_cost: float,
        deal_status: str,
        buyer_signature: Optional[str] = None,
        seller_signature: Optional[str] = None,
    ) -> int:
        """
        Record a deal in the ledger_deals table.

        Args:
            entry_id: Reference to the ledger entry
            deal_id: Unique deal identifier
            buyer_id: Buyer agent ID
            seller_id: Seller agent ID
            impressions: Number of impressions
            cpm: Cost per mille
            total_cost: Total deal cost
            deal_status: Status at time of recording
            buyer_signature: Optional buyer signature
            seller_signature: Optional seller signature

        Returns:
            ID of the deal record
        """
        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                """
                INSERT INTO ledger_deals (
                    entry_id, deal_id, buyer_id, seller_id,
                    impressions, cpm, total_cost, deal_status,
                    buyer_signature, seller_signature
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """,
                entry_id,
                deal_id,
                buyer_id,
                seller_id,
                impressions,
                cpm,
                total_cost,
                deal_status,
                buyer_signature,
                seller_signature,
            )

        logger.debug(
            "ledger_client.deal_recorded",
            record_id=record_id,
            deal_id=deal_id,
            buyer_id=buyer_id,
            seller_id=seller_id,
        )

        return record_id

    async def record_delivery(
        self,
        entry_id: str,
        deal_id: str,
        batch_number: int,
        impressions_in_batch: int,
        cumulative_impressions: int,
        delivery_proof_hash: Optional[str] = None,
    ) -> int:
        """
        Record an impression delivery batch.

        Args:
            entry_id: Reference to the ledger entry
            deal_id: Deal being delivered
            batch_number: Sequential batch number
            impressions_in_batch: Impressions in this batch
            cumulative_impressions: Total delivered so far
            delivery_proof_hash: Hash of delivery proof

        Returns:
            ID of the delivery record
        """
        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                """
                INSERT INTO ledger_deliveries (
                    entry_id, deal_id, batch_number,
                    impressions_in_batch, cumulative_impressions,
                    delivery_proof_hash, timestamp_proof
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                RETURNING id
                """,
                entry_id,
                deal_id,
                batch_number,
                impressions_in_batch,
                cumulative_impressions,
                delivery_proof_hash,
            )

        return record_id

    async def record_settlement(
        self,
        entry_id: str,
        deal_id: str,
        buyer_spend: float,
        seller_revenue: float,
        blockchain_costs: float,
        platform_fee: float = 0.0,
        settlement_hash: Optional[str] = None,
    ) -> int:
        """
        Record a financial settlement.

        Args:
            entry_id: Reference to the ledger entry
            deal_id: Deal being settled
            buyer_spend: Amount buyer paid
            seller_revenue: Gross seller revenue
            blockchain_costs: Blockchain/storage costs
            platform_fee: Platform fee (minimal for Alkimi)
            settlement_hash: Hash of settlement data

        Returns:
            ID of the settlement record
        """
        import hashlib

        if not settlement_hash:
            settlement_data = f"{deal_id}:{buyer_spend}:{seller_revenue}:{blockchain_costs}"
            settlement_hash = hashlib.sha256(settlement_data.encode()).hexdigest()

        net_to_seller = seller_revenue - blockchain_costs - platform_fee

        async with self.pool.acquire() as conn:
            record_id = await conn.fetchval(
                """
                INSERT INTO ledger_settlements (
                    entry_id, deal_id, buyer_spend, seller_revenue,
                    platform_fee, blockchain_costs, net_to_seller,
                    settlement_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                entry_id,
                deal_id,
                buyer_spend,
                seller_revenue,
                platform_fee,
                blockchain_costs,
                net_to_seller,
                settlement_hash,
            )

        return record_id

    # -------------------------------------------------------------------------
    # Cost Tracking
    # -------------------------------------------------------------------------

    async def get_blockchain_costs(
        self,
        simulation_day: Optional[int] = None,
    ) -> BlockchainCosts:
        """
        Get aggregated blockchain costs.

        Args:
            simulation_day: Optional day filter (None = all time)

        Returns:
            BlockchainCosts aggregate
        """
        async with self.pool.acquire() as conn:
            if simulation_day is not None:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        COALESCE(SUM(payload_size_bytes), 0) as total_bytes,
                        COALESCE(SUM(estimated_sui_gas), 0) as total_gas_sui,
                        COALESCE(SUM(estimated_walrus_cost), 0) as total_walrus_sui,
                        COALESCE(SUM(total_cost_sui), 0) as total_cost_sui,
                        COALESCE(SUM(total_cost_usd), 0) as total_cost_usd
                    FROM ledger_entries
                    WHERE simulation_day = $1
                    """,
                    simulation_day,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        COALESCE(SUM(payload_size_bytes), 0) as total_bytes,
                        COALESCE(SUM(estimated_sui_gas), 0) as total_gas_sui,
                        COALESCE(SUM(estimated_walrus_cost), 0) as total_walrus_sui,
                        COALESCE(SUM(total_cost_sui), 0) as total_cost_sui,
                        COALESCE(SUM(total_cost_usd), 0) as total_cost_usd
                    FROM ledger_entries
                    """
                )

        return BlockchainCosts(
            total_entries=row["total_entries"],
            total_bytes=row["total_bytes"],
            total_gas_sui=row["total_gas_sui"],
            total_walrus_sui=row["total_walrus_sui"],
            total_cost_sui=row["total_cost_sui"],
            total_cost_usd=row["total_cost_usd"],
        )

    async def get_cost_per_1k_impressions(self, deal_id: str) -> Optional[float]:
        """
        Get blockchain cost per 1000 impressions for a deal.

        Args:
            deal_id: Deal to calculate for

        Returns:
            Cost per 1k impressions in USD, or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT cost_per_1k_impressions
                FROM ledger_cost_per_1k_impressions
                WHERE deal_id = $1
                """,
                deal_id,
            )

        return float(row["cost_per_1k_impressions"]) if row else None

    async def get_blockchain_state(self) -> dict[str, Any]:
        """
        Get current blockchain state.

        Returns:
            Dict with current block number, totals, etc.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM blockchain_state LIMIT 1")

        return {
            "current_block_number": row["current_block_number"],
            "total_entries": row["total_entries"],
            "total_gas_used": float(row["total_gas_used"]),
            "total_storage_used_bytes": row["total_storage_used_bytes"],
            "total_cost_sui": float(row["total_cost_sui"]),
            "total_cost_usd": float(row["total_cost_usd"]),
            "last_entry_hash": row["last_entry_hash"],
            "updated_at": row["updated_at"],
        }


# -------------------------------------------------------------------------
# Convenience factory
# -------------------------------------------------------------------------

async def create_ledger_client(
    dsn: Optional[str] = None,
) -> LedgerClient:
    """
    Create and connect a ledger client.

    Args:
        dsn: PostgreSQL connection string (default: from DATABASE_URL env)

    Returns:
        Connected LedgerClient instance
    """
    client = LedgerClient(dsn=dsn)
    await client.connect()
    return client
