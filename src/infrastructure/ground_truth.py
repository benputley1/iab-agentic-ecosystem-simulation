"""
Ground Truth Repository for RTB Simulation.

Provides the authoritative source of truth that agents CANNOT read directly.
Used for post-hoc verification of agent claims and hallucination detection.

Tables managed:
- inventory_reality: True inventory levels (agents see estimates)
- campaign_delivery_reality: Actual campaign delivery metrics
- agent_claims: Record of agent claims vs reality
- agent_decisions: All agent decisions for verification
- context_rot_events: Context loss events (Scenario B)
- hallucination_injections: Injected false data log
- fact_registry: Authoritative facts for verification
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
import structlog
import asyncpg

logger = structlog.get_logger()


class ClaimSeverity(str, Enum):
    """Hallucination severity levels based on discrepancy."""
    MINOR = "minor"         # < 5% discrepancy
    MODERATE = "moderate"   # 5-20% discrepancy
    SEVERE = "severe"       # > 20% discrepancy


class ContextRotEventType(str, Enum):
    """Types of context rot events."""
    DECAY = "decay"              # Gradual memory loss
    RESTART = "restart"          # Full context wipe
    MEMORY_PRESSURE = "memory_pressure"  # Forced eviction


@dataclass
class Fact:
    """A verifiable fact in the simulation."""
    fact_id: str
    fact_type: str
    entity_id: str
    entity_type: str
    fact_key: str
    fact_value: str
    fact_value_numeric: Optional[Decimal]
    fact_unit: Optional[str]
    valid_from: datetime
    valid_until: Optional[datetime]
    simulation_day: int
    created_at: datetime


@dataclass
class ClaimVerificationResult:
    """Result of verifying a claim against ground truth."""
    is_valid: bool
    actual_value: Optional[str]
    discrepancy_pct: Optional[float]
    severity: Optional[ClaimSeverity]
    fact_id: Optional[str] = None


@dataclass
class InventoryReality:
    """True inventory state (hidden from agents)."""
    publisher_id: str
    channel: str
    date: date
    actual_avails: int
    actual_fill_rate: Optional[Decimal]
    actual_viewability: Optional[Decimal]
    actual_ctr: Optional[Decimal]
    true_floor_cpm: Decimal


@dataclass
class CampaignDeliveryReality:
    """Actual campaign delivery (hidden from agents)."""
    campaign_id: str
    date: date
    actual_impressions: int
    actual_clicks: int
    actual_conversions: int
    actual_spend: Decimal
    actual_viewability: Optional[Decimal]
    actual_ctr: Optional[Decimal]
    actual_conversion_rate: Optional[Decimal]


@dataclass
class AgentClaim:
    """Record of an agent's claim for verification."""
    claim_id: Optional[int]
    agent_id: str
    agent_type: str
    scenario: str
    claim_type: str
    claim_context: Optional[dict]
    claimed_value: str
    actual_value: str
    is_hallucination: bool
    discrepancy_pct: Optional[float]
    severity: Optional[ClaimSeverity]
    decision_id: Optional[str]
    impact_description: Optional[str]
    simulation_day: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentDecision:
    """Record of an agent decision for post-hoc analysis."""
    decision_id: str
    agent_id: str
    agent_type: str
    scenario: str
    decision_type: str
    decision_input: dict
    decision_output: dict
    decision_reasoning: Optional[str]
    claimed_fact_id: Optional[str]
    decision_basis_verified: Optional[bool]
    simulation_day: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContextRotEvent:
    """Record of context loss event."""
    agent_id: str
    agent_type: str
    scenario: str
    event_type: ContextRotEventType
    keys_lost: list[str]
    keys_corrupted: list[str]
    recovery_attempted: bool
    recovery_successful: Optional[bool]
    recovery_accuracy: Optional[float]
    recovery_source: Optional[str]
    impact_description: Optional[str]
    decisions_affected: int
    simulation_day: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GroundTruthRepository:
    """
    Repository for ground truth data that agents cannot access.

    Provides methods to:
    - Record true inventory and delivery metrics
    - Register facts for claim verification
    - Record and verify agent claims
    - Track agent decisions
    - Log context rot events
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        """
        Initialize ground truth repository.

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
            raise RuntimeError("GroundTruthRepository not connected. Call connect() first.")
        return self._pool

    async def connect(self) -> "GroundTruthRepository":
        """Connect to the database."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_connections,
                max_size=self.max_connections,
            )
            logger.info("ground_truth.connected", dsn=self.dsn[:30] + "...")
        return self

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("ground_truth.disconnected")

    # -------------------------------------------------------------------------
    # Fact Registry
    # -------------------------------------------------------------------------

    async def register_fact(
        self,
        fact_type: str,
        entity_id: str,
        entity_type: str,
        fact_key: str,
        fact_value: str,
        simulation_day: int,
        fact_value_numeric: Optional[Decimal] = None,
        fact_unit: Optional[str] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
    ) -> str:
        """
        Register an authoritative fact for later verification.

        Args:
            fact_type: Type of fact (inventory, price, delivery, transaction)
            entity_id: ID of related entity
            entity_type: Type of entity (publisher, campaign, deal)
            fact_key: Key for the fact (e.g., avail_impressions, floor_cpm)
            fact_value: String value of the fact
            simulation_day: Current simulation day
            fact_value_numeric: Numeric value for comparison
            fact_unit: Unit of measure (e.g., impressions, USD)
            valid_from: When fact becomes valid
            valid_until: When fact expires

        Returns:
            Generated fact_id
        """
        valid_from = valid_from or datetime.utcnow()

        fact_id = await self.pool.fetchval(
            """
            INSERT INTO fact_registry (
                fact_type, entity_id, entity_type, fact_key, fact_value,
                fact_value_numeric, fact_unit, valid_from, valid_until, simulation_day
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING fact_id
            """,
            fact_type, entity_id, entity_type, fact_key, fact_value,
            fact_value_numeric, fact_unit, valid_from, valid_until, simulation_day
        )

        logger.debug(
            "ground_truth.fact_registered",
            fact_id=fact_id,
            fact_type=fact_type,
            entity_id=entity_id,
            fact_key=fact_key,
        )

        return fact_id

    async def get_fact(
        self,
        entity_id: str,
        fact_key: str,
        simulation_day: int,
    ) -> Optional[Fact]:
        """
        Get a fact from the registry.

        Args:
            entity_id: Entity ID to look up
            fact_key: Fact key to retrieve
            simulation_day: Simulation day for validity check

        Returns:
            Fact if found, None otherwise
        """
        row = await self.pool.fetchrow(
            """
            SELECT * FROM fact_registry
            WHERE entity_id = $1 AND fact_key = $2 AND simulation_day = $3
            ORDER BY created_at DESC LIMIT 1
            """,
            entity_id, fact_key, simulation_day
        )

        if not row:
            return None

        return Fact(
            fact_id=row["fact_id"],
            fact_type=row["fact_type"],
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            fact_key=row["fact_key"],
            fact_value=row["fact_value"],
            fact_value_numeric=row["fact_value_numeric"],
            fact_unit=row["fact_unit"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            simulation_day=row["simulation_day"],
            created_at=row["created_at"],
        )

    # -------------------------------------------------------------------------
    # Inventory Reality
    # -------------------------------------------------------------------------

    async def record_inventory_reality(
        self,
        publisher_id: str,
        channel: str,
        record_date: date,
        actual_avails: int,
        true_floor_cpm: Decimal,
        actual_fill_rate: Optional[Decimal] = None,
        actual_viewability: Optional[Decimal] = None,
        actual_ctr: Optional[Decimal] = None,
    ) -> int:
        """
        Record true inventory state (hidden from agents).

        Args:
            publisher_id: Publisher identifier
            channel: Ad channel (display, video, ctv)
            record_date: Date of the inventory
            actual_avails: True available impressions
            true_floor_cpm: True floor CPM price
            actual_fill_rate: True historical fill rate
            actual_viewability: True viewability rate
            actual_ctr: True click-through rate

        Returns:
            Record ID
        """
        record_id = await self.pool.fetchval(
            """
            INSERT INTO inventory_reality (
                publisher_id, channel, date, actual_avails, true_floor_cpm,
                actual_fill_rate, actual_viewability, actual_ctr
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (publisher_id, channel, date) DO UPDATE SET
                actual_avails = EXCLUDED.actual_avails,
                true_floor_cpm = EXCLUDED.true_floor_cpm,
                actual_fill_rate = EXCLUDED.actual_fill_rate,
                actual_viewability = EXCLUDED.actual_viewability,
                actual_ctr = EXCLUDED.actual_ctr
            RETURNING id
            """,
            publisher_id, channel, record_date, actual_avails, true_floor_cpm,
            actual_fill_rate, actual_viewability, actual_ctr
        )

        logger.debug(
            "ground_truth.inventory_recorded",
            publisher_id=publisher_id,
            channel=channel,
            actual_avails=actual_avails,
        )

        return record_id

    async def get_inventory_reality(
        self,
        publisher_id: str,
        channel: str,
        record_date: date,
    ) -> Optional[InventoryReality]:
        """Get true inventory state for verification."""
        row = await self.pool.fetchrow(
            """
            SELECT * FROM inventory_reality
            WHERE publisher_id = $1 AND channel = $2 AND date = $3
            """,
            publisher_id, channel, record_date
        )

        if not row:
            return None

        return InventoryReality(
            publisher_id=row["publisher_id"],
            channel=row["channel"],
            date=row["date"],
            actual_avails=row["actual_avails"],
            actual_fill_rate=row["actual_fill_rate"],
            actual_viewability=row["actual_viewability"],
            actual_ctr=row["actual_ctr"],
            true_floor_cpm=row["true_floor_cpm"],
        )

    # -------------------------------------------------------------------------
    # Campaign Delivery Reality
    # -------------------------------------------------------------------------

    async def record_delivery_reality(
        self,
        campaign_id: str,
        record_date: date,
        actual_impressions: int,
        actual_clicks: int,
        actual_conversions: int,
        actual_spend: Decimal,
        actual_viewability: Optional[Decimal] = None,
        actual_ctr: Optional[Decimal] = None,
        actual_conversion_rate: Optional[Decimal] = None,
    ) -> int:
        """Record true campaign delivery metrics."""
        record_id = await self.pool.fetchval(
            """
            INSERT INTO campaign_delivery_reality (
                campaign_id, date, actual_impressions, actual_clicks,
                actual_conversions, actual_spend, actual_viewability,
                actual_ctr, actual_conversion_rate
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (campaign_id, date) DO UPDATE SET
                actual_impressions = EXCLUDED.actual_impressions,
                actual_clicks = EXCLUDED.actual_clicks,
                actual_conversions = EXCLUDED.actual_conversions,
                actual_spend = EXCLUDED.actual_spend,
                actual_viewability = EXCLUDED.actual_viewability,
                actual_ctr = EXCLUDED.actual_ctr,
                actual_conversion_rate = EXCLUDED.actual_conversion_rate
            RETURNING id
            """,
            campaign_id, record_date, actual_impressions, actual_clicks,
            actual_conversions, actual_spend, actual_viewability,
            actual_ctr, actual_conversion_rate
        )

        return record_id

    # -------------------------------------------------------------------------
    # Claim Verification
    # -------------------------------------------------------------------------

    async def verify_claim(
        self,
        claim_type: str,
        entity_id: str,
        claimed_value: str,
        simulation_day: int,
    ) -> ClaimVerificationResult:
        """
        Verify a claim against ground truth.

        Uses the verify_claim SQL function to compare claimed vs actual values.

        Args:
            claim_type: Type of claim (matches fact_key)
            entity_id: Entity being claimed about
            claimed_value: The value being claimed
            simulation_day: Current simulation day

        Returns:
            ClaimVerificationResult with is_valid, actual_value, discrepancy
        """
        row = await self.pool.fetchrow(
            "SELECT * FROM verify_claim($1, $2, $3, $4)",
            claim_type, entity_id, claimed_value, simulation_day
        )

        if not row or row["actual_value"] is None:
            return ClaimVerificationResult(
                is_valid=False,
                actual_value=None,
                discrepancy_pct=None,
                severity=None,
            )

        discrepancy = row["discrepancy_pct"]
        severity = None

        if discrepancy is not None:
            if discrepancy < 5:
                severity = ClaimSeverity.MINOR
            elif discrepancy < 20:
                severity = ClaimSeverity.MODERATE
            else:
                severity = ClaimSeverity.SEVERE

        return ClaimVerificationResult(
            is_valid=row["is_valid"],
            actual_value=row["actual_value"],
            discrepancy_pct=float(discrepancy) if discrepancy else None,
            severity=severity,
        )

    async def record_claim(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        claim_type: str,
        claimed_value: str,
        simulation_day: int,
        entity_id: Optional[str] = None,
        claim_context: Optional[dict] = None,
        decision_id: Optional[str] = None,
        impact_description: Optional[str] = None,
    ) -> int:
        """
        Record an agent claim with automatic verification.

        Args:
            agent_id: Agent making the claim
            agent_type: Type of agent (buyer, seller, exchange)
            scenario: Scenario code (A, B, C)
            claim_type: Type of claim
            claimed_value: The claimed value
            simulation_day: Current simulation day
            entity_id: Related entity ID for verification
            claim_context: Additional context as JSON
            decision_id: Related decision ID
            impact_description: Description of impact

        Returns:
            Claim ID
        """
        # Verify claim if entity_id provided
        actual_value = "UNVERIFIED"
        is_hallucination = False
        discrepancy_pct = None
        severity = None

        if entity_id:
            verification = await self.verify_claim(
                claim_type, entity_id, claimed_value, simulation_day
            )
            if verification.actual_value:
                actual_value = verification.actual_value
                is_hallucination = not verification.is_valid
                discrepancy_pct = verification.discrepancy_pct
                severity = verification.severity

        claim_id = await self.pool.fetchval(
            """
            INSERT INTO agent_claims (
                agent_id, agent_type, scenario, claim_type, claim_context,
                claimed_value, actual_value, is_hallucination, discrepancy_pct,
                severity, decision_id, impact_description, simulation_day
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            """,
            agent_id, agent_type, scenario, claim_type,
            json.dumps(claim_context) if claim_context else None,
            claimed_value, actual_value, is_hallucination, discrepancy_pct,
            severity.value if severity else None,
            decision_id, impact_description, simulation_day
        )

        if is_hallucination:
            logger.warning(
                "ground_truth.hallucination_detected",
                agent_id=agent_id,
                claim_type=claim_type,
                claimed=claimed_value,
                actual=actual_value,
                discrepancy_pct=discrepancy_pct,
                severity=severity.value if severity else None,
            )

        return claim_id

    # -------------------------------------------------------------------------
    # Agent Decisions
    # -------------------------------------------------------------------------

    async def record_decision(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        decision_type: str,
        decision_input: dict,
        decision_output: dict,
        simulation_day: int,
        decision_reasoning: Optional[str] = None,
        claimed_fact_id: Optional[str] = None,
        decision_basis_verified: Optional[bool] = None,
    ) -> str:
        """
        Record an agent decision for post-hoc analysis.

        Args:
            agent_id: Agent making the decision
            agent_type: Type of agent
            scenario: Scenario code
            decision_type: Type of decision (bid, accept, reject, counter, allocate)
            decision_input: Input data the agent used
            decision_output: The decision made
            simulation_day: Current simulation day
            decision_reasoning: Agent's stated reasoning
            claimed_fact_id: FK to a specific claim
            decision_basis_verified: Was the basis factually correct?

        Returns:
            Decision ID
        """
        decision_id = await self.pool.fetchval(
            """
            INSERT INTO agent_decisions (
                agent_id, agent_type, scenario, decision_type,
                decision_input, decision_output, decision_reasoning,
                claimed_fact_id, decision_basis_verified, simulation_day
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """,
            agent_id, agent_type, scenario, decision_type,
            json.dumps(decision_input), json.dumps(decision_output),
            decision_reasoning, claimed_fact_id, decision_basis_verified,
            simulation_day
        )

        logger.debug(
            "ground_truth.decision_recorded",
            decision_id=decision_id,
            agent_id=agent_id,
            decision_type=decision_type,
            verified=decision_basis_verified,
        )

        return decision_id

    # -------------------------------------------------------------------------
    # Context Rot Events
    # -------------------------------------------------------------------------

    async def record_context_rot(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        event_type: ContextRotEventType,
        simulation_day: int,
        keys_lost: Optional[list[str]] = None,
        keys_corrupted: Optional[list[str]] = None,
        recovery_attempted: bool = False,
        recovery_successful: Optional[bool] = None,
        recovery_accuracy: Optional[float] = None,
        recovery_source: Optional[str] = None,
        impact_description: Optional[str] = None,
        decisions_affected: int = 0,
    ) -> int:
        """
        Record a context rot event.

        Args:
            agent_id: Agent that lost context
            agent_type: Type of agent
            scenario: Scenario code
            event_type: Type of rot (decay, restart, memory_pressure)
            simulation_day: Current simulation day
            keys_lost: List of memory keys lost
            keys_corrupted: List of memory keys corrupted
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery succeeded
            recovery_accuracy: How accurate the recovery was (0-1)
            recovery_source: Source of recovery (ledger, checkpoint, peer, none)
            impact_description: Description of impact
            decisions_affected: Number of decisions affected

        Returns:
            Event ID
        """
        event_id = await self.pool.fetchval(
            """
            INSERT INTO context_rot_events (
                agent_id, agent_type, scenario, event_type,
                keys_lost, keys_corrupted, recovery_attempted,
                recovery_successful, recovery_accuracy, recovery_source,
                impact_description, decisions_affected, simulation_day
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            """,
            agent_id, agent_type, scenario, event_type.value,
            keys_lost or [], keys_corrupted or [],
            recovery_attempted, recovery_successful, recovery_accuracy,
            recovery_source, impact_description, decisions_affected,
            simulation_day
        )

        logger.info(
            "ground_truth.context_rot_recorded",
            event_id=event_id,
            agent_id=agent_id,
            event_type=event_type.value,
            keys_lost=len(keys_lost or []),
            recovery_attempted=recovery_attempted,
        )

        return event_id

    # -------------------------------------------------------------------------
    # Hallucination Injection Tracking
    # -------------------------------------------------------------------------

    async def record_hallucination_injection(
        self,
        agent_id: str,
        scenario: str,
        injection_type: str,
        original_value: str,
        injected_value: str,
        injection_factor: float,
        simulation_day: int,
    ) -> int:
        """
        Record an injected hallucination (Scenario B).

        Args:
            agent_id: Agent receiving corrupted data
            scenario: Scenario code (should be B)
            injection_type: Type of injection (inventory, price, history)
            original_value: True value
            injected_value: Corrupted value
            injection_factor: Multiplier applied
            simulation_day: Current simulation day

        Returns:
            Injection ID
        """
        injection_id = await self.pool.fetchval(
            """
            INSERT INTO hallucination_injections (
                agent_id, scenario, injection_type,
                original_value, injected_value, injection_factor,
                simulation_day
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            agent_id, scenario, injection_type,
            original_value, injected_value, injection_factor,
            simulation_day
        )

        logger.debug(
            "ground_truth.injection_recorded",
            injection_id=injection_id,
            agent_id=agent_id,
            injection_type=injection_type,
        )

        return injection_id

    async def mark_injection_detected(
        self,
        injection_id: int,
        detected_by: str,
    ) -> None:
        """Mark an injection as detected."""
        await self.pool.execute(
            """
            UPDATE hallucination_injections
            SET was_detected = TRUE,
                detected_by = $1,
                detection_delay_seconds = EXTRACT(EPOCH FROM (NOW() - timestamp))::INTEGER
            WHERE id = $2
            """,
            detected_by, injection_id
        )

    # -------------------------------------------------------------------------
    # Analytics Queries
    # -------------------------------------------------------------------------

    async def get_hallucination_rates(self) -> list[dict]:
        """Get hallucination rates by scenario and agent type."""
        rows = await self.pool.fetch("SELECT * FROM hallucination_rates")
        return [dict(row) for row in rows]

    async def get_hallucination_severity_dist(self) -> list[dict]:
        """Get hallucination severity distribution."""
        rows = await self.pool.fetch("SELECT * FROM hallucination_severity_dist")
        return [dict(row) for row in rows]

    async def get_context_rot_impact(self) -> list[dict]:
        """Get context rot impact by day."""
        rows = await self.pool.fetch("SELECT * FROM context_rot_impact")
        return [dict(row) for row in rows]

    async def get_injection_detection_rates(self) -> list[dict]:
        """Get hallucination injection detection rates."""
        rows = await self.pool.fetch("SELECT * FROM injection_detection_rates")
        return [dict(row) for row in rows]

    async def get_summary(self, scenario: Optional[str] = None) -> dict:
        """
        Get summary statistics for ground truth data.

        Args:
            scenario: Optional scenario filter

        Returns:
            Summary dict with counts and rates
        """
        where_clause = "WHERE scenario = $1" if scenario else ""
        params = [scenario] if scenario else []

        # Total claims
        total_claims = await self.pool.fetchval(
            f"SELECT COUNT(*) FROM agent_claims {where_clause}",
            *params
        )

        # Hallucinated claims
        hallucinated = await self.pool.fetchval(
            f"SELECT COUNT(*) FROM agent_claims {where_clause} {'AND' if where_clause else 'WHERE'} is_hallucination = TRUE",
            *params
        )

        # Total decisions
        total_decisions = await self.pool.fetchval(
            f"SELECT COUNT(*) FROM agent_decisions {where_clause}",
            *params
        )

        # Verified decisions
        verified_decisions = await self.pool.fetchval(
            f"SELECT COUNT(*) FROM agent_decisions {where_clause} {'AND' if where_clause else 'WHERE'} decision_basis_verified = TRUE",
            *params
        )

        # Context rot events
        rot_events = await self.pool.fetchval(
            f"SELECT COUNT(*) FROM context_rot_events {where_clause}",
            *params
        )

        # Injections
        injections = await self.pool.fetchval(
            f"SELECT COUNT(*) FROM hallucination_injections {where_clause}",
            *params
        )

        detected_injections = await self.pool.fetchval(
            f"SELECT COUNT(*) FROM hallucination_injections {where_clause} {'AND' if where_clause else 'WHERE'} was_detected = TRUE",
            *params
        )

        return {
            "scenario": scenario or "all",
            "total_claims": total_claims or 0,
            "hallucinated_claims": hallucinated or 0,
            "hallucination_rate": (
                (hallucinated / total_claims * 100) if total_claims else 0.0
            ),
            "total_decisions": total_decisions or 0,
            "verified_decisions": verified_decisions or 0,
            "verification_rate": (
                (verified_decisions / total_decisions * 100) if total_decisions else 0.0
            ),
            "context_rot_events": rot_events or 0,
            "total_injections": injections or 0,
            "detected_injections": detected_injections or 0,
            "injection_detection_rate": (
                (detected_injections / injections * 100) if injections else 0.0
            ),
        }


# Factory function for convenience
async def create_ground_truth_repo(
    dsn: Optional[str] = None,
) -> GroundTruthRepository:
    """Create and connect a GroundTruthRepository."""
    repo = GroundTruthRepository(dsn=dsn)
    await repo.connect()
    return repo
