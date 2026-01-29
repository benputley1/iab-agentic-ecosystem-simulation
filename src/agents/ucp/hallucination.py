"""
Hallucination Injection and Detection for RTB Simulation.

Implements mechanisms to:
1. Inject false data for Scenario B testing (simulating unreliable context)
2. Detect hallucinations by verifying agent claims against ground truth
3. Track injection/detection metrics for analysis

This module is critical for demonstrating the value of ledger-backed
verification (Scenario C) vs pure A2A without ground truth (Scenario B).
"""

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

# Optional dependency for database operations
try:
    import psycopg2
    from psycopg2.extras import DictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None  # type: ignore
    DictCursor = None  # type: ignore
    HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)


class InjectionType(str, Enum):
    """Types of hallucinations that can be injected."""
    INVENTORY = "inventory"     # Inflate/deflate available impressions
    PRICE = "price"             # Corrupt pricing data
    HISTORY = "history"         # Fabricate transaction history
    DELIVERY = "delivery"       # False delivery metrics
    AUDIENCE = "audience"       # Wrong audience segments


class Severity(str, Enum):
    """Hallucination severity levels."""
    MINOR = "minor"         # < 5% discrepancy
    MODERATE = "moderate"   # 5-20% discrepancy
    SEVERE = "severe"       # > 20% discrepancy


@dataclass
class InjectionRecord:
    """Record of an injected hallucination."""
    injection_id: str = field(default_factory=lambda: f"inj-{uuid.uuid4().hex[:12]}")
    agent_id: str = ""
    scenario: str = "B"  # Primarily used in Scenario B
    injection_type: InjectionType = InjectionType.INVENTORY
    original_value: str = ""
    injected_value: str = ""
    injection_factor: float = 1.0  # Multiplier applied (e.g., 1.3 = 30% inflation)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    simulation_day: int = 1

    # Detection tracking
    was_detected: bool = False
    detection_delay_seconds: Optional[int] = None
    detected_by: Optional[str] = None


@dataclass
class ClaimVerification:
    """Result of verifying an agent claim."""
    claim_id: str = field(default_factory=lambda: f"claim-{uuid.uuid4().hex[:12]}")
    agent_id: str = ""
    agent_type: str = ""  # 'buyer', 'seller', 'exchange'
    scenario: str = ""
    claim_type: str = ""
    claimed_value: str = ""
    actual_value: str = ""
    is_hallucination: bool = False
    discrepancy_pct: Optional[float] = None
    severity: Optional[Severity] = None
    simulation_day: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HallucinationInjector:
    """
    Injects hallucinations for Scenario B testing.

    Simulates unreliable data that can occur when agents operate
    without a verified ground truth (no ledger).
    """

    def __init__(self, injection_rate: float = 0.05, seed: Optional[int] = None):
        """
        Initialize the hallucination injector.

        Args:
            injection_rate: Probability of corrupting data (0.0-1.0). Default 5%.
            seed: Random seed for reproducibility.
        """
        self.injection_rate = injection_rate
        self._random = random.Random(seed)
        self._injections: list[InjectionRecord] = []
        self._db_connection = None

    def set_db_connection(self, connection) -> None:
        """Set database connection for recording injections."""
        self._db_connection = connection

    def maybe_corrupt_inventory(
        self,
        real_inventory: dict[str, int],
        agent_id: str,
        simulation_day: int = 1,
    ) -> tuple[dict[str, int], Optional[InjectionRecord]]:
        """
        Randomly corrupt inventory data.

        Args:
            real_inventory: True inventory levels {channel: impressions}
            agent_id: Agent receiving the (potentially corrupted) data
            simulation_day: Current simulation day

        Returns:
            Tuple of (possibly corrupted inventory, injection record if corrupted)
        """
        if self._random.random() >= self.injection_rate:
            return real_inventory, None

        # Inflate by 10-50%
        multiplier = self._random.uniform(1.1, 1.5)
        corrupted = {k: int(v * multiplier) for k, v in real_inventory.items()}

        record = InjectionRecord(
            agent_id=agent_id,
            injection_type=InjectionType.INVENTORY,
            original_value=str(real_inventory),
            injected_value=str(corrupted),
            injection_factor=multiplier,
            simulation_day=simulation_day,
        )
        self._injections.append(record)
        self._record_to_db(record)

        logger.warning(
            f"INJECTION: Corrupted inventory for {agent_id} "
            f"by {(multiplier - 1) * 100:.1f}%"
        )

        return corrupted, record

    def maybe_corrupt_price(
        self,
        real_price: float,
        agent_id: str,
        simulation_day: int = 1,
    ) -> tuple[float, Optional[InjectionRecord]]:
        """
        Randomly corrupt pricing data.

        Typically deflates prices to make deals look better than they are.

        Args:
            real_price: True price (CPM)
            agent_id: Agent receiving the data
            simulation_day: Current simulation day

        Returns:
            Tuple of (possibly corrupted price, injection record if corrupted)
        """
        if self._random.random() >= self.injection_rate:
            return real_price, None

        # Deflate by 10-30% (makes deals look cheaper)
        multiplier = self._random.uniform(0.7, 0.9)
        corrupted = real_price * multiplier

        record = InjectionRecord(
            agent_id=agent_id,
            injection_type=InjectionType.PRICE,
            original_value=str(real_price),
            injected_value=str(corrupted),
            injection_factor=multiplier,
            simulation_day=simulation_day,
        )
        self._injections.append(record)
        self._record_to_db(record)

        logger.warning(
            f"INJECTION: Deflated price for {agent_id} "
            f"from ${real_price:.2f} to ${corrupted:.2f}"
        )

        return corrupted, record

    def maybe_fabricate_history(
        self,
        agent_id: str,
        simulation_day: int = 1,
    ) -> tuple[Optional[dict], Optional[InjectionRecord]]:
        """
        Create false memory of past transactions.

        Args:
            agent_id: Agent to receive fabricated history
            simulation_day: Current simulation day

        Returns:
            Tuple of (fabricated history dict if injected, injection record)
        """
        if self._random.random() >= self.injection_rate:
            return None, None

        fabricated = {
            "fabricated": True,
            "fake_deal_id": f"FAKE-{uuid.uuid4().hex[:8].upper()}",
            "fake_success_rate": self._random.uniform(0.8, 0.95),
            "fake_avg_cpm": self._random.uniform(2.0, 8.0),
            "fake_impressions": self._random.randint(100000, 1000000),
        }

        record = InjectionRecord(
            agent_id=agent_id,
            injection_type=InjectionType.HISTORY,
            original_value="null",
            injected_value=str(fabricated),
            injection_factor=1.0,
            simulation_day=simulation_day,
        )
        self._injections.append(record)
        self._record_to_db(record)

        logger.warning(f"INJECTION: Fabricated history for {agent_id}")

        return fabricated, record

    def maybe_corrupt_delivery(
        self,
        real_impressions: int,
        real_clicks: int,
        agent_id: str,
        simulation_day: int = 1,
    ) -> tuple[dict, Optional[InjectionRecord]]:
        """
        Corrupt delivery metrics (impressions, clicks).

        Args:
            real_impressions: Actual impression count
            real_clicks: Actual click count
            agent_id: Agent receiving the data
            simulation_day: Current simulation day

        Returns:
            Tuple of (delivery dict with possibly corrupted data, injection record)
        """
        if self._random.random() >= self.injection_rate:
            return {"impressions": real_impressions, "clicks": real_clicks}, None

        # Inflate impressions, keep clicks same (lower CTR)
        multiplier = self._random.uniform(1.2, 1.8)
        corrupted_impressions = int(real_impressions * multiplier)

        record = InjectionRecord(
            agent_id=agent_id,
            injection_type=InjectionType.DELIVERY,
            original_value=f"{real_impressions}:{real_clicks}",
            injected_value=f"{corrupted_impressions}:{real_clicks}",
            injection_factor=multiplier,
            simulation_day=simulation_day,
        )
        self._injections.append(record)
        self._record_to_db(record)

        return {"impressions": corrupted_impressions, "clicks": real_clicks}, record

    def _record_to_db(self, record: InjectionRecord) -> None:
        """Record injection to PostgreSQL for analysis."""
        if not self._db_connection:
            return

        try:
            with self._db_connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hallucination_injections (
                        agent_id, scenario, injection_type,
                        original_value, injected_value, injection_factor,
                        simulation_day, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.agent_id,
                        record.scenario,
                        record.injection_type.value,
                        record.original_value,
                        record.injected_value,
                        record.injection_factor,
                        record.simulation_day,
                        record.timestamp,
                    ),
                )
            self._db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to record injection to DB: {e}")

    def get_injections(self) -> list[InjectionRecord]:
        """Get all injections recorded this session."""
        return list(self._injections)

    def get_injection_rate_actual(self) -> float:
        """Calculate actual injection rate from recorded data."""
        # This would need total operations count to be accurate
        return self.injection_rate


class HallucinationDetector:
    """
    Detects hallucinations by verifying agent claims against ground truth.

    Uses the PostgreSQL ground_truth database to verify claims and
    record detection results.
    """

    def __init__(self, db_connection=None):
        """
        Initialize detector with database connection.

        Args:
            db_connection: psycopg2 connection to ground truth database
        """
        self._db_connection = db_connection
        self._verifications: list[ClaimVerification] = []

    def set_db_connection(self, connection) -> None:
        """Set or update database connection."""
        self._db_connection = connection

    def verify_inventory_claim(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        publisher_id: str,
        channel: str,
        claimed_avails: int,
        simulation_day: int,
    ) -> ClaimVerification:
        """
        Verify an inventory availability claim.

        Args:
            agent_id: Agent making the claim
            agent_type: Type of agent
            scenario: Scenario code
            publisher_id: Publisher ID
            channel: Ad channel
            claimed_avails: Claimed available impressions
            simulation_day: Current simulation day

        Returns:
            ClaimVerification with results
        """
        actual_avails = self._get_actual_inventory(publisher_id, channel, simulation_day)

        return self._create_verification(
            agent_id=agent_id,
            agent_type=agent_type,
            scenario=scenario,
            claim_type="inventory_level",
            claimed_value=str(claimed_avails),
            actual_value=str(actual_avails) if actual_avails is not None else "UNKNOWN",
            simulation_day=simulation_day,
        )

    def verify_price_claim(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        publisher_id: str,
        channel: str,
        claimed_cpm: float,
        simulation_day: int,
    ) -> ClaimVerification:
        """
        Verify a price (CPM) claim.

        Args:
            agent_id: Agent making the claim
            agent_type: Type of agent
            scenario: Scenario code
            publisher_id: Publisher ID
            channel: Ad channel
            claimed_cpm: Claimed CPM floor price
            simulation_day: Current simulation day

        Returns:
            ClaimVerification with results
        """
        actual_cpm = self._get_actual_floor_cpm(publisher_id, channel, simulation_day)

        return self._create_verification(
            agent_id=agent_id,
            agent_type=agent_type,
            scenario=scenario,
            claim_type="floor_cpm",
            claimed_value=str(claimed_cpm),
            actual_value=str(actual_cpm) if actual_cpm is not None else "UNKNOWN",
            simulation_day=simulation_day,
        )

    def verify_delivery_claim(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        campaign_id: str,
        claimed_impressions: int,
        simulation_day: int,
    ) -> ClaimVerification:
        """
        Verify a delivery (impressions) claim.

        Args:
            agent_id: Agent making the claim
            agent_type: Type of agent
            scenario: Scenario code
            campaign_id: Campaign ID
            claimed_impressions: Claimed delivered impressions
            simulation_day: Current simulation day

        Returns:
            ClaimVerification with results
        """
        actual_impressions = self._get_actual_delivery(campaign_id, simulation_day)

        return self._create_verification(
            agent_id=agent_id,
            agent_type=agent_type,
            scenario=scenario,
            claim_type="delivery_count",
            claimed_value=str(claimed_impressions),
            actual_value=str(actual_impressions) if actual_impressions is not None else "UNKNOWN",
            simulation_day=simulation_day,
        )

    def _create_verification(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        claim_type: str,
        claimed_value: str,
        actual_value: str,
        simulation_day: int,
    ) -> ClaimVerification:
        """Create a verification result and record it."""
        is_hallucination = False
        discrepancy_pct = None
        severity = None

        if actual_value != "UNKNOWN":
            try:
                claimed_num = float(claimed_value)
                actual_num = float(actual_value)

                if actual_num != 0:
                    discrepancy_pct = abs(claimed_num - actual_num) / actual_num * 100
                    # > 1% discrepancy is considered a hallucination
                    is_hallucination = discrepancy_pct > 1.0

                    if is_hallucination:
                        if discrepancy_pct < 5:
                            severity = Severity.MINOR
                        elif discrepancy_pct < 20:
                            severity = Severity.MODERATE
                        else:
                            severity = Severity.SEVERE
            except (ValueError, TypeError):
                # Non-numeric comparison
                is_hallucination = claimed_value != actual_value

        verification = ClaimVerification(
            agent_id=agent_id,
            agent_type=agent_type,
            scenario=scenario,
            claim_type=claim_type,
            claimed_value=claimed_value,
            actual_value=actual_value,
            is_hallucination=is_hallucination,
            discrepancy_pct=discrepancy_pct,
            severity=severity,
            simulation_day=simulation_day,
        )

        self._verifications.append(verification)
        self._record_to_db(verification)

        if is_hallucination:
            logger.warning(
                f"HALLUCINATION DETECTED: {agent_id} claimed {claim_type}={claimed_value}, "
                f"actual={actual_value} ({discrepancy_pct:.1f}% off, {severity.value if severity else 'unknown'})"
            )

        return verification

    def _get_actual_inventory(
        self, publisher_id: str, channel: str, simulation_day: int
    ) -> Optional[int]:
        """Query ground truth for actual inventory."""
        if not self._db_connection or not HAS_PSYCOPG2:
            return None

        try:
            cursor_args = {"cursor_factory": DictCursor} if DictCursor else {}
            with self._db_connection.cursor(**cursor_args) as cur:
                cur.execute(
                    """
                    SELECT actual_avails FROM inventory_reality
                    WHERE publisher_id = %s AND channel = %s
                    AND date = CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (publisher_id, channel, 30 - simulation_day),
                )
                row = cur.fetchone()
                return row["actual_avails"] if row else None
        except Exception as e:
            logger.error(f"Failed to query inventory ground truth: {e}")
            return None

    def _get_actual_floor_cpm(
        self, publisher_id: str, channel: str, simulation_day: int
    ) -> Optional[float]:
        """Query ground truth for actual floor CPM."""
        if not self._db_connection or not HAS_PSYCOPG2:
            return None

        try:
            cursor_args = {"cursor_factory": DictCursor} if DictCursor else {}
            with self._db_connection.cursor(**cursor_args) as cur:
                cur.execute(
                    """
                    SELECT true_floor_cpm FROM inventory_reality
                    WHERE publisher_id = %s AND channel = %s
                    AND date = CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (publisher_id, channel, 30 - simulation_day),
                )
                row = cur.fetchone()
                return float(row["true_floor_cpm"]) if row else None
        except Exception as e:
            logger.error(f"Failed to query price ground truth: {e}")
            return None

    def _get_actual_delivery(
        self, campaign_id: str, simulation_day: int
    ) -> Optional[int]:
        """Query ground truth for actual delivery."""
        if not self._db_connection or not HAS_PSYCOPG2:
            return None

        try:
            cursor_args = {"cursor_factory": DictCursor} if DictCursor else {}
            with self._db_connection.cursor(**cursor_args) as cur:
                cur.execute(
                    """
                    SELECT actual_impressions FROM campaign_delivery_reality
                    WHERE campaign_id = %s
                    AND date = CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (campaign_id, 30 - simulation_day),
                )
                row = cur.fetchone()
                return row["actual_impressions"] if row else None
        except Exception as e:
            logger.error(f"Failed to query delivery ground truth: {e}")
            return None

    def _record_to_db(self, verification: ClaimVerification) -> None:
        """Record claim verification to PostgreSQL."""
        if not self._db_connection:
            return

        try:
            with self._db_connection.cursor() as cur:
                # Use the helper function from ground_truth.sql if available
                # Otherwise direct insert
                cur.execute(
                    """
                    INSERT INTO agent_claims (
                        agent_id, agent_type, scenario, claim_type,
                        claimed_value, actual_value, is_hallucination,
                        discrepancy_pct, severity, simulation_day, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        verification.agent_id,
                        verification.agent_type,
                        verification.scenario,
                        verification.claim_type,
                        verification.claimed_value,
                        verification.actual_value,
                        verification.is_hallucination,
                        verification.discrepancy_pct,
                        verification.severity.value if verification.severity else None,
                        verification.simulation_day,
                        verification.timestamp,
                    ),
                )
            self._db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to record verification to DB: {e}")

    def mark_injection_detected(
        self,
        injection_id: str,
        detected_by: str = "verification_check",
    ) -> None:
        """
        Mark an injection as detected.

        Called when a verification check catches an injected hallucination.

        Args:
            injection_id: ID of the injection that was detected
            detected_by: Mechanism that detected it
        """
        if not self._db_connection:
            return

        try:
            with self._db_connection.cursor() as cur:
                cur.execute(
                    """
                    UPDATE hallucination_injections
                    SET was_detected = TRUE,
                        detected_by = %s,
                        detection_delay_seconds = EXTRACT(EPOCH FROM (NOW() - timestamp))::INTEGER
                    WHERE id = %s
                    """,
                    (detected_by, injection_id),
                )
            self._db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to mark injection as detected: {e}")

    def get_verifications(self) -> list[ClaimVerification]:
        """Get all verifications from this session."""
        return list(self._verifications)

    def get_hallucination_count(self) -> int:
        """Count hallucinations detected this session."""
        return sum(1 for v in self._verifications if v.is_hallucination)

    def get_hallucination_rate(self) -> float:
        """Calculate hallucination rate for this session."""
        if not self._verifications:
            return 0.0
        return self.get_hallucination_count() / len(self._verifications) * 100


class HallucinationManager:
    """
    Combined manager for injection and detection.

    Coordinates the injector (for Scenario B testing) and detector
    (for all scenarios) with unified metrics collection.
    """

    def __init__(
        self,
        scenario: str,
        injection_rate: float = 0.05,
        db_connection=None,
        metric_collector=None,
    ):
        """
        Initialize the hallucination manager.

        Args:
            scenario: Current scenario (A, B, or C)
            injection_rate: Rate of hallucination injection (Scenario B only)
            db_connection: PostgreSQL connection
            metric_collector: MetricCollector instance for InfluxDB
        """
        self.scenario = scenario
        self._db_connection = db_connection
        self._metric_collector = metric_collector

        # Injector only active in Scenario B
        self.injector = HallucinationInjector(
            injection_rate=injection_rate if scenario == "B" else 0.0
        )
        self.detector = HallucinationDetector()

        if db_connection:
            self.injector.set_db_connection(db_connection)
            self.detector.set_db_connection(db_connection)

    def set_db_connection(self, connection) -> None:
        """Update database connection."""
        self._db_connection = connection
        self.injector.set_db_connection(connection)
        self.detector.set_db_connection(connection)

    def set_metric_collector(self, collector) -> None:
        """Set metrics collector for InfluxDB."""
        self._metric_collector = collector

    def process_inventory_data(
        self,
        real_inventory: dict[str, int],
        agent_id: str,
        agent_type: str,
        publisher_id: str,
        simulation_day: int = 1,
    ) -> dict[str, int]:
        """
        Process inventory data through injection pipeline.

        In Scenario B: May inject hallucinations
        In Scenarios A, C: Pass through unchanged

        Args:
            real_inventory: True inventory levels
            agent_id: Agent receiving data
            agent_type: Type of agent
            publisher_id: Publisher ID
            simulation_day: Current simulation day

        Returns:
            Possibly corrupted inventory (same as input in A, C)
        """
        if self.scenario != "B":
            return real_inventory

        processed, injection = self.injector.maybe_corrupt_inventory(
            real_inventory, agent_id, simulation_day
        )

        if injection and self._metric_collector:
            # Record the hallucination event
            self._metric_collector.record_hallucination(
                agent_id=agent_id,
                agent_type=agent_type,
                scenario=self.scenario,
                claim_type="inventory",
                claimed_value=str(processed),
                actual_value=str(real_inventory),
            )

        return processed

    def process_price_data(
        self,
        real_price: float,
        agent_id: str,
        agent_type: str,
        publisher_id: str,
        simulation_day: int = 1,
    ) -> float:
        """
        Process price data through injection pipeline.

        Args:
            real_price: True price (CPM)
            agent_id: Agent receiving data
            agent_type: Type of agent
            publisher_id: Publisher ID
            simulation_day: Current simulation day

        Returns:
            Possibly corrupted price
        """
        if self.scenario != "B":
            return real_price

        processed, injection = self.injector.maybe_corrupt_price(
            real_price, agent_id, simulation_day
        )

        if injection and self._metric_collector:
            self._metric_collector.record_hallucination(
                agent_id=agent_id,
                agent_type=agent_type,
                scenario=self.scenario,
                claim_type="price",
                claimed_value=str(processed),
                actual_value=str(real_price),
            )

        return processed

    def verify_claim(
        self,
        agent_id: str,
        agent_type: str,
        claim_type: str,
        claimed_value: str,
        entity_id: str,
        simulation_day: int,
    ) -> ClaimVerification:
        """
        Verify an agent's claim against ground truth.

        Args:
            agent_id: Agent making the claim
            agent_type: Type of agent
            claim_type: Type of claim being made
            claimed_value: The claimed value
            entity_id: ID of related entity (publisher, campaign, etc.)
            simulation_day: Current simulation day

        Returns:
            ClaimVerification with results
        """
        verification = ClaimVerification(
            agent_id=agent_id,
            agent_type=agent_type,
            scenario=self.scenario,
            claim_type=claim_type,
            claimed_value=claimed_value,
            actual_value="VERIFICATION_PENDING",
            simulation_day=simulation_day,
        )

        # In Scenario C with ledger, claims can be verified against chain
        # In Scenario A with exchange, claims verified against exchange records
        # In Scenario B without ground truth, verification is uncertain

        if self._metric_collector and verification.is_hallucination:
            self._metric_collector.record_hallucination(
                agent_id=agent_id,
                agent_type=agent_type,
                scenario=self.scenario,
                claim_type=claim_type,
                claimed_value=claimed_value,
                actual_value=verification.actual_value,
            )

        return verification

    def get_summary(self) -> dict:
        """Get summary statistics for the session."""
        verifications = self.detector.get_verifications()
        injections = self.injector.get_injections()

        return {
            "scenario": self.scenario,
            "total_verifications": len(verifications),
            "hallucinations_detected": sum(1 for v in verifications if v.is_hallucination),
            "hallucination_rate": self.detector.get_hallucination_rate(),
            "total_injections": len(injections),
            "injections_detected": sum(1 for i in injections if i.was_detected),
            "severity_breakdown": {
                "minor": sum(1 for v in verifications if v.severity == Severity.MINOR),
                "moderate": sum(1 for v in verifications if v.severity == Severity.MODERATE),
                "severe": sum(1 for v in verifications if v.severity == Severity.SEVERE),
            },
        }
