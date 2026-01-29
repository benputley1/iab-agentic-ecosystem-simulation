"""
Base scenario interface for RTB simulation.

All scenarios (A, B, C) implement this interface, enabling consistent
simulation orchestration and metric comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional
import structlog

from ..infrastructure.redis_bus import RedisBus
from ..infrastructure.message_schemas import DealConfirmation

if TYPE_CHECKING:
    from ..infrastructure.ground_truth import GroundTruthRepository
    from ..metrics.collector import MetricCollector


logger = structlog.get_logger()


@dataclass
class ScenarioMetrics:
    """Metrics collected during scenario execution."""

    scenario_id: str
    scenario_name: str

    # Transaction metrics
    total_deals: int = 0
    total_impressions: int = 0
    total_buyer_spend: float = 0.0
    total_seller_revenue: float = 0.0
    total_exchange_fees: float = 0.0

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    simulation_days_completed: int = 0

    # Campaign metrics
    campaigns_started: int = 0
    campaigns_completed: int = 0
    goal_achievement_rate: float = 0.0

    # Ground truth tracking
    total_decisions: int = 0
    verified_decisions: int = 0
    total_claims: int = 0
    hallucinated_claims: int = 0

    # Context rot tracking (Scenario B)
    context_rot_events: int = 0
    keys_lost_total: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0

    # Tracking details
    deals: list[DealConfirmation] = field(default_factory=list)

    @property
    def duration(self) -> Optional[timedelta]:
        """Total scenario duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def intermediary_take_rate(self) -> float:
        """Percentage of spend taken by intermediaries."""
        if self.total_buyer_spend == 0:
            return 0.0
        return (self.total_exchange_fees / self.total_buyer_spend) * 100

    @property
    def average_cpm(self) -> float:
        """Average CPM across all deals."""
        if self.total_impressions == 0:
            return 0.0
        return (self.total_buyer_spend / self.total_impressions) * 1000

    def record_deal(self, deal: DealConfirmation) -> None:
        """Record a completed deal."""
        self.total_deals += 1
        self.total_impressions += deal.impressions
        self.total_buyer_spend += deal.total_cost
        self.total_seller_revenue += deal.seller_revenue
        self.total_exchange_fees += deal.exchange_fee
        self.deals.append(deal)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for reporting."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "total_deals": self.total_deals,
            "total_impressions": self.total_impressions,
            "total_buyer_spend": round(self.total_buyer_spend, 2),
            "total_seller_revenue": round(self.total_seller_revenue, 2),
            "total_exchange_fees": round(self.total_exchange_fees, 2),
            "intermediary_take_rate": round(self.intermediary_take_rate, 2),
            "average_cpm": round(self.average_cpm, 2),
            "simulation_days_completed": self.simulation_days_completed,
            "campaigns_started": self.campaigns_started,
            "campaigns_completed": self.campaigns_completed,
            "goal_achievement_rate": round(self.goal_achievement_rate, 2),
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            # Ground truth metrics
            "total_decisions": self.total_decisions,
            "verified_decisions": self.verified_decisions,
            "decision_verification_rate": (
                round(self.verified_decisions / self.total_decisions * 100, 2)
                if self.total_decisions > 0 else 0.0
            ),
            "total_claims": self.total_claims,
            "hallucinated_claims": self.hallucinated_claims,
            "hallucination_rate": (
                round(self.hallucinated_claims / self.total_claims * 100, 2)
                if self.total_claims > 0 else 0.0
            ),
            # Context rot metrics
            "context_rot_events": self.context_rot_events,
            "keys_lost_total": self.keys_lost_total,
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "recovery_success_rate": (
                round(self.recovery_successes / self.recovery_attempts * 100, 2)
                if self.recovery_attempts > 0 else 100.0
            ),
        }


@dataclass
class ScenarioConfig:
    """Configuration for running a scenario."""

    # Scenario identification
    scenario_code: str = "A"
    name: str = "Unnamed Scenario"
    description: str = ""

    # Agent counts
    num_buyers: int = 5
    num_sellers: int = 5
    campaigns_per_buyer: int = 10

    # Simulation parameters
    simulation_days: int = 30
    time_acceleration: float = 100.0  # 100x = 7.2 hours for 30 days

    # Timing
    bid_collection_timeout_ms: int = 5000  # How long to wait for bids
    auction_interval_ms: int = 1000  # How often to run auctions

    # Mock mode
    mock_llm: bool = True  # Use mock LLM to avoid API costs

    # Scenario-specific
    exchange_fee_pct: float = 0.15  # Scenario A: 15% default

    # Scenario B specific (context rot simulation)
    context_decay_rate: float = 0.05  # Daily context decay rate
    hallucination_rate: float = 0.1  # Probability of hallucination injection


class BaseScenario(ABC):
    """
    Base class for all simulation scenarios.

    Provides common infrastructure for buyer/seller/exchange coordination,
    metric collection, and simulation time management.
    """

    def __init__(
        self,
        scenario_id: str,
        scenario_name: str,
        config: Optional[ScenarioConfig] = None,
        redis_bus: Optional[RedisBus] = None,
        ground_truth_repo: Optional["GroundTruthRepository"] = None,
        metric_collector: Optional["MetricCollector"] = None,
    ):
        """
        Initialize scenario.

        Args:
            scenario_id: Unique identifier (e.g., "A", "B", "C")
            scenario_name: Human-readable name
            config: Scenario configuration
            redis_bus: Optional pre-configured Redis bus
            ground_truth_repo: Optional ground truth repository for verification
            metric_collector: Optional metrics collector for InfluxDB
        """
        self.scenario_id = scenario_id
        self.scenario_name = scenario_name
        self.config = config or ScenarioConfig()
        self._bus = redis_bus
        self._owned_bus = False

        # Ground truth and metrics infrastructure
        self._ground_truth_repo = ground_truth_repo
        self._metrics_collector = metric_collector
        self._owned_ground_truth = False

        # Current simulation state
        self._current_day: int = 0

        self.metrics = ScenarioMetrics(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
        )

        self._running = False

    @abstractmethod
    async def setup(self) -> None:
        """Set up scenario resources (agents, connections, etc.)."""
        pass

    @abstractmethod
    async def run_day(self, day: int) -> list[DealConfirmation]:
        """
        Run one simulation day.

        Args:
            day: Simulation day (1-30)

        Returns:
            List of deals made during this day
        """
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up scenario resources."""
        pass

    async def run(self, days: Optional[int] = None) -> ScenarioMetrics:
        """
        Run the full scenario simulation.

        Args:
            days: Number of days to simulate (default: config.simulation_days)

        Returns:
            Collected metrics
        """
        days = days or self.config.simulation_days
        self._running = True

        logger.info(
            "scenario.starting",
            scenario_id=self.scenario_id,
            days=days,
            buyers=self.config.num_buyers,
            sellers=self.config.num_sellers,
        )

        self.metrics.start_time = datetime.utcnow()

        try:
            await self.setup()

            for day in range(1, days + 1):
                if not self._running:
                    break

                logger.info(
                    "scenario.day_starting",
                    scenario_id=self.scenario_id,
                    day=day,
                )

                deals = await self.run_day(day)

                for deal in deals:
                    self.metrics.record_deal(deal)

                self.metrics.simulation_days_completed = day

                logger.info(
                    "scenario.day_completed",
                    scenario_id=self.scenario_id,
                    day=day,
                    deals_today=len(deals),
                    total_deals=self.metrics.total_deals,
                )

        finally:
            await self.teardown()
            self.metrics.end_time = datetime.utcnow()
            self._running = False

        logger.info(
            "scenario.completed",
            scenario_id=self.scenario_id,
            **self.metrics.to_dict(),
        )

        return self.metrics

    def stop(self) -> None:
        """Signal scenario to stop."""
        self._running = False

    async def connect_bus(self) -> RedisBus:
        """Connect to Redis bus if not already connected."""
        if self._bus is None:
            from ..infrastructure.redis_bus import create_redis_bus

            self._bus = await create_redis_bus(
                consumer_id=f"scenario-{self.scenario_id}"
            )
            self._owned_bus = True
        return self._bus

    async def disconnect_bus(self) -> None:
        """Disconnect from Redis bus if we own it."""
        if self._bus and self._owned_bus:
            await self._bus.disconnect()
            self._bus = None

    # -------------------------------------------------------------------------
    # Ground Truth Infrastructure
    # -------------------------------------------------------------------------

    @property
    def current_day(self) -> int:
        """Get current simulation day."""
        return self._current_day

    @current_day.setter
    def current_day(self, value: int) -> None:
        """Set current simulation day."""
        self._current_day = value

    async def connect_ground_truth(self) -> "GroundTruthRepository":
        """Connect to ground truth repository if not already connected."""
        if self._ground_truth_repo is None:
            from ..infrastructure.ground_truth import create_ground_truth_repo

            self._ground_truth_repo = await create_ground_truth_repo()
            self._owned_ground_truth = True
            logger.info("scenario.ground_truth_connected", scenario_id=self.scenario_id)
        return self._ground_truth_repo

    async def disconnect_ground_truth(self) -> None:
        """Disconnect from ground truth repository if we own it."""
        if self._ground_truth_repo and self._owned_ground_truth:
            await self._ground_truth_repo.disconnect()
            self._ground_truth_repo = None

    def record_deal(self, deal: DealConfirmation) -> None:
        """Record a completed deal to metrics."""
        self.metrics.record_deal(deal)

    async def record_decision(
        self,
        verified: bool,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        decision_type: str = "bid",
        decision_input: Optional[dict] = None,
        decision_output: Optional[dict] = None,
        decision_reasoning: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record an agent decision for post-hoc verification.

        Args:
            verified: Whether the decision basis was verified as correct
            agent_id: Agent making the decision (optional)
            agent_type: Type of agent (optional)
            decision_type: Type of decision (bid, accept, reject, counter, allocate)
            decision_input: Input data the agent used
            decision_output: The decision made
            decision_reasoning: Agent's stated reasoning

        Returns:
            Decision ID if recorded to ground truth, None otherwise
        """
        # Update metrics
        self.metrics.total_decisions += 1
        if verified:
            self.metrics.verified_decisions += 1

        # Record to ground truth if connected
        if self._ground_truth_repo and agent_id:
            try:
                decision_id = await self._ground_truth_repo.record_decision(
                    agent_id=agent_id,
                    agent_type=agent_type or "unknown",
                    scenario=self.scenario_id,
                    decision_type=decision_type,
                    decision_input=decision_input or {},
                    decision_output=decision_output or {},
                    simulation_day=self._current_day,
                    decision_reasoning=decision_reasoning,
                    decision_basis_verified=verified,
                )
                return decision_id
            except Exception as e:
                logger.error(
                    "scenario.record_decision_failed",
                    error=str(e),
                    agent_id=agent_id,
                )

        return None

    async def record_context_rot(
        self,
        agent_id: str,
        keys_lost: int,
        is_decay: bool,
        agent_type: str = "unknown",
        keys_lost_names: Optional[list[str]] = None,
        recovery_attempted: bool = False,
        recovery_successful: Optional[bool] = None,
        recovery_accuracy: Optional[float] = None,
        recovery_source: Optional[str] = None,
    ) -> Optional[int]:
        """
        Record a context rot event.

        Args:
            agent_id: Agent that lost context
            keys_lost: Number of memory keys lost
            is_decay: True if gradual decay, False if restart/wipe
            agent_type: Type of agent
            keys_lost_names: List of specific keys lost
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery succeeded
            recovery_accuracy: How accurate the recovery was (0-1)
            recovery_source: Source of recovery (ledger, checkpoint, peer, none)

        Returns:
            Event ID if recorded to ground truth, None otherwise
        """
        # Update metrics
        self.metrics.context_rot_events += 1
        self.metrics.keys_lost_total += keys_lost

        if recovery_attempted:
            self.metrics.recovery_attempts += 1
            if recovery_successful:
                self.metrics.recovery_successes += 1

        # Record to ground truth if connected
        if self._ground_truth_repo:
            try:
                from ..infrastructure.ground_truth import ContextRotEventType

                event_type = ContextRotEventType.DECAY if is_decay else ContextRotEventType.RESTART

                event_id = await self._ground_truth_repo.record_context_rot(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    scenario=self.scenario_id,
                    event_type=event_type,
                    simulation_day=self._current_day,
                    keys_lost=keys_lost_names or [],
                    recovery_attempted=recovery_attempted,
                    recovery_successful=recovery_successful,
                    recovery_accuracy=recovery_accuracy,
                    recovery_source=recovery_source,
                )
                return event_id
            except Exception as e:
                logger.error(
                    "scenario.record_context_rot_failed",
                    error=str(e),
                    agent_id=agent_id,
                )

        return None

    async def record_claim(
        self,
        agent_id: str,
        agent_type: str,
        claim_type: str,
        claimed_value: str,
        entity_id: Optional[str] = None,
        claim_context: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Record an agent claim for verification.

        Args:
            agent_id: Agent making the claim
            agent_type: Type of agent
            claim_type: Type of claim
            claimed_value: The claimed value
            entity_id: Related entity ID for verification
            claim_context: Additional context

        Returns:
            Claim ID if recorded, None otherwise
        """
        self.metrics.total_claims += 1

        if self._ground_truth_repo:
            try:
                claim_id = await self._ground_truth_repo.record_claim(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    scenario=self.scenario_id,
                    claim_type=claim_type,
                    claimed_value=claimed_value,
                    simulation_day=self._current_day,
                    entity_id=entity_id,
                    claim_context=claim_context,
                )

                # Check if it was a hallucination by querying back
                # (The record_claim function auto-verifies)
                if entity_id:
                    verification = await self._ground_truth_repo.verify_claim(
                        claim_type, entity_id, claimed_value, self._current_day
                    )
                    if not verification.is_valid:
                        self.metrics.hallucinated_claims += 1

                return claim_id
            except Exception as e:
                logger.error(
                    "scenario.record_claim_failed",
                    error=str(e),
                    agent_id=agent_id,
                )

        return None

    def get_summary(self) -> dict:
        """
        Get summary statistics for the scenario.

        Returns:
            Dict with metrics and ground truth statistics
        """
        summary = self.metrics.to_dict()

        # Add live statistics if available
        summary["running"] = self._running
        summary["current_day"] = self._current_day
        summary["has_ground_truth"] = self._ground_truth_repo is not None
        summary["has_metrics_collector"] = self._metrics_collector is not None

        return summary

    async def get_ground_truth_summary(self) -> Optional[dict]:
        """
        Get summary from ground truth repository.

        Returns:
            Ground truth summary dict, or None if not connected
        """
        if self._ground_truth_repo:
            return await self._ground_truth_repo.get_summary(scenario=self.scenario_id)
        return None
