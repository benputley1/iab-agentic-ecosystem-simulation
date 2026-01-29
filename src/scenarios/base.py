"""
Base scenario interface for RTB simulation.

All scenarios (A, B, C) implement this interface, enabling consistent
simulation orchestration and metric comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import structlog

from ..infrastructure.redis_bus import RedisBus
from ..infrastructure.message_schemas import DealConfirmation


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
        }


@dataclass
class ScenarioConfig:
    """Configuration for running a scenario."""

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
    ):
        """
        Initialize scenario.

        Args:
            scenario_id: Unique identifier (e.g., "A", "B", "C")
            scenario_name: Human-readable name
            config: Scenario configuration
            redis_bus: Optional pre-configured Redis bus
        """
        self.scenario_id = scenario_id
        self.scenario_name = scenario_name
        self.config = config or ScenarioConfig()
        self._bus = redis_bus
        self._owned_bus = False

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
