"""
Simulation Runner for RTB Simulation.

Orchestrates the entire simulation across multiple scenarios with:
- Time acceleration
- Event injection (chaos testing)
- Metric collection
- Checkpoint/resume capability

Usage:
    runner = SimulationRunner(
        scenarios=["a", "b", "c"],
        days=30,
        buyers=5,
        sellers=5,
        time_acceleration=100.0,
    )
    await runner.run()
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Awaitable
import structlog

from .time_controller import TimeController, TimeControllerState
from .event_injector import EventInjector, EventType, InjectedEvent

logger = structlog.get_logger()


class SimulationState(str, Enum):
    """State of the simulation."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""

    # Scenarios to run
    scenarios: list[str] = field(default_factory=lambda: ["a", "b", "c"])

    # Agent configuration
    num_buyers: int = 5
    num_sellers: int = 5
    campaigns_per_buyer: int = 10

    # Simulation parameters
    simulation_days: int = 30
    time_acceleration: float = 100.0

    # Chaos testing
    enable_chaos: bool = False
    chaos_agent_failure_rate: float = 0.01
    chaos_network_issue_rate: float = 0.005
    chaos_market_shock_rate: float = 0.002

    # Checkpointing
    checkpoint_interval_days: int = 1
    checkpoint_dir: str = "checkpoints"

    # LLM configuration
    mock_llm: bool = True  # Use mock LLM by default to avoid costs

    # Logging
    log_level: str = "INFO"
    log_events: bool = True
    log_narratives: bool = True


@dataclass
class ScenarioResult:
    """Result from running a single scenario."""

    scenario_id: str
    scenario_name: str
    status: str
    metrics: dict = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class SimulationResult:
    """Result from the full simulation run."""

    config: SimulationConfig
    state: SimulationState
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_events_injected: int = 0
    checkpoints_created: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state.value,
            "scenarios": [r.scenario_id for r in self.scenario_results],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_events_injected": self.total_events_injected,
            "checkpoints_created": self.checkpoints_created,
            "error": self.error,
            "scenario_results": [
                {
                    "scenario_id": r.scenario_id,
                    "scenario_name": r.scenario_name,
                    "status": r.status,
                    "metrics": r.metrics,
                }
                for r in self.scenario_results
            ],
        }


class SimulationRunner:
    """
    Main orchestrator for RTB simulation.

    Coordinates:
    - Multiple scenario engines (A, B, C)
    - Time acceleration
    - Event/chaos injection
    - Checkpoint/resume
    - Metric collection

    Example:
        runner = SimulationRunner(
            scenarios=["a", "b", "c"],
            days=30,
            buyers=5,
            sellers=5,
            time_acceleration=100.0,
        )

        # Run full simulation
        result = await runner.run()

        # Or run step by step
        await runner.initialize()
        while runner.current_day <= 30:
            await runner.run_day()
        result = await runner.finalize()
    """

    def __init__(
        self,
        scenarios: Optional[list[str]] = None,
        days: int = 30,
        buyers: int = 5,
        sellers: int = 5,
        campaigns_per_buyer: int = 10,
        time_acceleration: float = 100.0,
        mock_llm: bool = True,
        enable_chaos: bool = False,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Initialize simulation runner.

        Args:
            scenarios: List of scenario codes to run (a, b, c)
            days: Number of simulation days
            buyers: Number of buyer agents
            sellers: Number of seller agents
            campaigns_per_buyer: Campaigns per buyer
            time_acceleration: Time multiplier (100 = 100x speed)
            mock_llm: Use mock LLM (no API costs)
            enable_chaos: Enable chaos testing
            checkpoint_dir: Directory for checkpoints
        """
        self.config = SimulationConfig(
            scenarios=[s.lower() for s in (scenarios or ["a", "b", "c"])],
            num_buyers=buyers,
            num_sellers=sellers,
            campaigns_per_buyer=campaigns_per_buyer,
            simulation_days=days,
            time_acceleration=time_acceleration,
            mock_llm=mock_llm,
            enable_chaos=enable_chaos,
            checkpoint_dir=checkpoint_dir,
        )

        # Core components
        self.time_controller = TimeController(acceleration=time_acceleration)
        self.event_injector = EventInjector(time_controller=self.time_controller)

        # State
        self._state = SimulationState.INITIALIZING
        self._current_day = 0
        self._scenarios_cache: dict = {}
        self._scenario_results: list[ScenarioResult] = []

        # Callbacks
        self._day_start_callbacks: list[Callable[[int], Awaitable[None]]] = []
        self._day_end_callbacks: list[Callable[[int], Awaitable[None]]] = []
        self._event_callbacks: list[Callable[[InjectedEvent], Awaitable[None]]] = []

        # Tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._chaos_task: Optional[asyncio.Task] = None

        # Checkpointing
        self._checkpoint_dir = Path(checkpoint_dir)
        self._last_checkpoint_day = 0

        # Result tracking
        self._result: Optional[SimulationResult] = None

    @property
    def state(self) -> SimulationState:
        """Get current simulation state."""
        return self._state

    @property
    def current_day(self) -> int:
        """Get current simulation day."""
        return self.time_controller.sim_day

    @property
    def current_sim_time(self) -> datetime:
        """Get current simulation time."""
        return self.time_controller.current_sim_time

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._state == SimulationState.RUNNING

    def on_day_start(self, callback: Callable[[int], Awaitable[None]]) -> None:
        """Register callback for day start events."""
        self._day_start_callbacks.append(callback)

    def on_day_end(self, callback: Callable[[int], Awaitable[None]]) -> None:
        """Register callback for day end events."""
        self._day_end_callbacks.append(callback)

    def on_event(self, callback: Callable[[InjectedEvent], Awaitable[None]]) -> None:
        """Register callback for injected events."""
        self._event_callbacks.append(callback)
        self.event_injector.on_any_event(callback)

    async def initialize(self) -> None:
        """
        Initialize the simulation.

        Sets up:
        - Scenario engines
        - Time controller
        - Event injector
        - Checkpoint directory
        """
        logger.info(
            "simulation.initializing",
            scenarios=self.config.scenarios,
            days=self.config.simulation_days,
            buyers=self.config.num_buyers,
            sellers=self.config.num_sellers,
            acceleration=self.config.time_acceleration,
        )

        # Create checkpoint directory
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scenarios
        await self._initialize_scenarios()

        # Start time controller
        self.time_controller.start()

        # Start scheduler for timed events
        self._scheduler_task = self.time_controller.start_scheduler()

        # Setup chaos mode if enabled
        if self.config.enable_chaos:
            self.event_injector.enable_chaos_mode(
                agent_failure_rate=self.config.chaos_agent_failure_rate,
                network_issue_rate=self.config.chaos_network_issue_rate,
                market_shock_rate=self.config.chaos_market_shock_rate,
            )

            # Get list of agent IDs for chaos targeting
            agent_ids = self._get_all_agent_ids()
            self._chaos_task = self.event_injector.start_chaos_loop(agent_ids)

        self._state = SimulationState.RUNNING

        logger.info(
            "simulation.initialized",
            real_duration=str(
                self.time_controller.get_real_duration_for_sim_days(
                    self.config.simulation_days
                )
            ),
        )

    async def _initialize_scenarios(self) -> None:
        """Initialize scenario engines."""
        from ..scenarios.base import ScenarioConfig

        base_config = ScenarioConfig(
            num_buyers=self.config.num_buyers,
            num_sellers=self.config.num_sellers,
            campaigns_per_buyer=self.config.campaigns_per_buyer,
            simulation_days=self.config.simulation_days,
            time_acceleration=self.config.time_acceleration,
            mock_llm=self.config.mock_llm,
        )

        for scenario_code in self.config.scenarios:
            scenario_code = scenario_code.lower()

            if scenario_code == "a":
                from ..scenarios.scenario_a import ScenarioA

                self._scenarios_cache["a"] = ScenarioA(config=base_config)

            elif scenario_code == "b":
                from ..scenarios.scenario_b import ScenarioB

                self._scenarios_cache["b"] = ScenarioB(config=base_config)

            elif scenario_code == "c":
                # Import ScenarioC if it exists, otherwise create placeholder
                try:
                    from ..scenarios.scenario_c import ScenarioC

                    self._scenarios_cache["c"] = ScenarioC(config=base_config)
                except ImportError:
                    logger.warning(
                        "simulation.scenario_c_not_implemented",
                        message="Scenario C not yet implemented, skipping",
                    )

    def _get_all_agent_ids(self) -> list[str]:
        """Get list of all agent IDs for chaos testing."""
        agent_ids = []

        for i in range(self.config.num_buyers):
            agent_ids.append(f"buyer-{i+1:03d}")

        for i in range(self.config.num_sellers):
            agent_ids.append(f"seller-{i+1:03d}")

        # Add exchange agent for Scenario A
        if "a" in self.config.scenarios:
            agent_ids.append("exchange-001")

        return agent_ids

    async def run_day(self, day: Optional[int] = None) -> dict:
        """
        Run a single simulation day across all scenarios.

        Args:
            day: Specific day to run (default: current day)

        Returns:
            Dict with day results from each scenario
        """
        day = day or self.current_day

        logger.info("simulation.day_starting", day=day)

        # Fire day start callbacks
        for callback in self._day_start_callbacks:
            try:
                await callback(day)
            except Exception as e:
                logger.error("simulation.day_start_callback_error", error=str(e))

        results = {}

        # Run each scenario's day
        for scenario_code, scenario in self._scenarios_cache.items():
            try:
                scenario.current_day = day
                deals = await scenario.run_day(day)

                results[scenario_code] = {
                    "deals": len(deals),
                    "metrics": scenario.metrics.to_dict(),
                }

                logger.info(
                    "simulation.scenario_day_completed",
                    scenario=scenario_code,
                    day=day,
                    deals=len(deals),
                )

            except Exception as e:
                logger.error(
                    "simulation.scenario_day_error",
                    scenario=scenario_code,
                    day=day,
                    error=str(e),
                )
                results[scenario_code] = {"error": str(e)}

        # Fire day end callbacks
        for callback in self._day_end_callbacks:
            try:
                await callback(day)
            except Exception as e:
                logger.error("simulation.day_end_callback_error", error=str(e))

        # Checkpoint if needed
        if day - self._last_checkpoint_day >= self.config.checkpoint_interval_days:
            await self._create_checkpoint(day)
            self._last_checkpoint_day = day

        logger.info("simulation.day_completed", day=day, results=results)

        return results

    async def run(self) -> SimulationResult:
        """
        Run the full simulation.

        Returns:
            SimulationResult with metrics from all scenarios
        """
        self._result = SimulationResult(
            config=self.config,
            state=SimulationState.INITIALIZING,
            start_time=datetime.utcnow(),
        )

        try:
            await self.initialize()

            self._result.state = SimulationState.RUNNING

            # Setup all scenarios
            for scenario_code, scenario in self._scenarios_cache.items():
                await scenario.setup()

            # Run each day
            for day in range(1, self.config.simulation_days + 1):
                if self._state != SimulationState.RUNNING:
                    break

                await self.run_day(day)

                # Wait for simulation time to reach end of day
                await self.time_controller.wait_until_day_end(day)

            # Finalize
            await self.finalize()

            self._result.state = SimulationState.COMPLETED

        except Exception as e:
            logger.error("simulation.run_error", error=str(e))
            self._result.state = SimulationState.FAILED
            self._result.error = str(e)

        self._result.end_time = datetime.utcnow()
        self._result.total_events_injected = len(self.event_injector.event_history)

        logger.info(
            "simulation.completed",
            state=self._result.state.value,
            scenarios=len(self._result.scenario_results),
            total_events=self._result.total_events_injected,
        )

        return self._result

    async def finalize(self) -> SimulationResult:
        """
        Finalize the simulation and collect results.

        Returns:
            SimulationResult
        """
        logger.info("simulation.finalizing")

        # Stop chaos mode
        if self._chaos_task:
            self._chaos_task.cancel()
            try:
                await self._chaos_task
            except asyncio.CancelledError:
                pass

        # Stop scheduler
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Stop time controller
        self.time_controller.stop()

        # Teardown scenarios and collect results
        for scenario_code, scenario in self._scenarios_cache.items():
            try:
                await scenario.teardown()

                result = ScenarioResult(
                    scenario_id=scenario.scenario_id,
                    scenario_name=scenario.scenario_name,
                    status="completed",
                    metrics=scenario.metrics.to_dict(),
                    start_time=scenario.metrics.start_time,
                    end_time=scenario.metrics.end_time,
                )
                self._scenario_results.append(result)

            except Exception as e:
                logger.error(
                    "simulation.scenario_teardown_error",
                    scenario=scenario_code,
                    error=str(e),
                )
                self._scenario_results.append(
                    ScenarioResult(
                        scenario_id=scenario_code,
                        scenario_name=scenario.scenario_name,
                        status="error",
                        error=str(e),
                    )
                )

        if self._result:
            self._result.scenario_results = self._scenario_results

        self._state = SimulationState.COMPLETED

        logger.info("simulation.finalized")

        return self._result or SimulationResult(
            config=self.config,
            state=SimulationState.COMPLETED,
            scenario_results=self._scenario_results,
        )

    def pause(self) -> None:
        """Pause the simulation."""
        if self._state != SimulationState.RUNNING:
            return

        self.time_controller.pause()
        self._state = SimulationState.PAUSED

        logger.info("simulation.paused", day=self.current_day)

    def resume(self) -> None:
        """Resume a paused simulation."""
        if self._state != SimulationState.PAUSED:
            return

        self.time_controller.resume()
        self._state = SimulationState.RUNNING

        logger.info("simulation.resumed", day=self.current_day)

    async def stop(self) -> SimulationResult:
        """
        Stop the simulation early.

        Returns:
            SimulationResult with current progress
        """
        self._state = SimulationState.COMPLETED
        return await self.finalize()

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    async def _create_checkpoint(self, day: int) -> str:
        """
        Create a checkpoint of current state.

        Args:
            day: Current simulation day

        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"checkpoint_day{day:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"

        state = {
            "checkpoint_id": checkpoint_id,
            "created_at": datetime.utcnow().isoformat(),
            "simulation_day": day,
            "config": {
                "scenarios": self.config.scenarios,
                "num_buyers": self.config.num_buyers,
                "num_sellers": self.config.num_sellers,
                "campaigns_per_buyer": self.config.campaigns_per_buyer,
                "simulation_days": self.config.simulation_days,
                "time_acceleration": self.config.time_acceleration,
            },
            "scenario_metrics": {
                code: scenario.metrics.to_dict()
                for code, scenario in self._scenarios_cache.items()
            },
            "event_statistics": self.event_injector.get_statistics(),
        }

        with open(checkpoint_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        if self._result:
            self._result.checkpoints_created += 1

        logger.info(
            "simulation.checkpoint_created",
            checkpoint_id=checkpoint_id,
            day=day,
        )

        # Cleanup old checkpoints (keep last 5)
        checkpoints = sorted(self._checkpoint_dir.glob("checkpoint_*.json"))
        for old in checkpoints[:-5]:
            old.unlink()

        return checkpoint_id

    @classmethod
    async def resume_from_checkpoint(cls, checkpoint_path: str) -> "SimulationRunner":
        """
        Resume simulation from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            SimulationRunner configured to resume
        """
        with open(checkpoint_path, "r") as f:
            state = json.load(f)

        config = state["config"]

        runner = cls(
            scenarios=config["scenarios"],
            days=config["simulation_days"],
            buyers=config["num_buyers"],
            sellers=config["num_sellers"],
            campaigns_per_buyer=config["campaigns_per_buyer"],
            time_acceleration=config["time_acceleration"],
        )

        # Initialize but skip to checkpoint day
        await runner.initialize()

        # Fast-forward time controller to checkpoint
        start_day = state["simulation_day"]

        logger.info(
            "simulation.resuming_from_checkpoint",
            checkpoint_id=state["checkpoint_id"],
            start_day=start_day,
        )

        return runner

    # -------------------------------------------------------------------------
    # Event injection helpers
    # -------------------------------------------------------------------------

    async def inject_event(
        self,
        event_type: EventType,
        target: Optional[str] = None,
        **parameters,
    ) -> InjectedEvent:
        """
        Inject an event into the simulation.

        Args:
            event_type: Type of event
            target: Optional target agent/component
            **parameters: Event-specific parameters

        Returns:
            InjectedEvent
        """
        return await self.event_injector.inject(
            event_type=event_type,
            target=target,
            **parameters,
        )

    async def inject_agent_failure(
        self, agent_id: str, duration_hours: float = 1.0
    ) -> InjectedEvent:
        """Convenience: inject agent failure."""
        return await self.event_injector.inject_agent_failure(
            agent_id, duration_hours
        )

    async def inject_market_shock(
        self, price_change_pct: float, duration_hours: float = 4.0
    ) -> InjectedEvent:
        """Convenience: inject market shock."""
        return await self.event_injector.inject_market_shock(
            price_change_pct, duration_hours
        )

    # -------------------------------------------------------------------------
    # Status and reporting
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current simulation status."""
        return {
            "state": self._state.value,
            "current_day": self.current_day,
            "simulation_time": self.current_sim_time.isoformat(),
            "time_controller": self.time_controller.get_status(),
            "events": self.event_injector.get_statistics(),
            "scenarios": {
                code: scenario.metrics.to_dict()
                for code, scenario in self._scenarios_cache.items()
            },
        }

    def get_comparison_report(self) -> dict:
        """
        Generate a comparison report across scenarios.

        Returns:
            Dict comparing key metrics across scenarios
        """
        report = {
            "simulation_days": self.config.simulation_days,
            "comparison": {},
        }

        metrics_to_compare = [
            "total_deals",
            "total_impressions",
            "total_buyer_spend",
            "total_seller_revenue",
            "total_exchange_fees",
            "intermediary_take_rate",
            "average_cpm",
            "goal_achievement_rate",
            "hallucination_rate",
        ]

        for metric in metrics_to_compare:
            report["comparison"][metric] = {}
            for code, scenario in self._scenarios_cache.items():
                scenario_metrics = scenario.metrics.to_dict()
                report["comparison"][metric][code] = scenario_metrics.get(metric, 0)

        return report
