"""
Event Injector for RTB Simulation.

Provides chaos testing and event injection capabilities for simulating
real-world conditions like agent failures, network issues, and market events.

Usage:
    injector = EventInjector(time_controller)

    # Inject specific events
    await injector.inject_agent_failure("buyer-001")
    await injector.inject_market_shock(0.3)  # 30% price increase

    # Enable continuous chaos
    injector.enable_chaos_mode(failure_rate=0.05)
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional, Awaitable, Any
import structlog

from .time_controller import TimeController

logger = structlog.get_logger()


class EventType(str, Enum):
    """Types of injectable events."""

    # Agent events
    AGENT_FAILURE = "agent_failure"  # Agent crashes/restarts
    AGENT_SLOWDOWN = "agent_slowdown"  # Agent becomes slow
    AGENT_MEMORY_LOSS = "agent_memory_loss"  # Context rot simulation

    # Network events
    NETWORK_PARTITION = "network_partition"  # Agents can't communicate
    NETWORK_LATENCY = "network_latency"  # High latency
    MESSAGE_LOSS = "message_loss"  # Messages dropped

    # Market events
    MARKET_SHOCK = "market_shock"  # Sudden price changes
    DEMAND_SPIKE = "demand_spike"  # Inventory shortage
    DEMAND_DROP = "demand_drop"  # Excess inventory
    BUDGET_FREEZE = "budget_freeze"  # Buyer budgets frozen

    # System events
    DATABASE_SLOW = "database_slow"  # DB latency
    EXCHANGE_DOWN = "exchange_down"  # Exchange unavailable
    LEDGER_DELAY = "ledger_delay"  # Blockchain confirmation delay


@dataclass
class InjectedEvent:
    """Record of an injected event."""

    event_type: EventType
    timestamp: datetime  # Simulation time when injected
    target: Optional[str] = None  # Target agent/component
    parameters: dict = field(default_factory=dict)
    duration: Optional[timedelta] = None  # How long event lasts
    resolved_at: Optional[datetime] = None  # When event was resolved
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.event_type.value}-{datetime.now().timestamp()}"


@dataclass
class ChaosConfig:
    """Configuration for chaos mode."""

    enabled: bool = False

    # Failure rates (probability per tick)
    agent_failure_rate: float = 0.01  # 1% chance per tick
    network_issue_rate: float = 0.005  # 0.5% chance
    market_shock_rate: float = 0.002  # 0.2% chance

    # Event parameters
    agent_failure_duration_hours: float = 1.0
    network_issue_duration_hours: float = 0.5
    market_shock_magnitude_range: tuple[float, float] = (-0.3, 0.3)

    # Scheduling
    tick_interval_sim_hours: float = 1.0  # Check every sim hour


EventCallback = Callable[[InjectedEvent], Awaitable[None]]


class EventInjector:
    """
    Injects events into the simulation for chaos testing.

    Supports:
    - Manual event injection
    - Scheduled events at specific times
    - Continuous chaos mode with random events
    - Event callbacks for handling

    Event types:
    - Agent events: failures, slowdowns, memory loss
    - Network events: partitions, latency, message loss
    - Market events: price shocks, demand changes
    - System events: database issues, exchange downtime
    """

    def __init__(
        self,
        time_controller: Optional[TimeController] = None,
    ):
        """
        Initialize event injector.

        Args:
            time_controller: Optional time controller for simulation time
        """
        self.time_controller = time_controller
        self.chaos_config = ChaosConfig()

        self._event_history: list[InjectedEvent] = []
        self._active_events: list[InjectedEvent] = []
        self._callbacks: dict[EventType, list[EventCallback]] = {}
        self._global_callbacks: list[EventCallback] = []

        self._chaos_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def current_time(self) -> datetime:
        """Get current simulation time."""
        if self.time_controller:
            return self.time_controller.current_sim_time
        return datetime.now()

    @property
    def active_events(self) -> list[InjectedEvent]:
        """Get currently active (unresolved) events."""
        return self._active_events.copy()

    @property
    def event_history(self) -> list[InjectedEvent]:
        """Get all injected events."""
        return self._event_history.copy()

    def on_event(
        self,
        event_type: EventType,
        callback: EventCallback,
    ) -> None:
        """
        Register a callback for a specific event type.

        Args:
            event_type: Type of event to handle
            callback: Async function to call when event is injected
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def on_any_event(self, callback: EventCallback) -> None:
        """
        Register a callback for any event type.

        Args:
            callback: Async function to call on any event
        """
        self._global_callbacks.append(callback)

    async def _fire_callbacks(self, event: InjectedEvent) -> None:
        """Fire all relevant callbacks for an event."""
        # Type-specific callbacks
        for callback in self._callbacks.get(event.event_type, []):
            try:
                await callback(event)
            except Exception as e:
                logger.error(
                    "event_injector.callback_error",
                    event_type=event.event_type.value,
                    error=str(e),
                )

        # Global callbacks
        for callback in self._global_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(
                    "event_injector.global_callback_error",
                    event_type=event.event_type.value,
                    error=str(e),
                )

    async def inject(
        self,
        event_type: EventType,
        target: Optional[str] = None,
        duration: Optional[timedelta] = None,
        **parameters,
    ) -> InjectedEvent:
        """
        Inject an event into the simulation.

        Args:
            event_type: Type of event to inject
            target: Optional target (agent ID, component name)
            duration: How long the event lasts (None = instant)
            **parameters: Event-specific parameters

        Returns:
            InjectedEvent record
        """
        event = InjectedEvent(
            event_type=event_type,
            timestamp=self.current_time,
            target=target,
            parameters=parameters,
            duration=duration,
        )

        self._event_history.append(event)

        if duration:
            self._active_events.append(event)

        logger.info(
            "event_injector.event_injected",
            event_type=event_type.value,
            target=target,
            duration=str(duration) if duration else "instant",
            parameters=parameters,
        )

        await self._fire_callbacks(event)

        return event

    async def resolve(self, event: InjectedEvent) -> None:
        """
        Resolve an active event.

        Args:
            event: Event to resolve
        """
        event.resolved_at = self.current_time

        if event in self._active_events:
            self._active_events.remove(event)

        logger.info(
            "event_injector.event_resolved",
            event_id=event.id,
            event_type=event.event_type.value,
            duration=str(event.resolved_at - event.timestamp),
        )

    # -------------------------------------------------------------------------
    # Convenience methods for common events
    # -------------------------------------------------------------------------

    async def inject_agent_failure(
        self,
        agent_id: str,
        duration_hours: float = 1.0,
    ) -> InjectedEvent:
        """
        Inject an agent failure (crash/restart).

        Args:
            agent_id: ID of the agent to fail
            duration_hours: How long agent is down (in sim hours)

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.AGENT_FAILURE,
            target=agent_id,
            duration=timedelta(hours=duration_hours),
            reason="simulated_crash",
        )

    async def inject_agent_slowdown(
        self,
        agent_id: str,
        slowdown_factor: float = 5.0,
        duration_hours: float = 2.0,
    ) -> InjectedEvent:
        """
        Make an agent slow (processing delays).

        Args:
            agent_id: ID of the agent
            slowdown_factor: How much slower (5.0 = 5x slower)
            duration_hours: How long the slowdown lasts

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.AGENT_SLOWDOWN,
            target=agent_id,
            duration=timedelta(hours=duration_hours),
            slowdown_factor=slowdown_factor,
        )

    async def inject_memory_loss(
        self,
        agent_id: str,
        keys_to_lose: Optional[list[str]] = None,
        loss_percentage: float = 0.2,
    ) -> InjectedEvent:
        """
        Inject memory/context loss for an agent (context rot simulation).

        Args:
            agent_id: ID of the agent
            keys_to_lose: Specific memory keys to remove
            loss_percentage: Percentage of memory to lose if keys not specified

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.AGENT_MEMORY_LOSS,
            target=agent_id,
            keys_to_lose=keys_to_lose,
            loss_percentage=loss_percentage,
        )

    async def inject_network_partition(
        self,
        agents: list[str],
        duration_hours: float = 0.5,
    ) -> InjectedEvent:
        """
        Inject a network partition (agents can't communicate).

        Args:
            agents: List of agent IDs affected
            duration_hours: How long partition lasts

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.NETWORK_PARTITION,
            duration=timedelta(hours=duration_hours),
            affected_agents=agents,
        )

    async def inject_network_latency(
        self,
        latency_ms: int = 500,
        duration_hours: float = 1.0,
        target: Optional[str] = None,
    ) -> InjectedEvent:
        """
        Inject high network latency.

        Args:
            latency_ms: Added latency in milliseconds
            duration_hours: How long latency lasts
            target: Optional specific target (agent or route)

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.NETWORK_LATENCY,
            target=target,
            duration=timedelta(hours=duration_hours),
            latency_ms=latency_ms,
        )

    async def inject_market_shock(
        self,
        price_change_pct: float,
        duration_hours: float = 4.0,
        affected_channels: Optional[list[str]] = None,
    ) -> InjectedEvent:
        """
        Inject a market shock (sudden price change).

        Args:
            price_change_pct: Price change as decimal (-0.3 = 30% decrease)
            duration_hours: How long the shock lasts
            affected_channels: Specific channels affected (None = all)

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.MARKET_SHOCK,
            duration=timedelta(hours=duration_hours),
            price_change_pct=price_change_pct,
            affected_channels=affected_channels or ["all"],
        )

    async def inject_demand_spike(
        self,
        spike_factor: float = 2.0,
        duration_hours: float = 6.0,
    ) -> InjectedEvent:
        """
        Inject a demand spike (inventory shortage).

        Args:
            spike_factor: Demand multiplier (2.0 = 2x demand)
            duration_hours: How long spike lasts

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.DEMAND_SPIKE,
            duration=timedelta(hours=duration_hours),
            spike_factor=spike_factor,
        )

    async def inject_exchange_outage(
        self,
        exchange_id: str,
        duration_hours: float = 0.5,
    ) -> InjectedEvent:
        """
        Take an exchange offline.

        Args:
            exchange_id: ID of the exchange
            duration_hours: How long outage lasts

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.EXCHANGE_DOWN,
            target=exchange_id,
            duration=timedelta(hours=duration_hours),
        )

    async def inject_ledger_delay(
        self,
        delay_seconds: int = 30,
        duration_hours: float = 1.0,
    ) -> InjectedEvent:
        """
        Inject blockchain/ledger confirmation delays.

        Args:
            delay_seconds: Confirmation delay in seconds
            duration_hours: How long the delays persist

        Returns:
            InjectedEvent
        """
        return await self.inject(
            EventType.LEDGER_DELAY,
            duration=timedelta(hours=duration_hours),
            delay_seconds=delay_seconds,
        )

    # -------------------------------------------------------------------------
    # Chaos mode
    # -------------------------------------------------------------------------

    def enable_chaos_mode(
        self,
        agent_failure_rate: float = 0.01,
        network_issue_rate: float = 0.005,
        market_shock_rate: float = 0.002,
    ) -> None:
        """
        Enable continuous chaos mode with random event injection.

        Args:
            agent_failure_rate: Probability of agent failure per tick
            network_issue_rate: Probability of network issue per tick
            market_shock_rate: Probability of market shock per tick
        """
        self.chaos_config.enabled = True
        self.chaos_config.agent_failure_rate = agent_failure_rate
        self.chaos_config.network_issue_rate = network_issue_rate
        self.chaos_config.market_shock_rate = market_shock_rate

        logger.info(
            "event_injector.chaos_mode_enabled",
            agent_failure_rate=agent_failure_rate,
            network_issue_rate=network_issue_rate,
            market_shock_rate=market_shock_rate,
        )

    def disable_chaos_mode(self) -> None:
        """Disable chaos mode."""
        self.chaos_config.enabled = False
        logger.info("event_injector.chaos_mode_disabled")

    async def _run_chaos_tick(
        self,
        available_agents: Optional[list[str]] = None,
    ) -> None:
        """Run one chaos mode tick, potentially injecting events."""
        if not self.chaos_config.enabled:
            return

        agents = available_agents or []

        # Check for agent failure
        if agents and random.random() < self.chaos_config.agent_failure_rate:
            agent = random.choice(agents)
            await self.inject_agent_failure(
                agent,
                duration_hours=self.chaos_config.agent_failure_duration_hours,
            )

        # Check for network issues
        if random.random() < self.chaos_config.network_issue_rate:
            # Randomly choose latency or partition
            if random.random() < 0.7:  # 70% latency, 30% partition
                await self.inject_network_latency(
                    latency_ms=random.randint(200, 1000),
                    duration_hours=self.chaos_config.network_issue_duration_hours,
                )
            elif agents and len(agents) >= 2:
                # Partition a subset of agents
                partition_size = random.randint(1, len(agents) // 2)
                partitioned = random.sample(agents, partition_size)
                await self.inject_network_partition(
                    partitioned,
                    duration_hours=self.chaos_config.network_issue_duration_hours,
                )

        # Check for market shock
        if random.random() < self.chaos_config.market_shock_rate:
            min_change, max_change = self.chaos_config.market_shock_magnitude_range
            change = random.uniform(min_change, max_change)
            await self.inject_market_shock(
                price_change_pct=change,
                duration_hours=random.uniform(2, 8),
            )

    async def run_chaos_loop(
        self,
        available_agents: Optional[list[str]] = None,
    ) -> None:
        """
        Run the chaos mode loop.

        Args:
            available_agents: List of agent IDs that can be targeted
        """
        self._running = True
        logger.info("event_injector.chaos_loop_started")

        try:
            while self._running:
                # Resolve expired events
                current = self.current_time
                for event in self._active_events.copy():
                    if event.duration and event.timestamp + event.duration <= current:
                        await self.resolve(event)

                # Run chaos tick
                await self._run_chaos_tick(available_agents)

                # Wait for next tick
                if self.time_controller:
                    await self.time_controller.wait_sim_hours(
                        self.chaos_config.tick_interval_sim_hours
                    )
                else:
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("event_injector.chaos_loop_cancelled")
            raise

    def start_chaos_loop(
        self,
        available_agents: Optional[list[str]] = None,
    ) -> asyncio.Task:
        """
        Start the chaos loop as a background task.

        Args:
            available_agents: List of agent IDs that can be targeted

        Returns:
            The chaos loop task
        """
        self._chaos_task = asyncio.create_task(
            self.run_chaos_loop(available_agents)
        )
        return self._chaos_task

    def stop(self) -> None:
        """Stop the event injector and chaos loop."""
        self._running = False
        if self._chaos_task and not self._chaos_task.done():
            self._chaos_task.cancel()

    def is_event_active(self, event_type: EventType, target: Optional[str] = None) -> bool:
        """
        Check if an event type is currently active.

        Args:
            event_type: Type of event to check
            target: Optional specific target to check

        Returns:
            True if event is active
        """
        for event in self._active_events:
            if event.event_type == event_type:
                if target is None or event.target == target:
                    return True
        return False

    def get_active_events_for_target(self, target: str) -> list[InjectedEvent]:
        """
        Get all active events affecting a specific target.

        Args:
            target: Target ID (agent, exchange, etc.)

        Returns:
            List of active events
        """
        return [e for e in self._active_events if e.target == target]

    def get_statistics(self) -> dict:
        """Get statistics about injected events."""
        type_counts: dict[str, int] = {}
        for event in self._event_history:
            key = event.event_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "total_events": len(self._event_history),
            "active_events": len(self._active_events),
            "events_by_type": type_counts,
            "chaos_mode_enabled": self.chaos_config.enabled,
        }
