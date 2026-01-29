"""
Time Controller for RTB Simulation.

Controls simulated time acceleration, enabling 30 simulated days
to run in a compressed real-time window.

100x acceleration = 1 real second = 100 simulated seconds
30 simulated days at 100x = 7.2 real hours
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional, Awaitable
import structlog

logger = structlog.get_logger()


class TimeControllerState(str, Enum):
    """State of the time controller."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class ScheduledEvent:
    """An event scheduled to run at a specific simulation time."""

    target_time: datetime
    callback: Callable[[], Awaitable[None]]
    name: str = ""
    executed: bool = False


@dataclass
class TimeControllerConfig:
    """Configuration for time controller."""

    acceleration: float = 100.0  # Time multiplier
    simulation_epoch: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1)
    )  # Start of simulation time
    tick_interval_real_seconds: float = 0.1  # How often to check scheduled events


class TimeController:
    """
    Controls simulated time acceleration.

    Provides:
    - Time acceleration (e.g., 100x means 1 real second = 100 sim seconds)
    - Pause/resume capability
    - Scheduled events at simulation times
    - Waiting for simulation time to pass

    Example:
        controller = TimeController(acceleration=100.0)
        controller.start()

        # Wait for 1 simulation hour (= 36 real seconds at 100x)
        await controller.wait_sim_hours(1)

        # Get current simulation time
        print(f"Simulation time: {controller.current_sim_time}")

        # Schedule event for day 5
        controller.schedule_at_day(5, my_callback)
    """

    def __init__(
        self,
        acceleration: float = 100.0,
        simulation_epoch: Optional[datetime] = None,
    ):
        """
        Initialize time controller.

        Args:
            acceleration: Time multiplier (100.0 = 100x speed)
            simulation_epoch: Start of simulation time (default: 2025-01-01)
        """
        self.config = TimeControllerConfig(
            acceleration=acceleration,
            simulation_epoch=simulation_epoch or datetime(2025, 1, 1),
        )

        self._state = TimeControllerState.STOPPED
        self._start_real_time: Optional[datetime] = None
        self._pause_real_time: Optional[datetime] = None
        self._total_pause_duration: timedelta = timedelta()

        self._scheduled_events: list[ScheduledEvent] = []
        self._scheduler_task: Optional[asyncio.Task] = None

    @property
    def acceleration(self) -> float:
        """Get current time acceleration factor."""
        return self.config.acceleration

    @acceleration.setter
    def acceleration(self, value: float) -> None:
        """Set time acceleration factor."""
        if value <= 0:
            raise ValueError("Acceleration must be positive")
        self.config.acceleration = value

    @property
    def state(self) -> TimeControllerState:
        """Get current state of the time controller."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if time controller is running (not paused or stopped)."""
        return self._state == TimeControllerState.RUNNING

    @property
    def current_sim_time(self) -> datetime:
        """
        Get current simulated time.

        Takes into account:
        - Time since start
        - Acceleration factor
        - Any pause duration
        """
        if self._start_real_time is None:
            return self.config.simulation_epoch

        now = datetime.now()

        # Account for current pause
        if self._state == TimeControllerState.PAUSED and self._pause_real_time:
            now = self._pause_real_time

        # Calculate real elapsed time minus pauses
        real_elapsed = now - self._start_real_time - self._total_pause_duration
        real_seconds = max(0, real_elapsed.total_seconds())

        # Apply acceleration
        sim_seconds = real_seconds * self.config.acceleration

        return self.config.simulation_epoch + timedelta(seconds=sim_seconds)

    @property
    def sim_day(self) -> int:
        """
        Get current simulation day (1-indexed).

        Day 1 = epoch to epoch + 24 hours
        """
        elapsed = self.current_sim_time - self.config.simulation_epoch
        return max(1, elapsed.days + 1)

    @property
    def sim_hour(self) -> int:
        """Get current hour within the simulation day (0-23)."""
        return self.current_sim_time.hour

    def start(self) -> None:
        """Start the time controller."""
        if self._state == TimeControllerState.RUNNING:
            return

        if self._state == TimeControllerState.PAUSED:
            # Resuming from pause
            if self._pause_real_time:
                pause_duration = datetime.now() - self._pause_real_time
                self._total_pause_duration += pause_duration
            self._pause_real_time = None
        else:
            # Fresh start
            self._start_real_time = datetime.now()
            self._total_pause_duration = timedelta()
            self._pause_real_time = None

        self._state = TimeControllerState.RUNNING

        logger.info(
            "time_controller.started",
            acceleration=self.config.acceleration,
            sim_time=self.current_sim_time.isoformat(),
        )

    def pause(self) -> None:
        """Pause the time controller."""
        if self._state != TimeControllerState.RUNNING:
            return

        self._pause_real_time = datetime.now()
        self._state = TimeControllerState.PAUSED

        logger.info(
            "time_controller.paused",
            sim_time=self.current_sim_time.isoformat(),
            sim_day=self.sim_day,
        )

    def resume(self) -> None:
        """Resume from paused state."""
        self.start()

    def stop(self) -> None:
        """Stop the time controller."""
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()

        self._state = TimeControllerState.STOPPED

        logger.info(
            "time_controller.stopped",
            sim_time=self.current_sim_time.isoformat(),
            sim_day=self.sim_day,
        )

    def reset(self) -> None:
        """Reset the time controller to initial state."""
        self.stop()
        self._start_real_time = None
        self._pause_real_time = None
        self._total_pause_duration = timedelta()
        self._scheduled_events.clear()

    async def wait_sim_seconds(self, sim_seconds: float) -> None:
        """
        Wait for specified simulation seconds to pass.

        Args:
            sim_seconds: Number of simulation seconds to wait
        """
        if sim_seconds <= 0:
            return

        real_seconds = sim_seconds / self.config.acceleration
        await asyncio.sleep(real_seconds)

    async def wait_sim_minutes(self, sim_minutes: float) -> None:
        """Wait for specified simulation minutes to pass."""
        await self.wait_sim_seconds(sim_minutes * 60)

    async def wait_sim_hours(self, sim_hours: float) -> None:
        """Wait for specified simulation hours to pass."""
        await self.wait_sim_seconds(sim_hours * 3600)

    async def wait_until_sim_time(self, target: datetime) -> None:
        """
        Wait until simulation reaches specified time.

        Args:
            target: Target simulation datetime
        """
        while self.current_sim_time < target:
            # Check every tick interval
            await asyncio.sleep(self.config.tick_interval_real_seconds)

            # Handle pause state
            while self._state == TimeControllerState.PAUSED:
                await asyncio.sleep(self.config.tick_interval_real_seconds)

            if self._state == TimeControllerState.STOPPED:
                break

    async def wait_until_day(self, day: int) -> None:
        """
        Wait until simulation reaches start of specified day.

        Args:
            day: Target simulation day (1-indexed)
        """
        target = self.config.simulation_epoch + timedelta(days=day - 1)
        await self.wait_until_sim_time(target)

    async def wait_until_day_end(self, day: int) -> None:
        """
        Wait until end of specified simulation day.

        Args:
            day: Target simulation day (1-indexed)
        """
        # End of day = start of next day
        target = self.config.simulation_epoch + timedelta(days=day)
        await self.wait_until_sim_time(target)

    def schedule_at_sim_time(
        self,
        target: datetime,
        callback: Callable[[], Awaitable[None]],
        name: str = "",
    ) -> ScheduledEvent:
        """
        Schedule an async callback to run at a specific simulation time.

        Args:
            target: Target simulation datetime
            callback: Async function to call
            name: Optional name for logging

        Returns:
            ScheduledEvent that can be cancelled
        """
        event = ScheduledEvent(
            target_time=target,
            callback=callback,
            name=name,
        )
        self._scheduled_events.append(event)

        logger.debug(
            "time_controller.event_scheduled",
            target=target.isoformat(),
            name=name,
        )

        return event

    def schedule_at_day(
        self,
        day: int,
        callback: Callable[[], Awaitable[None]],
        name: str = "",
    ) -> ScheduledEvent:
        """
        Schedule callback at start of specified simulation day.

        Args:
            day: Target simulation day (1-indexed)
            callback: Async function to call
            name: Optional name for logging

        Returns:
            ScheduledEvent
        """
        target = self.config.simulation_epoch + timedelta(days=day - 1)
        return self.schedule_at_sim_time(target, callback, name or f"day_{day}_start")

    def schedule_at_day_hour(
        self,
        day: int,
        hour: int,
        callback: Callable[[], Awaitable[None]],
        name: str = "",
    ) -> ScheduledEvent:
        """
        Schedule callback at specific hour of a simulation day.

        Args:
            day: Target simulation day (1-indexed)
            hour: Hour of day (0-23)
            callback: Async function to call
            name: Optional name for logging

        Returns:
            ScheduledEvent
        """
        target = self.config.simulation_epoch + timedelta(days=day - 1, hours=hour)
        return self.schedule_at_sim_time(target, callback, name or f"day_{day}_hour_{hour}")

    def cancel_event(self, event: ScheduledEvent) -> bool:
        """
        Cancel a scheduled event.

        Args:
            event: Event to cancel

        Returns:
            True if event was found and removed
        """
        try:
            self._scheduled_events.remove(event)
            return True
        except ValueError:
            return False

    async def run_scheduler(self) -> None:
        """
        Run the event scheduler loop.

        Call this in a task to enable scheduled events.
        """
        logger.info("time_controller.scheduler_started")

        try:
            while self._state != TimeControllerState.STOPPED:
                if self._state == TimeControllerState.PAUSED:
                    await asyncio.sleep(self.config.tick_interval_real_seconds)
                    continue

                current = self.current_sim_time

                # Find and execute due events
                pending_events = [
                    e
                    for e in self._scheduled_events
                    if not e.executed and e.target_time <= current
                ]

                for event in pending_events:
                    try:
                        logger.debug(
                            "time_controller.executing_event",
                            name=event.name,
                            target=event.target_time.isoformat(),
                        )
                        await event.callback()
                        event.executed = True
                    except Exception as e:
                        logger.error(
                            "time_controller.event_error",
                            name=event.name,
                            error=str(e),
                        )

                # Clean up executed events
                self._scheduled_events = [
                    e for e in self._scheduled_events if not e.executed
                ]

                await asyncio.sleep(self.config.tick_interval_real_seconds)

        except asyncio.CancelledError:
            logger.info("time_controller.scheduler_cancelled")
            raise

    def start_scheduler(self) -> asyncio.Task:
        """
        Start the scheduler as a background task.

        Returns:
            The scheduler task
        """
        if self._scheduler_task and not self._scheduler_task.done():
            return self._scheduler_task

        self._scheduler_task = asyncio.create_task(self.run_scheduler())
        return self._scheduler_task

    def get_real_duration_for_sim_days(self, sim_days: int) -> timedelta:
        """
        Calculate real-world duration for given simulation days.

        Args:
            sim_days: Number of simulation days

        Returns:
            Real-world timedelta
        """
        sim_seconds = sim_days * 24 * 60 * 60
        real_seconds = sim_seconds / self.config.acceleration
        return timedelta(seconds=real_seconds)

    def get_status(self) -> dict:
        """Get current status as a dictionary."""
        return {
            "state": self._state.value,
            "acceleration": self.config.acceleration,
            "sim_time": self.current_sim_time.isoformat(),
            "sim_day": self.sim_day,
            "sim_hour": self.sim_hour,
            "scheduled_events": len(self._scheduled_events),
            "real_duration_for_30_days": str(
                self.get_real_duration_for_sim_days(30)
            ),
        }
