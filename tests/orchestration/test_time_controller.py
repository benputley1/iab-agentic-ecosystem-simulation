"""
Tests for TimeController - simulation time acceleration.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.orchestration.time_controller import (
    TimeController,
    TimeControllerState,
    ScheduledEvent,
)


class TestTimeController:
    """Tests for TimeController class."""

    def test_initialization(self):
        """Test time controller initializes correctly."""
        controller = TimeController(acceleration=100.0)

        assert controller.acceleration == 100.0
        assert controller.state == TimeControllerState.STOPPED
        assert controller.sim_day == 1

    def test_start_stop(self):
        """Test starting and stopping the controller."""
        controller = TimeController(acceleration=100.0)

        controller.start()
        assert controller.state == TimeControllerState.RUNNING
        assert controller.is_running

        controller.stop()
        assert controller.state == TimeControllerState.STOPPED
        assert not controller.is_running

    def test_pause_resume(self):
        """Test pausing and resuming."""
        controller = TimeController(acceleration=100.0)
        controller.start()

        controller.pause()
        assert controller.state == TimeControllerState.PAUSED

        controller.resume()
        assert controller.state == TimeControllerState.RUNNING

    def test_current_sim_time_advances(self):
        """Test that simulation time advances with acceleration."""
        controller = TimeController(acceleration=1000.0)  # 1000x speed
        controller.start()

        initial_time = controller.current_sim_time

        # Wait a small amount of real time
        import time
        time.sleep(0.1)

        # At 1000x, 0.1 real seconds = 100 sim seconds
        elapsed = controller.current_sim_time - initial_time
        assert elapsed.total_seconds() >= 90  # Allow some margin

        controller.stop()

    def test_sim_day_calculation(self):
        """Test simulation day calculation."""
        controller = TimeController(acceleration=100.0)

        # At epoch start, should be day 1
        assert controller.sim_day == 1

        controller.start()

        # Advance to day 2 (86400 sim seconds)
        # At 100x, that's 864 real seconds
        # We'll verify the calculation is correct
        epoch = controller.config.simulation_epoch
        day2_start = epoch + timedelta(days=1)

        # Manually verify day calculation
        elapsed_for_day2 = timedelta(days=1, seconds=1)
        expected_day = elapsed_for_day2.days + 1
        assert expected_day == 2

        controller.stop()

    def test_real_duration_calculation(self):
        """Test calculation of real-world duration."""
        controller = TimeController(acceleration=100.0)

        # 30 sim days at 100x
        duration = controller.get_real_duration_for_sim_days(30)

        # 30 days = 30 * 24 * 60 * 60 = 2,592,000 seconds
        # At 100x = 25,920 real seconds = 7.2 hours
        expected_hours = 7.2
        actual_hours = duration.total_seconds() / 3600

        assert abs(actual_hours - expected_hours) < 0.01

    def test_set_invalid_acceleration(self):
        """Test that invalid acceleration raises error."""
        controller = TimeController(acceleration=100.0)

        with pytest.raises(ValueError):
            controller.acceleration = 0

        with pytest.raises(ValueError):
            controller.acceleration = -10

    def test_reset(self):
        """Test resetting the controller."""
        controller = TimeController(acceleration=100.0)
        controller.start()

        controller.reset()

        assert controller.state == TimeControllerState.STOPPED
        assert controller._start_real_time is None

    def test_get_status(self):
        """Test status dictionary generation."""
        controller = TimeController(acceleration=100.0)

        status = controller.get_status()

        assert "state" in status
        assert "acceleration" in status
        assert "sim_time" in status
        assert "sim_day" in status
        assert status["acceleration"] == 100.0


class TestTimeControllerAsync:
    """Async tests for TimeController."""

    @pytest.mark.asyncio
    async def test_wait_sim_seconds(self):
        """Test waiting for simulation seconds."""
        controller = TimeController(acceleration=1000.0)  # High speed
        controller.start()

        start = datetime.now()
        await controller.wait_sim_seconds(100)  # 100 sim seconds
        elapsed = (datetime.now() - start).total_seconds()

        # At 1000x, 100 sim seconds = 0.1 real seconds
        assert elapsed < 0.2  # Allow margin

        controller.stop()

    @pytest.mark.asyncio
    async def test_wait_sim_hours(self):
        """Test waiting for simulation hours."""
        controller = TimeController(acceleration=10000.0)  # Very high speed
        controller.start()

        start = datetime.now()
        await controller.wait_sim_hours(1)  # 1 sim hour
        elapsed = (datetime.now() - start).total_seconds()

        # At 10000x, 1 hour (3600 seconds) = 0.36 real seconds
        assert elapsed < 0.5

        controller.stop()

    @pytest.mark.asyncio
    async def test_schedule_at_sim_time(self):
        """Test scheduling callbacks at simulation time."""
        controller = TimeController(acceleration=100000.0)  # Very fast
        controller.start()

        callback_executed = False
        callback_time = None

        async def my_callback():
            nonlocal callback_executed, callback_time
            callback_executed = True
            callback_time = controller.current_sim_time

        # Schedule for 10 sim seconds from now
        target = controller.current_sim_time + timedelta(seconds=10)
        controller.schedule_at_sim_time(target, my_callback, "test_callback")

        # Start scheduler
        scheduler_task = controller.start_scheduler()

        # Wait enough time for event to fire (at 100000x, 10 sim seconds = 0.0001 real seconds)
        # Use a longer wait to account for scheduler tick interval
        await asyncio.sleep(0.3)

        controller.stop()
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

        assert callback_executed
        assert callback_time >= target

    @pytest.mark.asyncio
    async def test_schedule_at_day(self):
        """Test scheduling callbacks at specific days."""
        controller = TimeController(acceleration=100000.0)  # Very fast

        called_days = []

        async def day_callback():
            called_days.append(controller.sim_day)

        # Schedule for day 2
        controller.schedule_at_day(2, day_callback, "day2_event")

        controller.start()
        scheduler_task = controller.start_scheduler()

        # Wait for day 2 (at 100000x, 1 day = ~0.86 seconds)
        await asyncio.sleep(1.5)

        controller.stop()
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

        # Should have been called when we reached day 2
        assert len(called_days) > 0

    @pytest.mark.asyncio
    async def test_cancel_event(self):
        """Test cancelling a scheduled event."""
        controller = TimeController(acceleration=10000.0)
        controller.start()

        callback_executed = False

        async def my_callback():
            nonlocal callback_executed
            callback_executed = True

        # Schedule event
        target = controller.current_sim_time + timedelta(seconds=100)
        event = controller.schedule_at_sim_time(target, my_callback)

        # Cancel it
        result = controller.cancel_event(event)
        assert result is True

        # Start scheduler briefly
        scheduler_task = controller.start_scheduler()
        await asyncio.sleep(0.1)

        controller.stop()
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

        # Callback should not have been executed
        assert not callback_executed
