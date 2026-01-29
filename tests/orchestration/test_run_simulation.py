"""
Tests for SimulationRunner - main simulation orchestrator.
"""

import asyncio
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestration.run_simulation import (
    SimulationRunner,
    SimulationConfig,
    SimulationState,
    SimulationResult,
)
from src.orchestration.event_injector import EventType


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()

        assert config.scenarios == ["a", "b", "c"]
        assert config.num_buyers == 5
        assert config.num_sellers == 5
        assert config.simulation_days == 30
        assert config.time_acceleration == 100.0
        assert config.mock_llm is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(
            scenarios=["a"],
            num_buyers=3,
            simulation_days=7,
            enable_chaos=True,
        )

        assert config.scenarios == ["a"]
        assert config.num_buyers == 3
        assert config.simulation_days == 7
        assert config.enable_chaos is True


class TestSimulationRunner:
    """Tests for SimulationRunner class."""

    def test_initialization(self):
        """Test runner initialization."""
        runner = SimulationRunner(
            scenarios=["a", "b"],
            days=10,
            buyers=3,
            sellers=3,
            time_acceleration=1000.0,
        )

        assert runner.config.scenarios == ["a", "b"]
        assert runner.config.simulation_days == 10
        assert runner.config.num_buyers == 3
        assert runner.config.time_acceleration == 1000.0
        assert runner.state == SimulationState.INITIALIZING

    def test_time_controller_integration(self):
        """Test time controller is properly configured."""
        runner = SimulationRunner(
            scenarios=["a"],
            time_acceleration=500.0,
        )

        assert runner.time_controller.acceleration == 500.0

    def test_event_injector_integration(self):
        """Test event injector is properly integrated."""
        runner = SimulationRunner(scenarios=["a"])

        assert runner.event_injector is not None
        assert runner.event_injector.time_controller is runner.time_controller

    def test_get_status(self):
        """Test status retrieval."""
        runner = SimulationRunner(scenarios=["a"])

        status = runner.get_status()

        assert "state" in status
        assert "current_day" in status
        assert "time_controller" in status
        assert status["state"] == SimulationState.INITIALIZING.value


class TestSimulationRunnerCallbacks:
    """Tests for callback registration."""

    def test_on_day_start(self):
        """Test day start callback registration."""
        runner = SimulationRunner(scenarios=["a"])

        async def callback(day: int):
            pass

        runner.on_day_start(callback)

        assert callback in runner._day_start_callbacks

    def test_on_day_end(self):
        """Test day end callback registration."""
        runner = SimulationRunner(scenarios=["a"])

        async def callback(day: int):
            pass

        runner.on_day_end(callback)

        assert callback in runner._day_end_callbacks


class TestSimulationRunnerPauseResume:
    """Tests for pause/resume functionality."""

    @pytest.mark.asyncio
    async def test_pause(self):
        """Test pausing the simulation."""
        runner = SimulationRunner(scenarios=["a"])
        runner._state = SimulationState.RUNNING
        runner.time_controller.start()

        runner.pause()

        assert runner.state == SimulationState.PAUSED

    @pytest.mark.asyncio
    async def test_resume(self):
        """Test resuming the simulation."""
        runner = SimulationRunner(scenarios=["a"])
        runner._state = SimulationState.PAUSED
        runner.time_controller.start()
        runner.time_controller.pause()

        runner.resume()

        assert runner.state == SimulationState.RUNNING


class TestEventInjectionHelpers:
    """Tests for event injection helper methods."""

    @pytest.mark.asyncio
    async def test_inject_event(self):
        """Test injecting events through runner."""
        runner = SimulationRunner(scenarios=["a"])

        event = await runner.inject_event(
            EventType.AGENT_FAILURE,
            target="buyer-001",
        )

        assert event.event_type == EventType.AGENT_FAILURE
        assert event in runner.event_injector.event_history

    @pytest.mark.asyncio
    async def test_inject_agent_failure(self):
        """Test agent failure injection helper."""
        runner = SimulationRunner(scenarios=["a"])

        event = await runner.inject_agent_failure(
            "seller-001",
            duration_hours=2.0,
        )

        assert event.event_type == EventType.AGENT_FAILURE
        assert event.target == "seller-001"

    @pytest.mark.asyncio
    async def test_inject_market_shock(self):
        """Test market shock injection helper."""
        runner = SimulationRunner(scenarios=["a"])

        event = await runner.inject_market_shock(
            price_change_pct=0.15,
            duration_hours=4.0,
        )

        assert event.event_type == EventType.MARKET_SHOCK
        assert event.parameters["price_change_pct"] == 0.15


class TestCheckpointing:
    """Tests for checkpoint functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_creation(self, tmp_path: Path):
        """Test checkpoint file creation."""
        runner = SimulationRunner(
            scenarios=["a"],
            checkpoint_dir=str(tmp_path),
        )
        runner._scenarios_cache = {}  # Mock empty scenarios
        runner.time_controller.start()

        checkpoint_id = await runner._create_checkpoint(5)

        assert checkpoint_id.startswith("checkpoint_day005")
        assert (tmp_path / f"{checkpoint_id}.json").exists()

        runner.time_controller.stop()

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, tmp_path: Path):
        """Test old checkpoint cleanup (keep last 5)."""
        runner = SimulationRunner(
            scenarios=["a"],
            checkpoint_dir=str(tmp_path),
        )
        runner._scenarios_cache = {}
        runner.time_controller.start()

        # Create 7 checkpoints
        for day in range(1, 8):
            await runner._create_checkpoint(day)
            await asyncio.sleep(0.01)  # Ensure unique timestamps

        runner.time_controller.stop()

        # Should only keep last 5
        checkpoints = list(tmp_path.glob("checkpoint_*.json"))
        assert len(checkpoints) == 5


class TestSimulationResult:
    """Tests for SimulationResult."""

    def test_to_dict(self):
        """Test result serialization."""
        result = SimulationResult(
            config=SimulationConfig(scenarios=["a", "b"]),
            state=SimulationState.COMPLETED,
            total_events_injected=10,
            checkpoints_created=3,
        )

        data = result.to_dict()

        assert data["state"] == "completed"
        assert data["total_events_injected"] == 10
        assert data["checkpoints_created"] == 3


class TestComparisonReport:
    """Tests for comparison report generation."""

    def test_get_comparison_report(self):
        """Test comparison report structure."""
        runner = SimulationRunner(scenarios=["a", "b"])

        # Mock scenario with metrics
        mock_scenario_a = MagicMock()
        mock_scenario_a.metrics.to_dict.return_value = {
            "total_deals": 100,
            "total_impressions": 1000000,
            "total_buyer_spend": 15000.0,
            "total_seller_revenue": 12750.0,
            "total_exchange_fees": 2250.0,
            "intermediary_take_rate": 15.0,
            "average_cpm": 15.0,
            "hallucination_rate": 0.0,
        }

        mock_scenario_b = MagicMock()
        mock_scenario_b.metrics.to_dict.return_value = {
            "total_deals": 95,
            "total_impressions": 950000,
            "total_buyer_spend": 14250.0,
            "total_seller_revenue": 13537.5,
            "total_exchange_fees": 712.5,
            "intermediary_take_rate": 5.0,
            "average_cpm": 15.0,
            "hallucination_rate": 2.5,
        }

        runner._scenarios_cache = {
            "a": mock_scenario_a,
            "b": mock_scenario_b,
        }

        report = runner.get_comparison_report()

        assert "comparison" in report
        assert "total_deals" in report["comparison"]
        assert report["comparison"]["total_deals"]["a"] == 100
        assert report["comparison"]["total_deals"]["b"] == 95
        assert report["comparison"]["intermediary_take_rate"]["a"] == 15.0
        assert report["comparison"]["intermediary_take_rate"]["b"] == 5.0
