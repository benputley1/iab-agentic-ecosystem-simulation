"""
Tests for EventInjector - chaos testing and event injection.
"""

import asyncio
from datetime import timedelta

import pytest

from src.orchestration.time_controller import TimeController
from src.orchestration.event_injector import (
    EventInjector,
    EventType,
    InjectedEvent,
    ChaosConfig,
)


class TestEventInjector:
    """Tests for EventInjector class."""

    def test_initialization(self):
        """Test event injector initializes correctly."""
        injector = EventInjector()

        assert injector.chaos_config.enabled is False
        assert len(injector.active_events) == 0
        assert len(injector.event_history) == 0

    def test_initialization_with_time_controller(self):
        """Test initialization with time controller."""
        tc = TimeController(acceleration=100.0)
        injector = EventInjector(time_controller=tc)

        assert injector.time_controller is tc

    @pytest.mark.asyncio
    async def test_inject_event(self):
        """Test basic event injection."""
        injector = EventInjector()

        event = await injector.inject(
            EventType.AGENT_FAILURE,
            target="buyer-001",
            duration=timedelta(hours=1),
            reason="test",
        )

        assert event.event_type == EventType.AGENT_FAILURE
        assert event.target == "buyer-001"
        assert event.parameters["reason"] == "test"
        assert event in injector.event_history
        assert event in injector.active_events

    @pytest.mark.asyncio
    async def test_instant_event_not_active(self):
        """Test that instant events (no duration) aren't active."""
        injector = EventInjector()

        event = await injector.inject(
            EventType.MARKET_SHOCK,
            price_change_pct=0.1,
        )

        assert event in injector.event_history
        assert event not in injector.active_events  # No duration = not active

    @pytest.mark.asyncio
    async def test_resolve_event(self):
        """Test resolving an active event."""
        injector = EventInjector()

        event = await injector.inject(
            EventType.NETWORK_LATENCY,
            duration=timedelta(hours=2),
        )

        assert event in injector.active_events

        await injector.resolve(event)

        assert event not in injector.active_events
        assert event.resolved_at is not None

    @pytest.mark.asyncio
    async def test_event_callback(self):
        """Test event callbacks are fired."""
        injector = EventInjector()

        received_events = []

        async def my_callback(event: InjectedEvent):
            received_events.append(event)

        injector.on_event(EventType.AGENT_FAILURE, my_callback)

        event = await injector.inject(
            EventType.AGENT_FAILURE,
            target="seller-001",
        )

        assert len(received_events) == 1
        assert received_events[0] == event

    @pytest.mark.asyncio
    async def test_global_callback(self):
        """Test global callbacks receive all events."""
        injector = EventInjector()

        received_events = []

        async def global_callback(event: InjectedEvent):
            received_events.append(event)

        injector.on_any_event(global_callback)

        await injector.inject(EventType.AGENT_FAILURE)
        await injector.inject(EventType.MARKET_SHOCK)
        await injector.inject(EventType.NETWORK_LATENCY)

        assert len(received_events) == 3


class TestEventInjectorConvenienceMethods:
    """Tests for convenience injection methods."""

    @pytest.mark.asyncio
    async def test_inject_agent_failure(self):
        """Test agent failure injection."""
        injector = EventInjector()

        event = await injector.inject_agent_failure(
            "buyer-001",
            duration_hours=2.0,
        )

        assert event.event_type == EventType.AGENT_FAILURE
        assert event.target == "buyer-001"
        assert event.duration == timedelta(hours=2.0)

    @pytest.mark.asyncio
    async def test_inject_agent_slowdown(self):
        """Test agent slowdown injection."""
        injector = EventInjector()

        event = await injector.inject_agent_slowdown(
            "seller-001",
            slowdown_factor=5.0,
            duration_hours=1.0,
        )

        assert event.event_type == EventType.AGENT_SLOWDOWN
        assert event.parameters["slowdown_factor"] == 5.0

    @pytest.mark.asyncio
    async def test_inject_memory_loss(self):
        """Test memory loss injection (context rot)."""
        injector = EventInjector()

        event = await injector.inject_memory_loss(
            "buyer-002",
            keys_to_lose=["campaign_history", "deal_cache"],
        )

        assert event.event_type == EventType.AGENT_MEMORY_LOSS
        assert event.parameters["keys_to_lose"] == ["campaign_history", "deal_cache"]

    @pytest.mark.asyncio
    async def test_inject_network_partition(self):
        """Test network partition injection."""
        injector = EventInjector()

        event = await injector.inject_network_partition(
            agents=["buyer-001", "buyer-002"],
            duration_hours=0.5,
        )

        assert event.event_type == EventType.NETWORK_PARTITION
        assert event.parameters["affected_agents"] == ["buyer-001", "buyer-002"]

    @pytest.mark.asyncio
    async def test_inject_market_shock(self):
        """Test market shock injection."""
        injector = EventInjector()

        event = await injector.inject_market_shock(
            price_change_pct=-0.2,  # 20% price drop
            duration_hours=4.0,
            affected_channels=["display"],
        )

        assert event.event_type == EventType.MARKET_SHOCK
        assert event.parameters["price_change_pct"] == -0.2
        assert event.parameters["affected_channels"] == ["display"]

    @pytest.mark.asyncio
    async def test_inject_exchange_outage(self):
        """Test exchange outage injection."""
        injector = EventInjector()

        event = await injector.inject_exchange_outage(
            "exchange-001",
            duration_hours=0.5,
        )

        assert event.event_type == EventType.EXCHANGE_DOWN
        assert event.target == "exchange-001"

    @pytest.mark.asyncio
    async def test_inject_ledger_delay(self):
        """Test ledger delay injection."""
        injector = EventInjector()

        event = await injector.inject_ledger_delay(
            delay_seconds=60,
            duration_hours=2.0,
        )

        assert event.event_type == EventType.LEDGER_DELAY
        assert event.parameters["delay_seconds"] == 60


class TestChaosMode:
    """Tests for chaos mode."""

    def test_enable_chaos_mode(self):
        """Test enabling chaos mode."""
        injector = EventInjector()

        injector.enable_chaos_mode(
            agent_failure_rate=0.05,
            network_issue_rate=0.02,
            market_shock_rate=0.01,
        )

        assert injector.chaos_config.enabled is True
        assert injector.chaos_config.agent_failure_rate == 0.05
        assert injector.chaos_config.network_issue_rate == 0.02
        assert injector.chaos_config.market_shock_rate == 0.01

    def test_disable_chaos_mode(self):
        """Test disabling chaos mode."""
        injector = EventInjector()
        injector.enable_chaos_mode()

        injector.disable_chaos_mode()

        assert injector.chaos_config.enabled is False

    @pytest.mark.asyncio
    async def test_chaos_loop_injects_events(self):
        """Test that chaos loop actually injects events."""
        tc = TimeController(acceleration=100000.0)  # Very fast
        tc.start()
        injector = EventInjector(time_controller=tc)

        # High rates to ensure events happen
        injector.enable_chaos_mode(
            agent_failure_rate=0.9,  # 90% chance
            network_issue_rate=0.9,
            market_shock_rate=0.0,
        )

        # Set very fast tick interval
        injector.chaos_config.tick_interval_sim_hours = 0.01

        agents = ["buyer-001", "buyer-002", "seller-001"]

        # Run chaos loop briefly
        task = injector.start_chaos_loop(agents)
        await asyncio.sleep(0.3)

        injector.stop()
        tc.stop()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have injected some events
        assert len(injector.event_history) > 0


class TestEventQueries:
    """Tests for event query methods."""

    @pytest.mark.asyncio
    async def test_is_event_active(self):
        """Test checking if event type is active."""
        injector = EventInjector()

        await injector.inject(
            EventType.AGENT_FAILURE,
            target="buyer-001",
            duration=timedelta(hours=1),
        )

        assert injector.is_event_active(EventType.AGENT_FAILURE) is True
        assert injector.is_event_active(EventType.AGENT_FAILURE, "buyer-001") is True
        assert injector.is_event_active(EventType.AGENT_FAILURE, "buyer-002") is False
        assert injector.is_event_active(EventType.NETWORK_LATENCY) is False

    @pytest.mark.asyncio
    async def test_get_active_events_for_target(self):
        """Test getting active events for a target."""
        injector = EventInjector()

        await injector.inject(
            EventType.AGENT_FAILURE,
            target="buyer-001",
            duration=timedelta(hours=1),
        )
        await injector.inject(
            EventType.AGENT_SLOWDOWN,
            target="buyer-001",
            duration=timedelta(hours=2),
        )
        await injector.inject(
            EventType.AGENT_FAILURE,
            target="buyer-002",
            duration=timedelta(hours=1),
        )

        events = injector.get_active_events_for_target("buyer-001")

        assert len(events) == 2
        assert all(e.target == "buyer-001" for e in events)

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test statistics generation."""
        injector = EventInjector()

        await injector.inject(EventType.AGENT_FAILURE, duration=timedelta(hours=1))
        await injector.inject(EventType.AGENT_FAILURE, duration=timedelta(hours=1))
        await injector.inject(EventType.MARKET_SHOCK)

        stats = injector.get_statistics()

        assert stats["total_events"] == 3
        assert stats["active_events"] == 2  # Only failures have duration
        assert stats["events_by_type"]["agent_failure"] == 2
        assert stats["events_by_type"]["market_shock"] == 1
