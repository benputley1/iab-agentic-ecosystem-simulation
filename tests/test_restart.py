"""Comprehensive tests for Agent Restart Simulator."""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.resilience.restart import (
    AgentRestartSimulator,
    AgentState,
    RecoveryResult,
    RestartEvent,
    MockAgent,
)


class TestAgentState:
    """Tests for AgentState dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        state = AgentState()
        assert state.budget_remaining == 0.0
        assert state.impressions_delivered == 0
        assert state.active_deals == {}
        assert state.frequency_caps == {}
        assert state.price_history == []
        assert state.campaign_id == ""
        assert state.timestamp is None
        assert state.metadata == {}
    
    def test_custom_values(self):
        """Test initialization with custom values."""
        state = AgentState(
            budget_remaining=5000.0,
            impressions_delivered=100000,
            active_deals={"deal-1": {"cpm": 15.0}},
            frequency_caps={"user-1": 3},
            price_history=[12.5, 13.0, 14.0],
            campaign_id="camp-123",
            metadata={"source": "test"},
        )
        
        assert state.budget_remaining == 5000.0
        assert state.impressions_delivered == 100000
        assert state.active_deals == {"deal-1": {"cpm": 15.0}}
        assert state.campaign_id == "camp-123"
    
    def test_to_dict(self):
        """Test state serialization to dict."""
        state = AgentState(
            budget_remaining=1000.0,
            impressions_delivered=5000,
            campaign_id="camp-test",
        )
        
        d = state.to_dict()
        assert d["budget_remaining"] == 1000.0
        assert d["impressions_delivered"] == 5000
        assert d["campaign_id"] == "camp-test"
        assert "timestamp" not in d  # timestamp excluded


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        result = RecoveryResult(
            mode="ledger",
            accuracy=0.998,
            recovered_state=AgentState(),
        )
        
        assert result.mode == "ledger"
        assert result.accuracy == 0.998
        assert result.fields_recovered == 0
        assert result.fields_total == 0
        assert result.field_errors == {}
        assert result.recovery_time_ms == 0.0
    
    def test_with_errors(self):
        """Test with field errors."""
        result = RecoveryResult(
            mode="private_db",
            accuracy=0.873,
            recovered_state=AgentState(),
            fields_recovered=5,
            fields_total=6,
            field_errors={"frequency_caps": (150, 100)},
            recovery_time_ms=2300,
        )
        
        assert result.fields_recovered == 5
        assert result.fields_total == 6
        assert "frequency_caps" in result.field_errors


class TestRestartEvent:
    """Tests for RestartEvent dataclass."""
    
    def test_basic_event(self):
        """Test basic restart event creation."""
        pre_state = AgentState(budget_remaining=5000.0)
        event = RestartEvent(
            hour=120,
            pre_state=pre_state,
            recovery_accuracy={"private_db": 0.87, "ledger": 0.998},
        )
        
        assert event.hour == 120
        assert event.pre_state.budget_remaining == 5000.0
        assert event.recovery_accuracy["ledger"] == 0.998
        assert event.crash_reason == "random_failure"


class TestMockAgent:
    """Tests for MockAgent class."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = MockAgent(
            agent_id="test-001",
            initial_budget=10000.0,
            campaign_id="camp-test",
        )
        
        assert agent.agent_id == "test-001"
        assert agent.campaign_id == "camp-test"
        
        state = agent.get_internal_state()
        assert state.budget_remaining == 10000.0
        assert state.impressions_delivered == 0
    
    def test_record_spend(self):
        """Test spend recording."""
        agent = MockAgent(initial_budget=10000.0)
        agent.record_spend(150.0, impressions=10000)
        
        state = agent.get_internal_state()
        assert state.budget_remaining == 9850.0
        assert state.impressions_delivered == 10000
        assert len(state.price_history) == 1
        assert state.price_history[0] == 15.0  # CPM
    
    def test_multiple_spends(self):
        """Test multiple spend events."""
        agent = MockAgent(initial_budget=10000.0)
        
        agent.record_spend(100.0, impressions=5000)  # CPM 20
        agent.record_spend(200.0, impressions=20000)  # CPM 10
        agent.record_spend(150.0, impressions=10000)  # CPM 15
        
        state = agent.get_internal_state()
        assert state.budget_remaining == 9550.0
        assert state.impressions_delivered == 35000
        assert len(state.price_history) == 3
    
    def test_add_deal(self):
        """Test deal management."""
        agent = MockAgent()
        agent.add_deal("deal-001", {"cpm": 12.0, "volume": 100000})
        agent.add_deal("deal-002", {"cpm": 15.0, "volume": 50000})
        
        state = agent.get_internal_state()
        assert len(state.active_deals) == 2
        assert state.active_deals["deal-001"]["cpm"] == 12.0
    
    def test_frequency_caps(self):
        """Test frequency cap tracking."""
        agent = MockAgent()
        
        agent.record_frequency("user-001")
        agent.record_frequency("user-001")
        agent.record_frequency("user-002")
        
        state = agent.get_internal_state()
        assert state.frequency_caps["user-001"] == 2
        assert state.frequency_caps["user-002"] == 1
    
    def test_metadata(self):
        """Test metadata storage."""
        agent = MockAgent()
        agent.set_metadata("strategy", "aggressive")
        agent.set_metadata("last_optimization", "2024-01-15")
        
        state = agent.get_internal_state()
        assert state.metadata["strategy"] == "aggressive"
    
    def test_restart_clears_state(self):
        """Test that restart clears volatile state."""
        agent = MockAgent(initial_budget=10000.0)
        
        # Build up state
        agent.record_spend(500.0, impressions=30000)
        agent.add_deal("deal-001", {"cpm": 12.0})
        agent.record_frequency("user-001")
        
        state_before = agent.get_internal_state()
        assert state_before.budget_remaining == 9500.0
        assert len(state_before.active_deals) == 1
        
        # Restart
        agent.restart()
        
        state_after = agent.get_internal_state()
        assert state_after.budget_remaining == 10000.0  # Reset to initial
        assert state_after.impressions_delivered == 0
        assert len(state_after.active_deals) == 0
        assert len(state_after.frequency_caps) == 0
    
    def test_private_db_recovery(self):
        """Test private DB recovery mode."""
        agent = MockAgent(initial_budget=10000.0)
        
        # Build up state
        agent.record_spend(500.0, impressions=30000)
        agent.add_deal("deal-001", {"cpm": 12.0})
        
        state_before = agent.get_internal_state()
        
        # Restart and recover
        agent.restart()
        recovered = agent.recover_state("private_db")
        
        # Should recover core data
        assert recovered.budget_remaining == state_before.budget_remaining
        assert recovered.impressions_delivered == state_before.impressions_delivered
    
    def test_ledger_recovery(self):
        """Test ledger recovery mode (near-perfect)."""
        agent = MockAgent(initial_budget=10000.0)
        
        # Build up state - frequency caps need a persist trigger
        for i in range(50):
            agent.record_frequency(f"user-{i:03d}")
        
        # This triggers persist, saving the frequency caps
        agent.record_spend(500.0, impressions=30000)
        agent.add_deal("deal-001", {"cpm": 12.0})
        
        state_before = agent.get_internal_state()
        
        # Restart and recover
        agent.restart()
        recovered = agent.recover_state("ledger")
        
        # Ledger should recover everything
        assert recovered.budget_remaining == state_before.budget_remaining
        assert recovered.impressions_delivered == state_before.impressions_delivered
        assert recovered.active_deals == state_before.active_deals
        assert len(recovered.frequency_caps) == len(state_before.frequency_caps)
    
    def test_invalid_recovery_mode(self):
        """Test invalid recovery mode raises error."""
        agent = MockAgent()
        
        with pytest.raises(ValueError, match="Unknown recovery mode"):
            agent.recover_state("invalid_mode")
    
    def test_custom_recovery_accuracy(self):
        """Test custom recovery accuracy configuration."""
        agent = MockAgent(
            recovery_accuracy={"private_db": 0.5, "ledger": 0.99}
        )
        
        assert agent._recovery_accuracy["private_db"] == 0.5
        assert agent._recovery_accuracy["ledger"] == 0.99


class TestAgentRestartSimulator:
    """Tests for AgentRestartSimulator class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        sim = AgentRestartSimulator()
        
        assert sim.crash_probability == 0.01
        assert "private_db" in sim.recovery_modes
        assert "ledger" in sim.recovery_modes
        assert sim.restart_events == []
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        sim = AgentRestartSimulator(
            crash_probability=0.05,
            recovery_modes=["ledger"],
            random_seed=42,
        )
        
        assert sim.crash_probability == 0.05
        assert sim.recovery_modes == ["ledger"]
    
    def test_invalid_crash_probability(self):
        """Test invalid crash probability raises error."""
        with pytest.raises(ValueError, match="crash_probability must be between"):
            AgentRestartSimulator(crash_probability=1.5)
        
        with pytest.raises(ValueError, match="crash_probability must be between"):
            AgentRestartSimulator(crash_probability=-0.1)
    
    def test_maybe_crash_no_crash(self):
        """Test maybe_crash when no crash occurs."""
        sim = AgentRestartSimulator(crash_probability=0.0)  # Never crash
        agent = MockAgent(initial_budget=10000.0)
        agent.record_spend(500.0, impressions=30000)
        
        result = sim.maybe_crash(agent, hour=100)
        
        assert result is None
        assert len(sim.restart_events) == 0
        # Agent state should be unchanged
        assert agent.get_internal_state().budget_remaining == 9500.0
    
    def test_maybe_crash_force_crash(self):
        """Test maybe_crash with forced crash."""
        sim = AgentRestartSimulator(crash_probability=0.0)  # Normally no crash
        agent = MockAgent(initial_budget=10000.0)
        agent.record_spend(500.0, impressions=30000)
        
        result = sim.maybe_crash(agent, hour=100, force_crash=True)
        
        assert result is not None
        assert isinstance(result, RestartEvent)
        assert result.hour == 100
        assert result.pre_state.budget_remaining == 9500.0
        assert "private_db" in result.recovery_accuracy
        assert "ledger" in result.recovery_accuracy
    
    def test_deterministic_crash_with_seed(self):
        """Test deterministic crash pattern with seed."""
        # Run twice with same seed
        results_1 = []
        sim1 = AgentRestartSimulator(crash_probability=0.5, random_seed=12345)
        agent1 = MockAgent()
        
        for hour in range(100):
            if sim1.maybe_crash(agent1, hour):
                results_1.append(hour)
        
        results_2 = []
        sim2 = AgentRestartSimulator(crash_probability=0.5, random_seed=12345)
        agent2 = MockAgent()
        
        for hour in range(100):
            if sim2.maybe_crash(agent2, hour):
                results_2.append(hour)
        
        # Should have same crash pattern
        assert results_1 == results_2
    
    def test_crash_probability_distribution(self):
        """Test that crash probability is approximately correct."""
        sim = AgentRestartSimulator(crash_probability=0.1, random_seed=42)
        agent = MockAgent()
        
        crashes = 0
        iterations = 1000
        
        for hour in range(iterations):
            # Reset agent each time to avoid state buildup issues
            agent = MockAgent()
            if sim.maybe_crash(agent, hour):
                crashes += 1
        
        # Should be approximately 10% (with some tolerance for randomness)
        crash_rate = crashes / iterations
        assert 0.05 < crash_rate < 0.20  # Wide tolerance for randomness
    
    def test_recovery_accuracy_comparison(self):
        """Test that ledger recovery is more accurate than private_db."""
        sim = AgentRestartSimulator(random_seed=42)
        agent = MockAgent(initial_budget=10000.0)
        
        # Build up complex state
        for i in range(10):
            agent.record_spend(100.0, impressions=5000)
            agent.add_deal(f"deal-{i:03d}", {"cpm": 12.0 + i, "volume": 10000})
            for j in range(20):
                agent.record_frequency(f"user-{i}-{j}")
        
        event = sim.maybe_crash(agent, hour=100, force_crash=True)
        
        # Ledger should be at least as accurate as private_db
        assert event.recovery_accuracy["ledger"] >= event.recovery_accuracy["private_db"]
    
    def test_restart_events_list(self):
        """Test that restart events are properly tracked."""
        sim = AgentRestartSimulator(crash_probability=1.0, random_seed=42)  # Always crash
        agent = MockAgent(initial_budget=10000.0)
        
        # Trigger multiple crashes
        for hour in [10, 50, 100]:
            agent.record_spend(50.0, impressions=3000)
            sim.maybe_crash(agent, hour)
        
        assert len(sim.restart_events) == 3
        assert sim.restart_events[0].hour == 10
        assert sim.restart_events[1].hour == 50
        assert sim.restart_events[2].hour == 100
    
    def test_crash_reasons(self):
        """Test that crash reasons are assigned."""
        sim = AgentRestartSimulator(random_seed=42)
        agent = MockAgent()
        
        event = sim.maybe_crash(agent, hour=1, force_crash=True)
        
        assert event.crash_reason in [
            "memory_overflow",
            "network_timeout",
            "process_killed",
            "container_restart",
            "dependency_failure",
            "rate_limit_exceeded",
        ]
    
    def test_get_summary_empty(self):
        """Test summary with no restart events."""
        sim = AgentRestartSimulator()
        
        summary = sim.get_summary()
        
        assert summary["total_restarts"] == 0
        assert summary["avg_accuracy_by_mode"] == {}
        assert summary["decisions_affected_total"] == 0
    
    def test_get_summary_with_events(self):
        """Test summary with restart events."""
        sim = AgentRestartSimulator(crash_probability=1.0, random_seed=42)
        agent = MockAgent(initial_budget=10000.0)
        
        # Generate some events
        for hour in range(5):
            agent.record_spend(100.0, impressions=5000)
            sim.maybe_crash(agent, hour * 24)
        
        summary = sim.get_summary()
        
        assert summary["total_restarts"] == 5
        assert "private_db" in summary["avg_accuracy_by_mode"]
        assert "ledger" in summary["avg_accuracy_by_mode"]
        assert summary["total_restarts"] == len(summary["crash_hours"])
        assert "crash_reasons" in summary
    
    def test_format_report(self):
        """Test report formatting."""
        sim = AgentRestartSimulator(crash_probability=1.0, random_seed=42)
        agent = MockAgent(initial_budget=10000.0)
        
        agent.record_spend(500.0, impressions=30000)
        sim.maybe_crash(agent, hour=100, force_crash=True)
        
        report = sim.format_report()
        
        assert "AGENT RESTART SIMULATION REPORT" in report
        assert "Total Restart Events: 1" in report
        assert "private_db" in report
        assert "ledger" in report
        assert "Hour 100" in report
    
    def test_format_report_empty(self):
        """Test report formatting with no events."""
        sim = AgentRestartSimulator()
        
        report = sim.format_report()
        
        assert "Total Restart Events: 0" in report
    
    def test_recovery_time_tracking(self):
        """Test that recovery time is tracked."""
        sim = AgentRestartSimulator()
        agent = MockAgent()
        
        event = sim.maybe_crash(agent, hour=1, force_crash=True)
        
        # Ledger should be faster
        assert event.recovery_results["ledger"].recovery_time_ms < event.recovery_results["private_db"].recovery_time_ms


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_agent_crash(self):
        """Test crashing an agent with no state."""
        sim = AgentRestartSimulator()
        agent = MockAgent()  # Fresh agent, no activity
        
        event = sim.maybe_crash(agent, hour=0, force_crash=True)
        
        assert event is not None
        assert event.pre_state.budget_remaining == 10000.0  # Default budget
        assert event.pre_state.impressions_delivered == 0
    
    def test_recovery_with_no_persisted_state(self):
        """Test recovery when no state was persisted."""
        agent = MockAgent()
        # Don't do any operations that trigger persistence
        agent.restart()
        
        # Directly test recovery - should return empty state
        agent._persisted_state = None
        agent._ledger_state = None
        
        recovered = agent.recover_state("private_db")
        assert recovered.budget_remaining == 0.0
        
        recovered = agent.recover_state("ledger")
        assert recovered.budget_remaining == 0.0
    
    def test_large_frequency_cap_list(self):
        """Test handling of large frequency cap lists."""
        agent = MockAgent()
        
        # Add many frequency caps
        for i in range(200):
            agent.record_frequency(f"user-{i:05d}")
        
        state = agent.get_internal_state()
        assert len(state.frequency_caps) == 200
        
        # Trigger persistence (happens on spend/deal)
        agent.record_spend(10.0, impressions=1000)
        
        # Private DB should truncate to 100
        assert len(agent._persisted_state.frequency_caps) == 100
        
        # Ledger should keep all
        assert len(agent._ledger_state.frequency_caps) == 200
    
    def test_long_price_history(self):
        """Test handling of long price history."""
        agent = MockAgent(initial_budget=100000.0)
        
        # Build up long price history
        for i in range(100):
            agent.record_spend(100.0, impressions=5000)
        
        state = agent.get_internal_state()
        assert len(state.price_history) == 100
        
        # Private DB should truncate to 50
        assert len(agent._persisted_state.price_history) == 50
        
        # Ledger should keep all
        assert len(agent._ledger_state.price_history) == 100
    
    def test_zero_crash_probability(self):
        """Test with zero crash probability."""
        sim = AgentRestartSimulator(crash_probability=0.0)
        agent = MockAgent()
        
        # Should never crash
        for hour in range(1000):
            result = sim.maybe_crash(agent, hour)
            assert result is None
    
    def test_full_crash_probability(self):
        """Test with 100% crash probability."""
        sim = AgentRestartSimulator(crash_probability=1.0)
        agent = MockAgent()
        
        # Should always crash
        for hour in range(10):
            result = sim.maybe_crash(agent, hour)
            assert result is not None


class TestIntegrationScenarios:
    """Integration tests simulating realistic scenarios."""
    
    def test_30_day_campaign(self):
        """Test a realistic 30-day campaign simulation."""
        sim = AgentRestartSimulator(
            crash_probability=0.01,  # 1% per hour
            random_seed=42,
        )
        agent = MockAgent(initial_budget=100000.0)
        
        hours = 24 * 30  # 30 days
        
        for hour in range(hours):
            # Simulate activity
            if hour % 4 == 0:  # Every 4 hours
                agent.record_spend(50.0, impressions=3000)
            
            if hour % 24 == 0:  # Daily deal update
                agent.add_deal(f"deal-day-{hour // 24}", {"cpm": 15.0})
            
            # Check for crash
            sim.maybe_crash(agent, hour)
        
        summary = sim.get_summary()
        
        # With 1% crash rate over 720 hours, expect ~7 crashes
        assert 0 < summary["total_restarts"] < 30
        
        # Ledger should always be more accurate
        if summary["total_restarts"] > 0:
            assert summary["avg_accuracy_by_mode"]["ledger"] >= summary["avg_accuracy_by_mode"]["private_db"]
    
    def test_high_frequency_scenario(self):
        """Test high-frequency trading scenario with many crashes."""
        sim = AgentRestartSimulator(
            crash_probability=0.05,  # 5% per hour - unstable system
            random_seed=123,
        )
        agent = MockAgent(initial_budget=50000.0)
        
        total_spend = 0
        total_impressions = 0
        
        for hour in range(168):  # 1 week
            # High activity
            for _ in range(10):
                spend = 10.0
                imps = 1000
                agent.record_spend(spend, impressions=imps)
                total_spend += spend
                total_impressions += imps
            
            # Crash check
            event = sim.maybe_crash(agent, hour)
            
            if event:
                # Verify crash was recorded
                assert event.hour == hour
                assert event.pre_state.impressions_delivered > 0
        
        summary = sim.get_summary()
        report = sim.format_report()
        
        # Should have several crashes with 5% rate
        assert summary["total_restarts"] > 0
        assert "AGENT RESTART SIMULATION REPORT" in report
    
    def test_comparison_private_db_vs_ledger(self):
        """Test that demonstrates the value proposition of ledger recovery."""
        sim = AgentRestartSimulator(random_seed=999)
        
        # Run multiple scenarios and compare
        total_private_accuracy = 0
        total_ledger_accuracy = 0
        scenarios = 10
        
        for s in range(scenarios):
            agent = MockAgent(initial_budget=10000.0)
            
            # Build up realistic state
            for i in range(50):
                agent.record_spend(20.0, impressions=1000)
                agent.record_frequency(f"scenario-{s}-user-{i}")
            
            agent.add_deal(f"deal-{s}", {"cpm": 14.0, "volume": 50000})
            
            event = sim.maybe_crash(agent, hour=s * 10, force_crash=True)
            
            total_private_accuracy += event.recovery_accuracy["private_db"]
            total_ledger_accuracy += event.recovery_accuracy["ledger"]
        
        avg_private = total_private_accuracy / scenarios
        avg_ledger = total_ledger_accuracy / scenarios
        
        # Ledger should be significantly better
        assert avg_ledger >= avg_private
        
        # Print comparison for visibility
        print(f"\nRecovery Comparison ({scenarios} scenarios):")
        print(f"  Private DB: {avg_private:.1%}")
        print(f"  Ledger:     {avg_ledger:.1%}")
        print(f"  Improvement: {(avg_ledger - avg_private):.1%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
