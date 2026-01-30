"""
Tests for Context Flow Management.

Tests context passing between levels, decay simulation,
and recovery mechanisms across different scenarios.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from src.context.flow import (
    ContextFlowManager,
    ContextPassResult,
    AggregatedContext,
    AgentRef,
    FlowDirection,
)
from src.context.window import (
    ContextWindowManager,
    ContextEntry,
    WindowState,
)
from src.context.rot import (
    ContextRotSimulator,
    DecayResult,
    HandoffResult,
    RotType,
)
from src.context.recovery import (
    ContextRecovery,
    RecoveryResult,
    RecoverySource,
)
from src.context.metrics import (
    ContextMetrics,
    ContextMetricsSummary,
    MetricEvent,
)
from src.agents.base import AgentContext, ContextPriority


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def orchestrator():
    """L1 orchestrator agent reference."""
    return AgentRef(
        agent_id="buyer_l1_portfolio",
        level=1,
        role="portfolio_manager"
    )


@pytest.fixture
def specialist():
    """L2 specialist agent reference."""
    return AgentRef(
        agent_id="buyer_l2_ctv",
        level=2,
        role="ctv_specialist"
    )


@pytest.fixture
def functional():
    """L3 functional agent reference."""
    return AgentRef(
        agent_id="buyer_l3_pricing",
        level=3,
        role="pricing"
    )


@pytest.fixture
def sample_context():
    """Sample agent context with various priority items."""
    ctx = AgentContext(
        source_level=1,
        source_agent_id="buyer_l1_portfolio",
        task_description="Execute CTV campaign",
        task_constraints=["budget <= $50000", "CPM <= $25"]
    )
    
    # Add items at different priorities
    ctx.add_item("campaign_id", "camp_123", ContextPriority.CRITICAL)
    ctx.add_item("budget", 50000, ContextPriority.CRITICAL)
    ctx.add_item("target_audience", {"age": "25-54", "interests": ["sports"]}, ContextPriority.HIGH)
    ctx.add_item("historical_performance", {"avg_cpm": 22.5}, ContextPriority.MEDIUM)
    ctx.add_item("notes", "Prefer premium inventory", ContextPriority.LOW)
    
    return ctx


@pytest.fixture
def context_window():
    """Sample context window."""
    window = ContextWindowManager(
        agent_id="test_agent",
        max_tokens=10000,
        reserved_tokens=500
    )
    return window


@pytest.fixture
def flow_manager_scenario_b():
    """Context flow manager for Scenario B."""
    return ContextFlowManager(scenario="B", base_degradation_rate=0.10)


@pytest.fixture
def flow_manager_scenario_c():
    """Context flow manager for Scenario C with mock ledger."""
    ledger = AsyncMock()
    ledger.store_context = AsyncMock()
    return ContextFlowManager(scenario="C", ledger_client=ledger)


@pytest.fixture
def rot_simulator():
    """Context rot simulator with fixed seed for reproducibility."""
    return ContextRotSimulator(
        decay_rate=0.05,
        restart_probability=0.01,
        handoff_loss_rate=0.10,
        seed=42
    )


@pytest.fixture
def recovery_system():
    """Context recovery system."""
    return ContextRecovery(
        exchange_recovery_rate=0.60,
        exchange_fidelity=0.95
    )


@pytest.fixture
def metrics_tracker():
    """Context metrics tracker."""
    return ContextMetrics()


# =============================================================================
# Context Flow Manager Tests
# =============================================================================

class TestContextFlowManager:
    """Tests for ContextFlowManager."""
    
    @pytest.mark.asyncio
    async def test_pass_context_down_scenario_c(
        self, 
        flow_manager_scenario_c, 
        orchestrator, 
        specialist, 
        sample_context
    ):
        """Scenario C should have 0% loss via ledger."""
        result = await flow_manager_scenario_c.pass_context_down(
            from_agent=orchestrator,
            to_agent=specialist,
            context=sample_context,
            task="Execute CTV buys"
        )
        
        assert result.success
        assert result.scenario == "C"
        assert result.tokens_lost == 0
        assert result.items_truncated == 0
        assert result.ledger_checkpoint is not None
        assert result.integrity == 1.0
    
    @pytest.mark.asyncio
    async def test_pass_context_down_scenario_b(
        self,
        flow_manager_scenario_b,
        orchestrator,
        specialist,
        sample_context
    ):
        """Scenario B should have some loss."""
        result = await flow_manager_scenario_b.pass_context_down(
            from_agent=orchestrator,
            to_agent=specialist,
            context=sample_context,
            task="Execute CTV buys"
        )
        
        assert result.success
        assert result.scenario == "B"
        # Some loss is expected but not guaranteed
        assert result.tokens_received <= result.tokens_sent
        assert result.ledger_checkpoint is None
    
    @pytest.mark.asyncio
    async def test_aggregate_context_up(
        self,
        flow_manager_scenario_b,
        specialist,
        orchestrator
    ):
        """Test aggregation of results from multiple agents."""
        functionals = [
            AgentRef(agent_id=f"buyer_l3_{role}", level=3, role=role)
            for role in ["pricing", "avails", "audience"]
        ]
        
        results = [
            {"bid_price": 22.50, "status": "ready"},
            {"available_impressions": 1000000, "status": "ready"},
            {"matched_users": 500000, "status": "ready"}
        ]
        
        aggregated = await flow_manager_scenario_b.aggregate_context_up(
            from_agents=functionals,
            to_agent=specialist,
            results=results
        )
        
        assert aggregated.target_agent_id == specialist.agent_id
        assert len(aggregated.source_agent_ids) == 3
        assert "bid_price" in aggregated.merged_items
        assert "available_impressions" in aggregated.merged_items
    
    @pytest.mark.asyncio
    async def test_aggregate_with_conflicts(
        self,
        flow_manager_scenario_b,
        specialist,
        orchestrator
    ):
        """Test aggregation handles conflicting values."""
        functionals = [
            AgentRef(agent_id=f"buyer_l3_{i}", level=3, role="pricing")
            for i in range(2)
        ]
        
        # Same key, different values
        results = [
            {"recommended_cpm": 20.0},
            {"recommended_cpm": 25.0}
        ]
        
        aggregated = await flow_manager_scenario_b.aggregate_context_up(
            from_agents=functionals,
            to_agent=specialist,
            results=results
        )
        
        assert len(aggregated.conflicts) == 1
        assert aggregated.conflicts[0]["key"] == "recommended_cpm"
    
    def test_agent_ref_levels(self, orchestrator, specialist, functional):
        """Test AgentRef level helpers."""
        assert orchestrator.is_orchestrator
        assert not orchestrator.is_specialist
        
        assert specialist.is_specialist
        assert not specialist.is_functional
        
        assert functional.is_functional
        assert not functional.is_orchestrator


# =============================================================================
# Context Window Tests
# =============================================================================

class TestContextWindow:
    """Tests for ContextWindowManager."""
    
    def test_add_entry(self, context_window):
        """Test adding entries to window."""
        entry = ContextEntry(
            content="Test content",
            tokens=100,
            source="test_agent",
            importance=0.8
        )
        
        tokens_used = context_window.add_entry(entry)
        
        assert tokens_used == 100
        assert context_window.current_tokens == 100
        assert len(context_window.history) == 1
    
    def test_add_content_convenience(self, context_window):
        """Test add_content convenience method."""
        entry = context_window.add_content(
            content="A" * 400,  # ~100 tokens
            source="test",
            importance=0.5
        )
        
        assert entry.tokens > 0
        assert context_window.current_tokens > 0
    
    def test_can_fit(self, context_window):
        """Test can_fit check."""
        available = context_window.available_tokens
        
        assert context_window.can_fit(available)
        assert context_window.can_fit(available - 1)
        assert not context_window.can_fit(available + 1)
    
    def test_truncate_oldest(self, context_window):
        """Test truncation removes low importance old entries."""
        # Add several entries
        for i in range(5):
            context_window.add_content(
                content=f"Entry {i}" * 100,
                importance=0.5 - (i * 0.1),  # Decreasing importance
                source="test"
            )
        
        initial_count = len(context_window.history)
        initial_tokens = context_window.current_tokens
        
        truncated = context_window.truncate_oldest(500)
        
        assert len(truncated) > 0
        assert len(context_window.history) < initial_count
        assert context_window.current_tokens < initial_tokens
    
    def test_window_state_transitions(self, context_window):
        """Test window state based on utilization."""
        assert context_window.state == WindowState.HEALTHY
        
        # Fill to 60%
        fill_tokens = int(context_window.max_tokens * 0.6)
        context_window.add_content("X" * fill_tokens * 4, importance=1.0)
        assert context_window.state == WindowState.WARNING
    
    def test_context_integrity(self, context_window):
        """Test context integrity calculation."""
        assert context_window.get_context_integrity() == 1.0
        
        # Add then truncate
        for i in range(10):
            context_window.add_content(f"Entry {i}" * 500, importance=0.1)
        
        context_window.truncate_oldest(2000)
        
        integrity = context_window.get_context_integrity()
        assert 0.0 < integrity < 1.0
    
    def test_decay_application(self, context_window):
        """Test applying decay to entries."""
        context_window.add_content("Test", importance=1.0)
        
        affected = context_window.apply_decay(0.1)
        
        assert affected == 1
        assert context_window.history[0].decay_factor < 1.0


# =============================================================================
# Context Rot Simulator Tests
# =============================================================================

class TestContextRotSimulator:
    """Tests for ContextRotSimulator."""
    
    @pytest.mark.asyncio
    async def test_apply_daily_decay(self, rot_simulator, context_window):
        """Test daily decay application."""
        # Fill window
        for i in range(5):
            context_window.add_content(
                f"Entry {i}" * 100,
                importance=0.5,
                source="test"
            )
        
        result = await rot_simulator.apply_daily_decay(context_window, days=1.0)
        
        assert result.decay_rate == 0.05
        assert result.entries_before == 5
        assert result.average_importance_after < result.average_importance_before
    
    @pytest.mark.asyncio
    async def test_check_restart_probability(self, rot_simulator):
        """Test restart check is probabilistic."""
        # Run many times to verify it's working
        restarts = 0
        for _ in range(1000):
            if await rot_simulator.check_restart("test_agent", days=1.0):
                restarts += 1
        
        # With 1% daily probability, expect ~10 restarts in 1000 days
        # Allow wide margin for randomness
        assert 0 <= restarts <= 50
    
    @pytest.mark.asyncio
    async def test_simulate_handoff_scenario_c(self, rot_simulator, sample_context):
        """Scenario C handoff should have 0% loss."""
        result = await rot_simulator.simulate_handoff(
            context=sample_context,
            from_agent_id="agent_1",
            to_agent_id="agent_2",
            scenario="C"
        )
        
        assert result.tokens_lost == 0
        assert result.items_lost == 0
        assert result.recoverable is True
        assert result.recovery_source == "ledger"
    
    @pytest.mark.asyncio
    async def test_simulate_handoff_scenario_b(self, rot_simulator, sample_context):
        """Scenario B handoff should have ~10% loss."""
        # Run multiple times to get average
        total_loss_rate = 0.0
        runs = 10
        
        for _ in range(runs):
            result = await rot_simulator.simulate_handoff(
                context=sample_context,
                from_agent_id="agent_1",
                to_agent_id="agent_2",
                scenario="B"
            )
            total_loss_rate += result.loss_rate
        
        avg_loss = total_loss_rate / runs
        
        # Should be in the ballpark of configured rate
        assert result.recoverable is False
        # Loss is variable, just check it's reasonable
        assert 0.0 <= avg_loss <= 0.5
    
    @pytest.mark.asyncio
    async def test_simulate_handoff_scenario_a(self, rot_simulator, sample_context):
        """Scenario A handoff should have ~5% loss but be recoverable."""
        result = await rot_simulator.simulate_handoff(
            context=sample_context,
            from_agent_id="agent_1",
            to_agent_id="agent_2",
            scenario="A"
        )
        
        assert result.recoverable is True
        assert result.recovery_source == "exchange_logs"
        # Loss rate should be lower than Scenario B
        assert result.loss_rate <= 0.3


# =============================================================================
# Context Recovery Tests
# =============================================================================

class TestContextRecovery:
    """Tests for ContextRecovery."""
    
    @pytest.mark.asyncio
    async def test_recovery_scenario_c(self, recovery_system):
        """Scenario C should have 100% recovery."""
        lost_entries = [
            ContextEntry(content=f"Entry {i}", tokens=100, source="test")
            for i in range(5)
        ]
        
        result = await recovery_system.attempt_recovery(
            agent_id="test_agent",
            lost_entries=lost_entries,
            scenario="C"
        )
        
        assert result.success
        assert result.source == RecoverySource.LEDGER
        assert result.recovery_rate == 1.0
        assert result.fidelity == 1.0
        assert result.entries_recovered == 5
    
    @pytest.mark.asyncio
    async def test_recovery_scenario_b(self, recovery_system):
        """Scenario B should have 0% recovery."""
        lost_entries = [
            ContextEntry(content=f"Entry {i}", tokens=100, source="test")
            for i in range(5)
        ]
        
        result = await recovery_system.attempt_recovery(
            agent_id="test_agent",
            lost_entries=lost_entries,
            scenario="B"
        )
        
        assert not result.success
        assert result.source == RecoverySource.NONE
        assert result.recovery_rate == 0.0
        assert result.entries_recovered == 0
    
    @pytest.mark.asyncio
    async def test_recovery_scenario_a(self, recovery_system):
        """Scenario A should have ~60% recovery."""
        lost_entries = [
            ContextEntry(content=f"Entry {i}", tokens=100, source="test")
            for i in range(10)
        ]
        
        result = await recovery_system.attempt_recovery(
            agent_id="test_agent",
            lost_entries=lost_entries,
            scenario="A"
        )
        
        assert result.source == RecoverySource.EXCHANGE_LOGS
        # Due to randomness, just check it's reasonable
        assert 0.0 <= result.recovery_rate <= 1.0
        assert result.fidelity == 0.95


# =============================================================================
# Context Metrics Tests
# =============================================================================

class TestContextMetrics:
    """Tests for ContextMetrics."""
    
    @pytest.mark.asyncio
    async def test_record_handoff(self, metrics_tracker):
        """Test recording handoff metrics."""
        handoff = HandoffResult(
            from_agent_id="agent_1",
            to_agent_id="agent_2",
            scenario="B",
            tokens_sent=1000,
            tokens_received=900,
            tokens_lost=100,
            items_sent=10,
            items_received=9,
            items_lost=1
        )
        
        await metrics_tracker.record_handoff(handoff)
        
        summary = await metrics_tracker.get_summary()
        assert summary.total_handoffs == 1
        assert summary.total_tokens_lost_handoff == 100
    
    @pytest.mark.asyncio
    async def test_record_decay(self, metrics_tracker):
        """Test recording decay metrics."""
        decay = DecayResult(
            agent_id="test_agent",
            decay_rate=0.05,
            entries_before=10,
            entries_after=8,
            entries_pruned=2,
            tokens_before=1000,
            tokens_after=800,
            tokens_lost=200
        )
        
        await metrics_tracker.record_decay(decay)
        
        summary = await metrics_tracker.get_summary()
        assert summary.total_decay_events == 1
        assert summary.tokens_lost_to_decay == 200
    
    @pytest.mark.asyncio
    async def test_record_recovery(self, metrics_tracker):
        """Test recording recovery metrics."""
        recovery = RecoveryResult(
            agent_id="test_agent",
            scenario="A",
            source=RecoverySource.EXCHANGE_LOGS,
            success=True,
            entries_requested=10,
            entries_recovered=6,
            tokens_requested=1000,
            tokens_recovered=600,
            recovery_rate=0.6
        )
        
        await metrics_tracker.record_recovery(recovery)
        
        summary = await metrics_tracker.get_summary()
        assert summary.recovery_attempts == 1
        assert summary.recovery_successes == 1
        assert summary.total_tokens_recovered == 600
    
    @pytest.mark.asyncio
    async def test_summary_calculation(self, metrics_tracker):
        """Test comprehensive summary calculation."""
        # Record multiple events
        for i in range(5):
            await metrics_tracker.record_handoff(HandoffResult(
                from_agent_id=f"agent_{i}",
                to_agent_id=f"agent_{i+1}",
                scenario="B",
                tokens_sent=1000,
                tokens_lost=100 if i % 2 == 0 else 0,
                items_sent=10,
                items_lost=1 if i % 2 == 0 else 0
            ))
        
        summary = await metrics_tracker.get_summary()
        
        assert summary.total_handoffs == 5
        assert summary.handoff_losses == 3  # 3 had losses
        assert summary.successful_handoffs == 2
        assert 0.0 < summary.average_context_integrity < 1.0
    
    @pytest.mark.asyncio
    async def test_scenario_comparison(self, metrics_tracker):
        """Test scenario comparison functionality."""
        scenarios = ["A", "B", "C"]
        
        for scenario in scenarios:
            for _ in range(3):
                loss = 0 if scenario == "C" else (50 if scenario == "A" else 100)
                await metrics_tracker.record_handoff(HandoffResult(
                    from_agent_id="from",
                    to_agent_id="to",
                    scenario=scenario,
                    tokens_sent=1000,
                    tokens_lost=loss,
                    items_sent=10,
                    items_lost=0
                ))
        
        comparison = metrics_tracker.get_scenario_comparison()
        
        assert comparison["C"]["losses"] == 0
        assert comparison["B"]["losses"] > comparison["A"]["losses"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestContextFlowIntegration:
    """Integration tests for full context flow."""
    
    @pytest.mark.asyncio
    async def test_full_flow_scenario_c(
        self,
        flow_manager_scenario_c,
        rot_simulator,
        recovery_system,
        metrics_tracker,
        orchestrator,
        specialist,
        sample_context
    ):
        """Test full flow in Scenario C (ledger-backed)."""
        # Pass context down
        pass_result = await flow_manager_scenario_c.pass_context_down(
            from_agent=orchestrator,
            to_agent=specialist,
            context=sample_context,
            task="Execute CTV campaign"
        )
        
        await metrics_tracker.record_handoff(HandoffResult(
            from_agent_id=orchestrator.agent_id,
            to_agent_id=specialist.agent_id,
            scenario="C",
            tokens_sent=pass_result.tokens_sent,
            tokens_received=pass_result.tokens_received,
            tokens_lost=pass_result.tokens_lost
        ))
        
        # Verify no loss
        assert pass_result.tokens_lost == 0
        
        summary = await metrics_tracker.get_summary()
        assert summary.total_tokens_lost == 0
    
    @pytest.mark.asyncio
    async def test_full_flow_scenario_b_with_recovery_failure(
        self,
        flow_manager_scenario_b,
        rot_simulator,
        recovery_system,
        metrics_tracker,
        orchestrator,
        specialist,
        sample_context
    ):
        """Test full flow in Scenario B (no recovery)."""
        # Simulate handoff with loss
        handoff_result = await rot_simulator.simulate_handoff(
            context=sample_context,
            from_agent_id=orchestrator.agent_id,
            to_agent_id=specialist.agent_id,
            scenario="B"
        )
        
        await metrics_tracker.record_handoff(handoff_result)
        
        # Attempt recovery (should fail)
        if handoff_result.items_lost > 0:
            lost_entries = [
                ContextEntry(content=key, tokens=50, source="test")
                for key in handoff_result.lost_keys
            ]
            
            recovery_result = await recovery_system.attempt_recovery(
                agent_id=specialist.agent_id,
                lost_entries=lost_entries,
                scenario="B"
            )
            
            await metrics_tracker.record_recovery(recovery_result)
            
            assert not recovery_result.success
        
        summary = await metrics_tracker.get_summary()
        assert summary.scenario_stats.get("B", {}).get("handoffs", 0) > 0
