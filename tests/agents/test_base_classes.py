"""
Tests for multi-agent hierarchy base classes.

Tests context management, state management, and agent base classes
without requiring actual Anthropic API calls.
"""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base import (
    # Context
    AgentContext,
    ContextItem,
    ContextPriority,
    ContextWindow,
    StandardContextPassing,
    measure_degradation,
    # State
    AgentState,
    StateSnapshot,
    StateManager,
    VolatileStateBackend,
    FileStateBackend,
    # Agents
    FunctionalAgent,
    FunctionalAgentState,
    ToolDefinition,
    ToolResult,
    SpecialistAgent,
    SpecialistAgentState,
    DelegationRequest,
    OrchestratorAgent,
    OrchestratorAgentState,
    CampaignState,
    SpecialistAssignment,
)


# ============================================================================
# Context Tests
# ============================================================================

class TestAgentContext:
    """Tests for AgentContext."""
    
    def test_create_context(self):
        """Test basic context creation."""
        ctx = AgentContext(
            source_level=1,
            source_agent_id="test-agent",
            task_description="Test task"
        )
        
        assert ctx.source_level == 1
        assert ctx.source_agent_id == "test-agent"
        assert ctx.task_description == "Test task"
        assert ctx.current_tokens == 0
    
    def test_add_item_within_budget(self):
        """Test adding items within token budget."""
        ctx = AgentContext(token_budget=1000)
        
        result = ctx.add_item(
            key="test_key",
            value="test_value",
            priority=ContextPriority.HIGH
        )
        
        assert result is True
        assert "test_key" in ctx.items
        assert ctx.items["test_key"].value == "test_value"
        assert ctx.current_tokens > 0
    
    def test_add_item_exceeds_budget(self):
        """Test adding item that exceeds budget."""
        ctx = AgentContext(token_budget=10)
        
        # Large value should exceed budget
        result = ctx.add_item(
            key="large_key",
            value="x" * 1000,
            priority=ContextPriority.LOW
        )
        
        assert result is False
        assert "large_key" not in ctx.items
    
    def test_get_by_priority(self):
        """Test getting items by priority."""
        ctx = AgentContext(token_budget=5000)
        
        ctx.add_item("critical1", "val1", ContextPriority.CRITICAL)
        ctx.add_item("high1", "val2", ContextPriority.HIGH)
        ctx.add_item("medium1", "val3", ContextPriority.MEDIUM)
        ctx.add_item("low1", "val4", ContextPriority.LOW)
        
        critical = ctx.get_by_priority(ContextPriority.CRITICAL)
        assert "critical1" in critical
        assert len(critical) == 1
    
    def test_remove_item(self):
        """Test removing items."""
        ctx = AgentContext(token_budget=1000)
        ctx.add_item("key1", "value1", ContextPriority.MEDIUM)
        
        initial_tokens = ctx.current_tokens
        result = ctx.remove_item("key1")
        
        assert result is True
        assert "key1" not in ctx.items
        assert ctx.current_tokens < initial_tokens
    
    def test_to_prompt_dict(self):
        """Test converting to prompt dictionary."""
        ctx = AgentContext(
            task_description="Test task",
            task_constraints=["constraint1"],
            token_budget=1000
        )
        ctx.add_item("data", "value", ContextPriority.HIGH)
        
        prompt_dict = ctx.to_prompt_dict()
        
        assert prompt_dict["task"] == "Test task"
        assert "constraint1" in prompt_dict["constraints"]
        assert "data" in prompt_dict["context"]


class TestContextWindow:
    """Tests for ContextWindow."""
    
    def test_utilization(self):
        """Test token utilization tracking."""
        window = ContextWindow(max_tokens=10000)
        window.system_tokens = 1000
        window.context_tokens = 2000
        window.history_tokens = 3000
        
        assert window.used_tokens == 6000
        assert window.utilization == 0.6
    
    def test_needs_summarization(self):
        """Test summarization threshold."""
        window = ContextWindow(max_tokens=10000)
        window.history_tokens = 6000
        
        assert window.needs_summarization(threshold=0.5) is True
        assert window.needs_summarization(threshold=0.8) is False
    
    def test_record_usage(self):
        """Test recording usage."""
        window = ContextWindow(max_tokens=10000)
        window.record_usage(100, 50)
        
        assert len(window.usage_history) == 1
        assert window.history_tokens == 150


class TestStandardContextPassing:
    """Tests for context passing between levels."""
    
    def test_context_to_child_basic(self):
        """Test basic context passing to child."""
        passing = StandardContextPassing(
            degradation_rate=0.0,  # No degradation for predictable test
            child_budget_ratio=0.5
        )
        
        parent = AgentContext(
            source_level=1,
            source_agent_id="parent",
            token_budget=1000
        )
        parent.add_item("key1", "value1", ContextPriority.CRITICAL)
        
        child = passing.context_to_child(
            parent_context=parent,
            child_agent_id="child",
            task="Child task"
        )
        
        assert child.source_level == 2
        assert child.parent_context_id == parent.context_id
        assert child.token_budget == 500  # 50% of parent
        assert "key1" in child.items
    
    def test_context_degradation(self):
        """Test context degradation at handoff."""
        # High degradation rate
        passing = StandardContextPassing(
            degradation_rate=1.0,  # 100% drop rate for LOW
            child_budget_ratio=0.5
        )
        
        parent = AgentContext(source_level=1, token_budget=1000)
        parent.add_item("critical", "val", ContextPriority.CRITICAL)
        parent.add_item("low1", "val1", ContextPriority.LOW)
        parent.add_item("low2", "val2", ContextPriority.LOW)
        
        child = passing.context_to_child(
            parent_context=parent,
            child_agent_id="child",
            task="Test"
        )
        
        # Critical should survive, LOW items should be dropped
        assert "critical" in child.items
        assert "low1" not in child.items
        assert "low2" not in child.items
    
    def test_context_from_child(self):
        """Test merging child results back."""
        passing = StandardContextPassing()
        
        parent = AgentContext(source_level=1, token_budget=5000)
        child = AgentContext(
            source_level=2,
            target_agent_id="child-agent"
        )
        child.add_item("finding", "important", ContextPriority.CRITICAL)
        
        result = {"status": "success"}
        
        updated = passing.context_from_child(parent, child, result)
        
        # Result should be added to parent
        assert "child_result_child-agent" in updated.items
        # Critical findings should be promoted
        assert "child_finding" in updated.items


class TestMeasureDegradation:
    """Tests for degradation measurement."""
    
    def test_no_degradation(self):
        """Test measuring zero degradation."""
        parent = AgentContext(token_budget=1000)
        parent.add_item("key", "value", ContextPriority.HIGH)
        
        # Child has same items
        child = AgentContext(token_budget=1000)
        child.add_item("key", "value", ContextPriority.HIGH)
        
        metrics = measure_degradation(parent, child)
        
        assert metrics.item_loss_rate == 0.0
        assert metrics.critical_items_lost == 0
    
    def test_full_degradation(self):
        """Test measuring full item loss."""
        parent = AgentContext(token_budget=1000)
        parent.add_item("key1", "val1", ContextPriority.HIGH)
        parent.add_item("key2", "val2", ContextPriority.MEDIUM)
        
        child = AgentContext(token_budget=1000)  # Empty
        
        metrics = measure_degradation(parent, child)
        
        assert metrics.item_loss_rate == 1.0
        assert metrics.items_passed == 2
        assert metrics.items_received == 0


# ============================================================================
# State Tests
# ============================================================================

class TestAgentState:
    """Tests for AgentState."""
    
    def test_create_state(self):
        """Test basic state creation."""
        state = AgentState(
            agent_id="test-agent",
            agent_type="test",
            data={"key": "value"}
        )
        
        assert state.agent_id == "test-agent"
        assert state.agent_type == "test"
        assert state.get("key") == "value"
    
    def test_update_state(self):
        """Test state update."""
        state = AgentState(agent_id="test")
        state.data["count"] = 0
        
        initial_version = state.version.version
        state.update(count=5)
        
        assert state.data["count"] == 5
        assert state.version.version == initial_version + 1
    
    def test_to_snapshot(self):
        """Test creating snapshot."""
        state = AgentState(
            agent_id="test",
            data={"important": "data"}
        )
        
        snapshot = state.to_snapshot()
        
        assert snapshot.state_id == state.state_id
        assert snapshot.agent_id == state.agent_id
        assert "important" in snapshot.state_data.get("data", {})
    
    def test_snapshot_restore(self):
        """Test restoring from snapshot."""
        original = AgentState(
            agent_id="test",
            data={"key": "value"}
        )
        
        snapshot = original.to_snapshot()
        restored = snapshot.restore()
        
        assert restored.agent_id == original.agent_id
        assert restored.get("key") == "value"


class TestVolatileStateBackend:
    """Tests for volatile (in-memory) state backend."""
    
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading state."""
        backend = VolatileStateBackend()
        state = AgentState(agent_id="test", data={"x": 1})
        
        await backend.save(state)
        loaded = await backend.load(state.state_id)
        
        assert loaded is not None
        assert loaded.agent_id == "test"
        assert loaded.get("x") == 1
    
    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        """Test loading non-existent state."""
        backend = VolatileStateBackend()
        loaded = await backend.load("nonexistent")
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing volatile state."""
        backend = VolatileStateBackend()
        state = AgentState(agent_id="test")
        
        await backend.save(state)
        backend.clear()
        
        loaded = await backend.load(state.state_id)
        assert loaded is None


class TestStateManager:
    """Tests for StateManager."""
    
    @pytest.mark.asyncio
    async def test_initialize_state(self):
        """Test state initialization."""
        backend = VolatileStateBackend()
        manager = StateManager(backend)
        
        state = await manager.initialize_state(
            agent_id="agent-1",
            agent_type="test",
            initial_data={"init": True}
        )
        
        assert state.agent_id == "agent-1"
        assert state.get("init") is True
    
    @pytest.mark.asyncio
    async def test_auto_snapshot(self):
        """Test automatic snapshotting."""
        backend = VolatileStateBackend()
        manager = StateManager(backend, snapshot_interval=3)
        
        state = await manager.initialize_state("agent", "test")
        
        # Update multiple times
        for i in range(5):
            await manager.update_state(state, count=i)
        
        snapshots = await backend.list_snapshots(state.state_id)
        assert len(snapshots) >= 1  # At least one auto-snapshot
    
    @pytest.mark.asyncio
    async def test_create_recovery_point(self):
        """Test creating recovery point."""
        backend = VolatileStateBackend()
        manager = StateManager(backend)
        
        state = await manager.initialize_state("agent", "test")
        
        snapshot = await manager.create_snapshot(
            state,
            description="Recovery point",
            recovery_point=True
        )
        
        assert snapshot.recovery_point is True
        
        latest = await manager.get_latest_recovery_point(state.state_id)
        assert latest is not None
        assert latest.snapshot_id == snapshot.snapshot_id


# ============================================================================
# Agent Tests (Mocked)
# ============================================================================

class MockFunctionalAgent(FunctionalAgent):
    """Mock functional agent for testing."""
    
    def _register_tools(self):
        self.register_tool(
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"input": {"type": "string"}},
                required_params=["input"]
            ),
            handler=self._handle_test_tool
        )
    
    def get_system_prompt(self) -> str:
        return "You are a test functional agent."
    
    def _handle_test_tool(self, input: str) -> str:
        return f"Processed: {input}"


class MockSpecialistAgent(SpecialistAgent):
    """Mock specialist agent for testing."""
    
    def get_system_prompt(self) -> str:
        return "You are a test specialist agent."
    
    def get_channel_capabilities(self) -> list[str]:
        return ["capability1", "capability2"]
    
    async def plan_execution(self, context, objective):
        # Return empty list for tests
        return []


class MockOrchestratorAgent(OrchestratorAgent):
    """Mock orchestrator agent for testing."""
    
    def get_system_prompt(self) -> str:
        return "You are a test orchestrator agent."
    
    async def make_strategic_decision(self, context, decision_request):
        from src.agents.base import StrategicDecision
        return StrategicDecision(
            decision_type="test",
            description="Test decision",
            rationale="For testing"
        )
    
    async def plan_specialist_assignments(self, context, campaign):
        return []


class TestFunctionalAgent:
    """Tests for FunctionalAgent base class."""
    
    def test_tool_registration(self):
        """Test tool registration."""
        agent = MockFunctionalAgent(name="TestAgent")
        
        assert "test_tool" in agent._tools
        assert "test_tool" in agent._tool_handlers
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test agent initialization."""
        agent = MockFunctionalAgent(name="TestAgent")
        await agent.initialize()
        
        assert agent.state is not None
        assert agent.state.agent_type == "functional"
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test direct tool execution."""
        agent = MockFunctionalAgent(name="TestAgent")
        
        result = await agent._execute_tool("test_tool", {"input": "hello"})
        
        assert result.success is True
        assert result.result == "Processed: hello"
    
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Test handling unknown tool."""
        agent = MockFunctionalAgent(name="TestAgent")
        
        result = await agent._execute_tool("nonexistent", {})
        
        assert result.success is False
        assert "Unknown tool" in result.error


class TestSpecialistAgent:
    """Tests for SpecialistAgent base class."""
    
    def test_creation(self):
        """Test specialist creation."""
        agent = MockSpecialistAgent(
            name="TestSpecialist",
            channel="test_channel"
        )
        
        assert agent.name == "TestSpecialist"
        assert agent.channel == "test_channel"
    
    def test_register_functional(self):
        """Test registering functional agents."""
        specialist = MockSpecialistAgent(name="Specialist", channel="test")
        functional = MockFunctionalAgent(name="Functional")
        
        specialist.register_functional_agent(functional)
        
        assert functional.agent_id in specialist._functional_agents
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test specialist initialization."""
        agent = MockSpecialistAgent(name="TestSpecialist", channel="test")
        await agent.initialize()
        
        assert agent.state is not None
        assert agent.state.channel == "test"


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent base class."""
    
    def test_creation(self):
        """Test orchestrator creation."""
        agent = MockOrchestratorAgent(name="TestOrchestrator")
        
        assert agent.name == "TestOrchestrator"
        assert agent.model == "claude-opus-4-20250514"
    
    def test_register_specialist(self):
        """Test registering specialists."""
        orchestrator = MockOrchestratorAgent(name="Orchestrator")
        specialist = MockSpecialistAgent(name="Specialist", channel="display")
        
        orchestrator.register_specialist(specialist)
        
        assert specialist.agent_id in orchestrator._specialists
        assert orchestrator._channel_specialists["display"] == specialist.agent_id
    
    @pytest.mark.asyncio
    async def test_add_campaign(self):
        """Test adding campaigns."""
        orchestrator = MockOrchestratorAgent(name="Orchestrator")
        
        campaign = CampaignState(
            campaign_id="camp-1",
            name="Test Campaign",
            budget_total=10000.0,
            objectives=["awareness", "conversion"]
        )
        
        await orchestrator.add_campaign(campaign)
        
        assert "camp-1" in orchestrator.state.campaigns
        assert orchestrator.state.total_budget == 10000.0
    
    @pytest.mark.asyncio
    async def test_get_portfolio_status(self):
        """Test portfolio status."""
        orchestrator = MockOrchestratorAgent(name="Orchestrator")
        await orchestrator.initialize()
        
        status = await orchestrator.get_portfolio_status()
        
        assert status["orchestrator_name"] == "Orchestrator"
        assert "campaigns" in status


class TestCampaignState:
    """Tests for CampaignState."""
    
    def test_budget_initialization(self):
        """Test budget auto-initialization."""
        campaign = CampaignState(
            campaign_id="camp-1",
            name="Test",
            budget_total=5000.0
        )
        
        assert campaign.budget_remaining == 5000.0
        assert campaign.budget_spent == 0.0
    
    def test_channel_allocations(self):
        """Test channel allocation tracking."""
        campaign = CampaignState(
            campaign_id="camp-1",
            name="Test",
            budget_total=10000.0,
            channel_allocations={
                "display": 0.4,
                "video": 0.3,
                "mobile": 0.3
            }
        )
        
        assert sum(campaign.channel_allocations.values()) == 1.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestAgentHierarchyIntegration:
    """Integration tests for the agent hierarchy."""
    
    @pytest.mark.asyncio
    async def test_context_flow_l1_to_l2(self):
        """Test context flowing from orchestrator to specialist."""
        passing = StandardContextPassing(degradation_rate=0.0)
        
        # L1 creates context
        l1_context = AgentContext(
            source_level=1,
            source_agent_id="orchestrator",
            token_budget=10000,
            task_description="Campaign execution"
        )
        l1_context.add_item("budget", 50000, ContextPriority.CRITICAL)
        l1_context.add_item("target_audience", "millennials", ContextPriority.HIGH)
        
        # Pass to L2
        l2_context = passing.context_to_child(
            l1_context,
            child_agent_id="specialist",
            task="Execute display campaign"
        )
        
        assert l2_context.source_level == 2
        assert l2_context.get_item("budget") == 50000
        assert l2_context.task_description == "Execute display campaign"
    
    @pytest.mark.asyncio
    async def test_context_flow_l2_to_l3(self):
        """Test context flowing from specialist to functional."""
        passing = StandardContextPassing(
            degradation_rate=0.1,
            child_budget_ratio=0.6
        )
        
        # L2 creates context
        l2_context = AgentContext(
            source_level=2,
            source_agent_id="specialist",
            token_budget=7000,
            task_description="Display channel execution"
        )
        l2_context.add_item("channel", "display", ContextPriority.CRITICAL)
        l2_context.add_item("format", "300x250", ContextPriority.HIGH)
        
        # Pass to L3
        l3_context = passing.context_to_child(
            l2_context,
            child_agent_id="functional",
            task="Execute product search"
        )
        
        assert l3_context.source_level == 3
        assert l3_context.token_budget == int(7000 * 0.6)  # 4200
    
    @pytest.mark.asyncio
    async def test_full_hierarchy_initialization(self):
        """Test initializing complete hierarchy."""
        orchestrator = MockOrchestratorAgent(name="Portfolio Manager")
        specialist = MockSpecialistAgent(name="Display Specialist", channel="display")
        functional = MockFunctionalAgent(name="Product Search")
        
        # Wire up hierarchy
        specialist.register_functional_agent(functional)
        orchestrator.register_specialist(specialist)
        
        # Initialize
        await orchestrator.initialize()
        
        # Verify all initialized
        assert orchestrator.state is not None
        assert specialist.state is not None
        assert functional.state is not None
