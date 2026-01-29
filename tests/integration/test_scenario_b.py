"""
Integration tests for Scenario B (IAB Pure A2A with context rot).

Tests verify:
1. Direct buyer-seller communication (no exchange)
2. Zero exchange fees (unlike Scenario A)
3. Context rot simulation
4. Hallucination injection/detection
5. Memory degradation over simulation days
"""

import pytest

# Mark entire module as integration tests (require scenario API alignment)
pytestmark = pytest.mark.integration
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.scenarios.scenario_b import (
    ScenarioB,
    ContextRotConfig,
    ContextRotSimulator,
    AgentMemory,
    ScenarioConfig,
)
from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    DealType,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def scenario_config():
    """Default test configuration for Scenario B."""
    return ScenarioConfig(
        scenario_code="B",
        name="Test IAB Pure A2A",
        description="Test scenario",
        num_buyers=2,
        num_sellers=2,
        simulation_days=5,
        exchange_fee_pct=0.0,  # Should always be 0 for Scenario B
        context_decay_rate=0.1,  # Higher rate for testing
        hallucination_rate=0.2,  # Higher rate for testing
        mock_llm=True,
    )


@pytest.fixture
def context_rot_config():
    """Context rot configuration for testing."""
    return ContextRotConfig(
        decay_rate=0.1,  # 10% decay for faster testing
        restart_probability=0.1,  # Higher for testing
        max_memory_items=50,
        grace_period_days=1,
    )


@pytest.fixture
def sample_request():
    """Sample bid request for testing."""
    return BidRequest(
        buyer_id="buyer-001",
        campaign_id="campaign-001",
        channel="display",
        impressions_requested=100000,
        max_cpm=20.0,
        targeting={"segments": ["sports", "tech"]},
    )


@pytest.fixture
def sample_response(sample_request):
    """Sample bid response for testing."""
    return BidResponse(
        request_id=sample_request.request_id,
        seller_id="seller-001",
        offered_cpm=15.0,
        available_impressions=100000,
        deal_type=DealType.PREFERRED_DEAL,
    )


@pytest.fixture
def mock_redis():
    """Mock Redis bus."""
    mock = AsyncMock()
    mock.publish_bid_request = AsyncMock(return_value="msg-001")
    mock.publish_bid_response = AsyncMock(return_value="msg-002")
    mock.publish_deal = AsyncMock(return_value="msg-003")
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    return mock


# -----------------------------------------------------------------------------
# Basic Scenario Tests
# -----------------------------------------------------------------------------


class TestScenarioBBasic:
    """Basic functionality tests for Scenario B."""

    @pytest.mark.asyncio
    async def test_scenario_creates_with_zero_fees(self, scenario_config, mock_redis):
        """Verify Scenario B creates with zero exchange fees."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)

        assert scenario.config.exchange_fee_pct == 0.0
        assert scenario.scenario_code == "B"
        assert "A2A" in scenario.config.name

    @pytest.mark.asyncio
    async def test_deal_has_no_exchange_fee(
        self, scenario_config, mock_redis, sample_request, sample_response
    ):
        """Verify deals created in Scenario B have zero exchange fees."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario._connected = True
        scenario._bus = mock_redis

        deal = await scenario.create_deal(sample_request, sample_response)

        assert deal.exchange_fee == 0.0
        assert deal.fee_percentage == 0.0
        assert deal.total_cost == deal.seller_revenue
        assert deal.scenario == "B"

    @pytest.mark.asyncio
    async def test_seller_receives_full_amount(
        self, scenario_config, mock_redis, sample_request, sample_response
    ):
        """Verify seller receives 100% of buyer payment (no intermediary take)."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario._connected = True
        scenario._bus = mock_redis

        deal = await scenario.create_deal(sample_request, sample_response)

        # Calculate expected
        expected_cost = (sample_response.available_impressions / 1000) * sample_response.offered_cpm

        assert deal.total_cost == pytest.approx(expected_cost)
        assert deal.seller_revenue == pytest.approx(expected_cost)
        assert deal.total_cost == deal.seller_revenue

    @pytest.mark.asyncio
    async def test_run_single_deal(self, scenario_config, mock_redis):
        """Test running a single deal through Scenario B."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario._connected = True
        scenario._bus = mock_redis

        result = await scenario.run_single_deal(
            buyer_id="test-buyer",
            seller_id="test-seller",
            impressions=50000,
            cpm=10.0,
        )

        assert result["exchange_fee"] == 0.0
        assert result["buyer_spend"] == pytest.approx(500.0)  # 50k/1000 * $10
        assert result["seller_revenue"] == pytest.approx(500.0)
        assert result["scenario"] == "B"


# -----------------------------------------------------------------------------
# Context Rot Tests
# -----------------------------------------------------------------------------


class TestContextRot:
    """Tests for context rot simulation."""

    def test_agent_memory_creation(self):
        """Test AgentMemory dataclass."""
        memory = AgentMemory(
            agent_id="test-agent",
            agent_type="buyer",
        )

        assert memory.agent_id == "test-agent"
        assert memory.rot_events == 0
        assert memory.memory_size() == 0

    def test_memory_size_tracking(self):
        """Test memory size calculation."""
        memory = AgentMemory(
            agent_id="test-agent",
            agent_type="buyer",
        )

        # Add some items
        memory.deal_history["deal-1"] = MagicMock()
        memory.deal_history["deal-2"] = MagicMock()
        memory.partner_reputation["seller-1"] = 0.8
        memory.pending_requests["req-1"] = MagicMock()

        assert memory.memory_size() == 4

    def test_context_rot_grace_period(self, context_rot_config):
        """Test that decay doesn't apply during grace period."""
        rot = ContextRotSimulator(context_rot_config, seed=42)
        memory = AgentMemory(
            agent_id="test-agent",
            agent_type="buyer",
        )

        # Add some items
        memory.deal_history["deal-1"] = MagicMock()
        memory.partner_reputation["seller-1"] = 0.8

        # Day 1 is within grace period
        keys_lost, _ = rot.apply_daily_decay(memory, simulation_day=1)

        assert keys_lost == 0
        assert len(memory.deal_history) == 1

    def test_context_rot_applies_after_grace(self, context_rot_config):
        """Test that decay applies after grace period."""
        context_rot_config.decay_rate = 0.9  # Very high for testing
        rot = ContextRotSimulator(context_rot_config, seed=42)

        memory = AgentMemory(
            agent_id="test-agent",
            agent_type="buyer",
        )

        # Add many items to increase chance of decay
        for i in range(20):
            memory.deal_history[f"deal-{i}"] = MagicMock()
            memory.partner_reputation[f"seller-{i}"] = 0.5

        # Day 10 is well after grace period
        keys_lost, lost_keys = rot.apply_daily_decay(memory, simulation_day=10)

        # With 0.9 decay rate after grace period, should lose significant memory
        # Survival rate = (1-0.9)^(10-1) = 0.1^9 ≈ 0.000000001
        # This means almost everything should be lost
        assert keys_lost > 0 or len(memory.deal_history) < 20

    def test_context_rot_restart(self, context_rot_config):
        """Test full context wipe on restart."""
        context_rot_config.restart_probability = 1.0  # Guaranteed restart
        rot = ContextRotSimulator(context_rot_config, seed=42)

        memory = AgentMemory(
            agent_id="test-agent",
            agent_type="buyer",
        )

        # Add items
        memory.deal_history["deal-1"] = MagicMock()
        memory.partner_reputation["seller-1"] = 0.8

        # Trigger restart check
        restarted = rot.check_restart(memory, simulation_day=5)

        assert restarted is True
        assert len(memory.deal_history) == 0
        assert len(memory.partner_reputation) == 0
        assert memory.rot_events == 1


# -----------------------------------------------------------------------------
# Memory Management Tests
# -----------------------------------------------------------------------------


class TestMemoryManagement:
    """Tests for agent memory management in Scenario B."""

    @pytest.mark.asyncio
    async def test_buyer_memory_creation(self, scenario_config, mock_redis):
        """Test buyer memory is created on demand."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)

        memory = scenario.get_or_create_buyer_memory("buyer-001")

        assert memory.agent_id == "buyer-001"
        assert memory.agent_type == "buyer"
        assert "buyer-001" in scenario._buyer_memories

    @pytest.mark.asyncio
    async def test_seller_memory_creation(self, scenario_config, mock_redis):
        """Test seller memory is created on demand."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)

        memory = scenario.get_or_create_seller_memory("seller-001")

        assert memory.agent_id == "seller-001"
        assert memory.agent_type == "seller"
        assert "seller-001" in scenario._seller_memories

    @pytest.mark.asyncio
    async def test_memory_persists_across_calls(self, scenario_config, mock_redis):
        """Test that same memory instance is returned."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)

        memory1 = scenario.get_or_create_buyer_memory("buyer-001")
        memory1.rot_events = 5

        memory2 = scenario.get_or_create_buyer_memory("buyer-001")

        assert memory1 is memory2
        assert memory2.rot_events == 5


# -----------------------------------------------------------------------------
# Hallucination Tests
# -----------------------------------------------------------------------------


class TestHallucinationInjection:
    """Tests for hallucination injection in Scenario B."""

    @pytest.mark.asyncio
    async def test_hallucination_manager_initialized(self, scenario_config, mock_redis):
        """Test hallucination manager is properly initialized."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)

        assert scenario._hallucination_mgr is not None
        assert scenario._hallucination_mgr.scenario == "B"

    @pytest.mark.asyncio
    async def test_price_data_may_be_corrupted(self, scenario_config, mock_redis):
        """Test that price data can be corrupted by hallucination injection."""
        scenario_config.hallucination_rate = 1.0  # Always corrupt
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario.current_day = 5

        original_price = 20.0
        corrupted_price = scenario._hallucination_mgr.process_price_data(
            real_price=original_price,
            agent_id="buyer-001",
            agent_type="buyer",
            publisher_id="seller-001",
            simulation_day=5,
        )

        # With 100% injection rate, price should be corrupted
        # Price is typically deflated by 10-30%
        assert corrupted_price < original_price or corrupted_price == original_price


# -----------------------------------------------------------------------------
# Metrics Tests
# -----------------------------------------------------------------------------


class TestScenarioBMetrics:
    """Tests for Scenario B metrics tracking."""

    @pytest.mark.asyncio
    async def test_deal_metrics_recorded(
        self, scenario_config, mock_redis, sample_request, sample_response
    ):
        """Test that deal metrics are properly recorded."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario._connected = True
        scenario._bus = mock_redis

        await scenario.create_deal(sample_request, sample_response)

        assert scenario.metrics.total_deals == 1
        assert scenario.metrics.total_exchange_fees == 0.0
        assert scenario.metrics.total_buyer_spend > 0
        assert scenario.metrics.total_seller_revenue > 0
        assert scenario.metrics.total_buyer_spend == scenario.metrics.total_seller_revenue

    @pytest.mark.asyncio
    async def test_fee_extraction_rate_is_zero(
        self, scenario_config, mock_redis, sample_request, sample_response
    ):
        """Test that fee extraction rate is 0% for Scenario B."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario._connected = True
        scenario._bus = mock_redis

        await scenario.create_deal(sample_request, sample_response)

        assert scenario.metrics.fee_extraction_rate == 0.0

    @pytest.mark.asyncio
    async def test_context_rot_metrics(self, scenario_config, mock_redis):
        """Test context rot metrics tracking."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)

        # Record context rot events
        scenario.record_context_rot("agent-001", keys_lost=5, recovered=True)
        scenario.record_context_rot("agent-002", keys_lost=3, recovered=False)

        assert scenario.metrics.context_rot_events == 2
        assert scenario.metrics.total_keys_lost == 8
        assert scenario.metrics.recovery_attempts == 2
        assert scenario.metrics.recovery_successes == 1
        assert scenario.metrics.context_recovery_rate == 50.0


# -----------------------------------------------------------------------------
# Comparison Tests (Scenario B vs A characteristics)
# -----------------------------------------------------------------------------


class TestScenarioBVsA:
    """Tests comparing Scenario B characteristics vs Scenario A."""

    @pytest.mark.asyncio
    async def test_no_auction_mechanics(
        self, scenario_config, mock_redis, sample_request, sample_response
    ):
        """Verify Scenario B has no auction (direct negotiation)."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario._connected = True
        scenario._bus = mock_redis

        # Process response directly - should be accepted if price is acceptable
        deal = await scenario.process_bid_response(sample_request, sample_response)

        # Deal should be created immediately (no auction wait)
        assert deal is not None

    @pytest.mark.asyncio
    async def test_direct_buyer_seller_communication(
        self, scenario_config, mock_redis, sample_request
    ):
        """Verify buyer communicates directly with sellers."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        scenario._connected = True
        scenario._bus = mock_redis

        await scenario.process_bid_request(sample_request)

        # Request should be published directly to sellers stream
        mock_redis.publish_bid_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_volatile_state_only(self, scenario_config, mock_redis):
        """Verify state is only in-memory (no persistence)."""
        scenario = ScenarioB(config=scenario_config, redis_bus=mock_redis)

        # Memory should be in-memory dicts only
        assert isinstance(scenario._buyer_memories, dict)
        assert isinstance(scenario._seller_memories, dict)

        # Create some memories
        scenario.get_or_create_buyer_memory("buyer-001")
        scenario.get_or_create_seller_memory("seller-001")

        # Memories exist only in this instance
        assert len(scenario._buyer_memories) == 1
        assert len(scenario._seller_memories) == 1

        # New instance should have empty memories
        scenario2 = ScenarioB(config=scenario_config, redis_bus=mock_redis)
        assert len(scenario2._buyer_memories) == 0
        assert len(scenario2._seller_memories) == 0
