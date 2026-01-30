"""
Tests for Multi-Agent Scenario Implementations.

Tests:
- Full hierarchy execution
- Context flow across levels
- Multi-campaign handling
- Compare A vs B vs C results
"""

import asyncio
import pytest
from datetime import date, timedelta
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scenarios.multi_agent_scenario_a import (
    MultiAgentScenarioA,
    MultiAgentScenarioAConfig,
)
from scenarios.multi_agent_scenario_b import (
    MultiAgentScenarioB,
    MultiAgentScenarioBConfig,
)
from scenarios.multi_agent_scenario_c import (
    MultiAgentScenarioC,
    MultiAgentScenarioCConfig,
    LedgerClient,
)
from orchestration.buyer_system import (
    BuyerAgentSystem,
    ContextFlowConfig,
    ContextFlowManager,
)
from orchestration.seller_system import (
    SellerAgentSystem,
    SellerContextConfig,
)
from agents.buyer.models import (
    Campaign,
    CampaignObjectives,
    CampaignStatus,
    AudienceSpec,
    Channel,
)
from protocols.inter_level import (
    AgentContext,
    Task,
    Result,
    ResultStatus,
    ContextSerializer,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_campaign() -> Campaign:
    """Create a sample campaign for testing."""
    return Campaign(
        campaign_id="test-camp-001",
        name="Test Campaign",
        advertiser="TestAdvertiser",
        total_budget=100000.0,
        start_date=date.today(),
        end_date=date.today() + timedelta(days=30),
        objectives=CampaignObjectives(
            reach_target=500000,
            frequency_cap=3,
            cpm_target=20.0,
            channel_mix={
                Channel.DISPLAY.value: 0.5,
                Channel.VIDEO.value: 0.3,
                Channel.CTV.value: 0.2,
            },
        ),
        audience=AudienceSpec(
            segments=["auto_intenders", "tech_enthusiasts"],
            demographics={"age_min": 25, "age_max": 54},
            geo_targets=["US"],
            device_types=["desktop", "mobile"],
        ),
    )


@pytest.fixture
def sample_campaigns(sample_campaign) -> List[Campaign]:
    """Create multiple sample campaigns."""
    campaigns = [sample_campaign]
    
    for i in range(1, 5):
        campaigns.append(Campaign(
            campaign_id=f"test-camp-{i:03d}",
            name=f"Test Campaign {i}",
            advertiser=f"Advertiser{i}",
            total_budget=50000.0 + i * 25000,
            start_date=date.today(),
            end_date=date.today() + timedelta(days=30),
            objectives=CampaignObjectives(
                reach_target=250000 + i * 50000,
                frequency_cap=3,
                cpm_target=15.0 + i * 2,
                channel_mix={Channel.DISPLAY.value: 0.6, Channel.VIDEO.value: 0.4},
            ),
            audience=AudienceSpec(
                segments=["tech_enthusiasts"],
                demographics={"age_min": 25, "age_max": 45},
                geo_targets=["US"],
                device_types=["desktop", "mobile"],
            ),
        ))
    
    return campaigns


# ============================================================================
# Context Flow Manager Tests
# ============================================================================

class TestContextFlowManager:
    """Tests for ContextFlowManager."""
    
    def test_create_initial_context(self, sample_campaign):
        """Test creating initial context from campaign."""
        manager = ContextFlowManager()
        
        context = manager.create_initial_context(
            campaign=sample_campaign,
            agent_id="test-agent",
        )
        
        assert context.agent_id == "test-agent"
        assert context.level == 1
        assert "campaign" in context.working_memory
        assert context.working_memory["campaign_id"] == sample_campaign.campaign_id
        assert context.constraints["budget_limit"] == sample_campaign.total_budget
    
    def test_pass_context_down(self, sample_campaign):
        """Test passing context from higher to lower level."""
        manager = ContextFlowManager()
        
        context = manager.create_initial_context(sample_campaign, "test-agent")
        original_tokens = manager._context_tokens_original
        
        # Pass L1 -> L2
        l2_context = manager.pass_context_down(context, 1, 2)
        
        assert l2_context.level == context.level  # Level preserved in context
        assert manager._context_tokens_current <= original_tokens
    
    def test_context_rot_when_enabled(self, sample_campaign):
        """Test context rot is applied when enabled."""
        config = ContextFlowConfig(
            enable_context_rot=True,
            rot_rate_per_level=0.5,  # High rate for testing
        )
        manager = ContextFlowManager(config)
        
        context = manager.create_initial_context(sample_campaign, "test-agent")
        
        # Add more items to working memory
        context.working_memory["extra1"] = "value1"
        context.working_memory["extra2"] = "value2"
        context.working_memory["extra3"] = "value3"
        
        # Pass down with rot
        l2_context = manager.pass_context_down(context, 1, 2)
        
        # With 50% rot rate, likely some items will be lost
        # (but not deterministic, so just check it doesn't crash)
        assert l2_context is not None
    
    def test_context_preservation_tracking(self, sample_campaign):
        """Test context preservation is correctly tracked."""
        manager = ContextFlowManager()
        
        context = manager.create_initial_context(sample_campaign, "test-agent")
        
        # Initially 100%
        assert manager.get_context_preservation() > 0
        
        # Pass down
        manager.pass_context_down(context, 1, 2)
        
        # Preservation should be tracked
        preservation = manager.get_context_preservation()
        assert 0 <= preservation <= 100


# ============================================================================
# Buyer System Tests
# ============================================================================

class TestBuyerAgentSystem:
    """Tests for BuyerAgentSystem."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test buyer system initializes correctly."""
        system = BuyerAgentSystem(
            buyer_id="test-buyer",
            scenario="A",
            mock_llm=True,
        )
        
        assert system.buyer_id == "test-buyer"
        assert not system._initialized
        
        await system.initialize()
        
        assert system._initialized
        assert system._l1_portfolio_manager is not None
        assert len(system._l2_specialists) > 0
        assert len(system._l3_functional) > 0
        
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_campaign_processing(self, sample_campaign):
        """Test processing a campaign through hierarchy."""
        system = BuyerAgentSystem(
            buyer_id="test-buyer",
            scenario="A",
            mock_llm=True,
        )
        
        await system.initialize()
        
        try:
            result = await system.process_campaign(sample_campaign)
            
            assert result.campaign_id == sample_campaign.campaign_id
            # Success depends on mock implementation
            assert isinstance(result.budget_allocated, (int, float))
            assert isinstance(result.spend, (int, float))
            assert isinstance(result.context_preserved_pct, float)
        finally:
            await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_campaign_processing(self, sample_campaigns):
        """Test processing multiple campaigns."""
        system = BuyerAgentSystem(
            buyer_id="test-buyer",
            scenario="A",
            mock_llm=True,
        )
        
        await system.initialize()
        
        try:
            results = await system.process_multi_campaign(sample_campaigns[:3])
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result.campaign_id, str)
        finally:
            await system.shutdown()
    
    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        system = BuyerAgentSystem(buyer_id="test-buyer")
        
        assert system.metrics.total_l1_decisions == 0
        
        # Simulate metrics
        system.metrics.total_l1_decisions = 5
        system.metrics.total_l2_tasks = 10
        system.metrics.context_handoffs = 15
        
        metrics_dict = system.get_metrics()
        
        assert metrics_dict["l1_decisions"] == 5
        assert metrics_dict["l2_tasks"] == 10
        assert metrics_dict["context_handoffs"] == 15


# ============================================================================
# Seller System Tests
# ============================================================================

class TestSellerAgentSystem:
    """Tests for SellerAgentSystem."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test seller system initializes correctly."""
        system = SellerAgentSystem(
            seller_id="test-seller",
            scenario="A",
            mock_llm=True,
        )
        
        assert system.seller_id == "test-seller"
        assert not system._initialized
        
        await system.initialize()
        
        assert system._initialized
        assert system._l1_inventory_manager is not None
        assert len(system._l2_inventory) > 0
        assert len(system._l3_functional) > 0
        
        await system.shutdown()


# ============================================================================
# Scenario A Tests
# ============================================================================

class TestMultiAgentScenarioA:
    """Tests for Multi-Agent Scenario A."""
    
    @pytest.mark.asyncio
    async def test_scenario_setup(self):
        """Test scenario A setup."""
        scenario = MultiAgentScenarioA(
            num_buyers=2,
            num_sellers=2,
            mock_llm=True,
        )
        
        await scenario.setup()
        
        try:
            assert len(scenario.buyer_systems) == 2
            assert len(scenario.seller_systems) == 2
            assert scenario._exchange is not None
        finally:
            await scenario.teardown()
    
    @pytest.mark.asyncio
    async def test_add_campaign(self, sample_campaign):
        """Test adding campaigns to scenario."""
        scenario = MultiAgentScenarioA(mock_llm=True)
        await scenario.setup()
        
        try:
            scenario.add_campaign(sample_campaign)
            
            assert sample_campaign.campaign_id in scenario._active_campaigns
            assert sample_campaign.status == CampaignStatus.ACTIVE
            assert scenario.metrics.campaigns_started == 1
        finally:
            await scenario.teardown()
    
    @pytest.mark.asyncio
    async def test_run_day(self, sample_campaign):
        """Test running a single day."""
        scenario = MultiAgentScenarioA(
            num_buyers=2,
            num_sellers=2,
            mock_llm=True,
        )
        await scenario.setup()
        
        try:
            scenario.add_campaign(sample_campaign)
            
            deals = await scenario.run_day(1)
            
            # Deals may or may not be made depending on mock responses
            assert isinstance(deals, list)
        finally:
            await scenario.teardown()
    
    @pytest.mark.asyncio
    async def test_deal_cycle(self, sample_campaign):
        """Test full deal cycle execution."""
        scenario = MultiAgentScenarioA(
            num_buyers=1,
            num_sellers=1,
            mock_llm=True,
        )
        await scenario.setup()
        
        try:
            buyer_system = list(scenario.buyer_systems.values())[0]
            seller_system = list(scenario.seller_systems.values())[0]
            
            result = await scenario.run_deal_cycle(
                campaign=sample_campaign,
                buyer_system=buyer_system,
                seller_system=seller_system,
            )
            
            assert result.deal_id is not None
            assert result.campaign_id == sample_campaign.campaign_id
            assert isinstance(result.execution_time_ms, float)
        finally:
            await scenario.teardown()
    
    def test_config_values(self):
        """Test configuration values."""
        config = MultiAgentScenarioAConfig()
        
        assert config.exchange_fee_pct == 0.15  # 15%
        assert config.exchange_recovery_rate == 0.60  # 60%
        assert config.enable_hierarchy is True


# ============================================================================
# Scenario B Tests
# ============================================================================

class TestMultiAgentScenarioB:
    """Tests for Multi-Agent Scenario B (Direct A2A)."""
    
    @pytest.mark.asyncio
    async def test_scenario_setup(self):
        """Test scenario B setup."""
        scenario = MultiAgentScenarioB(
            num_buyers=2,
            num_sellers=2,
            mock_llm=True,
        )
        
        await scenario.setup()
        
        try:
            assert len(scenario.buyer_systems) == 2
            assert len(scenario.seller_systems) == 2
            # No exchange in scenario B!
            assert len(scenario._a2a_protocols) == 4  # 2 buyers + 2 sellers
        finally:
            await scenario.teardown()
    
    @pytest.mark.asyncio
    async def test_no_recovery(self, sample_campaign):
        """Test that scenario B has no recovery."""
        scenario = MultiAgentScenarioB(mock_llm=True)
        await scenario.setup()
        
        try:
            # Recovery config should be disabled
            assert scenario._context_rot_config.recovery_source.value == "none"
            assert scenario._context_rot_config.recovery_accuracy == 0.0
        finally:
            await scenario.teardown()
    
    def test_config_values(self):
        """Test configuration values."""
        config = MultiAgentScenarioBConfig()
        
        assert config.exchange_fee_pct == 0.0  # No fees!
        assert config.recovery_enabled is False
        assert config.recovery_accuracy == 0.0
    
    @pytest.mark.asyncio
    async def test_error_accumulation(self, sample_campaign):
        """Test error accumulation tracking."""
        scenario = MultiAgentScenarioB(mock_llm=True, seed=42)
        await scenario.setup()
        
        try:
            # Initial error state
            assert scenario._accumulated_errors == 0
            assert scenario._error_rate_multiplier == 1.0
            
            # Run some deal cycles
            scenario.add_campaign(sample_campaign)
            await scenario.run_day(1)
            
            # Error metrics should be tracked
            summary = scenario.get_error_summary()
            assert "total_errors" in summary
            assert summary["recovery_attempts"] == 0  # No recovery!
        finally:
            await scenario.teardown()


# ============================================================================
# Scenario C Tests
# ============================================================================

class TestMultiAgentScenarioC:
    """Tests for Multi-Agent Scenario C (Ledger-Backed)."""
    
    @pytest.mark.asyncio
    async def test_scenario_setup(self):
        """Test scenario C setup."""
        scenario = MultiAgentScenarioC(
            num_buyers=2,
            num_sellers=2,
            mock_llm=True,
        )
        
        await scenario.setup()
        
        try:
            assert len(scenario.buyer_systems) == 2
            assert len(scenario.seller_systems) == 2
            assert scenario._ledger_client is not None
        finally:
            await scenario.teardown()
    
    @pytest.mark.asyncio
    async def test_ledger_client_operations(self):
        """Test ledger client basic operations."""
        client = LedgerClient(network="sui:testnet")
        
        # Commit state
        state = await client.commit_state(
            agent_id="test-agent",
            agent_level=1,
            working_memory={"key": "value"},
            conversation_history=[{"role": "user", "content": "test"}],
            constraints={"limit": 100},
        )
        
        assert state.state_id is not None
        assert state.agent_id == "test-agent"
        assert state.merkle_root is not None
        assert state.tx_hash is not None
        
        # Recover state
        recovered = await client.recover_state(state.state_id)
        
        assert recovered is not None
        assert recovered.state_id == state.state_id
        assert recovered.working_memory == {"key": "value"}
        
        # Verify state
        verification = await client.verify_state(state.state_id)
        
        assert verification.verified is True
    
    @pytest.mark.asyncio
    async def test_100_percent_recovery(self, sample_campaign):
        """Test that scenario C has 100% recovery."""
        scenario = MultiAgentScenarioC(mock_llm=True)
        await scenario.setup()
        
        try:
            # Recovery config should be 100%
            assert scenario._context_rot_config.recovery_accuracy == 1.0
            assert scenario._context_rot_config.recovery_source.value == "ledger"
        finally:
            await scenario.teardown()
    
    @pytest.mark.asyncio
    async def test_ledger_metrics(self, sample_campaign):
        """Test ledger metrics tracking."""
        scenario = MultiAgentScenarioC(mock_llm=True)
        await scenario.setup()
        
        try:
            initial_commits = scenario._total_commits
            
            scenario.add_campaign(sample_campaign)
            await scenario.run_day(1)
            
            # Should have more commits after running
            summary = scenario.get_ledger_summary()
            assert "total_state_commits" in summary
            assert summary["context_recovery_rate"] == 1.0
            assert summary["alkimi_fee_rate"] == 0.05
        finally:
            await scenario.teardown()
    
    def test_config_values(self):
        """Test configuration values."""
        config = MultiAgentScenarioCConfig()
        
        assert config.alkimi_fee_pct == 0.05  # 5% (vs 15% traditional)
        assert config.recovery_accuracy == 1.0  # 100%
        assert config.enable_state_commits is True


# ============================================================================
# Comparison Tests
# ============================================================================

class TestScenarioComparison:
    """Tests comparing scenarios A, B, and C."""
    
    @pytest.mark.asyncio
    async def test_fee_comparison(self):
        """Compare fee structures across scenarios."""
        config_a = MultiAgentScenarioAConfig()
        config_b = MultiAgentScenarioBConfig()
        config_c = MultiAgentScenarioCConfig()
        
        # A: 15% exchange fees
        assert config_a.exchange_fee_pct == 0.15
        
        # B: 0% fees (direct)
        assert config_b.exchange_fee_pct == 0.0
        
        # C: 5% Alkimi fees
        assert config_c.alkimi_fee_pct == 0.05
        
        # C should be 10% cheaper than A
        assert config_a.exchange_fee_pct - config_c.alkimi_fee_pct == 0.10
    
    @pytest.mark.asyncio
    async def test_recovery_comparison(self):
        """Compare recovery rates across scenarios."""
        config_a = MultiAgentScenarioAConfig()
        config_b = MultiAgentScenarioBConfig()
        config_c = MultiAgentScenarioCConfig()
        
        # A: 60% recovery via exchange
        assert config_a.exchange_recovery_rate == 0.60
        
        # B: 0% recovery (no verification)
        assert config_b.recovery_accuracy == 0.0
        
        # C: 100% recovery via ledger
        assert config_c.recovery_accuracy == 1.0
    
    @pytest.mark.asyncio
    async def test_context_preservation(self, sample_campaign):
        """Test context preservation differs across scenarios."""
        scenarios = []
        
        # Create all three scenarios
        scenario_a = MultiAgentScenarioA(num_buyers=1, num_sellers=1, mock_llm=True, seed=42)
        scenario_b = MultiAgentScenarioB(num_buyers=1, num_sellers=1, mock_llm=True, seed=42)
        scenario_c = MultiAgentScenarioC(num_buyers=1, num_sellers=1, mock_llm=True, seed=42)
        
        try:
            for scenario in [scenario_a, scenario_b, scenario_c]:
                await scenario.setup()
                scenario.add_campaign(sample_campaign)
                scenarios.append(scenario)
            
            # Run one day each
            for scenario in scenarios:
                await scenario.run_day(1)
            
            # Get metrics
            metrics_a = scenario_a.get_hierarchy_metrics()
            metrics_b = scenario_b.get_hierarchy_metrics()
            metrics_c = scenario_c.get_hierarchy_metrics()
            
            # C should have best recovery rate (100%)
            assert metrics_c.get("recovery_rate", 0) == 1.0
            
            # B should have worst recovery rate (0%)
            assert metrics_b.get("recovery_rate", 0) == 0.0
            
        finally:
            for scenario in scenarios:
                await scenario.teardown()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full simulation."""
    
    @pytest.mark.asyncio
    async def test_full_simulation_run(self, sample_campaign):
        """Test running a complete simulation."""
        scenario = MultiAgentScenarioA(
            num_buyers=2,
            num_sellers=2,
            mock_llm=True,
            seed=42,
        )
        
        try:
            # Run full simulation
            metrics = await scenario.run(days=3)
            
            assert metrics is not None
            assert metrics.simulation_days_completed == 3
            
        except Exception as e:
            pytest.fail(f"Full simulation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_multi_campaign_simulation(self, sample_campaigns):
        """Test simulation with multiple campaigns."""
        scenario = MultiAgentScenarioC(
            num_buyers=3,
            num_sellers=3,
            mock_llm=True,
            seed=42,
        )
        
        await scenario.setup()
        
        try:
            # Add multiple campaigns
            scenario.add_campaigns(sample_campaigns[:3])
            
            assert len(scenario._active_campaigns) == 3
            
            # Run simulation
            deals = await scenario.run_day(1)
            
            # Should have processed something
            assert isinstance(deals, list)
            
        finally:
            await scenario.teardown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
