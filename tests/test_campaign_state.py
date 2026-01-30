"""
Tests for multi-campaign state management.

Tests cover:
1. Multi-campaign portfolio management
2. Volatile vs ledger-backed state
3. State divergence detection
4. Cross-campaign coordination
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from src.state import (
    # Portfolio
    CampaignPortfolio,
    CampaignState,
    CampaignMetrics,
    PortfolioView,
    StateUpdate,
    Conflict,
    Deal,
    # Cross-campaign
    CrossCampaignState,
    Commit,
    PacingState,
    ContentionResult,
    # Volatile
    VolatileStateManager,
    # Ledger-backed
    LedgerBackedStateManager,
    StateSnapshot,
    VerificationResult,
    # Sync
    StateSync,
    SyncResult,
    Divergence,
)
from src.state.campaign_portfolio import Campaign, CampaignStatus, ConflictType


class TestCampaignPortfolio:
    """Tests for multi-campaign portfolio management."""
    
    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio."""
        return CampaignPortfolio(
            portfolio_id="test-portfolio",
            total_budget=1_000_000.0,
        )
    
    @pytest.mark.asyncio
    async def test_add_campaign(self, portfolio):
        """Test adding a campaign to portfolio."""
        campaign = Campaign(
            campaign_id="camp-001",
            advertiser_id="adv-001",
            budget=100_000.0,
            target_audience=["tech_professionals"],
            target_channels=["display", "video"],
        )
        
        state = await portfolio.add_campaign(campaign)
        
        assert state.campaign_id == "camp-001"
        assert state.budget_total == 100_000.0
        assert state.status == CampaignStatus.DRAFT
        assert portfolio.allocated_budget == 100_000.0
    
    @pytest.mark.asyncio
    async def test_add_campaign_budget_check(self, portfolio):
        """Test that adding campaign validates budget."""
        # Add campaign that uses most budget
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-001",
            advertiser_id="adv-001",
            budget=900_000.0,
        ))
        
        # Try to add another that exceeds remaining
        with pytest.raises(ValueError, match="Insufficient portfolio budget"):
            await portfolio.add_campaign(Campaign(
                campaign_id="camp-002",
                advertiser_id="adv-001",
                budget=200_000.0,
            ))
    
    @pytest.mark.asyncio
    async def test_update_campaign_state(self, portfolio):
        """Test updating campaign state."""
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-001",
            advertiser_id="adv-001",
            budget=100_000.0,
        ))
        
        # Update budget spend
        update = StateUpdate(
            campaign_id="camp-001",
            update_type="budget_spend",
            payload={"amount": 5000.0},
        )
        
        state = await portfolio.update_campaign_state("camp-001", update)
        
        assert state.budget_spent == 5000.0
        assert state.metrics.spend == 5000.0
        assert state.version == 2
    
    @pytest.mark.asyncio
    async def test_portfolio_view(self, portfolio):
        """Test getting aggregated portfolio view."""
        # Add multiple campaigns
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-001",
            advertiser_id="adv-001",
            budget=100_000.0,
            target_channels=["display"],
        ))
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-002",
            advertiser_id="adv-001",
            budget=150_000.0,
            target_channels=["video"],
        ))
        
        # Update one to active
        await portfolio.update_campaign_state("camp-001", StateUpdate(
            campaign_id="camp-001",
            update_type="status_change",
            payload={"status": "active"},
        ))
        
        view = await portfolio.get_portfolio_view()
        
        assert view.total_campaigns == 2
        assert view.active_campaigns == 1
        assert view.total_budget == 1_000_000.0
        assert view.channel_allocation["display"] == 100_000.0
        assert view.channel_allocation["video"] == 150_000.0
    
    @pytest.mark.asyncio
    async def test_budget_conflict_detection(self, portfolio):
        """Test detection of budget conflicts."""
        # Manually create overcommitment scenario
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-001",
            advertiser_id="adv-001",
            budget=600_000.0,
        ))
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-002",
            advertiser_id="adv-001",
            budget=400_000.0,
        ))
        
        # Spend beyond portfolio total (1M) - both campaigns overspend
        portfolio.campaigns["camp-001"].budget_spent = 650_000.0
        portfolio.campaigns["camp-002"].budget_spent = 450_000.0  # 1.1M total > 1M
        
        conflicts = await portfolio.check_budget_conflicts()
        
        assert len(conflicts) >= 1
        assert any(c.conflict_type == ConflictType.BUDGET_OVERCOMMIT for c in conflicts)
    
    @pytest.mark.asyncio
    async def test_audience_overlap_detection(self, portfolio):
        """Test detection of audience overlap."""
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-001",
            advertiser_id="adv-001",
            budget=100_000.0,
            target_audience=["tech", "finance", "crypto"],
        ))
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-002",
            advertiser_id="adv-001",
            budget=100_000.0,
            target_audience=["tech", "finance", "gaming"],  # 2/3 overlap with camp-001
        ))
        
        conflicts = await portfolio.check_budget_conflicts()
        
        overlap_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.AUDIENCE_OVERLAP]
        assert len(overlap_conflicts) >= 1
    
    @pytest.mark.asyncio
    async def test_deal_management(self, portfolio):
        """Test adding and updating deals."""
        await portfolio.add_campaign(Campaign(
            campaign_id="camp-001",
            advertiser_id="adv-001",
            budget=100_000.0,
        ))
        
        # Add a deal
        deal_data = {
            "deal_id": "deal-001",
            "campaign_id": "camp-001",
            "seller_id": "seller-001",
            "product_id": "prod-001",
            "impressions_committed": 1_000_000,
            "cpm_agreed": 5.0,
            "total_value": 5000.0,
            "status": "active",
        }
        
        await portfolio.update_campaign_state("camp-001", StateUpdate(
            campaign_id="camp-001",
            update_type="deal_add",
            payload={"deal": deal_data},
        ))
        
        state = await portfolio.get_campaign("camp-001")
        assert len(state.deals) == 1
        assert state.deals[0].deal_id == "deal-001"
        assert state.budget_committed == 5000.0


class TestCrossCampaignState:
    """Tests for cross-campaign state coordination."""
    
    @pytest.fixture
    def cross_state(self):
        """Create cross-campaign state manager."""
        return CrossCampaignState()
    
    @pytest.mark.asyncio
    async def test_audience_registration(self, cross_state):
        """Test registering audience usage."""
        # Campaign 1 uses audience
        others = await cross_state.register_audience_usage("tech_pros", "camp-001")
        assert len(others) == 0
        
        # Campaign 2 uses same audience
        others = await cross_state.register_audience_usage("tech_pros", "camp-002")
        assert "camp-001" in others
    
    @pytest.mark.asyncio
    async def test_audience_overlap_calculation(self, cross_state):
        """Test calculating audience overlap."""
        # Setup audiences
        await cross_state.register_audience_usage("tech", "camp-001")
        await cross_state.register_audience_usage("finance", "camp-001")
        await cross_state.register_audience_usage("crypto", "camp-001")
        
        await cross_state.register_audience_usage("tech", "camp-002")
        await cross_state.register_audience_usage("finance", "camp-002")
        await cross_state.register_audience_usage("gaming", "camp-002")
        
        overlap = await cross_state.check_audience_overlap("camp-001", "camp-002")
        
        # 2 out of 3 audiences overlap
        assert abs(overlap - 2/3) < 0.01
    
    @pytest.mark.asyncio
    async def test_inventory_contention(self, cross_state):
        """Test inventory contention detection."""
        product_id = "premium-video-001"
        
        # Add first commitment
        commit1 = Commit(
            commit_id="commit-001",
            campaign_id="camp-001",
            product_id=product_id,
            seller_id="seller-001",
            impressions_reserved=500_000,
            status="active",
        )
        
        result = await cross_state.add_inventory_commit(
            commit1,
            product_capacity=1_000_000,
        )
        
        assert result.available == 500_000
        assert result.contention_ratio == 0.5
        assert not result.is_constrained
        
        # Add second commitment - creates contention
        commit2 = Commit(
            commit_id="commit-002",
            campaign_id="camp-002",
            product_id=product_id,
            seller_id="seller-001",
            impressions_reserved=400_000,
            status="active",
        )
        
        result = await cross_state.add_inventory_commit(commit2)
        
        assert result.available == 100_000
        assert result.contention_ratio == 0.9
        assert result.is_constrained
        assert len(result.competing_campaigns) == 2
    
    @pytest.mark.asyncio
    async def test_pacing_update(self, cross_state):
        """Test budget pacing tracking."""
        pacing = await cross_state.update_pacing(
            campaign_id="camp-001",
            total_budget=100_000.0,
            spent_total=30_000.0,
            spent_today=3_000.0,
            days_elapsed=10,
            days_remaining=20,
        )
        
        assert pacing.budget_remaining == 70_000.0
        # Expected: 100k over 30 days = ~33.3k after 10 days
        # Actual: 30k spent - slightly behind pace
        assert pacing.status.value in ["on_track", "behind"]


class TestVolatileState:
    """Tests for volatile (in-memory) state - Scenario B."""
    
    @pytest.fixture
    def volatile(self):
        """Create volatile state manager."""
        return VolatileStateManager(agent_id="test-agent")
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, volatile):
        """Test basic get/set operations."""
        await volatile.set("key1", "value1")
        await volatile.set("key2", {"nested": "data"})
        
        assert await volatile.get("key1") == "value1"
        assert await volatile.get("key2") == {"nested": "data"}
        assert await volatile.get("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_restart_loses_state(self, volatile):
        """Test that restart loses all state."""
        # Set some state
        await volatile.set("important_key", "important_value")
        await volatile.set("campaign_state", {"budget": 100000})
        
        # Verify state exists
        assert await volatile.get("important_key") == "important_value"
        
        # Simulate restart
        loss_report = await volatile.simulate_restart()
        
        # State is gone
        assert await volatile.get("important_key") is None
        assert await volatile.get("campaign_state") is None
        assert loss_report["keys_lost"] == 2
    
    @pytest.mark.asyncio
    async def test_state_comparison(self):
        """Test comparing state between two agents."""
        agent_a = VolatileStateManager(agent_id="agent-a")
        agent_b = VolatileStateManager(agent_id="agent-b")
        
        # Set overlapping but different state
        await agent_a.set("shared_key", "value_a")
        await agent_a.set("only_a", "data")
        
        await agent_b.set("shared_key", "value_b")
        await agent_b.set("only_b", "data")
        
        comparison = await agent_a.compare_with(agent_b)
        
        assert comparison["divergence_detected"]
        assert "only_a" in comparison["only_in_mine"]
        assert "only_b" in comparison["only_in_theirs"]
        assert len(comparison["value_differences"]) == 1
    
    @pytest.mark.asyncio
    async def test_corruption_simulation(self, volatile):
        """Test partial data corruption."""
        # Set up state
        for i in range(10):
            await volatile.set(f"key_{i}", f"value_{i}")
        
        # Corrupt 50%
        report = await volatile.simulate_partial_corruption(corruption_rate=0.5)
        
        assert report["keys_corrupted"] == 5
        assert "UNDETECTABLE" in report["note"]


class TestLedgerBackedState:
    """Tests for ledger-backed state - Scenario C."""
    
    @pytest.fixture
    def ledger_state(self):
        """Create ledger-backed state manager."""
        return LedgerBackedStateManager(
            agent_id="test-agent",
            auto_persist=True,
        )
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, ledger_state):
        """Test basic get/set with persistence."""
        tx_id = await ledger_state.set("key1", "value1")
        
        assert tx_id is not None
        assert await ledger_state.get("key1") == "value1"
    
    @pytest.mark.asyncio
    async def test_recovery_from_ledger(self, ledger_state):
        """Test state recovery from ledger."""
        # Set some state
        await ledger_state.set("campaign_budget", 100000)
        await ledger_state.set("deals_active", ["deal-001", "deal-002"])
        
        # Persist snapshot
        snapshot = await ledger_state.persist_snapshot()
        
        assert snapshot.entry_count == 2
        assert snapshot.ledger_tx is not None
        
        # Clear local state (simulate restart)
        ledger_state._local_state = {}
        
        # Recover from ledger
        recovered = await ledger_state.recover_from_ledger()
        
        assert recovered.entry_count == 2
        assert await ledger_state.get("campaign_budget") == 100000
    
    @pytest.mark.asyncio
    async def test_verification(self, ledger_state):
        """Test state verification against ledger."""
        # Set and persist state
        await ledger_state.set("key1", "value1")
        await ledger_state.persist_snapshot()
        
        # Verify - should match
        result = await ledger_state.verify_state()
        assert result.is_verified
        
        # Modify local without persisting
        ledger_state._local_state["key2"] = "value2"
        
        # Verify - should show divergence
        result = await ledger_state.verify_state()
        assert not result.is_verified
        assert "key2" in result.missing_ledger
    
    @pytest.mark.asyncio
    async def test_sync_from_ledger(self, ledger_state):
        """Test syncing local state from ledger."""
        # Set and persist state
        await ledger_state.set("correct_value", 100)
        await ledger_state.persist_snapshot()
        
        # Corrupt local
        ledger_state._local_state["correct_value"] = 999
        ledger_state._local_state["orphan_key"] = "should_disappear"
        
        # Sync from ledger
        fixed_count = await ledger_state.sync_from_ledger()
        
        assert fixed_count > 0
        assert await ledger_state.get("correct_value") == 100
        # Note: orphan key is replaced by ledger state


class TestStateSync:
    """Tests for state synchronization."""
    
    @pytest.fixture
    def sync(self):
        """Create state sync manager."""
        return StateSync(
            message_loss_rate=0.1,
            max_retries=3,
            seed=42,
        )
    
    @pytest.mark.asyncio
    async def test_detect_divergence(self, sync):
        """Test divergence detection."""
        state_a = {
            "shared": "same_value",
            "different": "value_a",
            "only_a": "data",
        }
        state_b = {
            "shared": "same_value",
            "different": "value_b",
            "only_b": "data",
        }
        
        divergences = await sync.detect_divergence(state_a, state_b)
        
        assert len(divergences) == 3
        types = {d.divergence_type.value for d in divergences}
        assert "missing_b" in types  # only_a
        assert "missing_a" in types  # only_b
        assert "value_mismatch" in types  # different
    
    @pytest.mark.asyncio
    async def test_sync_with_ledger(self, sync):
        """Test sync using ledger (Scenario C)."""
        buyer_state = {"key1": "buyer_value", "key2": 100}
        seller_state = {"key1": "seller_value", "key3": 200}
        ledger_state = {"key1": "correct_value", "key2": 100, "key3": 200}
        
        result = await sync.sync_buyer_seller(
            buyer_state,
            seller_state,
            use_ledger=True,
            ledger_state=ledger_state,
        )
        
        assert result.is_success or result.status.value == "partial"
        # With ledger, all divergences should be resolved
        for div in result.divergences:
            if div.key in ledger_state:
                assert div.resolved
                assert div.resolution_source == "ledger"
    
    @pytest.mark.asyncio
    async def test_sync_without_ledger(self, sync):
        """Test sync without ledger (Scenario B) - may fail."""
        buyer_state = {"deal_1": {"imps": 1000, "spend": 100}}
        seller_state = {"deal_1": {"imps": 1100, "spend": 110}}  # Different counts
        
        result = await sync.sync_buyer_seller(
            buyer_state,
            seller_state,
            use_ledger=False,
        )
        
        # Without ledger, resolution is probabilistic
        assert result.keys_compared == 1
        # May or may not resolve depending on RNG
    
    @pytest.mark.asyncio
    async def test_drift_simulation(self, sync):
        """Test simulating state drift over time."""
        initial_state = {"base_key": "initial_value"}
        
        state_a, state_b, drifted = await sync.simulate_drift(
            initial_state,
            num_operations=50,
            drift_rate=0.2,  # 20% message loss
        )
        
        # States should have diverged
        divergences = await sync.detect_divergence(state_a, state_b)
        
        # With 20% drift over 50 operations, expect some divergence
        assert len(divergences) > 0 or len(drifted) > 0


class TestIntegration:
    """Integration tests combining multiple state components."""
    
    @pytest.mark.asyncio
    async def test_portfolio_with_volatile_state(self):
        """Test portfolio using volatile state manager."""
        portfolio = CampaignPortfolio("test", total_budget=500_000)
        volatile = VolatileStateManager("buyer-agent")
        
        # Add campaign to portfolio
        campaign = Campaign("camp-001", "adv-001", 100_000)
        state = await portfolio.add_campaign(campaign)
        
        # Store state in volatile manager
        await volatile.set(f"campaign_{state.campaign_id}", state.to_dict())
        
        # Simulate agent restart
        await volatile.simulate_restart()
        
        # State is lost
        assert await volatile.get(f"campaign_{state.campaign_id}") is None
        # But portfolio still has it (in-memory)
        assert await portfolio.get_campaign("camp-001") is not None
    
    @pytest.mark.asyncio
    async def test_portfolio_with_ledger_state(self):
        """Test portfolio using ledger-backed state manager."""
        portfolio = CampaignPortfolio("test", total_budget=500_000)
        ledger = LedgerBackedStateManager("buyer-agent")
        
        # Add campaign
        campaign = Campaign("camp-001", "adv-001", 100_000)
        state = await portfolio.add_campaign(campaign)
        
        # Persist to ledger
        await ledger.set(f"campaign_{state.campaign_id}", state.to_dict())
        await ledger.persist_snapshot()
        
        # Clear local (simulate restart)
        ledger._local_state = {}
        
        # Recover from ledger
        await ledger.recover_from_ledger()
        
        # State is recovered
        recovered = await ledger.get(f"campaign_{state.campaign_id}")
        assert recovered is not None
        assert recovered["campaign_id"] == "camp-001"
    
    @pytest.mark.asyncio
    async def test_cross_campaign_sync_scenario(self):
        """Test cross-campaign state synchronization scenario."""
        # Two agents with their own state
        buyer_volatile = VolatileStateManager("buyer")
        seller_volatile = VolatileStateManager("seller")
        sync = StateSync(message_loss_rate=0.15, seed=42)
        
        # Both track same deal differently (realistic scenario)
        await buyer_volatile.set("deal_001", {
            "impressions": 95000,
            "spend": 475.0,
            "status": "active",
        })
        
        await seller_volatile.set("deal_001", {
            "impressions": 100000,  # Seller counted more
            "spend": 500.0,
            "status": "active",
        })
        
        # Try to sync
        buyer_state = {"deal_001": await buyer_volatile.get("deal_001")}
        seller_state = {"deal_001": await seller_volatile.get("deal_001")}
        
        result = await sync.sync_buyer_seller(buyer_state, seller_state, use_ledger=False)
        
        # Without ledger, divergence exists
        assert len(result.divergences) > 0
        
        # Now try with ledger
        ledger_state = {
            "deal_001": {
                "impressions": 97500,  # Ground truth
                "spend": 487.50,
                "status": "active",
            }
        }
        
        result_ledger = await sync.sync_buyer_seller(
            buyer_state, seller_state,
            use_ledger=True,
            ledger_state=ledger_state,
        )
        
        # With ledger, should resolve
        for div in result_ledger.divergences:
            assert div.resolved
            assert div.confidence == 1.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
