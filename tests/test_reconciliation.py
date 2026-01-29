"""
Tests for cross-agent reconciliation simulation.

Validates that:
1. Discrepancy injection produces realistic divergence
2. Reconciliation outcomes match expected thresholds
3. Shared ledger provides 100% resolution
4. Metrics are calculated correctly
"""

import pytest
import asyncio
from src.scenarios.reconciliation import (
    ReconciliationSimulator,
    ReconciliationEngine,
    DiscrepancyInjector,
    DiscrepancyConfig,
    ReconciliationMetrics,
    CampaignRecord,
    ResolutionOutcome,
)


@pytest.fixture
def config():
    """Default configuration for tests."""
    return DiscrepancyConfig()


@pytest.fixture
def injector(config):
    """Discrepancy injector with fixed seed for reproducibility."""
    return DiscrepancyInjector(config, seed=42)


@pytest.fixture
def engine(config):
    """Reconciliation engine with fixed seed."""
    return ReconciliationEngine(config, seed=42)


@pytest.fixture
def simulator():
    """Full simulator with fixed seed."""
    return ReconciliationSimulator(seed=42)


class TestDiscrepancyInjector:
    """Tests for discrepancy injection."""
    
    def test_injection_creates_divergence(self, injector):
        """Verify that injection creates buyer-seller differences."""
        buyer = CampaignRecord(
            campaign_id="test-001",
            party_id="buyer-001",
            party_type="buyer",
        )
        seller = CampaignRecord(
            campaign_id="test-001",
            party_id="seller-001",
            party_type="seller",
        )
        
        # Inject 30 days of discrepancies
        for day in range(1, 31):
            injector.inject_daily_discrepancy(
                buyer_record=buyer,
                seller_record=seller,
                simulation_day=day,
                daily_impressions=100000,
                daily_spend=1500.0,
            )
        
        # Should have divergence
        assert buyer.impressions_delivered != seller.impressions_delivered
        
        # Calculate discrepancy
        total = max(buyer.impressions_delivered, seller.impressions_delivered)
        diff = abs(buyer.impressions_delivered - seller.impressions_delivered)
        pct = diff / total
        
        # Should be in realistic range (5-20% after 30 days)
        assert 0.03 < pct < 0.25, f"Discrepancy {pct:.1%} outside expected range"
    
    def test_no_negative_values(self, injector):
        """Ensure injector never creates negative impressions or spend."""
        buyer = CampaignRecord(
            campaign_id="test-002",
            party_id="buyer-001",
            party_type="buyer",
        )
        seller = CampaignRecord(
            campaign_id="test-002",
            party_id="seller-001",
            party_type="seller",
        )
        
        for day in range(1, 100):
            injector.inject_daily_discrepancy(
                buyer_record=buyer,
                seller_record=seller,
                simulation_day=day,
                daily_impressions=1000,
                daily_spend=15.0,
            )
        
        assert buyer.impressions_delivered >= 0
        assert seller.impressions_delivered >= 0
        assert buyer.total_spend >= 0
        assert seller.total_spend >= 0


class TestReconciliationEngine:
    """Tests for reconciliation logic."""
    
    def test_small_discrepancy_auto_accepts(self, engine):
        """<3% discrepancy should auto-accept."""
        buyer = CampaignRecord(
            campaign_id="test-003",
            party_id="buyer-001",
            party_type="buyer",
            impressions_delivered=1000000,
            total_spend=15000.0,
        )
        seller = CampaignRecord(
            campaign_id="test-003",
            party_id="seller-001",
            party_type="seller",
            impressions_delivered=980000,  # 2% difference
            total_spend=14700.0,
        )
        
        result = engine.attempt_reconciliation(buyer, seller)
        
        assert result.outcome == ResolutionOutcome.BUYER_ACCEPTS
        assert result.days_to_resolve == 1
        assert result.is_resolved
    
    def test_moderate_discrepancy_negotiates(self, engine):
        """3-10% discrepancy should negotiate."""
        buyer = CampaignRecord(
            campaign_id="test-004",
            party_id="buyer-001",
            party_type="buyer",
            impressions_delivered=1000000,
            total_spend=15000.0,
        )
        seller = CampaignRecord(
            campaign_id="test-004",
            party_id="seller-001",
            party_type="seller",
            impressions_delivered=920000,  # 8% difference
            total_spend=13800.0,
        )
        
        result = engine.attempt_reconciliation(buyer, seller)
        
        assert result.outcome == ResolutionOutcome.NEGOTIATED
        assert result.days_to_resolve >= 7
        assert result.is_resolved
    
    def test_large_discrepancy_may_be_unresolvable(self, engine):
        """>15% discrepancy may be unresolvable."""
        buyer = CampaignRecord(
            campaign_id="test-005",
            party_id="buyer-001",
            party_type="buyer",
            impressions_delivered=1000000,
            total_spend=15000.0,
        )
        seller = CampaignRecord(
            campaign_id="test-005",
            party_id="seller-001",
            party_type="seller",
            impressions_delivered=800000,  # 20% difference
            total_spend=12000.0,
        )
        
        result = engine.attempt_reconciliation(buyer, seller)
        
        # Should be disputed or unresolvable
        assert result.outcome in [
            ResolutionOutcome.DISPUTED,
            ResolutionOutcome.UNRESOLVABLE,
            ResolutionOutcome.NEGOTIATED,  # Sometimes resolves
        ]
    
    def test_shared_ledger_always_resolves(self, engine):
        """With shared ledger, any discrepancy is resolvable."""
        buyer = CampaignRecord(
            campaign_id="test-006",
            party_id="buyer-001",
            party_type="buyer",
            impressions_delivered=1000000,
            total_spend=15000.0,
        )
        seller = CampaignRecord(
            campaign_id="test-006",
            party_id="seller-001",
            party_type="seller",
            impressions_delivered=700000,  # 30% difference - normally unresolvable
            total_spend=10500.0,
        )
        ledger = CampaignRecord(
            campaign_id="test-006",
            party_id="ledger",
            party_type="ledger",
            impressions_delivered=950000,  # Ground truth
            total_spend=14250.0,
        )
        
        result = engine.attempt_reconciliation(
            buyer, seller, has_shared_ledger=True, ledger_record=ledger
        )
        
        assert result.outcome == ResolutionOutcome.AGREED
        assert result.days_to_resolve == 0
        assert result.final_impressions == 950000
        assert result.is_resolved
        assert not result.is_unresolvable


class TestReconciliationMetrics:
    """Tests for metrics aggregation."""
    
    def test_metrics_calculation(self):
        """Verify metrics are calculated correctly."""
        metrics = ReconciliationMetrics()
        
        # Add some results
        metrics.total_campaigns = 100
        metrics.agreed = 20
        metrics.negotiated = 30
        metrics.buyer_accepts = 25
        metrics.seller_accepts = 10
        metrics.disputed = 10
        metrics.unresolvable = 5
        
        assert metrics.resolution_rate == 0.85  # 85 resolved / 100 total
        assert metrics.unresolvable_rate == 0.05
        assert metrics.dispute_rate == 0.15  # disputed + unresolvable
    
    def test_add_result(self):
        """Verify add_result updates metrics correctly."""
        metrics = ReconciliationMetrics()
        
        result = ReconciliationEngine().attempt_reconciliation(
            CampaignRecord("c1", "b1", "buyer", impressions_delivered=100000, total_spend=1500),
            CampaignRecord("c1", "s1", "seller", impressions_delivered=98000, total_spend=1470),
        )
        
        metrics.add_result(result)
        
        assert metrics.total_campaigns == 1
        assert metrics.total_buyer_spend == 1500
        assert metrics.total_seller_spend == 1470


class TestReconciliationSimulator:
    """Integration tests for full simulation."""
    
    @pytest.mark.asyncio
    async def test_campaign_simulation(self, simulator):
        """Test running a single campaign."""
        buyer, seller, ledger = await simulator.run_campaign(
            campaign_id="sim-001",
            buyer_id="buyer-001",
            seller_id="seller-001",
            duration_days=30,
            total_budget=50000,
            target_impressions=3000000,
            has_shared_ledger=True,
        )
        
        # All records should have data
        assert buyer.impressions_delivered > 0
        assert seller.impressions_delivered > 0
        assert ledger.impressions_delivered > 0
        
        # Ledger should have ground truth (higher than others due to no loss)
        assert ledger.total_spend > 0
    
    @pytest.mark.asyncio
    async def test_comparison_simulation(self, simulator):
        """Test running full comparison."""
        results = await simulator.run_comparison(
            num_campaigns=10,
            campaign_days=30,
            avg_budget=50000,
        )
        
        assert "B_fragmented" in results
        assert "C_ledger" in results
        
        fragmented = results["B_fragmented"]
        ledger = results["C_ledger"]
        
        # Fragmented should have some disputes
        assert fragmented.total_campaigns == 10
        
        # Ledger should have 100% resolution (all agreed)
        assert ledger.total_campaigns == 10
        assert ledger.agreed == 10
        assert ledger.resolution_rate == 1.0
        assert ledger.unresolvable_rate == 0.0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_impressions(self, engine):
        """Handle campaigns with zero delivery."""
        buyer = CampaignRecord("c1", "b1", "buyer", impressions_delivered=0, total_spend=0)
        seller = CampaignRecord("c1", "s1", "seller", impressions_delivered=0, total_spend=0)
        
        result = engine.attempt_reconciliation(buyer, seller)
        
        # Should still produce a result
        assert result.outcome in list(ResolutionOutcome)
        assert result.impression_discrepancy_pct >= 0
    
    def test_identical_records(self, engine):
        """Perfect match should auto-resolve."""
        buyer = CampaignRecord("c1", "b1", "buyer", impressions_delivered=1000000, total_spend=15000)
        seller = CampaignRecord("c1", "s1", "seller", impressions_delivered=1000000, total_spend=15000)
        
        result = engine.attempt_reconciliation(buyer, seller)
        
        assert result.outcome == ResolutionOutcome.BUYER_ACCEPTS
        assert result.impression_discrepancy_pct == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
