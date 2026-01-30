"""
Cross-Agent Reconciliation Module.

Tests the core hypothesis: Multi-agent systems with private databases
cannot reliably reconcile campaign results without a shared source of truth.

Key concepts:
- **State Divergence**: Buyer and seller records drift apart over time
- **Reconciliation Attempt**: End-of-campaign comparison of records
- **Dispute**: When records differ beyond acceptable threshold
- **Unresolvable**: When no authoritative source can determine truth

This module implements the ACTUAL problem Ben identified:
Not single-agent context rot, but cross-agent state inconsistency.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal
from collections import defaultdict
from enum import Enum
import structlog

logger = structlog.get_logger()


class DiscrepancySource(Enum):
    """Root causes of buyer-seller discrepancies."""
    TIMING_DIFFERENCE = "timing"           # Impression counted at different times
    IVT_FILTERING = "ivt"                  # Different bot detection
    VIEWABILITY = "viewability"            # Different viewability standards
    AD_SERVING_LATENCY = "latency"         # Bid won vs ad rendered gap
    DATA_LOSS = "data_loss"                # Random record loss
    CURRENCY_CONVERSION = "currency"        # Exchange rate timing
    TIMEZONE_CUTOFF = "timezone"           # Day boundary differences
    ATTRIBUTION_WINDOW = "attribution"     # Lookback period differences


class ResolutionOutcome(Enum):
    """Possible outcomes of reconciliation attempt."""
    AGREED = "agreed"                       # Records match (within tolerance)
    NEGOTIATED = "negotiated"               # Settled by splitting difference
    BUYER_ACCEPTS = "buyer_accepts"         # Buyer accepts seller's count
    SELLER_ACCEPTS = "seller_accepts"       # Seller accepts buyer's count
    DISPUTED = "disputed"                   # Formal dispute filed
    UNRESOLVABLE = "unresolvable"          # No resolution possible


@dataclass
class CampaignRecord:
    """
    A single party's record of a campaign.
    
    Buyer and seller each maintain their own version of this.
    """
    campaign_id: str
    party_id: str
    party_type: Literal["buyer", "seller"]
    
    # Delivery metrics (where divergence occurs)
    impressions_delivered: int = 0
    clicks: int = 0
    conversions: int = 0
    
    # Financial metrics
    total_spend: float = 0.0
    average_cpm: float = 0.0
    
    # Timing
    campaign_start: Optional[datetime] = None
    campaign_end: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Record quality
    records_lost: int = 0
    sync_failures: int = 0
    
    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "party_id": self.party_id,
            "party_type": self.party_type,
            "impressions_delivered": self.impressions_delivered,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "total_spend": self.total_spend,
            "average_cpm": self.average_cpm,
            "records_lost": self.records_lost,
        }


@dataclass
class ReconciliationResult:
    """Result of attempting to reconcile buyer and seller records."""
    campaign_id: str
    buyer_id: str
    seller_id: str
    
    # Records being reconciled
    buyer_impressions: int
    seller_impressions: int
    buyer_spend: float
    seller_spend: float
    
    # Divergence metrics
    impression_discrepancy_pct: float
    spend_discrepancy_pct: float
    absolute_spend_difference: float
    
    # Resolution
    outcome: ResolutionOutcome
    final_impressions: Optional[int] = None
    final_spend: Optional[float] = None
    
    # Metadata
    days_to_resolve: int = 0
    resolution_method: Optional[str] = None
    discrepancy_sources: list[DiscrepancySource] = field(default_factory=list)
    
    @property
    def is_resolved(self) -> bool:
        return self.outcome not in [ResolutionOutcome.DISPUTED, ResolutionOutcome.UNRESOLVABLE]
    
    @property
    def is_unresolvable(self) -> bool:
        return self.outcome == ResolutionOutcome.UNRESOLVABLE
    
    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "buyer_impressions": self.buyer_impressions,
            "seller_impressions": self.seller_impressions,
            "impression_discrepancy_pct": round(self.impression_discrepancy_pct, 2),
            "spend_discrepancy_pct": round(self.spend_discrepancy_pct, 2),
            "absolute_spend_difference": round(self.absolute_spend_difference, 2),
            "outcome": self.outcome.value,
            "is_resolved": self.is_resolved,
            "days_to_resolve": self.days_to_resolve,
        }


@dataclass
class DiscrepancyConfig:
    """
    Configuration for realistic discrepancy injection.
    
    Based on industry research:
    - ISBA 2020: 15% "unknown delta"
    - ANA 2023: 5-10% typical discrepancy  
    - MRC: 5% acceptable threshold
    
    Calibrated to produce:
    - ~8% average discrepancy
    - ~33% of campaigns with >10% discrepancy
    - ~10% unresolvable disputes
    """
    # Systematic biases (per-campaign, compounds over time)
    ivt_bias_range: tuple[float, float] = (0.02, 0.10)     # 2-10% IVT disagreement
    viewability_bias_range: tuple[float, float] = (0.01, 0.06)  # 1-6% viewability gap
    timing_bias_range: tuple[float, float] = (-0.03, 0.03)  # ±3% timing variance
    
    # Probability of systematic bias applying
    ivt_apply_rate: float = 0.75         # 75% of campaigns have IVT disagreement
    viewability_apply_rate: float = 0.65  # 65% have viewability gap
    
    # Edge cases (critical for realistic unresolvable rate)
    major_sync_failure_rate: float = 0.10  # 10% have major sync issues
    major_sync_bias_add: float = 0.15      # Adds 15% to IVT bias
    technical_failure_rate: float = 0.08   # 8% have technical data loss
    technical_failure_magnitude: tuple[float, float] = (0.20, 0.40)  # 20-40% loss
    attribution_mismatch_rate: float = 0.05  # 5% have attribution window issues
    attribution_mismatch_bias: tuple[float, float] = (0.05, 0.12)  # 5-12% additional bias
    
    # Daily data loss
    daily_loss_rate: float = 0.03        # 3% daily chance
    daily_loss_magnitude: tuple[float, float] = (0.01, 0.05)  # 1-5% when occurs
    
    # Resolution thresholds
    auto_accept_threshold: float = 0.03  # <3%: auto-accept buyer's count
    negotiation_threshold: float = 0.10  # 3-10%: negotiate
    dispute_threshold: float = 0.15      # 10-15%: formal dispute
    unresolvable_threshold: float = 0.15  # >15%: often unresolvable
    extreme_threshold: float = 0.25       # >25%: always unresolvable
    
    # Resolution probabilities
    dispute_resolution_prob: float = 0.50   # 50% of 10-15% disputes resolve
    severe_resolution_prob: float = 0.30    # 30% of 15-25% disputes resolve
    
    # Timing discrepancy rate
    timing_rate: float = 0.15               # 15% of daily events have timing issues
    
    # Additional discrepancy rates (for inject_daily_discrepancy)
    ivt_disagreement_rate: float = 0.20     # 20% have IVT disagreement
    viewability_rate: float = 0.15          # 15% have viewability issues
    latency_loss_rate: float = 0.08         # 8% have latency losses
    data_loss_rate: float = 0.01            # 1% per day data loss
    currency_variance: float = 0.05         # 5% have currency variance
    unresolvable_above: float = 0.20        # >20% discrepancy = unresolvable


class DiscrepancyInjector:
    """
    Injects realistic discrepancies into buyer-seller records.
    
    Models the real-world causes of reporting differences:
    - Timing: When exactly is an impression counted?
    - IVT: Different bot detection algorithms
    - Viewability: MRC vs proprietary standards
    - Data loss: Random record loss over time
    """
    
    def __init__(
        self,
        config: Optional[DiscrepancyConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or DiscrepancyConfig()
        self._random = random.Random(seed)
    
    def inject_daily_discrepancy(
        self,
        buyer_record: CampaignRecord,
        seller_record: CampaignRecord,
        simulation_day: int,
        daily_impressions: int,
        daily_spend: float,
    ) -> list[DiscrepancySource]:
        """
        Inject discrepancies for a single day of campaign delivery.
        
        Both parties start with the same "ground truth" delivery,
        then we model how their records diverge.
        
        Returns list of discrepancy sources that occurred.
        """
        sources = []
        
        # Start with ground truth
        buyer_imps = daily_impressions
        seller_imps = daily_impressions
        buyer_spend = daily_spend
        seller_spend = daily_spend
        
        # 1. Timing differences
        if self._random.random() < self.config.timing_rate:
            # Buyer might count some impressions on next day
            timing_loss = int(daily_impressions * self._random.uniform(0.01, 0.05))
            buyer_imps -= timing_loss
            sources.append(DiscrepancySource.TIMING_DIFFERENCE)
        
        # 2. IVT filtering disagreement
        if self._random.random() < self.config.ivt_disagreement_rate:
            # Different bot detection - seller filters more
            ivt_diff = int(daily_impressions * self._random.uniform(0.02, 0.08))
            seller_imps -= ivt_diff
            sources.append(DiscrepancySource.IVT_FILTERING)
        
        # 3. Viewability disagreement
        if self._random.random() < self.config.viewability_rate:
            # Seller's viewability standard is stricter
            view_diff = int(daily_impressions * self._random.uniform(0.01, 0.06))
            seller_imps -= view_diff
            sources.append(DiscrepancySource.VIEWABILITY)
        
        # 4. Ad serving latency losses
        if self._random.random() < self.config.latency_loss_rate:
            # Bid won but ad never rendered
            latency_loss = int(daily_impressions * self._random.uniform(0.01, 0.03))
            buyer_imps -= latency_loss  # Buyer doesn't count unrendered
            sources.append(DiscrepancySource.AD_SERVING_LATENCY)
        
        # 5. Random data loss (accumulates over time)
        if self._random.random() < self.config.data_loss_rate * simulation_day:
            # Either party loses some records
            if self._random.random() < 0.5:
                loss = int(daily_impressions * self._random.uniform(0.001, 0.01))
                buyer_imps -= loss
                buyer_record.records_lost += loss
            else:
                loss = int(daily_impressions * self._random.uniform(0.001, 0.01))
                seller_imps -= loss
                seller_record.records_lost += loss
            sources.append(DiscrepancySource.DATA_LOSS)
        
        # 6. Currency conversion timing (for spend)
        if self._random.random() < self.config.currency_variance:
            variance = self._random.uniform(-0.02, 0.02)
            buyer_spend *= (1 + variance)
            sources.append(DiscrepancySource.CURRENCY_CONVERSION)
        
        # Update records
        buyer_record.impressions_delivered += max(0, buyer_imps)
        seller_record.impressions_delivered += max(0, seller_imps)
        buyer_record.total_spend += max(0, buyer_spend)
        seller_record.total_spend += max(0, seller_spend)
        buyer_record.last_updated = datetime.now()
        seller_record.last_updated = datetime.now()
        
        return sources


class ReconciliationEngine:
    """
    Simulates end-of-campaign reconciliation between buyer and seller.
    
    Models the real-world process:
    1. Campaign ends
    2. Both parties generate final reports
    3. Compare reports
    4. Attempt resolution:
       - <3%: Auto-accept
       - 3-10%: Negotiate (split difference)
       - 10-15%: Formal dispute process
       - >15%: Often unresolvable
    """
    
    def __init__(
        self,
        config: Optional[DiscrepancyConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or DiscrepancyConfig()
        self._random = random.Random(seed)
    
    def calculate_discrepancy(
        self,
        buyer_record: CampaignRecord,
        seller_record: CampaignRecord,
    ) -> tuple[float, float, float]:
        """
        Calculate discrepancy between buyer and seller records.
        
        Returns: (impression_pct, spend_pct, absolute_spend_diff)
        """
        # Impression discrepancy
        imp_total = max(buyer_record.impressions_delivered, seller_record.impressions_delivered, 1)
        imp_diff = abs(buyer_record.impressions_delivered - seller_record.impressions_delivered)
        imp_pct = imp_diff / imp_total
        
        # Spend discrepancy
        spend_total = max(buyer_record.total_spend, seller_record.total_spend, 0.01)
        spend_diff = abs(buyer_record.total_spend - seller_record.total_spend)
        spend_pct = spend_diff / spend_total
        
        return imp_pct, spend_pct, spend_diff
    
    def attempt_reconciliation(
        self,
        buyer_record: CampaignRecord,
        seller_record: CampaignRecord,
        has_shared_ledger: bool = False,
        ledger_record: Optional[CampaignRecord] = None,
    ) -> ReconciliationResult:
        """
        Attempt to reconcile buyer and seller records.
        
        Args:
            buyer_record: Buyer's campaign record
            seller_record: Seller's campaign record  
            has_shared_ledger: If True, use ledger as source of truth
            ledger_record: The ledger's authoritative record (if shared ledger exists)
        
        Returns:
            ReconciliationResult with outcome and final values
        """
        imp_pct, spend_pct, spend_diff = self.calculate_discrepancy(buyer_record, seller_record)
        
        result = ReconciliationResult(
            campaign_id=buyer_record.campaign_id,
            buyer_id=buyer_record.party_id,
            seller_id=seller_record.party_id,
            buyer_impressions=buyer_record.impressions_delivered,
            seller_impressions=seller_record.impressions_delivered,
            buyer_spend=buyer_record.total_spend,
            seller_spend=seller_record.total_spend,
            impression_discrepancy_pct=imp_pct * 100,
            spend_discrepancy_pct=spend_pct * 100,
            absolute_spend_difference=spend_diff,
        )
        
        # If we have a shared ledger, reconciliation is trivial
        if has_shared_ledger and ledger_record:
            result.outcome = ResolutionOutcome.AGREED
            result.final_impressions = ledger_record.impressions_delivered
            result.final_spend = ledger_record.total_spend
            result.days_to_resolve = 0
            result.resolution_method = "ledger_authoritative"
            return result
        
        # No shared ledger - must negotiate based on threshold
        if imp_pct < self.config.auto_accept_threshold:
            # Small discrepancy - buyer accepts seller's count (industry norm)
            result.outcome = ResolutionOutcome.BUYER_ACCEPTS
            result.final_impressions = seller_record.impressions_delivered
            result.final_spend = seller_record.total_spend
            result.days_to_resolve = 1
            result.resolution_method = "auto_accept"
            
        elif imp_pct < self.config.negotiation_threshold:
            # Moderate discrepancy - negotiate (typically split)
            result.outcome = ResolutionOutcome.NEGOTIATED
            result.final_impressions = (buyer_record.impressions_delivered + seller_record.impressions_delivered) // 2
            result.final_spend = (buyer_record.total_spend + seller_record.total_spend) / 2
            result.days_to_resolve = self._random.randint(7, 21)
            result.resolution_method = "negotiated_average"
            
        elif imp_pct < self.config.dispute_threshold:
            # Significant discrepancy - formal dispute
            result.outcome = ResolutionOutcome.DISPUTED
            result.days_to_resolve = self._random.randint(30, 60)
            result.resolution_method = "dispute_process"
            
            # 50% chance of resolution after dispute process
            if self._random.random() < 0.5:
                result.outcome = ResolutionOutcome.NEGOTIATED
                # Usually seller's number wins in disputes (they have delivery logs)
                result.final_impressions = seller_record.impressions_delivered
                result.final_spend = seller_record.total_spend
                
        elif imp_pct < self.config.unresolvable_above:
            # Major discrepancy - often unresolvable
            result.outcome = ResolutionOutcome.DISPUTED
            result.days_to_resolve = self._random.randint(45, 90)
            
            # Only 30% resolution rate at this level
            if self._random.random() < 0.3:
                result.outcome = ResolutionOutcome.NEGOTIATED
                result.final_impressions = seller_record.impressions_delivered
                result.final_spend = seller_record.total_spend
            else:
                result.outcome = ResolutionOutcome.UNRESOLVABLE
                result.resolution_method = "unresolvable_no_source_of_truth"
        else:
            # Extreme discrepancy - no resolution possible
            result.outcome = ResolutionOutcome.UNRESOLVABLE
            result.days_to_resolve = 90  # Max tracking period
            result.resolution_method = "unresolvable_extreme_divergence"
        
        return result


@dataclass
class ReconciliationMetrics:
    """Aggregate metrics for reconciliation simulation."""
    
    total_campaigns: int = 0
    
    # Outcome counts
    agreed: int = 0
    negotiated: int = 0
    buyer_accepts: int = 0
    seller_accepts: int = 0
    disputed: int = 0
    unresolvable: int = 0
    
    # Financial impact
    total_buyer_spend: float = 0.0
    total_seller_spend: float = 0.0
    total_disputed_spend: float = 0.0
    total_unresolvable_spend: float = 0.0
    
    # Time metrics
    total_resolution_days: int = 0
    
    # Discrepancy distribution
    discrepancies_under_3pct: int = 0
    discrepancies_3_to_10pct: int = 0
    discrepancies_10_to_15pct: int = 0
    discrepancies_over_15pct: int = 0
    
    @property
    def resolution_rate(self) -> float:
        if self.total_campaigns == 0:
            return 0.0
        resolved = self.agreed + self.negotiated + self.buyer_accepts + self.seller_accepts
        return resolved / self.total_campaigns
    
    @property
    def unresolvable_rate(self) -> float:
        if self.total_campaigns == 0:
            return 0.0
        return self.unresolvable / self.total_campaigns
    
    @property
    def dispute_rate(self) -> float:
        if self.total_campaigns == 0:
            return 0.0
        return (self.disputed + self.unresolvable) / self.total_campaigns
    
    @property
    def average_resolution_days(self) -> float:
        resolved = self.agreed + self.negotiated + self.buyer_accepts + self.seller_accepts
        if resolved == 0:
            return 0.0
        return self.total_resolution_days / resolved
    
    def add_result(self, result: ReconciliationResult):
        self.total_campaigns += 1
        self.total_buyer_spend += result.buyer_spend
        self.total_seller_spend += result.seller_spend
        self.total_resolution_days += result.days_to_resolve
        
        # Count outcome
        if result.outcome == ResolutionOutcome.AGREED:
            self.agreed += 1
        elif result.outcome == ResolutionOutcome.NEGOTIATED:
            self.negotiated += 1
        elif result.outcome == ResolutionOutcome.BUYER_ACCEPTS:
            self.buyer_accepts += 1
        elif result.outcome == ResolutionOutcome.SELLER_ACCEPTS:
            self.seller_accepts += 1
        elif result.outcome == ResolutionOutcome.DISPUTED:
            self.disputed += 1
            self.total_disputed_spend += result.absolute_spend_difference
        elif result.outcome == ResolutionOutcome.UNRESOLVABLE:
            self.unresolvable += 1
            self.total_unresolvable_spend += max(result.buyer_spend, result.seller_spend)
        
        # Categorize discrepancy
        pct = result.impression_discrepancy_pct
        if pct < 3:
            self.discrepancies_under_3pct += 1
        elif pct < 10:
            self.discrepancies_3_to_10pct += 1
        elif pct < 15:
            self.discrepancies_10_to_15pct += 1
        else:
            self.discrepancies_over_15pct += 1
    
    def to_dict(self) -> dict:
        return {
            "total_campaigns": self.total_campaigns,
            "resolution_rate": round(self.resolution_rate * 100, 1),
            "dispute_rate": round(self.dispute_rate * 100, 1),
            "unresolvable_rate": round(self.unresolvable_rate * 100, 1),
            "outcomes": {
                "agreed": self.agreed,
                "negotiated": self.negotiated,
                "buyer_accepts": self.buyer_accepts,
                "seller_accepts": self.seller_accepts,
                "disputed": self.disputed,
                "unresolvable": self.unresolvable,
            },
            "financial_impact": {
                "total_buyer_spend": round(self.total_buyer_spend, 2),
                "total_seller_spend": round(self.total_seller_spend, 2),
                "disputed_spend": round(self.total_disputed_spend, 2),
                "unresolvable_spend": round(self.total_unresolvable_spend, 2),
            },
            "discrepancy_distribution": {
                "under_3pct": self.discrepancies_under_3pct,
                "3_to_10pct": self.discrepancies_3_to_10pct,
                "10_to_15pct": self.discrepancies_10_to_15pct,
                "over_15pct": self.discrepancies_over_15pct,
            },
            "average_resolution_days": round(self.average_resolution_days, 1),
        }


class ReconciliationSimulator:
    """
    Full reconciliation simulation runner.
    
    Runs campaigns with discrepancy injection, then attempts reconciliation,
    comparing outcomes with and without shared ledger.
    """
    
    def __init__(
        self,
        config: Optional[DiscrepancyConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or DiscrepancyConfig()
        self._seed = seed
        self.injector = DiscrepancyInjector(config, seed)
        self.engine = ReconciliationEngine(config, seed)
        self._random = random.Random(seed)
    
    async def run_campaign(
        self,
        campaign_id: str,
        buyer_id: str,
        seller_id: str,
        duration_days: int,
        total_budget: float,
        target_impressions: int,
        has_shared_ledger: bool = False,
    ) -> tuple[CampaignRecord, CampaignRecord, Optional[CampaignRecord]]:
        """
        Simulate a full campaign with daily discrepancy injection.
        
        Returns: (buyer_record, seller_record, ledger_record or None)
        """
        # Initialize records
        buyer_record = CampaignRecord(
            campaign_id=campaign_id,
            party_id=buyer_id,
            party_type="buyer",
            campaign_start=datetime.now(),
        )
        seller_record = CampaignRecord(
            campaign_id=campaign_id,
            party_id=seller_id,
            party_type="seller",
            campaign_start=datetime.now(),
        )
        ledger_record = None
        if has_shared_ledger:
            ledger_record = CampaignRecord(
                campaign_id=campaign_id,
                party_id="ledger",
                party_type="ledger",
                campaign_start=datetime.now(),
            )
        
        # Daily delivery
        daily_budget = total_budget / duration_days
        daily_impressions_target = target_impressions // duration_days
        
        for day in range(1, duration_days + 1):
            # Ground truth delivery (with some variance)
            actual_imps = int(daily_impressions_target * self._random.uniform(0.8, 1.2))
            actual_spend = daily_budget * (actual_imps / daily_impressions_target)
            
            # Inject discrepancies into buyer/seller records
            self.injector.inject_daily_discrepancy(
                buyer_record=buyer_record,
                seller_record=seller_record,
                simulation_day=day,
                daily_impressions=actual_imps,
                daily_spend=actual_spend,
            )
            
            # Ledger gets ground truth (if exists)
            if ledger_record:
                ledger_record.impressions_delivered += actual_imps
                ledger_record.total_spend += actual_spend
        
        # Set end timestamps
        buyer_record.campaign_end = datetime.now()
        seller_record.campaign_end = datetime.now()
        if ledger_record:
            ledger_record.campaign_end = datetime.now()
        
        return buyer_record, seller_record, ledger_record
    
    async def run_comparison(
        self,
        num_campaigns: int = 50,
        campaign_days: int = 30,
        avg_budget: float = 50000,
        avg_impressions: int = 3000000,
    ) -> dict[str, ReconciliationMetrics]:
        """
        Run full comparison simulation.
        
        Runs same campaigns with and without shared ledger,
        comparing reconciliation outcomes.
        
        Returns metrics for both scenarios.
        """
        metrics_fragmented = ReconciliationMetrics()
        metrics_ledger = ReconciliationMetrics()
        
        logger.info(
            "reconciliation_simulation.start",
            num_campaigns=num_campaigns,
            campaign_days=campaign_days,
        )
        
        for i in range(num_campaigns):
            campaign_id = f"camp-{i+1:04d}"
            buyer_id = f"buyer-{(i % 5) + 1:03d}"
            seller_id = f"seller-{(i % 5) + 1:03d}"
            
            # Randomize campaign parameters
            budget = avg_budget * self._random.uniform(0.5, 2.0)
            impressions = int(avg_impressions * self._random.uniform(0.5, 2.0))
            
            # Run campaign WITHOUT shared ledger (Scenario B)
            buyer_b, seller_b, _ = await self.run_campaign(
                campaign_id=f"{campaign_id}-B",
                buyer_id=buyer_id,
                seller_id=seller_id,
                duration_days=campaign_days,
                total_budget=budget,
                target_impressions=impressions,
                has_shared_ledger=False,
            )
            
            # Attempt reconciliation without ledger
            result_b = self.engine.attempt_reconciliation(
                buyer_record=buyer_b,
                seller_record=seller_b,
                has_shared_ledger=False,
            )
            metrics_fragmented.add_result(result_b)
            
            # Run same campaign WITH shared ledger (Scenario C)
            buyer_c, seller_c, ledger_c = await self.run_campaign(
                campaign_id=f"{campaign_id}-C",
                buyer_id=buyer_id,
                seller_id=seller_id,
                duration_days=campaign_days,
                total_budget=budget,
                target_impressions=impressions,
                has_shared_ledger=True,
            )
            
            # Attempt reconciliation with ledger
            result_c = self.engine.attempt_reconciliation(
                buyer_record=buyer_c,
                seller_record=seller_c,
                has_shared_ledger=True,
                ledger_record=ledger_c,
            )
            metrics_ledger.add_result(result_c)
            
            if (i + 1) % 10 == 0:
                logger.info(
                    "reconciliation_simulation.progress",
                    completed=i + 1,
                    total=num_campaigns,
                )
        
        logger.info(
            "reconciliation_simulation.complete",
            fragmented_resolution_rate=f"{metrics_fragmented.resolution_rate:.1%}",
            ledger_resolution_rate=f"{metrics_ledger.resolution_rate:.1%}",
        )
        
        return {
            "B_fragmented": metrics_fragmented,
            "C_ledger": metrics_ledger,
        }


# CLI entry point for testing
async def main():
    """Run reconciliation simulation from command line."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Cross-Agent Reconciliation Simulation")
    parser.add_argument("--campaigns", type=int, default=50, help="Number of campaigns")
    parser.add_argument("--days", type=int, default=30, help="Campaign duration in days")
    parser.add_argument("--budget", type=float, default=50000, help="Average campaign budget")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    simulator = ReconciliationSimulator(seed=args.seed)
    results = await simulator.run_comparison(
        num_campaigns=args.campaigns,
        campaign_days=args.days,
        avg_budget=args.budget,
    )
    
    output = {
        "simulation_params": {
            "campaigns": args.campaigns,
            "days": args.days,
            "avg_budget": args.budget,
            "seed": args.seed,
        },
        "results": {
            "B_fragmented": results["B_fragmented"].to_dict(),
            "C_ledger": results["C_ledger"].to_dict(),
        },
        "summary": {
            "fragmented_unresolvable_rate": f"{results['B_fragmented'].unresolvable_rate:.1%}",
            "ledger_unresolvable_rate": f"{results['C_ledger'].unresolvable_rate:.1%}",
            "fragmented_disputed_spend": f"${results['B_fragmented'].total_disputed_spend:,.2f}",
            "fragmented_unresolvable_spend": f"${results['B_fragmented'].total_unresolvable_spend:,.2f}",
        }
    }
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
