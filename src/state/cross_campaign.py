"""
Cross-Campaign State Management.

State that spans multiple campaigns, including:
- Shared audiences (which campaigns target which audiences)
- Inventory commitments (which products are committed to which campaigns)
- Budget pacing across campaigns

This is critical for context rot demonstration:
- Without ledger: this state can diverge between agents
- With ledger: always consistent via shared source of truth
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from enum import Enum
import structlog

logger = structlog.get_logger()


class PacingStatus(str, Enum):
    """Budget pacing status."""
    ON_TRACK = "on_track"
    BEHIND = "behind"
    AHEAD = "ahead"
    OVERSPENT = "overspent"
    COMPLETE = "complete"


@dataclass
class Commit:
    """
    A commitment to inventory from a campaign.
    
    Commits reserve inventory capacity for a campaign.
    Multiple campaigns competing for the same inventory
    creates contention that must be managed.
    """
    commit_id: str
    campaign_id: str
    product_id: str
    seller_id: str
    
    # Commitment details
    impressions_reserved: int = 0
    spend_committed: float = 0.0
    cpm_rate: float = 0.0
    
    # Fulfillment
    impressions_delivered: int = 0
    spend_actual: float = 0.0
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # Status
    status: Literal["pending", "active", "fulfilled", "cancelled", "expired"] = "pending"
    
    @property
    def fulfillment_rate(self) -> float:
        """Percentage of commitment fulfilled."""
        if self.impressions_reserved == 0:
            return 0.0
        return self.impressions_delivered / self.impressions_reserved
    
    @property
    def is_active(self) -> bool:
        return self.status == "active"
    
    def to_dict(self) -> dict:
        return {
            "commit_id": self.commit_id,
            "campaign_id": self.campaign_id,
            "product_id": self.product_id,
            "seller_id": self.seller_id,
            "impressions_reserved": self.impressions_reserved,
            "impressions_delivered": self.impressions_delivered,
            "fulfillment_rate": round(self.fulfillment_rate * 100, 1),
            "spend_committed": self.spend_committed,
            "spend_actual": self.spend_actual,
            "status": self.status,
        }


@dataclass
class PacingState:
    """
    Budget pacing state for a campaign.
    
    Tracks how well the campaign is pacing against its budget
    and timeline. Critical for cross-campaign coordination.
    """
    campaign_id: str
    
    # Budget targets
    total_budget: float = 0.0
    daily_budget: float = 0.0
    
    # Current state
    spent_today: float = 0.0
    spent_total: float = 0.0
    
    # Timeline
    days_elapsed: int = 0
    days_remaining: int = 0
    
    # Status
    status: PacingStatus = PacingStatus.ON_TRACK
    
    # Calculated pacing
    expected_spend: float = 0.0
    actual_pacing: float = 0.0  # vs expected
    
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def budget_remaining(self) -> float:
        return max(0, self.total_budget - self.spent_total)
    
    @property
    def recommended_daily(self) -> float:
        """Recommended daily spend to stay on track."""
        if self.days_remaining <= 0:
            return 0.0
        return self.budget_remaining / self.days_remaining
    
    def calculate_status(self) -> PacingStatus:
        """Calculate pacing status based on current state."""
        if self.spent_total >= self.total_budget:
            return PacingStatus.OVERSPENT if self.days_remaining > 0 else PacingStatus.COMPLETE
        
        if self.days_elapsed == 0:
            return PacingStatus.ON_TRACK
        
        expected_daily_avg = self.total_budget / (self.days_elapsed + self.days_remaining)
        actual_daily_avg = self.spent_total / self.days_elapsed
        
        pacing_ratio = actual_daily_avg / expected_daily_avg if expected_daily_avg > 0 else 1.0
        
        if pacing_ratio < 0.8:
            return PacingStatus.BEHIND
        elif pacing_ratio > 1.2:
            return PacingStatus.AHEAD
        else:
            return PacingStatus.ON_TRACK
    
    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "total_budget": self.total_budget,
            "spent_total": round(self.spent_total, 2),
            "budget_remaining": round(self.budget_remaining, 2),
            "daily_budget": round(self.daily_budget, 2),
            "spent_today": round(self.spent_today, 2),
            "days_elapsed": self.days_elapsed,
            "days_remaining": self.days_remaining,
            "status": self.status.value,
            "recommended_daily": round(self.recommended_daily, 2),
        }


@dataclass
class ContentionResult:
    """
    Result of checking inventory contention.
    
    When multiple campaigns want the same inventory,
    there is contention that affects pricing and availability.
    """
    product_id: str
    total_capacity: int  # Available impressions
    total_committed: int  # Impressions already committed
    available: int  # Remaining availability
    
    # Campaigns competing for this inventory
    competing_campaigns: list[str] = field(default_factory=list)
    commits: list[Commit] = field(default_factory=list)
    
    # Contention level
    contention_ratio: float = 0.0  # committed / capacity
    is_constrained: bool = False
    
    # Recommendations
    recommended_max_commit: int = 0
    price_pressure: float = 1.0  # Multiplier on base CPM
    
    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "total_capacity": self.total_capacity,
            "total_committed": self.total_committed,
            "available": self.available,
            "competing_campaigns": self.competing_campaigns,
            "contention_ratio": round(self.contention_ratio, 2),
            "is_constrained": self.is_constrained,
            "recommended_max_commit": self.recommended_max_commit,
            "price_pressure": round(self.price_pressure, 2),
        }


class CrossCampaignState:
    """
    State that spans multiple campaigns.
    
    Critical for context rot demonstration:
    - Without ledger: this state can diverge between agents
    - With ledger: always consistent via shared source of truth
    
    Manages:
    - Shared audiences across campaigns
    - Inventory commitments by product
    - Budget pacing across portfolio
    """
    
    def __init__(self):
        # audience_id -> list of campaign_ids using this audience
        self.shared_audiences: dict[str, list[str]] = {}
        
        # product_id -> list of commits against this inventory
        self.inventory_commits: dict[str, list[Commit]] = {}
        
        # campaign_id -> pacing state
        self.budget_pacing: dict[str, PacingState] = {}
        
        # Product capacity (inventory available)
        self._product_capacity: dict[str, int] = {}
        
        self._lock = asyncio.Lock()
        
        logger.info("cross_campaign_state.initialized")
    
    async def register_audience_usage(
        self,
        audience_id: str,
        campaign_id: str,
    ) -> list[str]:
        """
        Register a campaign's use of an audience.
        
        Returns list of other campaigns using the same audience.
        """
        async with self._lock:
            if audience_id not in self.shared_audiences:
                self.shared_audiences[audience_id] = []
            
            if campaign_id not in self.shared_audiences[audience_id]:
                self.shared_audiences[audience_id].append(campaign_id)
            
            # Return other campaigns using this audience
            return [
                cid for cid in self.shared_audiences[audience_id]
                if cid != campaign_id
            ]
    
    async def check_audience_overlap(
        self,
        campaign_a: str,
        campaign_b: str,
    ) -> float:
        """
        Check audience overlap between two campaigns.
        
        Returns overlap ratio (0.0 to 1.0) - percentage of audiences
        used by campaign_a that are also used by campaign_b.
        """
        async with self._lock:
            audiences_a = set()
            audiences_b = set()
            
            for audience_id, campaigns in self.shared_audiences.items():
                if campaign_a in campaigns:
                    audiences_a.add(audience_id)
                if campaign_b in campaigns:
                    audiences_b.add(audience_id)
            
            if not audiences_a:
                return 0.0
            
            overlap = audiences_a & audiences_b
            return len(overlap) / len(audiences_a)
    
    async def add_inventory_commit(
        self,
        commit: Commit,
        product_capacity: Optional[int] = None,
    ) -> ContentionResult:
        """
        Add an inventory commitment and check contention.
        
        Returns contention result showing impact on availability.
        """
        async with self._lock:
            product_id = commit.product_id
            
            # Set capacity if provided
            if product_capacity is not None:
                self._product_capacity[product_id] = product_capacity
            
            # Initialize product tracking if needed
            if product_id not in self.inventory_commits:
                self.inventory_commits[product_id] = []
            
            # Add commit
            self.inventory_commits[product_id].append(commit)
            
            # Calculate contention
            return self._calculate_contention(product_id)
    
    async def check_inventory_contention(
        self,
        product_id: str,
    ) -> ContentionResult:
        """
        Check if multiple campaigns want the same inventory.
        
        Returns contention analysis showing:
        - How much is committed vs available
        - Which campaigns are competing
        - Price pressure from contention
        """
        async with self._lock:
            return self._calculate_contention(product_id)
    
    def _calculate_contention(self, product_id: str) -> ContentionResult:
        """Calculate contention for a product (internal, no lock)."""
        capacity = self._product_capacity.get(product_id, 0)
        commits = self.inventory_commits.get(product_id, [])
        
        # Sum active commitments
        total_committed = sum(
            c.impressions_reserved
            for c in commits
            if c.is_active
        )
        
        available = max(0, capacity - total_committed)
        
        # Get competing campaigns
        competing = list(set(c.campaign_id for c in commits if c.is_active))
        
        # Calculate contention ratio
        contention_ratio = total_committed / capacity if capacity > 0 else 0.0
        
        # Determine if constrained (>80% committed)
        is_constrained = contention_ratio > 0.8
        
        # Calculate price pressure (increases with contention)
        # 0% contention = 1.0x, 100% contention = 2.0x
        price_pressure = 1.0 + contention_ratio
        
        # Recommended max commit (leave 10% buffer)
        recommended_max = int(available * 0.9)
        
        return ContentionResult(
            product_id=product_id,
            total_capacity=capacity,
            total_committed=total_committed,
            available=available,
            competing_campaigns=competing,
            commits=[c for c in commits if c.is_active],
            contention_ratio=contention_ratio,
            is_constrained=is_constrained,
            recommended_max_commit=recommended_max,
            price_pressure=price_pressure,
        )
    
    async def update_pacing(
        self,
        campaign_id: str,
        total_budget: float,
        spent_total: float,
        spent_today: float,
        days_elapsed: int,
        days_remaining: int,
    ) -> PacingState:
        """
        Update pacing state for a campaign.
        
        Returns updated pacing state with recommendations.
        """
        async with self._lock:
            pacing = PacingState(
                campaign_id=campaign_id,
                total_budget=total_budget,
                spent_total=spent_total,
                spent_today=spent_today,
                days_elapsed=days_elapsed,
                days_remaining=days_remaining,
                daily_budget=total_budget / (days_elapsed + days_remaining) if (days_elapsed + days_remaining) > 0 else 0,
                last_updated=datetime.now(),
            )
            
            # Calculate expected spend and pacing
            if days_elapsed + days_remaining > 0:
                pacing.expected_spend = (total_budget * days_elapsed) / (days_elapsed + days_remaining)
                pacing.actual_pacing = spent_total / pacing.expected_spend if pacing.expected_spend > 0 else 1.0
            
            pacing.status = pacing.calculate_status()
            
            self.budget_pacing[campaign_id] = pacing
            
            logger.debug(
                "pacing.updated",
                campaign_id=campaign_id,
                status=pacing.status.value,
                spent=spent_total,
                expected=pacing.expected_spend,
            )
            
            return pacing
    
    async def get_pacing(self, campaign_id: str) -> Optional[PacingState]:
        """Get pacing state for a campaign."""
        return self.budget_pacing.get(campaign_id)
    
    async def get_portfolio_pacing_summary(self) -> dict:
        """Get summary of pacing across all campaigns."""
        async with self._lock:
            summary = {
                "on_track": 0,
                "behind": 0,
                "ahead": 0,
                "overspent": 0,
                "complete": 0,
            }
            
            for pacing in self.budget_pacing.values():
                summary[pacing.status.value] += 1
            
            return {
                "campaign_count": len(self.budget_pacing),
                "status_breakdown": summary,
                "campaigns": {
                    cid: p.to_dict()
                    for cid, p in self.budget_pacing.items()
                },
            }
    
    def to_dict(self) -> dict:
        """Serialize cross-campaign state."""
        return {
            "shared_audiences": self.shared_audiences,
            "inventory_products": list(self.inventory_commits.keys()),
            "inventory_commits": {
                pid: [c.to_dict() for c in commits]
                for pid, commits in self.inventory_commits.items()
            },
            "pacing_campaigns": list(self.budget_pacing.keys()),
        }
