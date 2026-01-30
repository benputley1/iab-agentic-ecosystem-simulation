"""
Campaign Portfolio State Management.

Manages state across multiple concurrent campaigns. This is where
context rot becomes critical:
- Each campaign has its own state
- Budget must be tracked across campaigns
- Performance data accumulates
- Agent decisions affect other campaigns
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Literal
from enum import Enum
import structlog

logger = structlog.get_logger()


class CampaignStatus(str, Enum):
    """Campaign lifecycle status."""
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ConflictType(str, Enum):
    """Types of budget/resource conflicts."""
    BUDGET_OVERCOMMIT = "budget_overcommit"
    PACING_CONFLICT = "pacing_conflict"
    AUDIENCE_OVERLAP = "audience_overlap"
    INVENTORY_CONTENTION = "inventory_contention"
    TIMING_CONFLICT = "timing_conflict"


@dataclass
class Deal:
    """
    Represents a negotiated deal within a campaign.
    
    Deals are commitments that affect budget and inventory.
    """
    deal_id: str
    campaign_id: str
    seller_id: str
    product_id: str
    
    # Deal terms
    impressions_committed: int = 0
    cpm_agreed: float = 0.0
    total_value: float = 0.0
    
    # Delivery status
    impressions_delivered: int = 0
    spend_to_date: float = 0.0
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # State
    status: Literal["pending", "active", "completed", "cancelled"] = "pending"
    
    @property
    def delivery_pct(self) -> float:
        """Percentage of committed impressions delivered."""
        if self.impressions_committed == 0:
            return 0.0
        return self.impressions_delivered / self.impressions_committed
    
    @property
    def budget_remaining(self) -> float:
        """Remaining budget on this deal."""
        return max(0, self.total_value - self.spend_to_date)
    
    def to_dict(self) -> dict:
        return {
            "deal_id": self.deal_id,
            "campaign_id": self.campaign_id,
            "seller_id": self.seller_id,
            "product_id": self.product_id,
            "impressions_committed": self.impressions_committed,
            "impressions_delivered": self.impressions_delivered,
            "delivery_pct": round(self.delivery_pct * 100, 1),
            "cpm_agreed": self.cpm_agreed,
            "total_value": self.total_value,
            "spend_to_date": self.spend_to_date,
            "budget_remaining": self.budget_remaining,
            "status": self.status,
        }


@dataclass
class CampaignMetrics:
    """
    Performance metrics for a campaign.
    
    These accumulate over time and are a source of context rot
    as they must be tracked accurately across agent restarts.
    """
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    
    spend: float = 0.0
    
    # Derived metrics
    @property
    def ctr(self) -> float:
        """Click-through rate."""
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions
    
    @property
    def cvr(self) -> float:
        """Conversion rate."""
        if self.clicks == 0:
            return 0.0
        return self.conversions / self.clicks
    
    @property
    def cpm(self) -> float:
        """Cost per mille (thousand impressions)."""
        if self.impressions == 0:
            return 0.0
        return (self.spend / self.impressions) * 1000
    
    @property
    def cpc(self) -> float:
        """Cost per click."""
        if self.clicks == 0:
            return 0.0
        return self.spend / self.clicks
    
    @property
    def cpa(self) -> float:
        """Cost per acquisition/conversion."""
        if self.conversions == 0:
            return 0.0
        return self.spend / self.conversions
    
    def to_dict(self) -> dict:
        return {
            "impressions": self.impressions,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "spend": round(self.spend, 2),
            "ctr": round(self.ctr * 100, 4),
            "cvr": round(self.cvr * 100, 4),
            "cpm": round(self.cpm, 2),
            "cpc": round(self.cpc, 2),
            "cpa": round(self.cpa, 2),
        }
    
    def merge(self, other: "CampaignMetrics") -> "CampaignMetrics":
        """Merge two metric sets."""
        return CampaignMetrics(
            impressions=self.impressions + other.impressions,
            clicks=self.clicks + other.clicks,
            conversions=self.conversions + other.conversions,
            spend=self.spend + other.spend,
        )


@dataclass
class CampaignState:
    """
    Complete state for a single campaign.
    
    This represents what an agent must track for one campaign.
    In a multi-campaign portfolio, context rot occurs as the agent
    must manage many of these simultaneously.
    """
    campaign_id: str
    advertiser_id: str
    
    # Budget
    budget_total: float = 0.0
    budget_spent: float = 0.0
    budget_committed: float = 0.0  # In active deals
    
    # Status
    status: CampaignStatus = CampaignStatus.DRAFT
    
    # Targeting
    target_audience: list[str] = field(default_factory=list)
    target_channels: list[str] = field(default_factory=list)
    target_geo: list[str] = field(default_factory=list)
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Deals and performance
    deals: list[Deal] = field(default_factory=list)
    metrics: CampaignMetrics = field(default_factory=CampaignMetrics)
    
    # State version (for conflict detection)
    version: int = 1
    
    @property
    def budget_available(self) -> float:
        """Budget available for new deals."""
        return max(0, self.budget_total - self.budget_spent - self.budget_committed)
    
    @property
    def budget_utilization(self) -> float:
        """Percentage of budget utilized (spent + committed)."""
        if self.budget_total == 0:
            return 0.0
        return (self.budget_spent + self.budget_committed) / self.budget_total
    
    @property
    def active_deals(self) -> list[Deal]:
        """Currently active deals."""
        return [d for d in self.deals if d.status == "active"]
    
    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "advertiser_id": self.advertiser_id,
            "status": self.status.value,
            "budget": {
                "total": self.budget_total,
                "spent": round(self.budget_spent, 2),
                "committed": round(self.budget_committed, 2),
                "available": round(self.budget_available, 2),
                "utilization": round(self.budget_utilization * 100, 1),
            },
            "targeting": {
                "audiences": self.target_audience,
                "channels": self.target_channels,
                "geo": self.target_geo,
            },
            "deals_count": len(self.deals),
            "active_deals": len(self.active_deals),
            "metrics": self.metrics.to_dict(),
            "version": self.version,
        }


@dataclass
class StateUpdate:
    """
    An update to apply to campaign state.
    
    Updates are the unit of state change. In Scenario B, updates
    may be lost on restart. In Scenario C, updates are logged to ledger.
    """
    campaign_id: str
    update_type: str  # budget_spend, deal_add, deal_update, metrics_update, status_change
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Update payload (type-specific)
    payload: dict = field(default_factory=dict)
    
    # Tracking
    applied: bool = False
    ledger_tx: Optional[str] = None  # Ledger transaction ID if persisted


@dataclass
class Conflict:
    """
    A detected conflict in campaign state.
    
    Conflicts arise from:
    - Budget overcommitment across campaigns
    - Overlapping audience targeting
    - Inventory contention
    """
    conflict_type: ConflictType
    campaign_ids: list[str]
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    
    # Resolution hints
    suggested_action: Optional[str] = None
    affected_budget: float = 0.0
    
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "type": self.conflict_type.value,
            "campaigns": self.campaign_ids,
            "description": self.description,
            "severity": self.severity,
            "suggested_action": self.suggested_action,
            "affected_budget": self.affected_budget,
        }


@dataclass
class PortfolioView:
    """
    Aggregated view across all campaigns in portfolio.
    
    This is what an agent sees when making cross-campaign decisions.
    """
    active_campaigns: int
    total_campaigns: int
    
    total_budget: float
    total_spent: float
    total_committed: float
    budget_utilization: float
    
    # Allocation by channel
    channel_allocation: dict[str, float] = field(default_factory=dict)
    
    # Performance summary
    performance_summary: dict = field(default_factory=dict)
    
    # Active conflicts
    conflicts: list[Conflict] = field(default_factory=list)
    
    # Timestamp
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "campaigns": {
                "active": self.active_campaigns,
                "total": self.total_campaigns,
            },
            "budget": {
                "total": self.total_budget,
                "spent": round(self.total_spent, 2),
                "committed": round(self.total_committed, 2),
                "available": round(self.total_budget - self.total_spent - self.total_committed, 2),
                "utilization": round(self.budget_utilization * 100, 1),
            },
            "channel_allocation": self.channel_allocation,
            "performance": self.performance_summary,
            "conflicts_count": len(self.conflicts),
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


class Campaign:
    """Input campaign definition for adding to portfolio."""
    def __init__(
        self,
        campaign_id: str,
        advertiser_id: str,
        budget: float,
        target_audience: Optional[list[str]] = None,
        target_channels: Optional[list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        self.campaign_id = campaign_id
        self.advertiser_id = advertiser_id
        self.budget = budget
        self.target_audience = target_audience or []
        self.target_channels = target_channels or []
        self.start_date = start_date
        self.end_date = end_date


class CampaignPortfolio:
    """
    Manages state across multiple concurrent campaigns.
    
    This is where context rot becomes critical:
    - Each campaign has its own state
    - Budget must be tracked across campaigns
    - Performance data accumulates
    - Agent decisions affect other campaigns
    
    In Scenario B: This state is volatile (lost on restart)
    In Scenario C: This state is backed by ledger
    """
    
    def __init__(
        self,
        portfolio_id: str,
        total_budget: float = 0.0,
    ):
        self.portfolio_id = portfolio_id
        self.campaigns: dict[str, CampaignState] = {}
        self.total_budget = total_budget
        self.allocated_budget: float = 0.0
        
        # Update log (for reconciliation)
        self._update_log: list[StateUpdate] = []
        self._lock = asyncio.Lock()
        
        logger.info(
            "portfolio.created",
            portfolio_id=portfolio_id,
            total_budget=total_budget,
        )
    
    async def add_campaign(self, campaign: Campaign) -> CampaignState:
        """
        Add a campaign to the portfolio.
        
        Validates budget availability and creates campaign state.
        """
        async with self._lock:
            # Check if campaign already exists
            if campaign.campaign_id in self.campaigns:
                raise ValueError(f"Campaign {campaign.campaign_id} already exists")
            
            # Check budget availability
            if self.allocated_budget + campaign.budget > self.total_budget:
                raise ValueError(
                    f"Insufficient portfolio budget. "
                    f"Available: {self.total_budget - self.allocated_budget}, "
                    f"Requested: {campaign.budget}"
                )
            
            # Create campaign state
            state = CampaignState(
                campaign_id=campaign.campaign_id,
                advertiser_id=campaign.advertiser_id,
                budget_total=campaign.budget,
                target_audience=campaign.target_audience,
                target_channels=campaign.target_channels,
                start_date=campaign.start_date,
                end_date=campaign.end_date,
                status=CampaignStatus.DRAFT,
            )
            
            self.campaigns[campaign.campaign_id] = state
            self.allocated_budget += campaign.budget
            
            # Log update
            self._update_log.append(StateUpdate(
                campaign_id=campaign.campaign_id,
                update_type="campaign_add",
                payload={"budget": campaign.budget},
                applied=True,
            ))
            
            logger.info(
                "portfolio.campaign_added",
                campaign_id=campaign.campaign_id,
                budget=campaign.budget,
                total_allocated=self.allocated_budget,
            )
            
            return state
    
    async def update_campaign_state(
        self,
        campaign_id: str,
        update: StateUpdate,
    ) -> CampaignState:
        """
        Update a campaign's state.
        
        Applies the update and logs it for reconciliation.
        """
        async with self._lock:
            if campaign_id not in self.campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            state = self.campaigns[campaign_id]
            
            # Apply update based on type
            if update.update_type == "budget_spend":
                amount = update.payload.get("amount", 0.0)
                state.budget_spent += amount
                state.metrics.spend += amount
                
            elif update.update_type == "deal_add":
                deal_data = update.payload.get("deal")
                if deal_data:
                    deal = Deal(**deal_data)
                    state.deals.append(deal)
                    state.budget_committed += deal.total_value
                    
            elif update.update_type == "deal_update":
                deal_id = update.payload.get("deal_id")
                for deal in state.deals:
                    if deal.deal_id == deal_id:
                        if "impressions_delivered" in update.payload:
                            deal.impressions_delivered = update.payload["impressions_delivered"]
                        if "spend_to_date" in update.payload:
                            deal.spend_to_date = update.payload["spend_to_date"]
                        if "status" in update.payload:
                            deal.status = update.payload["status"]
                        break
                        
            elif update.update_type == "metrics_update":
                metrics = update.payload.get("metrics", {})
                state.metrics.impressions += metrics.get("impressions", 0)
                state.metrics.clicks += metrics.get("clicks", 0)
                state.metrics.conversions += metrics.get("conversions", 0)
                
            elif update.update_type == "status_change":
                new_status = update.payload.get("status")
                if new_status:
                    state.status = CampaignStatus(new_status)
            
            # Update version and timestamp
            state.version += 1
            state.updated_at = datetime.now()
            
            # Mark update as applied and log
            update.applied = True
            self._update_log.append(update)
            
            logger.debug(
                "portfolio.campaign_updated",
                campaign_id=campaign_id,
                update_type=update.update_type,
                version=state.version,
            )
            
            return state
    
    async def get_portfolio_view(self) -> PortfolioView:
        """
        Get aggregated view across all campaigns.
        
        This is the "big picture" view an agent uses for cross-campaign decisions.
        """
        async with self._lock:
            active = sum(
                1 for c in self.campaigns.values()
                if c.status == CampaignStatus.ACTIVE
            )
            
            total_spent = sum(c.budget_spent for c in self.campaigns.values())
            total_committed = sum(c.budget_committed for c in self.campaigns.values())
            
            # Calculate channel allocation
            channel_allocation: dict[str, float] = {}
            for campaign in self.campaigns.values():
                for channel in campaign.target_channels:
                    channel_allocation[channel] = channel_allocation.get(channel, 0) + campaign.budget_total
            
            # Aggregate performance
            total_metrics = CampaignMetrics()
            for campaign in self.campaigns.values():
                total_metrics = total_metrics.merge(campaign.metrics)
            
            # Check for conflicts
            conflicts = await self.check_budget_conflicts()
            
            utilization = 0.0
            if self.total_budget > 0:
                utilization = (total_spent + total_committed) / self.total_budget
            
            return PortfolioView(
                active_campaigns=active,
                total_campaigns=len(self.campaigns),
                total_budget=self.total_budget,
                total_spent=total_spent,
                total_committed=total_committed,
                budget_utilization=utilization,
                channel_allocation=channel_allocation,
                performance_summary=total_metrics.to_dict(),
                conflicts=conflicts,
            )
    
    async def check_budget_conflicts(self) -> list[Conflict]:
        """
        Check for budget conflicts between campaigns.
        
        Detects:
        - Budget overcommitment
        - Pacing issues
        - Resource contention
        """
        conflicts = []
        
        # Check total budget overcommit
        total_committed = sum(
            c.budget_spent + c.budget_committed
            for c in self.campaigns.values()
        )
        
        if total_committed > self.total_budget:
            overcommit = total_committed - self.total_budget
            conflicts.append(Conflict(
                conflict_type=ConflictType.BUDGET_OVERCOMMIT,
                campaign_ids=list(self.campaigns.keys()),
                description=f"Portfolio budget overcommitted by ${overcommit:.2f}",
                severity="critical",
                suggested_action="Pause low-priority campaigns or increase portfolio budget",
                affected_budget=overcommit,
            ))
        
        # Check individual campaign pacing issues
        for campaign in self.campaigns.values():
            if campaign.status != CampaignStatus.ACTIVE:
                continue
                
            # Check if spending too fast
            if campaign.budget_utilization > 0.9 and campaign.end_date:
                days_elapsed = (datetime.now() - (campaign.start_date or campaign.created_at)).days
                days_total = (campaign.end_date - (campaign.start_date or campaign.created_at)).days
                
                if days_total > 0:
                    expected_pacing = days_elapsed / days_total
                    if campaign.budget_utilization > expected_pacing + 0.2:
                        conflicts.append(Conflict(
                            conflict_type=ConflictType.PACING_CONFLICT,
                            campaign_ids=[campaign.campaign_id],
                            description=f"Campaign {campaign.campaign_id} spending too fast",
                            severity="medium",
                            suggested_action="Reduce bid prices or pause temporarily",
                            affected_budget=campaign.budget_available,
                        ))
        
        # Check audience overlap (potential cannibalization)
        campaign_list = list(self.campaigns.values())
        for i, camp_a in enumerate(campaign_list):
            for camp_b in campaign_list[i+1:]:
                overlap = set(camp_a.target_audience) & set(camp_b.target_audience)
                if len(overlap) > len(camp_a.target_audience) * 0.5:
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.AUDIENCE_OVERLAP,
                        campaign_ids=[camp_a.campaign_id, camp_b.campaign_id],
                        description=f"Significant audience overlap: {overlap}",
                        severity="low",
                        suggested_action="Consider consolidating or differentiating targeting",
                    ))
        
        return conflicts
    
    async def get_campaign(self, campaign_id: str) -> Optional[CampaignState]:
        """Get a specific campaign's state."""
        return self.campaigns.get(campaign_id)
    
    async def get_update_log(self) -> list[StateUpdate]:
        """Get all updates for reconciliation."""
        return self._update_log.copy()
    
    def to_dict(self) -> dict:
        """Serialize portfolio state."""
        return {
            "portfolio_id": self.portfolio_id,
            "total_budget": self.total_budget,
            "allocated_budget": self.allocated_budget,
            "campaigns": {
                cid: state.to_dict()
                for cid, state in self.campaigns.items()
            },
            "update_count": len(self._update_log),
        }
