"""Data models for the Buyer Agent System.

This module defines the core data structures used by the buyer agent hierarchy,
from campaign specifications to budget allocations.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class Channel(str, Enum):
    """Available advertising channels."""
    
    DISPLAY = "display"
    VIDEO = "video"
    CTV = "ctv"
    MOBILE_APP = "mobile_app"
    NATIVE = "native"
    AUDIO = "audio"


class CampaignStatus(str, Enum):
    """Campaign lifecycle status."""
    
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class AudienceSpec:
    """Audience targeting specification."""
    
    segments: list[str] = field(default_factory=list)
    demographics: dict[str, Any] = field(default_factory=dict)
    geo_targets: list[str] = field(default_factory=list)
    device_types: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # contextual targeting
    
    # UCP/AA integration (for Scenario C)
    ucp_embeddings: Optional[list[float]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "segments": self.segments,
            "demographics": self.demographics,
            "geo_targets": self.geo_targets,
            "device_types": self.device_types,
            "contexts": self.contexts,
            "has_ucp": self.ucp_embeddings is not None,
        }


@dataclass
class CampaignObjectives:
    """Campaign objectives and KPIs."""
    
    reach_target: int  # Target unique reach
    frequency_cap: int  # Max impressions per user
    cpm_target: float  # Target CPM in dollars
    channel_mix: dict[str, float] = field(default_factory=dict)  # channel -> budget %
    
    # Additional KPI targets
    viewability_target: float = 0.7  # 70% viewability
    brand_safety_level: str = "high"  # brand safety requirement
    
    def validate(self) -> bool:
        """Validate objectives are internally consistent."""
        if self.channel_mix:
            total = sum(self.channel_mix.values())
            return abs(total - 1.0) < 0.01  # Should sum to 100%
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "reach_target": self.reach_target,
            "frequency_cap": self.frequency_cap,
            "cpm_target": self.cpm_target,
            "channel_mix": self.channel_mix,
            "viewability_target": self.viewability_target,
            "brand_safety_level": self.brand_safety_level,
        }


@dataclass
class Campaign:
    """A buyer's advertising campaign.
    
    Represents a complete campaign configuration including budget,
    schedule, objectives, and audience targeting.
    """
    
    campaign_id: str
    name: str
    advertiser: str
    total_budget: float
    start_date: date
    end_date: date
    objectives: CampaignObjectives
    audience: AudienceSpec
    
    # Optional fields
    status: CampaignStatus = CampaignStatus.PENDING
    priority: int = 1  # Higher = more important
    notes: str = ""
    
    # Runtime tracking
    spend: float = 0.0
    impressions_delivered: int = 0
    deals_made: int = 0
    
    @property
    def remaining_budget(self) -> float:
        """Budget remaining after spend."""
        return max(0, self.total_budget - self.spend)
    
    @property
    def budget_utilization(self) -> float:
        """Percentage of budget spent."""
        if self.total_budget == 0:
            return 0.0
        return self.spend / self.total_budget
    
    @property
    def campaign_duration_days(self) -> int:
        """Total campaign duration in days."""
        return (self.end_date - self.start_date).days
    
    @property
    def daily_budget(self) -> float:
        """Ideal daily budget for even pacing."""
        days = self.campaign_duration_days
        if days <= 0:
            return self.total_budget
        return self.total_budget / days
    
    @property
    def is_active(self) -> bool:
        """Whether campaign is currently active and has budget."""
        today = date.today()
        return (
            self.status == CampaignStatus.ACTIVE
            and self.start_date <= today <= self.end_date
            and self.remaining_budget > 0
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "advertiser": self.advertiser,
            "total_budget": self.total_budget,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "objectives": self.objectives.to_dict(),
            "audience": self.audience.to_dict(),
            "status": self.status.value,
            "priority": self.priority,
            "spend": self.spend,
            "impressions_delivered": self.impressions_delivered,
            "remaining_budget": self.remaining_budget,
        }


@dataclass
class ChannelAllocation:
    """Budget allocation for a single channel."""
    
    channel: str
    amount: float
    percentage: float  # Of campaign budget
    priority: int = 1
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAllocation:
    """Budget allocation across campaigns and channels.
    
    This is the output of the Portfolio Manager's strategic allocation decision.
    """
    
    # campaign_id -> channel -> amount
    allocations: dict[str, dict[str, float]] = field(default_factory=dict)
    reasoning: str = ""  # LLM's explanation of allocation strategy
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_allocated: float = 0.0
    
    def get_campaign_total(self, campaign_id: str) -> float:
        """Get total allocation for a campaign."""
        if campaign_id not in self.allocations:
            return 0.0
        return sum(self.allocations[campaign_id].values())
    
    def get_channel_total(self, channel: str) -> float:
        """Get total allocation across all campaigns for a channel."""
        total = 0.0
        for campaign_allocations in self.allocations.values():
            total += campaign_allocations.get(channel, 0.0)
        return total
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "allocations": self.allocations,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "total_allocated": self.total_allocated,
        }


@dataclass
class ChannelSelection:
    """Result of channel selection decision."""
    
    channel: str
    selected: bool
    allocation_pct: float  # Percentage of campaign budget
    rationale: str
    expected_reach: int = 0
    expected_cpm: float = 0.0


@dataclass
class PortfolioState:
    """Aggregated state of the buyer's campaign portfolio.
    
    This represents the complete view of all campaigns and their performance.
    """
    
    portfolio_id: str
    campaigns: dict[str, Campaign] = field(default_factory=dict)
    current_allocation: Optional[BudgetAllocation] = None
    
    # Aggregate metrics
    total_budget: float = 0.0
    total_spend: float = 0.0
    total_impressions: int = 0
    active_campaigns: int = 0
    
    # Performance tracking
    overall_cpm: float = 0.0
    budget_pacing: float = 1.0  # 1.0 = on track, <1 = underspent, >1 = overspent
    
    def add_campaign(self, campaign: Campaign) -> None:
        """Add a campaign to the portfolio."""
        self.campaigns[campaign.campaign_id] = campaign
        self._recalculate_totals()
    
    def remove_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Remove a campaign from the portfolio."""
        campaign = self.campaigns.pop(campaign_id, None)
        if campaign:
            self._recalculate_totals()
        return campaign
    
    def _recalculate_totals(self) -> None:
        """Recalculate aggregate metrics."""
        self.total_budget = sum(c.total_budget for c in self.campaigns.values())
        self.total_spend = sum(c.spend for c in self.campaigns.values())
        self.total_impressions = sum(c.impressions_delivered for c in self.campaigns.values())
        self.active_campaigns = sum(1 for c in self.campaigns.values() if c.is_active)
        
        if self.total_impressions > 0:
            self.overall_cpm = (self.total_spend / self.total_impressions) * 1000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "portfolio_id": self.portfolio_id,
            "campaigns": {k: v.to_dict() for k, v in self.campaigns.items()},
            "total_budget": self.total_budget,
            "total_spend": self.total_spend,
            "total_impressions": self.total_impressions,
            "active_campaigns": self.active_campaigns,
            "overall_cpm": self.overall_cpm,
            "budget_pacing": self.budget_pacing,
        }


@dataclass
class SpecialistTask:
    """Task to be delegated to a Level 2 specialist."""
    
    task_id: str
    campaign_id: str
    channel: str
    action: str  # e.g., "discover_inventory", "negotiate_deal", "execute_buy"
    parameters: dict[str, Any] = field(default_factory=dict)
    budget_limit: float = 0.0
    priority: int = 1


@dataclass 
class SpecialistResult:
    """Result returned from a Level 2 specialist."""
    
    task_id: str
    campaign_id: str
    channel: str
    success: bool
    impressions_secured: int = 0
    spend: float = 0.0
    deals: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
