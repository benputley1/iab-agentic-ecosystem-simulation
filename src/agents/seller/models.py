"""Data models for the Seller Agent System.

These models represent the core entities used by seller agents
for inventory management, deal negotiation, and yield optimization.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class DealAction(str, Enum):
    """Actions for deal decisions."""
    
    ACCEPT = "accept"
    REJECT = "reject"
    COUNTER = "counter"


class DealTypeEnum(str, Enum):
    """Types of programmatic deals."""
    
    PROGRAMMATIC_GUARANTEED = "PG"
    PRIVATE_MARKETPLACE = "PMP"
    OPEN_AUCTION = "open"
    PREFERRED_DEAL = "PD"


class BuyerTier(str, Enum):
    """Buyer relationship tiers affecting pricing/priority."""
    
    PUBLIC = "public"  # Open market buyer
    SEAT = "seat"  # Has a seat/account
    AGENCY = "agency"  # Agency relationship
    ADVERTISER = "advertiser"  # Direct advertiser relationship


class ChannelType(str, Enum):
    """Inventory channel types."""
    
    DISPLAY = "display"
    VIDEO = "video"
    CTV = "ctv"
    MOBILE_APP = "mobile_app"
    NATIVE = "native"


@dataclass
class AudienceSpec:
    """Audience targeting specification."""
    
    segments: list[str] = field(default_factory=list)
    demographics: dict[str, Any] = field(default_factory=dict)
    geo_targets: list[str] = field(default_factory=list)
    device_types: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "segments": self.segments,
            "demographics": self.demographics,
            "geo_targets": self.geo_targets,
            "device_types": self.device_types,
            "contexts": self.contexts,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudienceSpec":
        """Create from dictionary."""
        return cls(
            segments=data.get("segments", []),
            demographics=data.get("demographics", {}),
            geo_targets=data.get("geo_targets", []),
            device_types=data.get("device_types", []),
            contexts=data.get("contexts", []),
        )


@dataclass
class Product:
    """A publisher's ad product/placement."""
    
    product_id: str
    name: str
    description: str
    channel: ChannelType
    base_cpm: float
    floor_cpm: float
    daily_impressions: int
    supported_deal_types: list[DealTypeEnum] = field(default_factory=list)
    audience_segments: list[str] = field(default_factory=list)
    content_categories: list[str] = field(default_factory=list)
    currency: str = "USD"
    
    @property
    def is_premium(self) -> bool:
        """Check if this is premium inventory."""
        return self.floor_cpm >= 15.0 or self.base_cpm >= 25.0


@dataclass
class InventoryPortfolio:
    """Complete inventory portfolio for a seller."""
    
    seller_id: str
    products: list[Product] = field(default_factory=list)
    total_avails: dict[str, int] = field(default_factory=dict)  # channel -> impressions
    floor_prices: dict[str, float] = field(default_factory=dict)  # product_id -> floor
    fill_rate: dict[str, float] = field(default_factory=dict)  # channel -> %
    
    # Performance metrics
    avg_cpm: dict[str, float] = field(default_factory=dict)  # channel -> avg CPM
    revenue_ytd: float = 0.0
    
    def get_channel_products(self, channel: ChannelType) -> list[Product]:
        """Get all products for a specific channel."""
        return [p for p in self.products if p.channel == channel]
    
    def get_total_daily_impressions(self) -> int:
        """Calculate total daily impressions across all products."""
        return sum(p.daily_impressions for p in self.products)
    
    def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """Get a specific product by ID."""
        for product in self.products:
            if product.product_id == product_id:
                return product
        return None


@dataclass
class DealRequest:
    """Incoming deal request from a buyer."""
    
    request_id: str
    buyer_id: str
    buyer_tier: BuyerTier
    product_id: str
    impressions: int
    max_cpm: float
    deal_type: DealTypeEnum
    flight_dates: tuple[date, date]
    audience_spec: AudienceSpec
    
    # Optional fields
    buyer_name: Optional[str] = None
    campaign_name: Optional[str] = None
    priority: int = 0  # Higher = more important
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def duration_days(self) -> int:
        """Calculate campaign duration in days."""
        return (self.flight_dates[1] - self.flight_dates[0]).days + 1
    
    @property
    def total_value(self) -> float:
        """Calculate total deal value at max CPM."""
        return (self.impressions / 1000) * self.max_cpm


@dataclass
class CounterOffer:
    """A counter-offer to a deal request."""
    
    suggested_cpm: float
    suggested_impressions: int
    alternative_products: list[str] = field(default_factory=list)
    modified_flight_dates: Optional[tuple[date, date]] = None
    package_deal: bool = False  # Cross-sell opportunity
    reasoning: str = ""


@dataclass
class DealDecision:
    """Decision on a deal request."""
    
    request_id: str
    action: DealAction
    price: float
    impressions: int
    reasoning: str
    counter_offer: Optional[CounterOffer] = None
    confidence: float = 1.0  # 0-1 confidence in decision
    
    # Metadata
    decided_at: datetime = field(default_factory=datetime.utcnow)
    model_used: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class Deal:
    """An accepted/active deal."""
    
    deal_id: str
    request_id: str
    buyer_id: str
    buyer_tier: BuyerTier
    product_id: str
    agreed_cpm: float
    impressions: int
    deal_type: DealTypeEnum
    flight_dates: tuple[date, date]
    
    # Runtime tracking
    impressions_delivered: int = 0
    revenue: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_active(self) -> bool:
        """Check if deal is currently active."""
        today = date.today()
        return self.flight_dates[0] <= today <= self.flight_dates[1]
    
    @property
    def remaining_impressions(self) -> int:
        """Calculate remaining impressions to deliver."""
        return max(0, self.impressions - self.impressions_delivered)
    
    @property
    def fill_rate(self) -> float:
        """Calculate current fill rate."""
        if self.impressions == 0:
            return 0.0
        return self.impressions_delivered / self.impressions


@dataclass
class CrossSellOpportunity:
    """Cross-sell/upsell opportunity identified from a deal."""
    
    source_deal_id: str
    source_product_id: str
    recommended_product_id: str
    recommended_channel: ChannelType
    estimated_value: float
    confidence: float  # 0-1
    reasoning: str
    
    # Suggested terms
    suggested_impressions: int = 0
    suggested_cpm: float = 0.0


@dataclass 
class YieldStrategy:
    """Yield optimization strategy recommendations."""
    
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Floor price adjustments by product
    floor_adjustments: dict[str, float] = field(default_factory=dict)  # product_id -> multiplier
    
    # Allocation priorities (ordered list of channels/products)
    allocation_priorities: list[str] = field(default_factory=list)
    
    # Pacing recommendations by product
    pacing_recommendations: dict[str, str] = field(default_factory=dict)  # product_id -> "aggressive"|"steady"|"conservative"
    
    # Strategic insights
    insights: list[str] = field(default_factory=list)
    
    # Expected impact
    expected_revenue_lift: float = 0.0  # Percentage
    expected_fill_rate_change: float = 0.0  # Percentage points
    
    # Confidence and model info
    confidence: float = 1.0
    model_used: Optional[str] = None


@dataclass
class Task:
    """A task to be delegated to a lower-level agent."""
    
    task_id: str
    task_type: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskResult:
    """Result from a delegated task."""
    
    task_id: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)
