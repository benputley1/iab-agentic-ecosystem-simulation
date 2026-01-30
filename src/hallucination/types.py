"""
Type definitions for the Hallucination Classifier.

Provides enums and dataclasses for classifying and tracking hallucinations
in agent decisions during ad bidding simulations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol


class HallucinationType(Enum):
    """Types of hallucinations that can occur in agent decisions."""
    
    BUDGET_DRIFT = "budget_drift"
    """Agent misremembers the remaining budget amount."""
    
    FREQUENCY_VIOLATION = "frequency_cap"
    """Agent loses track of user exposure and violates frequency caps."""
    
    DEAL_INVENTION = "deal_invention"
    """Agent invents or misremembers deal terms that don't exist."""
    
    CAMPAIGN_CROSS_CONTAMINATION = "cross_campaign"
    """Agent attributes data to the wrong campaign."""
    
    PHANTOM_INVENTORY = "phantom_inventory"
    """Agent references inventory that doesn't exist."""
    
    PRICE_ANCHOR_ERROR = "price_anchor"
    """Agent misremembers floor prices or price history."""


@dataclass
class CampaignState:
    """
    Represents the true state of a campaign at a point in time.
    Used by GroundTruthDB for comparison.
    """
    campaign_id: str
    budget_total: float
    budget_spent: float
    budget_remaining: float
    impressions: int
    frequency_caps: dict[str, int] = field(default_factory=dict)
    """Maps user_id -> impression count for frequency tracking."""
    active_deals: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Maps deal_id -> deal terms."""
    floor_prices: dict[str, float] = field(default_factory=dict)
    """Maps inventory_id -> floor price."""
    available_inventory: set[str] = field(default_factory=set)
    """Set of valid inventory IDs."""
    last_updated: Optional[datetime] = None


@dataclass
class AgentDecision:
    """
    Represents a decision made by an agent that needs to be validated.
    
    This is a simplified representation capturing the key fields
    that can be checked for hallucinations.
    """
    id: str
    timestamp: datetime
    campaign_id: str
    
    # Budget-related fields
    budget_remaining: Optional[float] = None
    bid_amount: Optional[float] = None
    
    # Frequency-related fields
    user_id: Optional[str] = None
    user_exposure_count: Optional[int] = None
    frequency_cap: Optional[int] = None
    
    # Deal-related fields
    deal_id: Optional[str] = None
    deal_floor_price: Optional[float] = None
    deal_terms: Optional[dict[str, Any]] = None
    
    # Inventory-related fields
    inventory_id: Optional[str] = None
    expected_floor_price: Optional[float] = None
    
    # Decision metadata
    decision_type: str = "bid"
    """Type of decision: bid, pass, adjust, etc."""
    
    references: list[str] = field(default_factory=list)
    """IDs of previous decisions this one references."""


@dataclass
class Hallucination:
    """
    Represents a single detected hallucination in an agent decision.
    """
    type: HallucinationType
    expected: Any
    """The ground truth value."""
    actual: Any
    """The value the agent used/believed."""
    severity: float
    """Severity score between 0 and 1."""
    
    description: Optional[str] = None
    """Human-readable description of the hallucination."""
    
    field_name: Optional[str] = None
    """The specific field where hallucination was detected."""
    
    def __post_init__(self):
        if not 0 <= self.severity <= 1:
            raise ValueError(f"Severity must be between 0 and 1, got {self.severity}")


@dataclass
class HallucinationResult:
    """
    Result of checking a decision for hallucinations.
    """
    decision_id: str
    timestamp: datetime
    errors: list[Hallucination] = field(default_factory=list)
    
    @property
    def has_hallucinations(self) -> bool:
        """Returns True if any hallucinations were detected."""
        return len(self.errors) > 0
    
    @property
    def total_severity(self) -> float:
        """Sum of all hallucination severities."""
        return sum(h.severity for h in self.errors)
    
    @property
    def max_severity(self) -> float:
        """Maximum severity among all hallucinations."""
        return max((h.severity for h in self.errors), default=0.0)
    
    @property
    def hallucination_types(self) -> set[HallucinationType]:
        """Set of all hallucination types detected."""
        return {h.type for h in self.errors}


class GroundTruthDBProtocol(Protocol):
    """
    Protocol defining the interface for GroundTruthDB.
    
    This allows the HallucinationClassifier to work with any
    implementation that satisfies this interface.
    """
    
    def get_campaign_state(self, campaign_id: str, at_time: datetime) -> CampaignState:
        """Get the true campaign state at a specific point in time."""
        ...
    
    def get_deal(self, deal_id: str) -> Optional[dict[str, Any]]:
        """Get deal terms by ID, or None if deal doesn't exist."""
        ...
    
    def get_user_frequency(self, campaign_id: str, user_id: str, at_time: datetime) -> int:
        """Get actual impression count for a user in a campaign."""
        ...
    
    def inventory_exists(self, inventory_id: str, at_time: datetime) -> bool:
        """Check if inventory existed at the given time."""
        ...
    
    def get_floor_price(self, inventory_id: str, at_time: datetime) -> Optional[float]:
        """Get the floor price for inventory at the given time."""
        ...
