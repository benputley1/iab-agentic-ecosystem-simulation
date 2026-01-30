"""
Hallucination Classifier - Detect and classify hallucinations in agent decisions.

This module compares agent decisions against ground truth to detect
various types of hallucinations caused by context window limitations,
memory loss, and other AI reliability issues.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ground_truth.db import GroundTruthDB, CampaignState


@dataclass
class SeverityThresholds:
    """Configurable thresholds for hallucination severity detection."""
    
    budget_drift_pct: float = 0.01  # 1% threshold for budget drift
    price_anchor_pct: float = 0.05  # 5% threshold for price errors
    frequency_cap_violations: int = 1  # Number of violations before flagging
    inventory_confidence: float = 0.8  # Confidence threshold for inventory


class HallucinationType(str, Enum):
    """Classification of hallucination types in agent decisions."""
    
    BUDGET_DRIFT = "budget_drift"  # Misremembers spent amount
    FREQUENCY_VIOLATION = "frequency_cap"  # Loses user exposure tracking
    DEAL_INVENTION = "deal_invention"  # Invents deal terms that don't exist
    CAMPAIGN_CROSS_CONTAMINATION = "cross_campaign"  # Wrong campaign attribution
    PHANTOM_INVENTORY = "phantom_inventory"  # References non-existent supply
    PRICE_ANCHOR_ERROR = "price_anchor"  # Wrong price memory/floor price


@dataclass
class Hallucination:
    """A single detected hallucination instance."""
    
    type: HallucinationType
    expected: Any  # Ground truth value
    actual: Any  # What the agent claimed/used
    severity: float  # 0.0-1.0, how bad is this error
    decision_id: str = ""
    campaign_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def drift_amount(self) -> float:
        """Calculate the absolute drift amount for numeric values."""
        if isinstance(self.expected, (int, float)) and isinstance(self.actual, (int, float)):
            return abs(self.expected - self.actual)
        return 0.0
    
    @property
    def drift_percentage(self) -> float:
        """Calculate the drift as a percentage of expected value."""
        if isinstance(self.expected, (int, float)) and self.expected != 0:
            return abs(self.expected - self.actual) / abs(self.expected) * 100
        return 0.0


@dataclass
class HallucinationResult:
    """Result of checking a decision for hallucinations."""
    
    decision_id: str
    errors: List[Hallucination] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def has_hallucinations(self) -> bool:
        """Check if any hallucinations were detected."""
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
    def hallucination_types(self) -> List[HallucinationType]:
        """List of hallucination types detected."""
        return [h.type for h in self.errors]


@dataclass
class AgentDecisionForCheck:
    """Agent decision data needed for hallucination checking."""
    
    decision_id: str
    timestamp: datetime
    campaign_id: str
    agent_id: str
    
    # Values claimed by the agent (to compare against ground truth)
    budget_remaining: Optional[float] = None
    total_spend: Optional[float] = None
    impressions_claimed: Optional[int] = None
    floor_price_used: Optional[float] = None
    deal_id_referenced: Optional[str] = None
    deal_terms_claimed: Optional[Dict[str, Any]] = None
    user_frequency_claimed: Optional[Dict[str, int]] = None  # user_id -> exposure count
    publisher_id: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class HallucinationClassifier:
    """
    Classify and track hallucination types in agent decisions.
    
    Compares agent decisions against ground truth to detect various
    types of hallucinations. Tracks statistics over time for reporting.
    
    Thresholds are configurable - by default:
    - Budget drift: >1% of remaining budget
    - Price anchor: >5% difference from actual floor
    - Frequency: any overcounting is flagged
    """
    
    def __init__(
        self,
        ground_truth: "GroundTruthDB",
        budget_drift_threshold: float = 0.01,  # 1% tolerance
        price_anchor_threshold: float = 0.05,  # 5% tolerance
        frequency_strict: bool = True,  # Any violation is flagged
    ):
        """
        Initialize the hallucination classifier.
        
        Args:
            ground_truth: GroundTruthDB instance for comparison
            budget_drift_threshold: Relative threshold for budget errors
            price_anchor_threshold: Relative threshold for price errors
            frequency_strict: If True, any frequency violation is flagged
        """
        self.ground_truth = ground_truth
        self.budget_drift_threshold = budget_drift_threshold
        self.price_anchor_threshold = price_anchor_threshold
        self.frequency_strict = frequency_strict
        
        # Track all detected hallucinations
        self.hallucinations: List[Hallucination] = []
        self.decisions_checked: int = 0
        self.decisions_with_errors: int = 0
        
        # Track known entities for phantom detection
        self._known_deals: Dict[str, Dict[str, Any]] = {}
        self._known_publishers: set = set()
        self._actual_floor_prices: Dict[str, float] = {}  # request_id -> floor
        self._user_frequencies: Dict[str, Dict[str, int]] = {}  # campaign_id -> {user_id: count}
    
    def register_deal(self, deal_id: str, terms: Dict[str, Any]) -> None:
        """Register a known deal for phantom detection."""
        self._known_deals[deal_id] = terms
    
    def register_publisher(self, publisher_id: str) -> None:
        """Register a known publisher for phantom detection."""
        self._known_publishers.add(publisher_id)
    
    def register_floor_price(self, request_id: str, floor_price: float) -> None:
        """Register the actual floor price for a bid request."""
        self._actual_floor_prices[request_id] = floor_price
    
    def record_user_exposure(self, campaign_id: str, user_id: str) -> None:
        """Record an actual user exposure for frequency tracking."""
        if campaign_id not in self._user_frequencies:
            self._user_frequencies[campaign_id] = {}
        freq = self._user_frequencies[campaign_id]
        freq[user_id] = freq.get(user_id, 0) + 1
    
    def check_decision(
        self,
        decision: AgentDecisionForCheck,
        actual_floor_price: Optional[float] = None,
    ) -> HallucinationResult:
        """
        Compare a decision against ground truth to detect hallucinations.
        
        Args:
            decision: The agent decision to check
            actual_floor_price: Optional actual floor price if not registered
            
        Returns:
            HallucinationResult with any detected errors
        """
        errors: List[Hallucination] = []
        self.decisions_checked += 1
        
        # Get ground truth state
        truth = self.ground_truth.get_campaign_state(
            decision.campaign_id,
            decision.timestamp
        )
        
        # Check for budget drift
        if decision.budget_remaining is not None or decision.total_spend is not None:
            budget_error = self._check_budget_drift(decision, truth)
            if budget_error:
                errors.append(budget_error)
        
        # Check for price anchor errors
        if decision.floor_price_used is not None:
            price_error = self._check_price_anchor(decision, actual_floor_price)
            if price_error:
                errors.append(price_error)
        
        # Check for deal invention
        if decision.deal_id_referenced is not None:
            deal_error = self._check_deal_invention(decision)
            if deal_error:
                errors.append(deal_error)
        
        # Check for phantom inventory
        if decision.publisher_id is not None:
            phantom_error = self._check_phantom_inventory(decision)
            if phantom_error:
                errors.append(phantom_error)
        
        # Check for frequency violations
        if decision.user_frequency_claimed is not None:
            freq_errors = self._check_frequency_violation(decision)
            errors.extend(freq_errors)
        
        # Track results
        if errors:
            self.decisions_with_errors += 1
            self.hallucinations.extend(errors)
        
        return HallucinationResult(
            decision_id=decision.decision_id,
            errors=errors,
        )
    
    def _check_budget_drift(
        self,
        decision: AgentDecisionForCheck,
        truth: "CampaignState",
    ) -> Optional[Hallucination]:
        """Check for budget drift hallucination."""
        # Get the agent's claimed values
        claimed_spend = decision.total_spend
        if claimed_spend is None and decision.budget_remaining is not None:
            # Infer from budget remaining if we know initial budget
            initial_budget = decision.metadata.get("initial_budget")
            if initial_budget:
                claimed_spend = initial_budget - decision.budget_remaining
        
        if claimed_spend is None:
            return None
        
        # Compare against ground truth
        actual_spend = truth.total_spend
        
        # Check if drift exceeds threshold
        if actual_spend == 0:
            if claimed_spend > 0:
                drift = 1.0  # 100% error
            else:
                return None  # Both zero, no error
        else:
            drift = abs(claimed_spend - actual_spend) / actual_spend
        
        if drift > self.budget_drift_threshold:
            # Calculate severity (capped at 1.0)
            severity = min(1.0, drift)
            
            return Hallucination(
                type=HallucinationType.BUDGET_DRIFT,
                expected=actual_spend,
                actual=claimed_spend,
                severity=severity,
                decision_id=decision.decision_id,
                campaign_id=decision.campaign_id,
                timestamp=decision.timestamp,
                metadata={
                    "drift_percentage": drift * 100,
                    "threshold": self.budget_drift_threshold * 100,
                }
            )
        
        return None
    
    def _check_price_anchor(
        self,
        decision: AgentDecisionForCheck,
        actual_floor_price: Optional[float],
    ) -> Optional[Hallucination]:
        """Check for price anchor errors."""
        # Get actual floor price
        floor = actual_floor_price
        if floor is None:
            # Try to find from registered prices
            request_id = decision.metadata.get("request_id")
            if request_id and request_id in self._actual_floor_prices:
                floor = self._actual_floor_prices[request_id]
        
        if floor is None or floor == 0:
            return None
        
        claimed_floor = decision.floor_price_used
        drift = abs(claimed_floor - floor) / floor
        
        if drift > self.price_anchor_threshold:
            severity = min(1.0, drift)
            
            return Hallucination(
                type=HallucinationType.PRICE_ANCHOR_ERROR,
                expected=floor,
                actual=claimed_floor,
                severity=severity,
                decision_id=decision.decision_id,
                campaign_id=decision.campaign_id,
                timestamp=decision.timestamp,
                metadata={
                    "drift_percentage": drift * 100,
                    "threshold": self.price_anchor_threshold * 100,
                }
            )
        
        return None
    
    def _check_deal_invention(
        self,
        decision: AgentDecisionForCheck,
    ) -> Optional[Hallucination]:
        """Check for deal invention hallucination."""
        deal_id = decision.deal_id_referenced
        
        # Check if deal exists
        if deal_id not in self._known_deals:
            return Hallucination(
                type=HallucinationType.DEAL_INVENTION,
                expected=None,  # Deal doesn't exist
                actual=deal_id,
                severity=0.8,  # High severity - invented deal
                decision_id=decision.decision_id,
                campaign_id=decision.campaign_id,
                timestamp=decision.timestamp,
                metadata={
                    "claimed_deal_id": deal_id,
                    "error": "deal_not_found",
                }
            )
        
        # Check if deal terms match (if provided)
        if decision.deal_terms_claimed:
            actual_terms = self._known_deals[deal_id]
            for key, claimed_value in decision.deal_terms_claimed.items():
                actual_value = actual_terms.get(key)
                if actual_value is not None and actual_value != claimed_value:
                    return Hallucination(
                        type=HallucinationType.DEAL_INVENTION,
                        expected=actual_value,
                        actual=claimed_value,
                        severity=0.5,  # Medium severity - wrong terms
                        decision_id=decision.decision_id,
                        campaign_id=decision.campaign_id,
                        timestamp=decision.timestamp,
                        metadata={
                            "deal_id": deal_id,
                            "field": key,
                            "error": "wrong_deal_terms",
                        }
                    )
        
        return None
    
    def _check_phantom_inventory(
        self,
        decision: AgentDecisionForCheck,
    ) -> Optional[Hallucination]:
        """Check for phantom inventory hallucination."""
        publisher_id = decision.publisher_id
        
        # Only check if we have known publishers registered
        if not self._known_publishers:
            return None
        
        if publisher_id not in self._known_publishers:
            return Hallucination(
                type=HallucinationType.PHANTOM_INVENTORY,
                expected=None,
                actual=publisher_id,
                severity=0.6,  # Medium-high severity
                decision_id=decision.decision_id,
                campaign_id=decision.campaign_id,
                timestamp=decision.timestamp,
                metadata={
                    "claimed_publisher": publisher_id,
                    "error": "publisher_not_found",
                }
            )
        
        return None
    
    def _check_frequency_violation(
        self,
        decision: AgentDecisionForCheck,
    ) -> List[Hallucination]:
        """Check for frequency cap violations."""
        errors = []
        claimed_frequencies = decision.user_frequency_claimed or {}
        
        # Get actual frequencies for this campaign
        actual_frequencies = self._user_frequencies.get(decision.campaign_id, {})
        
        for user_id, claimed_count in claimed_frequencies.items():
            actual_count = actual_frequencies.get(user_id, 0)
            
            # Agent claims fewer exposures than actually occurred
            # This could lead to over-serving the user
            if claimed_count < actual_count:
                undercount = actual_count - claimed_count
                severity = min(1.0, undercount / max(actual_count, 1))
                
                errors.append(Hallucination(
                    type=HallucinationType.FREQUENCY_VIOLATION,
                    expected=actual_count,
                    actual=claimed_count,
                    severity=severity,
                    decision_id=decision.decision_id,
                    campaign_id=decision.campaign_id,
                    timestamp=decision.timestamp,
                    metadata={
                        "user_id": user_id,
                        "undercount": undercount,
                        "error": "frequency_undercount",
                    }
                ))
        
        return errors
    
    def check_campaign_contamination(
        self,
        decision_id: str,
        claimed_campaign_id: str,
        actual_campaign_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Hallucination]:
        """
        Check for campaign cross-contamination.
        
        This occurs when an agent attributes data to the wrong campaign.
        
        Args:
            decision_id: The decision being checked
            claimed_campaign_id: Campaign ID the agent claims
            actual_campaign_id: Actual campaign ID
            timestamp: When this occurred
            
        Returns:
            Hallucination if contamination detected, None otherwise
        """
        if claimed_campaign_id != actual_campaign_id:
            return Hallucination(
                type=HallucinationType.CAMPAIGN_CROSS_CONTAMINATION,
                expected=actual_campaign_id,
                actual=claimed_campaign_id,
                severity=0.7,  # High severity - wrong campaign
                decision_id=decision_id,
                campaign_id=actual_campaign_id,
                timestamp=timestamp or datetime.utcnow(),
                metadata={
                    "claimed_campaign": claimed_campaign_id,
                    "actual_campaign": actual_campaign_id,
                }
            )
        return None
    
    def get_hallucination_rate(self) -> float:
        """Calculate overall hallucination rate."""
        if self.decisions_checked == 0:
            return 0.0
        return self.decisions_with_errors / self.decisions_checked
    
    def get_type_distribution(self) -> Dict[HallucinationType, int]:
        """Get distribution of hallucination types."""
        distribution = {t: 0 for t in HallucinationType}
        for h in self.hallucinations:
            distribution[h.type] += 1
        return distribution
    
    def get_severity_stats(self) -> Dict[str, float]:
        """Get severity statistics."""
        if not self.hallucinations:
            return {
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "total": 0.0,
            }
        
        severities = [h.severity for h in self.hallucinations]
        return {
            "mean": sum(severities) / len(severities),
            "max": max(severities),
            "min": min(severities),
            "total": sum(severities),
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for reporting."""
        type_dist = self.get_type_distribution()
        severity_stats = self.get_severity_stats()
        
        return {
            "decisions_checked": self.decisions_checked,
            "decisions_with_errors": self.decisions_with_errors,
            "hallucination_rate": self.get_hallucination_rate(),
            "total_hallucinations": len(self.hallucinations),
            "type_distribution": {t.value: c for t, c in type_dist.items()},
            "severity_stats": severity_stats,
            "by_type_severity": self._get_severity_by_type(),
        }
    
    def _get_severity_by_type(self) -> Dict[str, float]:
        """Get average severity by hallucination type."""
        by_type: Dict[HallucinationType, List[float]] = {t: [] for t in HallucinationType}
        for h in self.hallucinations:
            by_type[h.type].append(h.severity)
        
        return {
            t.value: (sum(severities) / len(severities) if severities else 0.0)
            for t, severities in by_type.items()
        }
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.hallucinations.clear()
        self.decisions_checked = 0
        self.decisions_with_errors = 0
        self._known_deals.clear()
        self._known_publishers.clear()
        self._actual_floor_prices.clear()
        self._user_frequencies.clear()
