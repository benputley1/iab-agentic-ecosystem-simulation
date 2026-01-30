"""
Hallucination Classifier for detecting agent decision errors.

Compares agent decisions against ground truth to detect and classify
various types of hallucinations that occur due to context window
limitations, memory loss, or other agent failures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .types import (
    AgentDecision,
    CampaignState,
    GroundTruthDBProtocol,
    Hallucination,
    HallucinationResult,
    HallucinationType,
)


@dataclass
class SeverityThresholds:
    """
    Configurable thresholds for hallucination detection.
    
    Values represent the minimum deviation required to classify
    something as a hallucination.
    """
    # Budget drift: percentage of remaining budget
    budget_drift_pct: float = 0.01  # 1%
    
    # Price anchor: percentage deviation from actual floor
    price_anchor_pct: float = 0.05  # 5%
    
    # Frequency: number of impressions over the cap
    frequency_violation_threshold: int = 1  # Any over = violation
    
    # Deal floor price: percentage deviation
    deal_price_deviation_pct: float = 0.01  # 1%


@dataclass
class ClassifierStats:
    """
    Statistics tracked by the classifier over time.
    """
    total_decisions_checked: int = 0
    total_hallucinations: int = 0
    hallucinations_by_type: dict[HallucinationType, int] = field(default_factory=dict)
    severity_sum: float = 0.0
    
    @property
    def hallucination_rate(self) -> float:
        """Rate of decisions with at least one hallucination."""
        if self.total_decisions_checked == 0:
            return 0.0
        return self.total_hallucinations / self.total_decisions_checked
    
    @property
    def average_severity(self) -> float:
        """Average severity across all hallucinations."""
        if self.total_hallucinations == 0:
            return 0.0
        return self.severity_sum / self.total_hallucinations
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_decisions_checked": self.total_decisions_checked,
            "total_hallucinations": self.total_hallucinations,
            "hallucination_rate": self.hallucination_rate,
            "average_severity": self.average_severity,
            "by_type": {
                t.value: count 
                for t, count in self.hallucinations_by_type.items()
            }
        }


class HallucinationClassifier:
    """
    Classifies and tracks hallucination types in agent decisions.
    
    Compares agent decisions against a ground truth database to detect
    various types of hallucinations that can occur during ad bidding.
    
    Usage:
        classifier = HallucinationClassifier(ground_truth_db)
        result = classifier.check_decision(agent_decision)
        if result.has_hallucinations:
            for h in result.errors:
                print(f"{h.type}: expected {h.expected}, got {h.actual}")
    """
    
    def __init__(
        self,
        ground_truth: GroundTruthDBProtocol,
        thresholds: Optional[SeverityThresholds] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            ground_truth: Database providing authoritative state
            thresholds: Optional custom thresholds for detection
        """
        self.ground_truth = ground_truth
        self.thresholds = thresholds or SeverityThresholds()
        self.hallucinations: list[Hallucination] = []
        self.stats = ClassifierStats()
    
    def check_decision(self, decision: AgentDecision) -> HallucinationResult:
        """
        Compare a decision against ground truth and detect hallucinations.
        
        Args:
            decision: The agent decision to validate
            
        Returns:
            HallucinationResult with list of detected hallucinations
        """
        errors: list[Hallucination] = []
        
        # Get ground truth state for this decision's campaign
        truth = self.ground_truth.get_campaign_state(
            decision.campaign_id,
            decision.timestamp
        )
        
        # Run all checks
        if budget_error := self._check_budget_drift(decision, truth):
            errors.append(budget_error)
        
        if freq_error := self._check_frequency_violation(decision, truth):
            errors.append(freq_error)
        
        if deal_error := self._check_deal_invention(decision):
            errors.append(deal_error)
        
        if cross_error := self._check_cross_contamination(decision, truth):
            errors.append(cross_error)
        
        if phantom_error := self._check_phantom_inventory(decision):
            errors.append(phantom_error)
        
        if price_error := self._check_price_anchor_error(decision, truth):
            errors.append(price_error)
        
        # Update stats
        self._update_stats(errors)
        
        # Store hallucinations for later analysis
        self.hallucinations.extend(errors)
        
        return HallucinationResult(
            decision_id=decision.id,
            timestamp=decision.timestamp,
            errors=errors
        )
    
    def _check_budget_drift(
        self,
        decision: AgentDecision,
        truth: CampaignState
    ) -> Optional[Hallucination]:
        """Check if agent misremembers remaining budget."""
        if decision.budget_remaining is None:
            return None
        
        actual_remaining = truth.budget_remaining
        agent_remaining = decision.budget_remaining
        
        if actual_remaining == 0:
            # Avoid division by zero - any non-zero belief is 100% drift
            if agent_remaining != 0:
                return Hallucination(
                    type=HallucinationType.BUDGET_DRIFT,
                    expected=actual_remaining,
                    actual=agent_remaining,
                    severity=1.0,
                    description=f"Budget exhausted but agent believes {agent_remaining:.2f} remains",
                    field_name="budget_remaining"
                )
            return None
        
        drift = abs(agent_remaining - actual_remaining)
        drift_pct = drift / actual_remaining
        
        if drift_pct > self.thresholds.budget_drift_pct:
            # Severity scales with drift percentage, capped at 1.0
            severity = min(drift_pct, 1.0)
            
            return Hallucination(
                type=HallucinationType.BUDGET_DRIFT,
                expected=actual_remaining,
                actual=agent_remaining,
                severity=severity,
                description=f"Budget drift of {drift_pct:.1%} ({drift:.2f} difference)",
                field_name="budget_remaining"
            )
        
        return None
    
    def _check_frequency_violation(
        self,
        decision: AgentDecision,
        truth: CampaignState
    ) -> Optional[Hallucination]:
        """Check if agent loses track of user exposure."""
        if decision.user_id is None or decision.user_exposure_count is None:
            return None
        
        # Get actual exposure count from ground truth
        actual_count = self.ground_truth.get_user_frequency(
            decision.campaign_id,
            decision.user_id,
            decision.timestamp
        )
        
        agent_count = decision.user_exposure_count
        
        # Check if agent's count differs significantly
        count_diff = abs(agent_count - actual_count)
        
        if count_diff >= self.thresholds.frequency_violation_threshold:
            # Also check if this resulted in exceeding frequency cap
            frequency_cap = decision.frequency_cap or truth.frequency_caps.get(
                decision.user_id, float('inf')
            )
            
            # Severity based on how wrong the count is and if cap was violated
            if actual_count >= frequency_cap and agent_count < frequency_cap:
                # Agent thought they could bid but cap was already reached
                severity = min(0.5 + (count_diff / 10), 1.0)
                description = (
                    f"Frequency cap violation: actual count {actual_count} "
                    f"(cap: {frequency_cap}), agent believed {agent_count}"
                )
            else:
                severity = min(count_diff / 10, 0.5)
                description = (
                    f"User exposure tracking error: actual {actual_count}, "
                    f"agent believed {agent_count}"
                )
            
            return Hallucination(
                type=HallucinationType.FREQUENCY_VIOLATION,
                expected=actual_count,
                actual=agent_count,
                severity=severity,
                description=description,
                field_name="user_exposure_count"
            )
        
        return None
    
    def _check_deal_invention(
        self,
        decision: AgentDecision
    ) -> Optional[Hallucination]:
        """Check if agent invents or misremembers deal terms."""
        if decision.deal_id is None:
            return None
        
        # Check if deal exists at all
        actual_deal = self.ground_truth.get_deal(decision.deal_id)
        
        if actual_deal is None:
            # Deal doesn't exist - complete invention
            return Hallucination(
                type=HallucinationType.DEAL_INVENTION,
                expected=None,
                actual={"deal_id": decision.deal_id, "terms": decision.deal_terms},
                severity=1.0,
                description=f"Deal {decision.deal_id} does not exist",
                field_name="deal_id"
            )
        
        # Deal exists - check if terms match
        if decision.deal_terms is not None:
            mismatched_terms = []
            
            for key, agent_value in decision.deal_terms.items():
                if key in actual_deal:
                    actual_value = actual_deal[key]
                    if agent_value != actual_value:
                        mismatched_terms.append({
                            "field": key,
                            "expected": actual_value,
                            "actual": agent_value
                        })
            
            if mismatched_terms:
                # Severity based on number and importance of mismatches
                severity = min(len(mismatched_terms) * 0.2, 0.8)
                
                return Hallucination(
                    type=HallucinationType.DEAL_INVENTION,
                    expected=actual_deal,
                    actual=decision.deal_terms,
                    severity=severity,
                    description=f"Deal terms mismatch: {mismatched_terms}",
                    field_name="deal_terms"
                )
        
        # Check floor price if specified
        if decision.deal_floor_price is not None and "floor_price" in actual_deal:
            actual_floor = actual_deal["floor_price"]
            price_diff = abs(decision.deal_floor_price - actual_floor)
            
            if actual_floor > 0:
                deviation_pct = price_diff / actual_floor
                if deviation_pct > self.thresholds.deal_price_deviation_pct:
                    return Hallucination(
                        type=HallucinationType.DEAL_INVENTION,
                        expected=actual_floor,
                        actual=decision.deal_floor_price,
                        severity=min(deviation_pct, 1.0),
                        description=f"Deal floor price wrong: {deviation_pct:.1%} deviation",
                        field_name="deal_floor_price"
                    )
        
        return None
    
    def _check_cross_contamination(
        self,
        decision: AgentDecision,
        truth: CampaignState
    ) -> Optional[Hallucination]:
        """Check if agent attributes data to wrong campaign."""
        # This check requires the decision to reference the campaign
        # If the campaign_id in decision doesn't match what we're checking,
        # that's a cross-contamination issue
        
        if decision.campaign_id != truth.campaign_id:
            return Hallucination(
                type=HallucinationType.CAMPAIGN_CROSS_CONTAMINATION,
                expected=truth.campaign_id,
                actual=decision.campaign_id,
                severity=1.0,
                description=f"Decision attributed to wrong campaign",
                field_name="campaign_id"
            )
        
        # Additional check: if budget figures match a DIFFERENT campaign
        # This would require access to other campaigns, which is a more
        # complex check. For now, we rely on the basic campaign_id check.
        
        return None
    
    def _check_phantom_inventory(
        self,
        decision: AgentDecision
    ) -> Optional[Hallucination]:
        """Check if agent references non-existent inventory."""
        if decision.inventory_id is None:
            return None
        
        exists = self.ground_truth.inventory_exists(
            decision.inventory_id,
            decision.timestamp
        )
        
        if not exists:
            return Hallucination(
                type=HallucinationType.PHANTOM_INVENTORY,
                expected="inventory_exists=True",
                actual="inventory_exists=False",
                severity=0.8,
                description=f"Inventory {decision.inventory_id} does not exist",
                field_name="inventory_id"
            )
        
        return None
    
    def _check_price_anchor_error(
        self,
        decision: AgentDecision,
        truth: CampaignState
    ) -> Optional[Hallucination]:
        """Check if agent misremembers floor prices."""
        if decision.inventory_id is None or decision.expected_floor_price is None:
            return None
        
        # Get actual floor price
        actual_floor = self.ground_truth.get_floor_price(
            decision.inventory_id,
            decision.timestamp
        )
        
        if actual_floor is None:
            # No floor price data available - can't check
            return None
        
        agent_floor = decision.expected_floor_price
        
        if actual_floor == 0:
            # Handle zero floor case
            if agent_floor != 0:
                return Hallucination(
                    type=HallucinationType.PRICE_ANCHOR_ERROR,
                    expected=actual_floor,
                    actual=agent_floor,
                    severity=0.5,
                    description=f"Floor price is 0, agent expected {agent_floor:.4f}",
                    field_name="expected_floor_price"
                )
            return None
        
        deviation = abs(agent_floor - actual_floor)
        deviation_pct = deviation / actual_floor
        
        if deviation_pct > self.thresholds.price_anchor_pct:
            severity = min(deviation_pct, 1.0)
            
            return Hallucination(
                type=HallucinationType.PRICE_ANCHOR_ERROR,
                expected=actual_floor,
                actual=agent_floor,
                severity=severity,
                description=(
                    f"Floor price error: actual {actual_floor:.4f}, "
                    f"agent expected {agent_floor:.4f} ({deviation_pct:.1%} off)"
                ),
                field_name="expected_floor_price"
            )
        
        return None
    
    def _update_stats(self, errors: list[Hallucination]) -> None:
        """Update classifier statistics."""
        self.stats.total_decisions_checked += 1
        
        if errors:
            self.stats.total_hallucinations += len(errors)
            
            for error in errors:
                self.stats.severity_sum += error.severity
                
                if error.type not in self.stats.hallucinations_by_type:
                    self.stats.hallucinations_by_type[error.type] = 0
                self.stats.hallucinations_by_type[error.type] += 1
    
    def get_stats(self) -> ClassifierStats:
        """Get current classifier statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset statistics (keeps hallucination history)."""
        self.stats = ClassifierStats()
    
    def get_hallucinations_by_type(
        self,
        hallucination_type: HallucinationType
    ) -> list[Hallucination]:
        """Get all hallucinations of a specific type."""
        return [h for h in self.hallucinations if h.type == hallucination_type]
    
    def get_hallucination_summary(self) -> dict:
        """
        Get a summary of all detected hallucinations.
        
        Returns:
            Dictionary with counts and statistics by type
        """
        summary = {
            "total": len(self.hallucinations),
            "by_type": {},
            "severity_distribution": {
                "low": 0,    # 0.0 - 0.33
                "medium": 0,  # 0.34 - 0.66
                "high": 0     # 0.67 - 1.0
            }
        }
        
        for h in self.hallucinations:
            # Count by type
            type_name = h.type.value
            if type_name not in summary["by_type"]:
                summary["by_type"][type_name] = {
                    "count": 0,
                    "avg_severity": 0.0,
                    "total_severity": 0.0
                }
            summary["by_type"][type_name]["count"] += 1
            summary["by_type"][type_name]["total_severity"] += h.severity
            
            # Severity distribution
            if h.severity <= 0.33:
                summary["severity_distribution"]["low"] += 1
            elif h.severity <= 0.66:
                summary["severity_distribution"]["medium"] += 1
            else:
                summary["severity_distribution"]["high"] += 1
        
        # Calculate averages
        for type_data in summary["by_type"].values():
            if type_data["count"] > 0:
                type_data["avg_severity"] = (
                    type_data["total_severity"] / type_data["count"]
                )
        
        return summary
