#!/usr/bin/env python3
"""
Agent State Models for Multi-Agent Isolation Architecture.

Provides dataclasses for tracking individual agent state and reconciliation
between buyer/seller pairs. Key to demonstrating how agent memory divergence
leads to disputes without a shared ledger.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class DealRecord:
    """A single deal as recorded by one party."""
    deal_id: str
    day: int
    channel: str
    counterparty_id: str
    agreed_cpm: float  # What this agent believes was agreed
    impressions: int
    my_bid: float
    their_bid: Optional[float] = None
    raw_response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: Optional[str] = None


@dataclass
class AgentState:
    """
    Complete state snapshot of an agent.
    
    Used for monitoring and analysis of per-agent metrics.
    Each agent maintains isolated state that may diverge from others.
    """
    agent_id: str
    agent_type: str  # "buyer" or "seller"
    
    # Context tracking
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    context_tokens: int = 0
    
    # Deal records (agent's "database")
    deal_records: Dict[str, DealRecord] = field(default_factory=dict)
    
    # Metrics
    total_decisions: int = 0
    hallucinations: int = 0
    hallucination_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Reconciliation tracking (from this agent's perspective)
    reconciliation_attempts: int = 0
    reconciliation_failures: int = 0
    disputed_deals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "context_entries": len(self.context_history),
            "context_tokens": self.context_tokens,
            "deal_count": len(self.deal_records),
            "total_decisions": self.total_decisions,
            "hallucinations": self.hallucinations,
            "hallucination_rate": self.hallucination_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "reconciliation_attempts": self.reconciliation_attempts,
            "reconciliation_failures": self.reconciliation_failures,
            "disputed_deals": self.disputed_deals,
        }


@dataclass
class Discrepancy:
    """A single field discrepancy between buyer and seller records."""
    field_name: str
    buyer_value: Any
    seller_value: Any
    difference: Optional[float] = None  # For numeric fields
    severity: str = "minor"  # "minor", "moderate", "severe"
    
    def __post_init__(self):
        """Calculate severity based on difference magnitude."""
        if self.difference is not None:
            abs_diff = abs(self.difference)
            if abs_diff > 0.5:  # More than 50% difference
                self.severity = "severe"
            elif abs_diff > 0.1:  # More than 10% difference
                self.severity = "moderate"
            else:
                self.severity = "minor"


@dataclass
class ReconciliationResult:
    """
    Result of attempting to reconcile buyer and seller records for a deal.
    
    This is the core data structure for proving the IAB A2A problem:
    without a shared ledger, buyer and seller memories diverge,
    leading to unresolvable disputes.
    """
    deal_id: str
    matched: bool
    
    # The parties involved
    buyer_id: str
    seller_id: str
    
    # What each party recorded
    buyer_cpm: float
    seller_cpm: float
    buyer_impressions: int
    seller_impressions: int
    
    # Discrepancies found
    discrepancies: List[Discrepancy] = field(default_factory=list)
    
    # Context state at time of deal (important for correlation analysis)
    buyer_context_tokens: int = 0
    seller_context_tokens: int = 0
    
    # Was hallucination involved?
    buyer_hallucinated: bool = False
    seller_hallucinated: bool = False
    
    # Metadata
    day: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def cpm_discrepancy(self) -> float:
        """Calculate CPM discrepancy as percentage."""
        if self.buyer_cpm == 0 and self.seller_cpm == 0:
            return 0.0
        avg = (self.buyer_cpm + self.seller_cpm) / 2
        if avg == 0:
            return 1.0  # 100% discrepancy
        return abs(self.buyer_cpm - self.seller_cpm) / avg
    
    @property
    def impression_discrepancy(self) -> float:
        """Calculate impression discrepancy as percentage."""
        if self.buyer_impressions == 0 and self.seller_impressions == 0:
            return 0.0
        avg = (self.buyer_impressions + self.seller_impressions) / 2
        if avg == 0:
            return 1.0
        return abs(self.buyer_impressions - self.seller_impressions) / avg
    
    @property
    def total_discrepancy_count(self) -> int:
        """Total number of discrepancies found."""
        return len(self.discrepancies)
    
    @property
    def severity_level(self) -> str:
        """Overall severity level of this reconciliation failure."""
        if self.matched:
            return "none"
        
        severities = [d.severity for d in self.discrepancies]
        if "severe" in severities:
            return "severe"
        if "moderate" in severities:
            return "moderate"
        return "minor"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "deal_id": self.deal_id,
            "matched": self.matched,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "buyer_cpm": self.buyer_cpm,
            "seller_cpm": self.seller_cpm,
            "buyer_impressions": self.buyer_impressions,
            "seller_impressions": self.seller_impressions,
            "cpm_discrepancy_pct": round(self.cpm_discrepancy * 100, 2),
            "impression_discrepancy_pct": round(self.impression_discrepancy * 100, 2),
            "discrepancy_count": self.total_discrepancy_count,
            "severity": self.severity_level,
            "buyer_context_tokens": self.buyer_context_tokens,
            "seller_context_tokens": self.seller_context_tokens,
            "buyer_hallucinated": self.buyer_hallucinated,
            "seller_hallucinated": self.seller_hallucinated,
            "day": self.day,
            "timestamp": self.timestamp,
        }


def reconcile_deal(
    deal_id: str,
    buyer_record: dict,
    seller_record: dict,
    day: int = 0,
    cpm_tolerance: float = 0.01,  # 1% tolerance
    impression_tolerance: float = 0.0,  # Exact match required
) -> ReconciliationResult:
    """
    Attempt to reconcile buyer and seller records for a deal.
    
    Args:
        deal_id: The deal identifier
        buyer_record: Buyer's recorded deal data
        seller_record: Seller's recorded deal data
        day: Simulation day
        cpm_tolerance: Acceptable CPM difference as fraction (0.01 = 1%)
        impression_tolerance: Acceptable impression difference as fraction
    
    Returns:
        ReconciliationResult with match status and any discrepancies
    """
    discrepancies = []
    
    # Extract values
    buyer_cpm = buyer_record.get("agreed_cpm", 0.0)
    seller_cpm = seller_record.get("agreed_cpm", 0.0)
    buyer_imps = buyer_record.get("impressions", 0)
    seller_imps = seller_record.get("impressions", 0)
    
    # Check CPM match
    if buyer_cpm > 0 or seller_cpm > 0:
        avg_cpm = (buyer_cpm + seller_cpm) / 2 if (buyer_cpm + seller_cpm) > 0 else 1
        cpm_diff = abs(buyer_cpm - seller_cpm) / avg_cpm
        if cpm_diff > cpm_tolerance:
            discrepancies.append(Discrepancy(
                field_name="agreed_cpm",
                buyer_value=buyer_cpm,
                seller_value=seller_cpm,
                difference=cpm_diff,
            ))
    
    # Check impression match
    if buyer_imps > 0 or seller_imps > 0:
        avg_imps = (buyer_imps + seller_imps) / 2 if (buyer_imps + seller_imps) > 0 else 1
        imp_diff = abs(buyer_imps - seller_imps) / avg_imps
        if imp_diff > impression_tolerance:
            discrepancies.append(Discrepancy(
                field_name="impressions",
                buyer_value=buyer_imps,
                seller_value=seller_imps,
                difference=imp_diff,
            ))
    
    # Check channel match (should be exact)
    buyer_channel = buyer_record.get("channel", "")
    seller_channel = seller_record.get("channel", "")
    if buyer_channel != seller_channel:
        discrepancies.append(Discrepancy(
            field_name="channel",
            buyer_value=buyer_channel,
            seller_value=seller_channel,
            severity="severe",  # Channel mismatch is always severe
        ))
    
    return ReconciliationResult(
        deal_id=deal_id,
        matched=len(discrepancies) == 0,
        buyer_id=buyer_record.get("recorded_by", "unknown"),
        seller_id=seller_record.get("recorded_by", "unknown"),
        buyer_cpm=buyer_cpm,
        seller_cpm=seller_cpm,
        buyer_impressions=buyer_imps,
        seller_impressions=seller_imps,
        discrepancies=discrepancies,
        buyer_context_tokens=buyer_record.get("context_tokens", 0),
        seller_context_tokens=seller_record.get("context_tokens", 0),
        buyer_hallucinated=buyer_record.get("was_hallucination", False),
        seller_hallucinated=seller_record.get("was_hallucination", False),
        day=day,
    )


@dataclass
class MultiAgentMetrics:
    """
    Aggregate metrics across all agents for the multi-agent simulation.
    """
    # Agent counts
    num_buyers: int = 0
    num_sellers: int = 0
    
    # Per-agent summaries
    buyer_states: Dict[str, dict] = field(default_factory=dict)
    seller_states: Dict[str, dict] = field(default_factory=dict)
    
    # Reconciliation summary
    total_deals: int = 0
    reconciliation_attempts: int = 0
    reconciliation_successes: int = 0
    reconciliation_failures: int = 0
    failure_rate: float = 0.0
    
    # Discrepancy breakdown
    cpm_discrepancies: int = 0
    impression_discrepancies: int = 0
    channel_discrepancies: int = 0
    
    # Severity breakdown
    minor_failures: int = 0
    moderate_failures: int = 0
    severe_failures: int = 0
    
    # Context correlation
    avg_buyer_tokens_on_failure: float = 0.0
    avg_seller_tokens_on_failure: float = 0.0
    avg_buyer_tokens_on_success: float = 0.0
    avg_seller_tokens_on_success: float = 0.0
    
    # Hallucination involvement
    failures_with_hallucination: int = 0
    failures_without_hallucination: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_buyers": self.num_buyers,
            "num_sellers": self.num_sellers,
            "buyer_states": self.buyer_states,
            "seller_states": self.seller_states,
            "total_deals": self.total_deals,
            "reconciliation_attempts": self.reconciliation_attempts,
            "reconciliation_successes": self.reconciliation_successes,
            "reconciliation_failures": self.reconciliation_failures,
            "failure_rate": round(self.failure_rate * 100, 2),
            "cpm_discrepancies": self.cpm_discrepancies,
            "impression_discrepancies": self.impression_discrepancies,
            "channel_discrepancies": self.channel_discrepancies,
            "minor_failures": self.minor_failures,
            "moderate_failures": self.moderate_failures,
            "severe_failures": self.severe_failures,
            "avg_buyer_tokens_on_failure": round(self.avg_buyer_tokens_on_failure, 0),
            "avg_seller_tokens_on_failure": round(self.avg_seller_tokens_on_failure, 0),
            "avg_buyer_tokens_on_success": round(self.avg_buyer_tokens_on_success, 0),
            "avg_seller_tokens_on_success": round(self.avg_seller_tokens_on_success, 0),
            "failures_with_hallucination": self.failures_with_hallucination,
            "failures_without_hallucination": self.failures_without_hallucination,
        }
