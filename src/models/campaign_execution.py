#!/usr/bin/env python3
"""
Campaign Execution Models for Two-Level Context Pressure Simulation.

This module provides dataclasses for tracking campaign execution metrics,
particularly focusing on the theoretical vs actual context pressure problem
in A2A (agent-to-agent) advertising transactions.

The key insight: If agents logged every impression transaction, context
windows would explode. This creates a two-level pressure model:
1. Theoretical pressure: What WOULD happen if we logged everything
2. Actual pressure: What fits in the API context window

When theoretical pressure exceeds actual capacity, agents must summarize
or forget - leading to price drift and reconciliation failures.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


# Constants for pressure calculation
TOKENS_PER_IMPRESSION = 50  # bid request + response tokens
CONTEXT_LIMIT = 200_000  # Claude's context window limit


@dataclass
class RecallResult:
    """Result of querying an agent about a previously agreed term."""
    seller_id: str
    expected_cpm: float
    recalled_cpm: float
    drift: float  # Absolute percentage drift from expected
    is_correct: bool  # drift < 2%
    raw_response: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchResult:
    """Result of processing a single batch of impressions."""
    batch_number: int
    impressions_in_batch: int
    impressions_cumulative: int
    
    # Pressure metrics at this batch
    theoretical_tokens: int  # If we logged every impression
    actual_context_tokens: int  # Real API context
    pressure_ratio: float  # theoretical / context_limit
    
    # Memory check result
    recall_result: Optional[RecallResult] = None
    
    # Processing metrics
    processing_time_ms: float = 0.0
    api_cost_usd: float = 0.0
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        result = asdict(self)
        if self.recall_result:
            result['recall_result'] = self.recall_result.to_dict()
        return result


@dataclass
class CampaignExecution:
    """
    Tracks execution metrics for a single campaign.
    
    This models the core problem: A campaign with 1M+ impressions
    would generate ~50M tokens of bid/response data. Since context
    windows are limited (~200K), agents must operate under severe
    pressure, leading to memory failures and price drift.
    """
    campaign_id: str
    buyer_id: str
    seller_id: str
    agreed_cpm: float  # Negotiated price at campaign start
    impressions_total: int  # 1M+ per campaign
    impressions_processed: int = 0  # Running count
    
    # Context pressure tracking
    theoretical_tokens: int = 0  # If we logged every impression (~50 tokens each)
    actual_context_tokens: int = 0  # Real API context usage
    pressure_ratio: float = 0.0  # theoretical / 200K limit
    
    # Failure tracking
    memory_overflow_events: int = 0  # Times agent "forgot" deal terms
    price_drift_incidents: int = 0  # Recalled CPM ≠ agreed CPM
    reconciliation_failures: int = 0  # Buyer vs seller mismatch
    
    # Recall accuracy tracking
    recall_checks: int = 0
    recall_successes: int = 0
    recall_accuracy: float = 1.0
    total_drift: float = 0.0  # Sum of all drifts for averaging
    avg_drift: float = 0.0
    max_drift: float = 0.0
    
    # Batch tracking
    batches_processed: int = 0
    batch_size: int = 100_000  # 100K impressions per batch
    batch_results: List[BatchResult] = field(default_factory=list)
    
    # Financial tracking
    total_spend: float = 0.0  # impressions * agreed_cpm / 1000
    api_cost_usd: float = 0.0
    
    # Timing
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    
    def calculate_pressure(self) -> float:
        """Calculate current context pressure ratio."""
        self.theoretical_tokens = self.impressions_processed * TOKENS_PER_IMPRESSION
        self.pressure_ratio = self.theoretical_tokens / CONTEXT_LIMIT
        return self.pressure_ratio
    
    def record_batch(self, batch_result: BatchResult) -> None:
        """Record a completed batch and update metrics."""
        self.batch_results.append(batch_result)
        self.batches_processed += 1
        self.impressions_processed = batch_result.impressions_cumulative
        self.theoretical_tokens = batch_result.theoretical_tokens
        self.actual_context_tokens = batch_result.actual_context_tokens
        self.pressure_ratio = batch_result.pressure_ratio
        self.api_cost_usd += batch_result.api_cost_usd
        
        # Update recall metrics if we did a memory check
        if batch_result.recall_result:
            self.recall_checks += 1
            if batch_result.recall_result.is_correct:
                self.recall_successes += 1
            else:
                self.price_drift_incidents += 1
            
            self.total_drift += batch_result.recall_result.drift
            self.avg_drift = self.total_drift / self.recall_checks
            self.max_drift = max(self.max_drift, batch_result.recall_result.drift)
            self.recall_accuracy = self.recall_successes / self.recall_checks
    
    def calculate_spend(self) -> float:
        """Calculate total campaign spend."""
        self.total_spend = (self.impressions_processed * self.agreed_cpm) / 1000
        return self.total_spend
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "campaign_id": self.campaign_id,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "agreed_cpm": self.agreed_cpm,
            "impressions_total": self.impressions_total,
            "impressions_processed": self.impressions_processed,
            "theoretical_tokens": self.theoretical_tokens,
            "actual_context_tokens": self.actual_context_tokens,
            "pressure_ratio": round(self.pressure_ratio, 4),
            "pressure_percent": round(self.pressure_ratio * 100, 2),
            "memory_overflow_events": self.memory_overflow_events,
            "price_drift_incidents": self.price_drift_incidents,
            "reconciliation_failures": self.reconciliation_failures,
            "recall_checks": self.recall_checks,
            "recall_successes": self.recall_successes,
            "recall_accuracy": round(self.recall_accuracy, 4),
            "avg_drift": round(self.avg_drift, 4),
            "max_drift": round(self.max_drift, 4),
            "batches_processed": self.batches_processed,
            "batch_size": self.batch_size,
            "total_spend": round(self.total_spend, 2),
            "api_cost_usd": round(self.api_cost_usd, 6),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
        }
        return result


@dataclass
class PressureThreshold:
    """Defines expected behavior at a given pressure level."""
    min_pressure: float  # Minimum pressure ratio (e.g., 0.0)
    max_pressure: float  # Maximum pressure ratio (e.g., 0.25)
    label: str  # Human-readable label
    expected_drift: str  # Expected price drift description
    drift_tolerance: float  # Maximum acceptable drift (e.g., 0.02 for 2%)
    expected_recall_accuracy: float  # Expected recall accuracy (0.0-1.0)


# Default pressure thresholds based on task specification
DEFAULT_PRESSURE_THRESHOLDS = [
    PressureThreshold(0.00, 0.25, "low", "Clean recall", 0.02, 0.98),
    PressureThreshold(0.25, 0.50, "moderate", "Minor drift (±2%)", 0.05, 0.90),
    PressureThreshold(0.50, 0.75, "high", "Moderate drift (±5-10%)", 0.10, 0.75),
    PressureThreshold(0.75, 1.00, "critical", "Hallucinations expected", 0.25, 0.50),
    PressureThreshold(1.00, float('inf'), "overflow", "Breakdown expected", 1.00, 0.10),
]


def get_pressure_threshold(pressure_ratio: float) -> PressureThreshold:
    """Get the threshold definition for a given pressure ratio."""
    for threshold in DEFAULT_PRESSURE_THRESHOLDS:
        if threshold.min_pressure <= pressure_ratio < threshold.max_pressure:
            return threshold
    return DEFAULT_PRESSURE_THRESHOLDS[-1]  # overflow


@dataclass
class PressureSimulationResult:
    """Complete results from a context pressure simulation."""
    simulation_id: str
    start_time: str
    end_time: Optional[str] = None
    
    # Configuration
    num_buyers: int = 10
    num_campaigns_per_buyer: int = 10
    impressions_per_campaign: int = 1_000_000
    batch_size: int = 100_000
    
    # Campaign executions
    campaigns: List[CampaignExecution] = field(default_factory=list)
    
    # Aggregate metrics
    total_impressions: int = 0
    total_batches: int = 0
    total_recall_checks: int = 0
    total_recall_successes: int = 0
    total_price_drift_incidents: int = 0
    overall_recall_accuracy: float = 0.0
    overall_avg_drift: float = 0.0
    
    # Pressure-level breakdown
    pressure_level_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Cost tracking
    total_api_cost_usd: float = 0.0
    total_campaign_spend: float = 0.0
    
    status: str = "running"
    
    def finalize(self) -> None:
        """Calculate final aggregate metrics."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        
        for campaign in self.campaigns:
            self.total_impressions += campaign.impressions_processed
            self.total_batches += campaign.batches_processed
            self.total_recall_checks += campaign.recall_checks
            self.total_recall_successes += campaign.recall_successes
            self.total_price_drift_incidents += campaign.price_drift_incidents
            self.total_api_cost_usd += campaign.api_cost_usd
            self.total_campaign_spend += campaign.total_spend
        
        if self.total_recall_checks > 0:
            self.overall_recall_accuracy = self.total_recall_successes / self.total_recall_checks
            # Calculate weighted average drift
            total_drift = sum(c.total_drift for c in self.campaigns)
            self.overall_avg_drift = total_drift / self.total_recall_checks
        
        # Calculate per-pressure-level stats
        self._calculate_pressure_stats()
    
    def _calculate_pressure_stats(self) -> None:
        """Calculate statistics broken down by pressure level."""
        # Initialize buckets
        for threshold in DEFAULT_PRESSURE_THRESHOLDS:
            self.pressure_level_stats[threshold.label] = {
                "recall_checks": 0,
                "recall_successes": 0,
                "recall_accuracy": 0.0,
                "total_drift": 0.0,
                "avg_drift": 0.0,
                "max_drift": 0.0,
                "expected_accuracy": threshold.expected_recall_accuracy,
                "drift_tolerance": threshold.drift_tolerance,
            }
        
        # Populate from batch results
        for campaign in self.campaigns:
            for batch in campaign.batch_results:
                if batch.recall_result:
                    threshold = get_pressure_threshold(batch.pressure_ratio)
                    stats = self.pressure_level_stats[threshold.label]
                    stats["recall_checks"] += 1
                    if batch.recall_result.is_correct:
                        stats["recall_successes"] += 1
                    stats["total_drift"] += batch.recall_result.drift
                    stats["max_drift"] = max(stats["max_drift"], batch.recall_result.drift)
        
        # Calculate averages
        for label, stats in self.pressure_level_stats.items():
            if stats["recall_checks"] > 0:
                stats["recall_accuracy"] = stats["recall_successes"] / stats["recall_checks"]
                stats["avg_drift"] = stats["total_drift"] / stats["recall_checks"]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "simulation_id": self.simulation_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": {
                "num_buyers": self.num_buyers,
                "num_campaigns_per_buyer": self.num_campaigns_per_buyer,
                "impressions_per_campaign": self.impressions_per_campaign,
                "batch_size": self.batch_size,
            },
            "aggregate_metrics": {
                "total_impressions": self.total_impressions,
                "total_batches": self.total_batches,
                "total_recall_checks": self.total_recall_checks,
                "total_recall_successes": self.total_recall_successes,
                "total_price_drift_incidents": self.total_price_drift_incidents,
                "overall_recall_accuracy": round(self.overall_recall_accuracy, 4),
                "overall_avg_drift": round(self.overall_avg_drift, 4),
            },
            "pressure_level_stats": {
                label: {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}
                for label, stats in self.pressure_level_stats.items()
            },
            "cost": {
                "total_api_cost_usd": round(self.total_api_cost_usd, 4),
                "total_campaign_spend": round(self.total_campaign_spend, 2),
            },
            "campaigns": [c.to_dict() for c in self.campaigns],
            "status": self.status,
        }
