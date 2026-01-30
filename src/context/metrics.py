"""
Context Metrics Tracking.

Comprehensive metrics for analyzing context health across
the agent hierarchy and different scenarios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

if TYPE_CHECKING:
    from .rot import DecayResult, HandoffResult
    from .recovery import RecoveryResult

logger = structlog.get_logger(__name__)


class MetricEvent(str, Enum):
    """Types of metric events."""
    HANDOFF = "handoff"
    DECAY = "decay"
    RECOVERY = "recovery"
    TRUNCATION = "truncation"
    RESTART = "restart"
    AGGREGATION = "aggregation"


class ContextMetricsSummary(BaseModel):
    """Summary of context health metrics."""
    # Time range
    start_time: float = Field(default_factory=time.time)
    end_time: float = Field(default_factory=time.time)
    
    # Handoff metrics
    total_handoffs: int = 0
    successful_handoffs: int = 0
    handoff_losses: int = 0  # Handoffs with any context loss
    total_tokens_handed_off: int = 0
    total_tokens_lost_handoff: int = 0
    
    # Decay metrics
    total_decay_events: int = 0
    keys_lost_to_decay: int = 0
    tokens_lost_to_decay: int = 0
    
    # Recovery metrics
    recovery_attempts: int = 0
    recovery_successes: int = 0
    total_tokens_recovered: int = 0
    total_tokens_unrecoverable: int = 0
    
    # Truncation metrics
    truncation_events: int = 0
    tokens_truncated: int = 0
    
    # Restart metrics
    restart_events: int = 0
    
    # Aggregation metrics
    aggregation_events: int = 0
    conflicts_resolved: int = 0
    
    # Overall health
    average_context_integrity: float = 1.0
    average_handoff_loss_rate: float = 0.0
    average_recovery_rate: float = 0.0
    
    # Scenario breakdown
    scenario_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Duration of metrics collection."""
        return self.end_time - self.start_time
    
    @property
    def handoff_success_rate(self) -> float:
        """Percentage of handoffs with no loss."""
        if self.total_handoffs == 0:
            return 1.0
        return (self.total_handoffs - self.handoff_losses) / self.total_handoffs
    
    @property
    def recovery_success_rate(self) -> float:
        """Percentage of successful recovery attempts."""
        if self.recovery_attempts == 0:
            return 0.0
        return self.recovery_successes / self.recovery_attempts
    
    @property
    def total_tokens_lost(self) -> int:
        """Total tokens lost across all events."""
        return (
            self.total_tokens_lost_handoff +
            self.tokens_lost_to_decay +
            self.tokens_truncated
        )
    
    @property
    def net_token_loss(self) -> int:
        """Net token loss after recovery."""
        return self.total_tokens_lost - self.total_tokens_recovered


@dataclass
class MetricEntry:
    """Individual metric entry."""
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: MetricEvent = MetricEvent.HANDOFF
    agent_id: str = ""
    scenario: str = ""
    
    # Numeric values
    tokens_involved: int = 0
    tokens_lost: int = 0
    items_involved: int = 0
    items_lost: int = 0
    
    # Success/failure
    success: bool = True
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    
    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextMetrics:
    """
    Track context-related metrics for analysis.
    
    Collects detailed metrics about context flow, degradation,
    and recovery across the agent hierarchy.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self._entries: list[MetricEntry] = []
        self._start_time = time.time()
        self.log = structlog.get_logger(__name__)
    
    async def record_handoff(self, result: "HandoffResult") -> None:
        """
        Record a context handoff.
        
        Args:
            result: HandoffResult from the handoff operation
        """
        entry = MetricEntry(
            event_type=MetricEvent.HANDOFF,
            agent_id=f"{result.from_agent_id}->{result.to_agent_id}",
            scenario=result.scenario,
            tokens_involved=result.tokens_sent,
            tokens_lost=result.tokens_lost,
            items_involved=result.items_sent,
            items_lost=result.items_lost,
            success=result.items_lost == 0,
            duration_ms=result.duration_ms,
            metadata={
                "from_agent": result.from_agent_id,
                "to_agent": result.to_agent_id,
                "lost_keys": result.lost_keys,
                "recoverable": result.recoverable
            }
        )
        
        self._entries.append(entry)
        
        self.log.debug(
            "handoff_metric_recorded",
            handoff_id=result.handoff_id,
            loss_rate=result.loss_rate
        )
    
    async def record_decay(self, result: "DecayResult") -> None:
        """
        Record context decay event.
        
        Args:
            result: DecayResult from decay simulation
        """
        entry = MetricEntry(
            event_type=MetricEvent.DECAY,
            agent_id=result.agent_id,
            tokens_involved=result.tokens_before,
            tokens_lost=result.tokens_lost,
            items_involved=result.entries_before,
            items_lost=result.entries_pruned,
            success=True,
            metadata={
                "decay_rate": result.decay_rate,
                "importance_drop": result.importance_drop,
                "keys_affected": result.keys_affected
            }
        )
        
        self._entries.append(entry)
        
        self.log.debug(
            "decay_metric_recorded",
            agent_id=result.agent_id,
            entries_pruned=result.entries_pruned
        )
    
    async def record_recovery(self, result: "RecoveryResult") -> None:
        """
        Record recovery attempt.
        
        Args:
            result: RecoveryResult from recovery attempt
        """
        entry = MetricEntry(
            event_type=MetricEvent.RECOVERY,
            agent_id=result.agent_id,
            scenario=result.scenario,
            tokens_involved=result.tokens_requested,
            tokens_lost=result.tokens_requested - result.tokens_recovered,
            items_involved=result.entries_requested,
            items_lost=result.entries_requested - result.entries_recovered,
            success=result.success,
            duration_ms=result.duration_ms,
            metadata={
                "source": result.source.value,
                "recovery_rate": result.recovery_rate,
                "fidelity": result.fidelity,
                "recovered_keys": result.recovered_keys,
                "failed_keys": result.failed_keys
            }
        )
        
        self._entries.append(entry)
        
        self.log.debug(
            "recovery_metric_recorded",
            agent_id=result.agent_id,
            success=result.success,
            recovery_rate=result.recovery_rate
        )
    
    async def record_truncation(
        self,
        agent_id: str,
        entries_truncated: int,
        tokens_truncated: int
    ) -> None:
        """
        Record a truncation event.
        
        Args:
            agent_id: Agent that experienced truncation
            entries_truncated: Number of entries removed
            tokens_truncated: Number of tokens removed
        """
        entry = MetricEntry(
            event_type=MetricEvent.TRUNCATION,
            agent_id=agent_id,
            tokens_involved=tokens_truncated,
            tokens_lost=tokens_truncated,
            items_involved=entries_truncated,
            items_lost=entries_truncated,
            success=True
        )
        
        self._entries.append(entry)
        
        self.log.debug(
            "truncation_metric_recorded",
            agent_id=agent_id,
            tokens_truncated=tokens_truncated
        )
    
    async def record_restart(self, agent_id: str) -> None:
        """
        Record an agent restart event.
        
        Args:
            agent_id: Agent that restarted
        """
        entry = MetricEntry(
            event_type=MetricEvent.RESTART,
            agent_id=agent_id,
            success=True
        )
        
        self._entries.append(entry)
        
        self.log.debug(
            "restart_metric_recorded",
            agent_id=agent_id
        )
    
    async def record_aggregation(
        self,
        target_agent_id: str,
        source_count: int,
        conflicts: int
    ) -> None:
        """
        Record a context aggregation event.
        
        Args:
            target_agent_id: Agent receiving aggregated context
            source_count: Number of source agents
            conflicts: Number of conflicts resolved
        """
        entry = MetricEntry(
            event_type=MetricEvent.AGGREGATION,
            agent_id=target_agent_id,
            items_involved=source_count,
            metadata={
                "source_count": source_count,
                "conflicts_resolved": conflicts
            }
        )
        
        self._entries.append(entry)
        
        self.log.debug(
            "aggregation_metric_recorded",
            agent_id=target_agent_id,
            source_count=source_count,
            conflicts=conflicts
        )
    
    async def get_summary(self) -> ContextMetricsSummary:
        """
        Get summary of context health.
        
        Returns:
            ContextMetricsSummary with aggregated metrics
        """
        summary = ContextMetricsSummary(
            start_time=self._start_time,
            end_time=time.time()
        )
        
        # Process each entry
        handoff_integrities: list[float] = []
        recovery_rates: list[float] = []
        scenario_data: dict[str, dict[str, Any]] = {}
        
        for entry in self._entries:
            # Initialize scenario stats if needed
            if entry.scenario and entry.scenario not in scenario_data:
                scenario_data[entry.scenario] = {
                    "handoffs": 0,
                    "tokens_lost": 0,
                    "recoveries": 0,
                    "recovery_successes": 0
                }
            
            if entry.event_type == MetricEvent.HANDOFF:
                summary.total_handoffs += 1
                summary.total_tokens_handed_off += entry.tokens_involved
                summary.total_tokens_lost_handoff += entry.tokens_lost
                
                if entry.tokens_lost > 0:
                    summary.handoff_losses += 1
                else:
                    summary.successful_handoffs += 1
                
                # Calculate integrity for this handoff
                if entry.tokens_involved > 0:
                    integrity = 1.0 - (entry.tokens_lost / entry.tokens_involved)
                    handoff_integrities.append(integrity)
                
                if entry.scenario:
                    scenario_data[entry.scenario]["handoffs"] += 1
                    scenario_data[entry.scenario]["tokens_lost"] += entry.tokens_lost
                    
            elif entry.event_type == MetricEvent.DECAY:
                summary.total_decay_events += 1
                summary.keys_lost_to_decay += entry.items_lost
                summary.tokens_lost_to_decay += entry.tokens_lost
                
            elif entry.event_type == MetricEvent.RECOVERY:
                summary.recovery_attempts += 1
                if entry.success:
                    summary.recovery_successes += 1
                    # Tokens recovered = tokens_involved - tokens_lost
                    summary.total_tokens_recovered += (
                        entry.tokens_involved - entry.tokens_lost
                    )
                else:
                    summary.total_tokens_unrecoverable += entry.tokens_involved
                
                recovery_rate = entry.metadata.get("recovery_rate", 0.0)
                recovery_rates.append(recovery_rate)
                
                if entry.scenario:
                    scenario_data[entry.scenario]["recoveries"] += 1
                    if entry.success:
                        scenario_data[entry.scenario]["recovery_successes"] += 1
                
            elif entry.event_type == MetricEvent.TRUNCATION:
                summary.truncation_events += 1
                summary.tokens_truncated += entry.tokens_lost
                
            elif entry.event_type == MetricEvent.RESTART:
                summary.restart_events += 1
                
            elif entry.event_type == MetricEvent.AGGREGATION:
                summary.aggregation_events += 1
                summary.conflicts_resolved += entry.metadata.get(
                    "conflicts_resolved", 0
                )
        
        # Calculate averages
        if handoff_integrities:
            summary.average_context_integrity = (
                sum(handoff_integrities) / len(handoff_integrities)
            )
            summary.average_handoff_loss_rate = (
                1.0 - summary.average_context_integrity
            )
        
        if recovery_rates:
            summary.average_recovery_rate = (
                sum(recovery_rates) / len(recovery_rates)
            )
        
        summary.scenario_stats = scenario_data
        
        return summary
    
    def get_entries(
        self,
        event_type: MetricEvent | None = None,
        agent_id: str | None = None,
        scenario: str | None = None,
        since: float | None = None
    ) -> list[MetricEntry]:
        """
        Get metric entries with optional filtering.
        
        Args:
            event_type: Filter by event type
            agent_id: Filter by agent ID
            scenario: Filter by scenario
            since: Filter entries after this timestamp
            
        Returns:
            List of matching metric entries
        """
        entries = self._entries
        
        if event_type is not None:
            entries = [e for e in entries if e.event_type == event_type]
        
        if agent_id is not None:
            entries = [e for e in entries if agent_id in e.agent_id]
        
        if scenario is not None:
            entries = [e for e in entries if e.scenario == scenario]
        
        if since is not None:
            entries = [e for e in entries if e.timestamp >= since]
        
        return entries
    
    def get_scenario_comparison(self) -> dict[str, dict[str, Any]]:
        """
        Compare metrics across scenarios.
        
        Returns:
            Dict with scenario-by-scenario breakdown
        """
        scenarios: dict[str, dict[str, Any]] = {
            "A": {"handoffs": 0, "losses": 0, "recoveries": 0, "recovered": 0},
            "B": {"handoffs": 0, "losses": 0, "recoveries": 0, "recovered": 0},
            "C": {"handoffs": 0, "losses": 0, "recoveries": 0, "recovered": 0}
        }
        
        for entry in self._entries:
            if entry.scenario not in scenarios:
                continue
            
            if entry.event_type == MetricEvent.HANDOFF:
                scenarios[entry.scenario]["handoffs"] += 1
                scenarios[entry.scenario]["losses"] += entry.tokens_lost
                
            elif entry.event_type == MetricEvent.RECOVERY:
                scenarios[entry.scenario]["recoveries"] += 1
                if entry.success:
                    scenarios[entry.scenario]["recovered"] += (
                        entry.tokens_involved - entry.tokens_lost
                    )
        
        # Calculate net loss for each scenario
        for scenario, stats in scenarios.items():
            stats["net_loss"] = stats["losses"] - stats["recovered"]
            if stats["handoffs"] > 0:
                stats["avg_loss_rate"] = stats["losses"] / (
                    stats["handoffs"] * 1000  # Rough estimate per handoff
                )
            else:
                stats["avg_loss_rate"] = 0.0
        
        return scenarios
    
    def clear(self) -> None:
        """Clear all metrics."""
        self._entries.clear()
        self._start_time = time.time()
        self.log.info("metrics_cleared")
    
    def export_json(self) -> dict[str, Any]:
        """Export metrics as JSON-serializable dict."""
        return {
            "start_time": self._start_time,
            "entry_count": len(self._entries),
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "event_type": e.event_type.value,
                    "agent_id": e.agent_id,
                    "scenario": e.scenario,
                    "tokens_involved": e.tokens_involved,
                    "tokens_lost": e.tokens_lost,
                    "items_involved": e.items_involved,
                    "items_lost": e.items_lost,
                    "success": e.success,
                    "timestamp": e.timestamp,
                    "duration_ms": e.duration_ms,
                    "metadata": e.metadata
                }
                for e in self._entries
            ]
        }
