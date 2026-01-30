"""
V2 Report Generators for Context Window Hallucination Testing.

Implements specialized reports for tracking hallucination rates, types,
context overflow events, and recovery comparisons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Any


# =============================================================================
# Hallucination Types (mirrors hallucination/classifier.py spec)
# =============================================================================

class HallucinationType(str, Enum):
    """Types of hallucinations in agent decisions."""
    BUDGET_DRIFT = "budget_drift"
    FREQUENCY_VIOLATION = "frequency_cap"
    DEAL_INVENTION = "deal_invention"
    CROSS_CAMPAIGN = "cross_campaign"
    PHANTOM_INVENTORY = "phantom_inventory"
    PRICE_ANCHOR_ERROR = "price_anchor"


# =============================================================================
# Report Data Structures
# =============================================================================

@dataclass
class DailyHallucinationData:
    """Hallucination data for a single day."""
    day: int
    decisions: int
    hallucinations: int
    rate: float  # As percentage
    cumulative_rate: float  # As percentage


@dataclass
class HallucinationRateReportData:
    """Data for the hallucination rate over time report."""
    days: list[DailyHallucinationData] = field(default_factory=list)
    total_decisions: int = 0
    total_hallucinations: int = 0
    final_cumulative_rate: float = 0.0
    peak_daily_rate: float = 0.0
    peak_day: int = 0


@dataclass
class HallucinationTypeData:
    """Data for a single hallucination type."""
    type: HallucinationType
    type_name: str
    count: int
    percentage: float  # As percentage
    avg_severity: float  # 0.0 - 1.0


@dataclass
class HallucinationTypeReportData:
    """Data for hallucination type distribution report."""
    types: list[HallucinationTypeData] = field(default_factory=list)
    total_hallucinations: int = 0
    most_common_type: Optional[str] = None
    highest_severity_type: Optional[str] = None


@dataclass
class OverflowEventData:
    """Data for a single context overflow event."""
    event_number: int
    day: int
    hour: int
    context_tokens: int
    context_limit: int
    compression_pct: float  # Events dropped percentage
    events_lost: int
    quality_before: float  # As percentage
    quality_after: float  # As percentage
    recovery_decisions: int  # Decisions until quality recovered


@dataclass
class ContextOverflowReportData:
    """Data for context overflow impact report."""
    events: list[OverflowEventData] = field(default_factory=list)
    total_overflow_events: int = 0
    total_events_lost: int = 0
    avg_quality_degradation: float = 0.0
    avg_recovery_time: float = 0.0  # Decisions to recover


@dataclass
class RecoveryModeData:
    """Data for a single recovery mode."""
    mode: str
    accuracy: float  # As percentage
    time_seconds: float
    decisions_affected: int
    error_rate: float  # As percentage


@dataclass
class RestartEventData:
    """Data for a single restart event."""
    event_number: int
    day: int
    hour: int
    recovery_modes: list[RecoveryModeData] = field(default_factory=list)


@dataclass
class RecoveryComparisonReportData:
    """Data for recovery comparison report."""
    events: list[RestartEventData] = field(default_factory=list)
    total_restart_events: int = 0
    avg_private_db_accuracy: float = 0.0
    avg_ledger_accuracy: float = 0.0
    reliability_advantage: float = 0.0  # Ledger advantage in percentage points


# =============================================================================
# Simulation Results Types (expected input structure)
# =============================================================================

@dataclass
class Hallucination:
    """Individual hallucination record."""
    decision_id: str
    timestamp: datetime
    day: int
    type: HallucinationType
    expected: Any
    actual: Any
    severity: float  # 0.0 - 1.0


@dataclass
class OverflowEvent:
    """Context overflow event record."""
    event_id: str
    timestamp: datetime
    day: int
    hour: int
    tokens_before: int
    context_limit: int
    events_dropped: int
    total_events_before: int
    quality_before_pct: float
    quality_after_pct: float
    recovery_decisions: int


@dataclass
class RestartEvent:
    """Agent restart event record."""
    event_id: str
    timestamp: datetime
    day: int
    hour: int
    recovery_results: dict[str, dict[str, float]]  # mode -> {accuracy, time_seconds, decisions_affected}


@dataclass 
class SimulationResults:
    """Complete simulation results for V2 reports."""
    # Basic metrics
    total_days: int = 30
    decisions_per_day: int = 10000
    
    # Hallucination data
    hallucinations: list[Hallucination] = field(default_factory=list)
    decisions_by_day: dict[int, int] = field(default_factory=dict)
    
    # Context overflow data
    overflow_events: list[OverflowEvent] = field(default_factory=list)
    context_limit: int = 200_000
    
    # Restart/recovery data
    restart_events: list[RestartEvent] = field(default_factory=list)


# =============================================================================
# Report Base Class
# =============================================================================

class BaseReport:
    """Base class for V2 reports."""
    
    def __init__(self):
        self._data = None
        self._generated_at: Optional[datetime] = None
    
    @property
    def data(self):
        """Get the report data."""
        if self._data is None:
            raise ValueError("Report has not been generated. Call generate() first.")
        return self._data
    
    def generate(self, simulation_results: SimulationResults):
        """Generate report from simulation results. Override in subclasses."""
        raise NotImplementedError
    
    def to_markdown(self) -> str:
        """Render report as Markdown. Override in subclasses."""
        raise NotImplementedError
    
    def to_json(self) -> dict:
        """Convert report data to JSON-serializable dict."""
        if self._data is None:
            raise ValueError("Report has not been generated. Call generate() first.")
        return {
            "generated_at": self._generated_at.isoformat() if self._generated_at else None,
            "data": asdict(self._data) if hasattr(self._data, '__dataclass_fields__') else self._data,
        }


# =============================================================================
# Hallucination Rate Report
# =============================================================================

class HallucinationRateReport(BaseReport):
    """
    Report showing hallucinations by day with cumulative rate.
    
    Demonstrates the hypothesis that hallucination rate grows non-linearly
    as context window pressure increases over campaign duration.
    """
    
    def generate(self, simulation_results: SimulationResults) -> HallucinationRateReportData:
        """
        Generate hallucination rate report from simulation results.
        
        Args:
            simulation_results: Complete simulation results
            
        Returns:
            HallucinationRateReportData with daily breakdown
        """
        self._generated_at = datetime.utcnow()
        
        # Group hallucinations by day
        hallucinations_by_day: dict[int, list[Hallucination]] = {}
        for h in simulation_results.hallucinations:
            if h.day not in hallucinations_by_day:
                hallucinations_by_day[h.day] = []
            hallucinations_by_day[h.day].append(h)
        
        # Build daily data
        days_data: list[DailyHallucinationData] = []
        cumulative_decisions = 0
        cumulative_hallucinations = 0
        peak_rate = 0.0
        peak_day = 1
        
        for day in range(1, simulation_results.total_days + 1):
            # Get decisions for this day
            day_decisions = simulation_results.decisions_by_day.get(
                day, simulation_results.decisions_per_day
            )
            day_hallucinations = len(hallucinations_by_day.get(day, []))
            
            cumulative_decisions += day_decisions
            cumulative_hallucinations += day_hallucinations
            
            daily_rate = (day_hallucinations / day_decisions * 100) if day_decisions > 0 else 0.0
            cumulative_rate = (cumulative_hallucinations / cumulative_decisions * 100) if cumulative_decisions > 0 else 0.0
            
            if daily_rate > peak_rate:
                peak_rate = daily_rate
                peak_day = day
            
            days_data.append(DailyHallucinationData(
                day=day,
                decisions=day_decisions,
                hallucinations=day_hallucinations,
                rate=round(daily_rate, 2),
                cumulative_rate=round(cumulative_rate, 2),
            ))
        
        self._data = HallucinationRateReportData(
            days=days_data,
            total_decisions=cumulative_decisions,
            total_hallucinations=cumulative_hallucinations,
            final_cumulative_rate=round(
                (cumulative_hallucinations / cumulative_decisions * 100) if cumulative_decisions > 0 else 0.0, 
                2
            ),
            peak_daily_rate=round(peak_rate, 2),
            peak_day=peak_day,
        )
        
        return self._data
    
    def to_markdown(self) -> str:
        """Render hallucination rate report as Markdown table."""
        data = self.data
        
        lines = [
            "## Hallucination Rate Over Time",
            "",
            f"**Generated:** {self._generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') if self._generated_at else 'N/A'}",
            "",
            f"**Total Decisions:** {data.total_decisions:,}",
            f"**Total Hallucinations:** {data.total_hallucinations:,}",
            f"**Final Cumulative Rate:** {data.final_cumulative_rate:.2f}%",
            f"**Peak Daily Rate:** {data.peak_daily_rate:.2f}% (Day {data.peak_day})",
            "",
            "| Day | Decisions | Hallucinations | Rate | Cumulative |",
            "|-----|-----------|----------------|------|------------|",
        ]
        
        for day_data in data.days:
            lines.append(
                f"| {day_data.day:3d} | {day_data.decisions:,} | "
                f"{day_data.hallucinations:,} | {day_data.rate:.2f}% | "
                f"{day_data.cumulative_rate:.2f}% |"
            )
        
        return "\n".join(lines)
    
    def to_json(self) -> dict:
        """Convert to JSON with additional metadata."""
        base = super().to_json()
        base["report_type"] = "hallucination_rate"
        return base


# =============================================================================
# Hallucination Type Report
# =============================================================================

class HallucinationTypeReport(BaseReport):
    """
    Report showing distribution of hallucinations by type.
    
    Shows count, percentage of total, and average severity for each
    hallucination type, helping identify the most problematic error modes.
    """
    
    def generate(self, simulation_results: SimulationResults) -> HallucinationTypeReportData:
        """
        Generate hallucination type distribution report.
        
        Args:
            simulation_results: Complete simulation results
            
        Returns:
            HallucinationTypeReportData with type breakdown
        """
        self._generated_at = datetime.utcnow()
        
        # Group by type
        by_type: dict[HallucinationType, list[Hallucination]] = {}
        for h in simulation_results.hallucinations:
            if h.type not in by_type:
                by_type[h.type] = []
            by_type[h.type].append(h)
        
        total = len(simulation_results.hallucinations)
        types_data: list[HallucinationTypeData] = []
        
        most_common_type = None
        most_common_count = 0
        highest_severity_type = None
        highest_severity = 0.0
        
        # Process each type
        for h_type in HallucinationType:
            hallucinations = by_type.get(h_type, [])
            count = len(hallucinations)
            percentage = (count / total * 100) if total > 0 else 0.0
            avg_severity = (
                sum(h.severity for h in hallucinations) / count 
                if count > 0 else 0.0
            )
            
            # Human-readable type names
            type_names = {
                HallucinationType.BUDGET_DRIFT: "Budget Drift",
                HallucinationType.FREQUENCY_VIOLATION: "Frequency Violation",
                HallucinationType.DEAL_INVENTION: "Deal Invention",
                HallucinationType.CROSS_CAMPAIGN: "Cross-Contamination",
                HallucinationType.PHANTOM_INVENTORY: "Phantom Inventory",
                HallucinationType.PRICE_ANCHOR_ERROR: "Price Anchor Error",
            }
            
            types_data.append(HallucinationTypeData(
                type=h_type,
                type_name=type_names.get(h_type, h_type.value),
                count=count,
                percentage=round(percentage, 1),
                avg_severity=round(avg_severity, 2),
            ))
            
            if count > most_common_count:
                most_common_count = count
                most_common_type = type_names.get(h_type, h_type.value)
            
            if avg_severity > highest_severity and count > 0:
                highest_severity = avg_severity
                highest_severity_type = type_names.get(h_type, h_type.value)
        
        # Sort by count descending
        types_data.sort(key=lambda x: x.count, reverse=True)
        
        self._data = HallucinationTypeReportData(
            types=types_data,
            total_hallucinations=total,
            most_common_type=most_common_type,
            highest_severity_type=highest_severity_type,
        )
        
        return self._data
    
    def to_markdown(self) -> str:
        """Render hallucination type report as Markdown table."""
        data = self.data
        
        lines = [
            "## Hallucination Type Distribution",
            "",
            f"**Generated:** {self._generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') if self._generated_at else 'N/A'}",
            "",
            f"**Total Hallucinations:** {data.total_hallucinations:,}",
            f"**Most Common Type:** {data.most_common_type or 'N/A'}",
            f"**Highest Severity Type:** {data.highest_severity_type or 'N/A'}",
            "",
            "| Type | Count | % of Total | Avg Severity |",
            "|------|-------|------------|--------------|",
        ]
        
        for type_data in data.types:
            lines.append(
                f"| {type_data.type_name:<23} | {type_data.count:,} | "
                f"{type_data.percentage:.1f}% | {type_data.avg_severity:.2f} |"
            )
        
        return "\n".join(lines)
    
    def to_json(self) -> dict:
        """Convert to JSON with type serialization."""
        base = super().to_json()
        base["report_type"] = "hallucination_type"
        # Convert enum values to strings for JSON
        if "data" in base and "types" in base["data"]:
            for t in base["data"]["types"]:
                if isinstance(t.get("type"), HallucinationType):
                    t["type"] = t["type"].value
        return base


# =============================================================================
# Context Overflow Report
# =============================================================================

class ContextOverflowReport(BaseReport):
    """
    Report showing details of each context overflow event.
    
    Tracks context token accumulation, compression events, and the
    impact on decision quality before and after overflow.
    """
    
    def generate(self, simulation_results: SimulationResults) -> ContextOverflowReportData:
        """
        Generate context overflow impact report.
        
        Args:
            simulation_results: Complete simulation results
            
        Returns:
            ContextOverflowReportData with event details
        """
        self._generated_at = datetime.utcnow()
        
        events_data: list[OverflowEventData] = []
        total_events_lost = 0
        total_quality_drop = 0.0
        total_recovery_time = 0
        
        for i, event in enumerate(simulation_results.overflow_events, 1):
            compression_pct = (
                event.events_dropped / event.total_events_before * 100
                if event.total_events_before > 0 else 0.0
            )
            quality_drop = event.quality_before_pct - event.quality_after_pct
            
            events_data.append(OverflowEventData(
                event_number=i,
                day=event.day,
                hour=event.hour,
                context_tokens=event.tokens_before,
                context_limit=event.context_limit,
                compression_pct=round(compression_pct, 1),
                events_lost=event.events_dropped,
                quality_before=round(event.quality_before_pct, 1),
                quality_after=round(event.quality_after_pct, 1),
                recovery_decisions=event.recovery_decisions,
            ))
            
            total_events_lost += event.events_dropped
            total_quality_drop += quality_drop
            total_recovery_time += event.recovery_decisions
        
        num_events = len(events_data)
        
        self._data = ContextOverflowReportData(
            events=events_data,
            total_overflow_events=num_events,
            total_events_lost=total_events_lost,
            avg_quality_degradation=round(total_quality_drop / num_events, 1) if num_events > 0 else 0.0,
            avg_recovery_time=round(total_recovery_time / num_events, 0) if num_events > 0 else 0.0,
        )
        
        return self._data
    
    def to_markdown(self) -> str:
        """Render context overflow report as Markdown."""
        data = self.data
        
        lines = [
            "## Context Overflow Impact",
            "",
            f"**Generated:** {self._generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') if self._generated_at else 'N/A'}",
            "",
            f"**Total Overflow Events:** {data.total_overflow_events}",
            f"**Total Events Lost:** {data.total_events_lost:,}",
            f"**Avg Quality Degradation:** {data.avg_quality_degradation:.1f}%",
            f"**Avg Recovery Time:** {data.avg_recovery_time:.0f} decisions",
            "",
        ]
        
        for event in data.events:
            lines.extend([
                f"### Overflow Event #{event.event_number} (Day {event.day}, Hour {event.hour})",
                "",
                f"- **Context tokens:** {event.context_tokens:,} (limit: {event.context_limit:,})",
                f"- **Compression:** {event.compression_pct:.1f}% events dropped",
                f"- **Events lost:** {event.events_lost:,}",
                "",
                f"- **Decision quality before:** {event.quality_before:.1f}%",
                f"- **Decision quality after:** {event.quality_after:.1f}%",
                f"- **Recovery time:** {event.recovery_decisions:,} decisions",
                "",
            ])
        
        return "\n".join(lines)
    
    def to_json(self) -> dict:
        """Convert to JSON."""
        base = super().to_json()
        base["report_type"] = "context_overflow"
        return base


# =============================================================================
# Recovery Comparison Report
# =============================================================================

class RecoveryComparisonReport(BaseReport):
    """
    Report comparing private_db vs ledger recovery accuracy.
    
    Demonstrates the advantage of blockchain-backed state recovery
    over traditional private database approaches after agent restarts.
    """
    
    def generate(self, simulation_results: SimulationResults) -> RecoveryComparisonReportData:
        """
        Generate recovery comparison report.
        
        Args:
            simulation_results: Complete simulation results
            
        Returns:
            RecoveryComparisonReportData with comparison details
        """
        self._generated_at = datetime.utcnow()
        
        events_data: list[RestartEventData] = []
        total_private_db_accuracy = 0.0
        total_ledger_accuracy = 0.0
        private_db_count = 0
        ledger_count = 0
        
        for i, event in enumerate(simulation_results.restart_events, 1):
            recovery_modes: list[RecoveryModeData] = []
            
            for mode, results in event.recovery_results.items():
                accuracy = results.get("accuracy", 0.0)
                time_seconds = results.get("time_seconds", 0.0)
                decisions_affected = int(results.get("decisions_affected", 0))
                error_rate = 100.0 - accuracy
                
                recovery_modes.append(RecoveryModeData(
                    mode=mode,
                    accuracy=round(accuracy, 1),
                    time_seconds=round(time_seconds, 1),
                    decisions_affected=decisions_affected,
                    error_rate=round(error_rate, 1),
                ))
                
                # Track for averages
                if mode == "private_db":
                    total_private_db_accuracy += accuracy
                    private_db_count += 1
                elif mode == "ledger":
                    total_ledger_accuracy += accuracy
                    ledger_count += 1
            
            events_data.append(RestartEventData(
                event_number=i,
                day=event.day,
                hour=event.hour,
                recovery_modes=recovery_modes,
            ))
        
        avg_private_db = total_private_db_accuracy / private_db_count if private_db_count > 0 else 0.0
        avg_ledger = total_ledger_accuracy / ledger_count if ledger_count > 0 else 0.0
        
        self._data = RecoveryComparisonReportData(
            events=events_data,
            total_restart_events=len(events_data),
            avg_private_db_accuracy=round(avg_private_db, 1),
            avg_ledger_accuracy=round(avg_ledger, 1),
            reliability_advantage=round(avg_ledger - avg_private_db, 1),
        )
        
        return self._data
    
    def to_markdown(self) -> str:
        """Render recovery comparison report as Markdown."""
        data = self.data
        
        lines = [
            "## Recovery Comparison: Private DB vs Ledger",
            "",
            f"**Generated:** {self._generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') if self._generated_at else 'N/A'}",
            "",
            f"**Total Restart Events:** {data.total_restart_events}",
            f"**Avg Private DB Accuracy:** {data.avg_private_db_accuracy:.1f}%",
            f"**Avg Ledger Accuracy:** {data.avg_ledger_accuracy:.1f}%",
            f"**Ledger Reliability Advantage:** +{data.reliability_advantage:.1f} percentage points",
            "",
        ]
        
        for event in data.events:
            lines.extend([
                f"### Restart Event #{event.event_number} (Day {event.day}, Hour {event.hour})",
                "",
                "| Recovery Mode | Accuracy | Time | Decisions Affected |",
                "|---------------|----------|------|-------------------|",
            ])
            
            for mode in event.recovery_modes:
                mode_display = "Private DB" if mode.mode == "private_db" else "Ledger (Alkimi)"
                lines.append(
                    f"| {mode_display:<13} | {mode.accuracy:.1f}% | "
                    f"{mode.time_seconds:.1f}s | "
                    f"{mode.decisions_affected:,} ({mode.error_rate:.1f}% errors) |"
                )
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_json(self) -> dict:
        """Convert to JSON."""
        base = super().to_json()
        base["report_type"] = "recovery_comparison"
        return base


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_all_v2_reports(
    simulation_results: SimulationResults
) -> dict[str, BaseReport]:
    """
    Generate all V2 reports from simulation results.
    
    Args:
        simulation_results: Complete simulation results
        
    Returns:
        Dictionary mapping report name to generated report
    """
    reports = {
        "hallucination_rate": HallucinationRateReport(),
        "hallucination_type": HallucinationTypeReport(),
        "context_overflow": ContextOverflowReport(),
        "recovery_comparison": RecoveryComparisonReport(),
    }
    
    for report in reports.values():
        report.generate(simulation_results)
    
    return reports


def export_all_reports_markdown(reports: dict[str, BaseReport]) -> str:
    """
    Export all reports as a combined Markdown document.
    
    Args:
        reports: Dictionary of generated reports
        
    Returns:
        Combined Markdown string
    """
    sections = [
        "# V2 Simulation Reports: Context Window Hallucination Testing",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
    ]
    
    for name, report in reports.items():
        sections.append(report.to_markdown())
        sections.append("")
        sections.append("---")
        sections.append("")
    
    return "\n".join(sections)


def export_all_reports_json(reports: dict[str, BaseReport]) -> dict:
    """
    Export all reports as a combined JSON structure.
    
    Args:
        reports: Dictionary of generated reports
        
    Returns:
        Combined JSON-serializable dict
    """
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "reports": {
            name: report.to_json()
            for name, report in reports.items()
        },
    }
