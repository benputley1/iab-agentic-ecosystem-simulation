"""
File writers for event and narrative logs.

Handles output to structured event logs (JSONL) and
human-readable narrative logs (Markdown).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
from dataclasses import dataclass

from .events import SimulationEvent, EventIndex, Scenario
from .narratives import (
    NarrativeEngine,
    CampaignNarrative,
    DayNarrative,
    ScenarioNarrative,
)
from .comparison import ComparisonAnalyzer, ComparisonReport


@dataclass
class WriterConfig:
    """Configuration for log writers."""

    # Output directories
    base_dir: Path = Path("logs")
    events_subdir: str = "events"
    narratives_subdir: str = "narratives"

    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"

    # Event log options
    events_per_file: int = 10000  # Rotate after this many events
    compress_old_logs: bool = False

    # Narrative options
    include_day_narratives: bool = True
    include_campaign_narratives: bool = True

    def __post_init__(self):
        """Ensure directories exist."""
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.narratives_dir.mkdir(parents=True, exist_ok=True)

    @property
    def events_dir(self) -> Path:
        """Directory for event logs."""
        return self.base_dir / self.events_subdir

    @property
    def narratives_dir(self) -> Path:
        """Directory for narrative logs."""
        return self.base_dir / self.narratives_subdir


class EventWriter:
    """
    Writes simulation events to JSONL files.

    Produces machine-readable event logs suitable for analysis,
    replay, and integration with external systems.
    """

    def __init__(
        self,
        config: Optional[WriterConfig] = None,
        scenario: Optional[Scenario] = None,
    ):
        """
        Initialize event writer.

        Args:
            config: Writer configuration
            scenario: Optional scenario filter
        """
        self.config = config or WriterConfig()
        self.scenario = scenario

        self._file: Optional[TextIO] = None
        self._file_path: Optional[Path] = None
        self._event_count: int = 0
        self._session_id = datetime.utcnow().strftime(self.config.timestamp_format)

    def open(self, filename: Optional[str] = None) -> None:
        """
        Open event log file for writing.

        Args:
            filename: Optional custom filename
        """
        if filename:
            self._file_path = self.config.events_dir / filename
        else:
            scenario_suffix = f"_{self.scenario.value.lower()}" if self.scenario else ""
            self._file_path = self.config.events_dir / f"events_{self._session_id}{scenario_suffix}.jsonl"

        self._file = open(self._file_path, "a", encoding="utf-8")
        self._event_count = 0

    def close(self) -> None:
        """Close event log file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_event(self, event: SimulationEvent) -> None:
        """
        Write a single event to the log.

        Args:
            event: Event to write
        """
        if not self._file:
            raise RuntimeError("Writer not opened. Call open() first.")

        # Apply scenario filter if set
        if self.scenario and event.scenario != self.scenario:
            return

        line = event.to_json()
        self._file.write(line + "\n")
        self._file.flush()  # Ensure immediate write
        self._event_count += 1

        # Rotate if needed
        if self._event_count >= self.config.events_per_file:
            self._rotate()

    def write_events(self, events: list[SimulationEvent]) -> None:
        """
        Write multiple events to the log.

        Args:
            events: Events to write
        """
        for event in events:
            self.write_event(event)

    def write_index(self, index: EventIndex) -> None:
        """
        Write all events from an index.

        Args:
            index: Event index to write
        """
        self.write_events(index.events)

    def _rotate(self) -> None:
        """Rotate to a new log file."""
        self.close()

        # Generate new filename with sequence number
        timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
        scenario_suffix = f"_{self.scenario.value.lower()}" if self.scenario else ""
        seq = 1
        while True:
            new_path = self.config.events_dir / f"events_{timestamp}{scenario_suffix}_{seq:03d}.jsonl"
            if not new_path.exists():
                break
            seq += 1

        self._file_path = new_path
        self._file = open(self._file_path, "w", encoding="utf-8")
        self._event_count = 0

    @property
    def current_file(self) -> Optional[Path]:
        """Current log file path."""
        return self._file_path


class NarrativeWriter:
    """
    Writes narrative logs in Markdown format.

    Produces human-readable, publication-ready narratives
    suitable for articles, reports, and content generation.
    """

    def __init__(
        self,
        config: Optional[WriterConfig] = None,
    ):
        """
        Initialize narrative writer.

        Args:
            config: Writer configuration
        """
        self.config = config or WriterConfig()
        self._session_id = datetime.utcnow().strftime(self.config.timestamp_format)

    def write_scenario_narrative(
        self,
        narrative: ScenarioNarrative,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write scenario narrative to Markdown file.

        Args:
            narrative: Scenario narrative to write
            filename: Optional custom filename

        Returns:
            Path to written file
        """
        if filename:
            file_path = self.config.narratives_dir / filename
        else:
            file_path = self.config.narratives_dir / f"scenario_{narrative.scenario.value.lower()}_{self._session_id}.md"

        content = self._format_scenario_narrative(narrative)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path

    def _format_scenario_narrative(self, narrative: ScenarioNarrative) -> str:
        """Format scenario narrative as Markdown."""
        lines = [
            f"# {narrative.name}",
            "",
            f"*Simulation run: {self._session_id}*",
            "",
            "---",
            "",
            narrative.executive_summary,
            "",
            "## Key Findings",
            "",
        ]

        for finding in narrative.key_findings:
            lines.append(f"- {finding}")

        lines.extend([
            "",
            "## Conclusion",
            "",
            narrative.conclusion,
            "",
        ])

        # Day summaries
        if self.config.include_day_narratives and narrative.day_narratives:
            lines.extend([
                "---",
                "",
                "## Daily Summary",
                "",
            ])

            for day_narrative in narrative.day_narratives:
                lines.append(f"### Day {day_narrative.simulation_day}")
                lines.append("")
                lines.append(day_narrative.detail)
                lines.append("")

        return "\n".join(lines)

    def write_campaign_narrative(
        self,
        narrative: CampaignNarrative,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write campaign narrative to Markdown file.

        Args:
            narrative: Campaign narrative to write
            filename: Optional custom filename

        Returns:
            Path to written file
        """
        if filename:
            file_path = self.config.narratives_dir / filename
        else:
            file_path = self.config.narratives_dir / f"campaign_{narrative.campaign_id}_{self._session_id}.md"

        content = self._format_campaign_narrative(narrative)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path

    def _format_campaign_narrative(self, narrative: CampaignNarrative) -> str:
        """Format campaign narrative as Markdown."""
        lines = [
            f"# Campaign: {narrative.campaign_id}",
            "",
            f"*Scenario: {narrative.scenario.value} | Buyer: {narrative.buyer_id}*",
            "",
            "---",
            "",
            "## Summary",
            "",
            narrative.opening_summary,
            "",
            "## Journey",
            "",
            narrative.journey_narrative,
            "",
            "## Outcome",
            "",
            narrative.outcome_summary,
            "",
            "## Key Insight",
            "",
            f"> {narrative.key_insight}",
            "",
            "---",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Target Impressions | {narrative.target_impressions:,} |",
            f"| Delivered Impressions | {narrative.delivered_impressions:,} |",
            f"| Goal Attainment | {narrative.goal_attainment:.1f}% |",
            f"| Total Spend | ${narrative.total_spend:,.2f} |",
            f"| Effective CPM | ${narrative.effective_cpm:.2f} |",
            f"| Deals Made | {narrative.deals_made} |",
        ]

        if narrative.scenario == Scenario.A:
            lines.extend([
                f"| Exchange Fees | ${narrative.total_fees:,.2f} |",
                f"| Fee Percentage | {narrative.fee_percentage:.1f}% |",
            ])
        elif narrative.scenario == Scenario.B:
            lines.extend([
                f"| Context Losses | {narrative.context_losses} |",
                f"| Hallucinations | {narrative.hallucinations} |",
            ])

        lines.append("")

        return "\n".join(lines)

    def write_comparison_report(
        self,
        report: ComparisonReport,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write comparison report to Markdown file.

        Args:
            report: Comparison report to write
            filename: Optional custom filename

        Returns:
            Path to written file
        """
        if filename:
            file_path = self.config.narratives_dir / filename
        else:
            file_path = self.config.narratives_dir / f"comparison_report_{self._session_id}.md"

        content = self._format_comparison_report(report)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path

    def _format_comparison_report(self, report: ComparisonReport) -> str:
        """Format comparison report as Markdown."""
        lines = [
            "# RTB Simulation: Scenario Comparison Report",
            "",
            f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
            "---",
            "",
            report.executive_summary,
            "",
            "---",
            "",
            "## Key Takeaways",
            "",
        ]

        for takeaway in report.key_takeaways:
            lines.append(f"- {takeaway}")

        lines.extend([
            "",
            "---",
            "",
            "## Detailed Insights",
            "",
        ])

        for insight in report.insights:
            lines.extend([
                f"### {insight.headline}",
                "",
                f"**Category:** {insight.category.title()}",
                "",
                insight.detail,
                "",
            ])

        lines.extend([
            "---",
            "",
            report.recommendation,
            "",
        ])

        return "\n".join(lines)

    def write_day_narrative(
        self,
        narrative: DayNarrative,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write day narrative to Markdown file.

        Args:
            narrative: Day narrative to write
            filename: Optional custom filename

        Returns:
            Path to written file
        """
        if filename:
            file_path = self.config.narratives_dir / filename
        else:
            file_path = self.config.narratives_dir / f"day_{narrative.simulation_day}_{narrative.scenario.value.lower()}_{self._session_id}.md"

        content = self._format_day_narrative(narrative)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path

    def _format_day_narrative(self, narrative: DayNarrative) -> str:
        """Format day narrative as Markdown."""
        lines = [
            f"# Day {narrative.simulation_day} - Scenario {narrative.scenario.value}",
            "",
            f"*{narrative.summary}*",
            "",
            "---",
            "",
            narrative.detail,
            "",
        ]

        if narrative.notable_events:
            lines.extend([
                "## Notable Events",
                "",
            ])
            for event in narrative.notable_events:
                lines.append(f"- {event}")
            lines.append("")

        return "\n".join(lines)


class OrchestrationLogWriter:
    """
    High-level writer that coordinates event and narrative output.

    Provides a unified interface for the orchestration logger to
    write both structured events and human-readable narratives.
    """

    def __init__(
        self,
        config: Optional[WriterConfig] = None,
    ):
        """
        Initialize orchestration log writer.

        Args:
            config: Writer configuration
        """
        self.config = config or WriterConfig()
        self.event_writer = EventWriter(config=self.config)
        self.narrative_writer = NarrativeWriter(config=self.config)

        self._index = EventIndex()
        self._started = False

    def start(self) -> None:
        """Start the log writers."""
        if self._started:
            return

        self.event_writer.open()
        self._started = True

    def stop(self) -> None:
        """Stop the log writers and finalize output."""
        if not self._started:
            return

        self.event_writer.close()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def log_event(self, event: SimulationEvent) -> None:
        """
        Log a simulation event.

        Args:
            event: Event to log
        """
        # Add to index
        self._index.add(event)

        # Write to file
        if self._started:
            self.event_writer.write_event(event)

    def log_events(self, events: list[SimulationEvent]) -> None:
        """
        Log multiple simulation events.

        Args:
            events: Events to log
        """
        for event in events:
            self.log_event(event)

    def generate_narratives(self) -> dict[str, Path]:
        """
        Generate all narrative outputs from logged events.

        Returns:
            Dictionary mapping narrative type to file path
        """
        outputs = {}

        # Generate scenario narratives
        engine = NarrativeEngine(self._index)

        for scenario in [Scenario.A, Scenario.B, Scenario.C]:
            if self._index.get_scenario_events(scenario):
                narrative = engine.generate_scenario_narrative(scenario)
                path = self.narrative_writer.write_scenario_narrative(narrative)
                outputs[f"scenario_{scenario.value}"] = path

        # Generate comparison report
        if len([s for s in [Scenario.A, Scenario.B, Scenario.C] if self._index.get_scenario_events(s)]) > 1:
            analyzer = ComparisonAnalyzer(self._index)
            report = analyzer.generate_comparison_report()
            path = self.narrative_writer.write_comparison_report(report)
            outputs["comparison"] = path

        return outputs

    def get_index(self) -> EventIndex:
        """Get the event index."""
        return self._index

    def get_event_count(self) -> int:
        """Get total number of logged events."""
        return len(self._index.events)
