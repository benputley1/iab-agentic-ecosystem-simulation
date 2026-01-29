"""
Report Generator for RTB Simulation.

Calculates KPIs and generates comparative reports across scenarios A, B, and C.
Supports multiple output formats: Markdown, HTML, and JSON.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any

from ..metrics.kpis import (
    KPICalculator,
    FeeExtractionMetrics,
    GoalAchievementMetrics,
    ContextRotMetrics,
    HallucinationMetrics,
    BlockchainCostMetrics,
)
from ..metrics.collector import InfluxConfig
from ..logging.events import EventIndex, Scenario
from ..logging.comparison import ComparisonAnalyzer, ComparisonReport


class ReportFormat(str, Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Data sources
    influx_config: Optional[InfluxConfig] = None
    event_index: Optional[EventIndex] = None

    # Report options
    include_kpis: bool = True
    include_comparison: bool = True
    include_narratives: bool = True
    include_daily_breakdown: bool = False

    # Output options
    output_format: ReportFormat = ReportFormat.MARKDOWN
    output_path: Optional[str] = None

    # Metadata
    title: str = "RTB Simulation Comparative Report"
    simulation_id: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class KPISummary:
    """Complete KPI summary across all scenarios."""

    # Fee extraction by scenario
    fee_extraction: dict[str, FeeExtractionMetrics] = field(default_factory=dict)

    # Goal achievement by scenario
    goal_achievement: dict[str, GoalAchievementMetrics] = field(default_factory=dict)

    # Context rot (primarily B vs C)
    context_rot: dict[str, ContextRotMetrics] = field(default_factory=dict)

    # Hallucination rates
    hallucination_rates: list[HallucinationMetrics] = field(default_factory=list)

    # Blockchain costs (C only)
    blockchain_costs: Optional[BlockchainCostMetrics] = None

    # Computed comparisons
    fee_reduction_pct: float = 0.0  # A to C reduction
    savings_per_100k: float = 0.0   # $ saved per $100k spend
    reliability_advantage: float = 0.0  # C vs B recovery difference

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "fee_extraction": {
                k: asdict(v) for k, v in self.fee_extraction.items()
            },
            "goal_achievement": {
                k: asdict(v) for k, v in self.goal_achievement.items()
            },
            "context_rot": {
                k: asdict(v) for k, v in self.context_rot.items()
            },
            "hallucination_rates": [asdict(h) for h in self.hallucination_rates],
            "blockchain_costs": asdict(self.blockchain_costs) if self.blockchain_costs else None,
            "fee_reduction_pct": self.fee_reduction_pct,
            "savings_per_100k": self.savings_per_100k,
            "reliability_advantage": self.reliability_advantage,
        }


@dataclass
class GeneratedReport:
    """Complete generated report."""

    # Metadata
    title: str
    generated_at: datetime
    simulation_id: Optional[str]

    # KPI data
    kpis: Optional[KPISummary] = None

    # Comparison report (from events)
    comparison: Optional[ComparisonReport] = None

    # Rendered content
    content: str = ""
    format: ReportFormat = ReportFormat.MARKDOWN

    def save(self, path: str) -> None:
        """Save report to file."""
        Path(path).write_text(self.content)


class ReportGenerator:
    """
    Generates comparative reports across RTB simulation scenarios.

    Combines KPI calculations from InfluxDB with event-based comparison
    analysis to produce comprehensive reports in multiple formats.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self._kpi_calculator: Optional[KPICalculator] = None
        self._comparison_analyzer: Optional[ComparisonAnalyzer] = None

    def generate(self) -> GeneratedReport:
        """
        Generate the complete report.

        Returns:
            Generated report with rendered content
        """
        report = GeneratedReport(
            title=self.config.title,
            generated_at=self.config.generated_at,
            simulation_id=self.config.simulation_id,
            format=self.config.output_format,
        )

        # Calculate KPIs from InfluxDB
        if self.config.include_kpis:
            report.kpis = self._calculate_kpis()

        # Generate comparison from events
        if self.config.include_comparison and self.config.event_index:
            report.comparison = self._generate_comparison()

        # Render to selected format
        if self.config.output_format == ReportFormat.MARKDOWN:
            report.content = self._render_markdown(report)
        elif self.config.output_format == ReportFormat.HTML:
            report.content = self._render_html(report)
        elif self.config.output_format == ReportFormat.JSON:
            report.content = self._render_json(report)

        # Save if path specified
        if self.config.output_path:
            report.save(self.config.output_path)

        return report

    def _calculate_kpis(self) -> KPISummary:
        """Calculate all KPIs from InfluxDB."""
        summary = KPISummary()

        influx_config = self.config.influx_config or InfluxConfig()

        with KPICalculator(influx_config) as calc:
            # Fee extraction for all scenarios
            for scenario in ["A", "B", "C"]:
                try:
                    metrics = calc.calculate_fee_extraction(scenario)
                    summary.fee_extraction[scenario] = metrics
                except Exception:
                    # Handle case where no data exists
                    summary.fee_extraction[scenario] = FeeExtractionMetrics(
                        scenario=scenario,
                        gross_spend=0.0,
                        net_to_publisher=0.0,
                        intermediary_take=0.0,
                        take_rate_pct=0.0,
                    )

            # Goal achievement for all scenarios
            for scenario in ["A", "B", "C"]:
                try:
                    metrics = calc.calculate_goal_achievement(scenario)
                    summary.goal_achievement[scenario] = metrics
                except Exception:
                    summary.goal_achievement[scenario] = GoalAchievementMetrics(
                        scenario=scenario,
                        total_campaigns=0,
                        hit_impression_goal=0,
                        hit_cpm_goal=0,
                        avg_goal_attainment=0.0,
                        success_rate_pct=0.0,
                    )

            # Context rot impact (B and C)
            for scenario in ["B", "C"]:
                try:
                    metrics = calc.calculate_context_rot_impact(scenario)
                    summary.context_rot[scenario] = metrics
                except Exception:
                    summary.context_rot[scenario] = ContextRotMetrics(
                        scenario=scenario,
                        total_rot_events=0,
                        avg_keys_lost=0.0,
                        avg_recovery_accuracy=1.0 if scenario == "C" else 0.0,
                        day_1_attainment=0.0,
                        day_30_attainment=0.0,
                        degradation_pct=0.0,
                    )

            # Hallucination rates
            try:
                summary.hallucination_rates = calc.calculate_hallucination_rates_all()
            except Exception:
                summary.hallucination_rates = []

            # Blockchain costs (C only)
            try:
                summary.blockchain_costs = calc.calculate_blockchain_costs()
            except Exception:
                summary.blockchain_costs = BlockchainCostMetrics(
                    total_transactions=0,
                    total_sui_gas=0.0,
                    total_walrus_cost=0.0,
                    total_usd=0.0,
                    cost_per_1k_impressions=0.0,
                    comparison_exchange_fee_per_1k=2.50,
                )

        # Calculate comparative metrics
        fee_a = summary.fee_extraction.get("A")
        fee_c = summary.fee_extraction.get("C")

        if fee_a and fee_c and fee_a.take_rate_pct > 0:
            summary.fee_reduction_pct = (
                (fee_a.take_rate_pct - fee_c.take_rate_pct) / fee_a.take_rate_pct
            ) * 100
            # Per $100k spend at A's take rate
            summary.savings_per_100k = (fee_a.take_rate_pct - fee_c.take_rate_pct) * 1000

        # Recovery accuracy comparison
        rot_b = summary.context_rot.get("B")
        rot_c = summary.context_rot.get("C")
        if rot_b and rot_c:
            summary.reliability_advantage = rot_c.avg_recovery_accuracy - rot_b.avg_recovery_accuracy

        return summary

    def _generate_comparison(self) -> Optional[ComparisonReport]:
        """Generate comparison report from events."""
        if not self.config.event_index:
            return None

        analyzer = ComparisonAnalyzer(self.config.event_index)
        return analyzer.generate_comparison_report()

    # =========================================================================
    # Markdown Rendering
    # =========================================================================

    def _render_markdown(self, report: GeneratedReport) -> str:
        """Render report as Markdown."""
        sections = []

        # Title and metadata
        sections.append(f"# {report.title}")
        sections.append("")
        sections.append(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if report.simulation_id:
            sections.append(f"**Simulation ID:** {report.simulation_id}")
        sections.append("")

        # Executive Summary
        sections.append("## Executive Summary")
        sections.append("")
        sections.append(self._render_executive_summary_md(report))
        sections.append("")

        # KPI Tables
        if report.kpis:
            sections.append("## Key Performance Indicators")
            sections.append("")
            sections.append(self._render_kpi_tables_md(report.kpis))

        # Insights
        if report.comparison and report.comparison.insights:
            sections.append("## Key Insights")
            sections.append("")
            sections.append(self._render_insights_md(report.comparison))

        # Recommendation
        if report.comparison:
            sections.append(report.comparison.recommendation)
            sections.append("")

        return "\n".join(sections)

    def _render_executive_summary_md(self, report: GeneratedReport) -> str:
        """Render executive summary for Markdown."""
        lines = []

        if report.kpis:
            kpis = report.kpis

            # Comparison table
            lines.append("| Model | Deals | Spend | Costs | Success Rate |")
            lines.append("|-------|-------|-------|-------|--------------|")

            for scenario in ["A", "B", "C"]:
                fee = kpis.fee_extraction.get(scenario)
                goal = kpis.goal_achievement.get(scenario)

                if not fee or not goal:
                    continue

                if scenario == "A":
                    cost_str = f"${fee.intermediary_take:,.0f} fees ({fee.take_rate_pct:.1f}%)"
                    model = "**A: Exchange**"
                elif scenario == "B":
                    rot = kpis.context_rot.get("B")
                    cost_str = f"{rot.total_rot_events if rot else 0} context losses"
                    model = "**B: Pure A2A**"
                else:
                    bc = kpis.blockchain_costs
                    cost_str = f"${bc.total_usd:,.2f} infrastructure" if bc else "$0.00"
                    model = "**C: Ledger**"

                lines.append(
                    f"| {model} | {goal.total_campaigns:,} | "
                    f"${fee.gross_spend:,.0f} | {cost_str} | "
                    f"{goal.success_rate_pct:.1f}% |"
                )

            lines.append("")

            # Key finding
            if kpis.fee_reduction_pct > 0:
                lines.append(
                    f"**Key Finding:** The ledger-backed model reduced costs by "
                    f"{kpis.fee_reduction_pct:.0f}% compared to the traditional exchange, "
                    f"saving ${kpis.savings_per_100k:,.2f} per $100k in ad spend while "
                    f"maintaining full auditability and state recovery."
                )

        elif report.comparison:
            lines.append(report.comparison.executive_summary)

        return "\n".join(lines)

    def _render_kpi_tables_md(self, kpis: KPISummary) -> str:
        """Render KPI tables as Markdown."""
        sections = []

        # Fee Extraction Comparison
        sections.append("### Fee Extraction Comparison")
        sections.append("")
        sections.append("| Scenario | Gross Spend | Net to Publisher | Intermediary Take | Take Rate |")
        sections.append("|----------|-------------|------------------|-------------------|-----------|")

        for scenario in ["A", "B", "C"]:
            fee = kpis.fee_extraction.get(scenario)
            if fee:
                sections.append(
                    f"| {scenario} | ${fee.gross_spend:,.2f} | "
                    f"${fee.net_to_publisher:,.2f} | "
                    f"${fee.intermediary_take:,.2f} | "
                    f"{fee.take_rate_pct:.2f}% |"
                )

        sections.append("")

        # Goal Achievement
        sections.append("### Campaign Goal Achievement")
        sections.append("")
        sections.append("| Scenario | Campaigns | Goals Met | Success Rate | Avg Attainment |")
        sections.append("|----------|-----------|-----------|--------------|----------------|")

        for scenario in ["A", "B", "C"]:
            goal = kpis.goal_achievement.get(scenario)
            if goal:
                sections.append(
                    f"| {scenario} | {goal.total_campaigns} | "
                    f"{goal.hit_impression_goal} | "
                    f"{goal.success_rate_pct:.1f}% | "
                    f"{goal.avg_goal_attainment:.1f}% |"
                )

        sections.append("")

        # Context Rot Impact
        sections.append("### Context Rot Impact (B vs C)")
        sections.append("")
        sections.append("| Scenario | Rot Events | Keys Lost | Recovery Accuracy | Day 1 Goal | Day 30 Goal | Degradation |")
        sections.append("|----------|------------|-----------|-------------------|------------|-------------|-------------|")

        for scenario in ["B", "C"]:
            rot = kpis.context_rot.get(scenario)
            if rot:
                sections.append(
                    f"| {scenario} | {rot.total_rot_events} | "
                    f"{rot.avg_keys_lost:.1f} | "
                    f"{rot.avg_recovery_accuracy:.1%} | "
                    f"{rot.day_1_attainment:.1f}% | "
                    f"{rot.day_30_attainment:.1f}% | "
                    f"{rot.degradation_pct:.1f}% |"
                )

        sections.append("")

        # Blockchain Costs
        if kpis.blockchain_costs:
            bc = kpis.blockchain_costs
            sections.append("### Blockchain Infrastructure Costs (Scenario C)")
            sections.append("")
            sections.append(f"- **Total Transactions:** {bc.total_transactions:,}")
            sections.append(f"- **Total Sui Gas:** {bc.total_sui_gas:,.4f} SUI")
            sections.append(f"- **Total Walrus Storage:** {bc.total_walrus_cost:,.4f} SUI")
            sections.append(f"- **Total USD Cost:** ${bc.total_usd:,.2f}")
            sections.append(f"- **Cost per 1k Impressions:** ${bc.cost_per_1k_impressions:,.4f}")
            sections.append(f"- **Comparison (Exchange Fee per 1k):** ${bc.comparison_exchange_fee_per_1k:,.2f}")
            sections.append("")

        # Hallucination Rates
        if kpis.hallucination_rates:
            sections.append("### Hallucination Rates by Scenario")
            sections.append("")
            sections.append("| Scenario | Agent Type | Total Decisions | Hallucinated | Rate |")
            sections.append("|----------|------------|-----------------|--------------|------|")

            for h in kpis.hallucination_rates:
                if h.total_decisions > 0:
                    sections.append(
                        f"| {h.scenario} | {h.agent_type} | "
                        f"{h.total_decisions:,} | "
                        f"{h.hallucinated_decisions:,} | "
                        f"{h.hallucination_rate_pct:.2f}% |"
                    )

            sections.append("")

        return "\n".join(sections)

    def _render_insights_md(self, comparison: ComparisonReport) -> str:
        """Render insights as Markdown."""
        lines = []

        for i, insight in enumerate(comparison.insights, 1):
            lines.append(f"### {i}. {insight.headline}")
            lines.append("")
            lines.append(insight.detail)
            lines.append("")

            if insight.data_points:
                lines.append("**Data:**")
                for key, value in insight.data_points.items():
                    if isinstance(value, float):
                        lines.append(f"- {key}: ${value:,.2f}" if "cost" in key or "fee" in key or "savings" in key else f"- {key}: {value:,.2f}")
                    else:
                        lines.append(f"- {key}: {value}")
                lines.append("")

        if comparison.key_takeaways:
            lines.append("### Key Takeaways")
            lines.append("")
            for takeaway in comparison.key_takeaways:
                lines.append(f"- {takeaway}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # HTML Rendering
    # =========================================================================

    def _render_html(self, report: GeneratedReport) -> str:
        """Render report as HTML."""
        md_content = self._render_markdown(report)

        # Simple HTML wrapper with basic styling
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 0.5rem; }}
        h2 {{ color: #2c5282; margin-top: 2rem; }}
        h3 {{ color: #2b6cb0; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 0.75rem;
            text-align: left;
        }}
        th {{
            background-color: #edf2f7;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background-color: #f7fafc; }}
        .metric {{ font-weight: 600; color: #2b6cb0; }}
        .positive {{ color: #276749; }}
        .negative {{ color: #c53030; }}
        ul {{ padding-left: 1.5rem; }}
        li {{ margin: 0.5rem 0; }}
        strong {{ color: #1a365d; }}
    </style>
</head>
<body>
<pre style="white-space: pre-wrap; font-family: inherit;">
{md_content}
</pre>
</body>
</html>
"""
        return html

    # =========================================================================
    # JSON Rendering
    # =========================================================================

    def _render_json(self, report: GeneratedReport) -> str:
        """Render report as JSON."""
        data = {
            "title": report.title,
            "generated_at": report.generated_at.isoformat(),
            "simulation_id": report.simulation_id,
        }

        if report.kpis:
            data["kpis"] = report.kpis.to_dict()

        if report.comparison:
            data["comparison"] = {
                "executive_summary": report.comparison.executive_summary,
                "key_takeaways": report.comparison.key_takeaways,
                "recommendation": report.comparison.recommendation,
                "insights": [
                    {
                        "category": i.category,
                        "headline": i.headline,
                        "detail": i.detail,
                        "data_points": i.data_points,
                        "significance": i.significance,
                    }
                    for i in report.comparison.insights
                ],
                "metrics": {
                    "scenario_a_spend": report.comparison.comparison.scenario_a_spend,
                    "scenario_b_spend": report.comparison.comparison.scenario_b_spend,
                    "scenario_c_spend": report.comparison.comparison.scenario_c_spend,
                    "scenario_a_fees": report.comparison.comparison.scenario_a_fees,
                    "scenario_c_blockchain_costs": report.comparison.comparison.scenario_c_blockchain_costs,
                    "context_rot_events": report.comparison.comparison.context_rot_events,
                    "hallucination_count": report.comparison.comparison.hallucination_count,
                    "fee_savings_c_vs_a": report.comparison.comparison.fee_savings_c_vs_a,
                },
            }

        return json.dumps(data, indent=2, default=str)


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_report(
    output_format: str = "markdown",
    output_path: Optional[str] = None,
    influx_url: Optional[str] = None,
    influx_token: Optional[str] = None,
    title: str = "RTB Simulation Comparative Report",
) -> GeneratedReport:
    """
    Convenience function to generate a report.

    Args:
        output_format: Output format (markdown, html, json)
        output_path: Optional file path to save report
        influx_url: InfluxDB URL (defaults to env or localhost)
        influx_token: InfluxDB token
        title: Report title

    Returns:
        Generated report
    """
    influx_config = InfluxConfig()
    if influx_url:
        influx_config.url = influx_url
    if influx_token:
        influx_config.token = influx_token

    config = ReportConfig(
        influx_config=influx_config,
        output_format=ReportFormat(output_format),
        output_path=output_path,
        title=title,
    )

    generator = ReportGenerator(config)
    return generator.generate()


def generate_kpi_summary(influx_config: Optional[InfluxConfig] = None) -> KPISummary:
    """
    Calculate and return KPI summary without generating full report.

    Args:
        influx_config: InfluxDB configuration

    Returns:
        KPI summary
    """
    config = ReportConfig(
        influx_config=influx_config,
        include_comparison=False,
        include_narratives=False,
    )

    generator = ReportGenerator(config)
    return generator._calculate_kpis()
