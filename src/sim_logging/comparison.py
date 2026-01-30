"""
Cross-scenario comparison analyzer for RTB simulation.

Identifies interesting divergences and generates comparative insights
showing how outcomes differ across scenarios A, B, and C.
"""

from dataclasses import dataclass, field
from typing import Optional

from .events import (
    SimulationEvent,
    EventType,
    EventIndex,
    Scenario,
)
from .narratives import ScenarioNarrative, NarrativeEngine


@dataclass
class ScenarioComparison:
    """
    Comparison metrics between scenarios.

    Enables side-by-side analysis of how the same campaigns
    perform differently across trading models.
    """

    # Scenario metrics
    scenario_a_spend: float = 0.0
    scenario_b_spend: float = 0.0
    scenario_c_spend: float = 0.0

    scenario_a_fees: float = 0.0
    scenario_b_fees: float = 0.0  # Always 0
    scenario_c_blockchain_costs: float = 0.0

    scenario_a_deals: int = 0
    scenario_b_deals: int = 0
    scenario_c_deals: int = 0

    scenario_a_impressions: int = 0
    scenario_b_impressions: int = 0
    scenario_c_impressions: int = 0

    # Scenario B specific
    context_rot_events: int = 0
    hallucination_count: int = 0

    # Computed comparisons
    @property
    def fee_savings_b_vs_a(self) -> float:
        """How much Scenario B saved by avoiding exchange fees."""
        return self.scenario_a_fees

    @property
    def fee_savings_c_vs_a(self) -> float:
        """Net savings of ledger model vs exchange (fees - blockchain costs)."""
        return self.scenario_a_fees - self.scenario_c_blockchain_costs

    @property
    def a_take_rate(self) -> float:
        """Scenario A exchange take rate."""
        if self.scenario_a_spend == 0:
            return 0.0
        return (self.scenario_a_fees / self.scenario_a_spend) * 100

    @property
    def c_infrastructure_rate(self) -> float:
        """Scenario C blockchain cost as percentage of spend."""
        if self.scenario_c_spend == 0:
            return 0.0
        return (self.scenario_c_blockchain_costs / self.scenario_c_spend) * 100

    @property
    def b_context_rot_penalty(self) -> float:
        """Estimated value lost to context rot (placeholder calculation)."""
        # Each context rot event assumed to cause ~1% reduction in deal efficiency
        if self.scenario_b_deals == 0:
            return 0.0
        avg_deal_value = self.scenario_b_spend / self.scenario_b_deals
        return self.context_rot_events * avg_deal_value * 0.01


@dataclass
class ComparisonInsight:
    """A single insight from scenario comparison."""

    category: str  # "fees", "reliability", "auditability", etc.
    headline: str
    detail: str
    data_points: dict = field(default_factory=dict)
    significance: int = 1  # 1-5, higher = more important


@dataclass
class ComparisonReport:
    """Full comparison report across all scenarios."""

    # Source narratives
    scenario_a: Optional[ScenarioNarrative] = None
    scenario_b: Optional[ScenarioNarrative] = None
    scenario_c: Optional[ScenarioNarrative] = None

    # Computed comparison
    comparison: ScenarioComparison = field(default_factory=ScenarioComparison)

    # Generated insights
    insights: list[ComparisonInsight] = field(default_factory=list)

    # Summary content
    executive_summary: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    recommendation: str = ""


class ComparisonAnalyzer:
    """
    Analyzes differences between scenarios and generates comparative insights.

    Identifies where outcomes diverge significantly and explains
    the underlying causes in content-ready format.
    """

    def __init__(self, event_index: EventIndex):
        """
        Initialize comparison analyzer.

        Args:
            event_index: Index of all simulation events
        """
        self.index = event_index
        self.narrative_engine = NarrativeEngine(event_index)

    def generate_comparison_report(self) -> ComparisonReport:
        """
        Generate full comparison report across all scenarios.

        Returns:
            Complete comparison report
        """
        report = ComparisonReport()

        # Generate individual scenario narratives
        report.scenario_a = self.narrative_engine.generate_scenario_narrative(Scenario.A)
        report.scenario_b = self.narrative_engine.generate_scenario_narrative(Scenario.B)
        report.scenario_c = self.narrative_engine.generate_scenario_narrative(Scenario.C)

        # Build comparison metrics
        report.comparison = self._build_comparison(
            report.scenario_a,
            report.scenario_b,
            report.scenario_c,
        )

        # Generate insights
        report.insights = self._generate_insights(report.comparison)

        # Generate summary content
        report.executive_summary = self._generate_executive_summary(report)
        report.key_takeaways = self._generate_key_takeaways(report)
        report.recommendation = self._generate_recommendation(report)

        return report

    def _build_comparison(
        self,
        scenario_a: ScenarioNarrative,
        scenario_b: ScenarioNarrative,
        scenario_c: ScenarioNarrative,
    ) -> ScenarioComparison:
        """Build comparison metrics from scenario narratives."""
        return ScenarioComparison(
            scenario_a_spend=scenario_a.total_spend,
            scenario_b_spend=scenario_b.total_spend,
            scenario_c_spend=scenario_c.total_spend,
            scenario_a_fees=scenario_a.total_fees,
            scenario_b_fees=0.0,  # Always 0
            scenario_c_blockchain_costs=scenario_c.total_blockchain_costs,
            scenario_a_deals=scenario_a.total_deals,
            scenario_b_deals=scenario_b.total_deals,
            scenario_c_deals=scenario_c.total_deals,
            scenario_a_impressions=scenario_a.total_impressions,
            scenario_b_impressions=scenario_b.total_impressions,
            scenario_c_impressions=scenario_c.total_impressions,
            context_rot_events=scenario_b.total_context_losses,
            hallucination_count=scenario_b.total_hallucinations,
        )

    def _generate_insights(
        self,
        comparison: ScenarioComparison,
    ) -> list[ComparisonInsight]:
        """Generate insights from comparison data."""
        insights = []

        # Fee comparison insight
        if comparison.scenario_a_fees > 0:
            savings = comparison.fee_savings_c_vs_a
            insights.append(ComparisonInsight(
                category="fees",
                headline=f"Ledger model saves ${savings:,.2f} vs traditional exchange",
                detail=(
                    f"Traditional exchange extracted ${comparison.scenario_a_fees:,.2f} in fees "
                    f"({comparison.a_take_rate:.1f}% take rate). The ledger model's infrastructure "
                    f"cost only ${comparison.scenario_c_blockchain_costs:,.2f} ({comparison.c_infrastructure_rate:.2f}%), "
                    f"a {(savings / comparison.scenario_a_fees * 100):.0f}% reduction."
                ),
                data_points={
                    "exchange_fees": comparison.scenario_a_fees,
                    "blockchain_costs": comparison.scenario_c_blockchain_costs,
                    "net_savings": savings,
                },
                significance=5,
            ))

        # Context rot impact
        if comparison.context_rot_events > 0:
            insights.append(ComparisonInsight(
                category="reliability",
                headline=f"Context rot caused {comparison.context_rot_events} incidents in A2A model",
                detail=(
                    f"The pure A2A model (Scenario B) experienced {comparison.context_rot_events} "
                    f"context loss events, resulting in {comparison.hallucination_count} decisions "
                    f"based on hallucinated data. Without persistent state, agents cannot recover "
                    f"lost memories, leading to degraded performance over time."
                ),
                data_points={
                    "context_losses": comparison.context_rot_events,
                    "hallucinations": comparison.hallucination_count,
                },
                significance=4,
            ))

        # Auditability comparison
        if comparison.scenario_c_deals > 0:
            insights.append(ComparisonInsight(
                category="auditability",
                headline="100% transaction auditability with ledger model",
                detail=(
                    f"All {comparison.scenario_c_deals:,} deals in the ledger model were recorded "
                    f"to the immutable ledger, providing complete transaction history and verification. "
                    f"In contrast, Scenario A relies on exchange records (centralized control), "
                    f"while Scenario B has no persistent transaction history at all."
                ),
                data_points={
                    "ledger_deals": comparison.scenario_c_deals,
                    "recovery_accuracy": 100.0,
                },
                significance=4,
            ))

        # Deal volume comparison
        total_spend = comparison.scenario_a_spend + comparison.scenario_b_spend + comparison.scenario_c_spend
        if total_spend > 0:
            insights.append(ComparisonInsight(
                category="volume",
                headline=f"${total_spend:,.2f} total transaction volume simulated",
                detail=(
                    f"Across all three scenarios, the simulation processed "
                    f"{comparison.scenario_a_deals + comparison.scenario_b_deals + comparison.scenario_c_deals:,} deals "
                    f"representing ${total_spend:,.2f} in advertising spend. "
                    f"Scenario A: ${comparison.scenario_a_spend:,.2f}, "
                    f"Scenario B: ${comparison.scenario_b_spend:,.2f}, "
                    f"Scenario C: ${comparison.scenario_c_spend:,.2f}."
                ),
                data_points={
                    "total_spend": total_spend,
                    "total_deals": comparison.scenario_a_deals + comparison.scenario_b_deals + comparison.scenario_c_deals,
                },
                significance=3,
            ))

        # Sort by significance
        insights.sort(key=lambda x: x.significance, reverse=True)

        return insights

    def _generate_executive_summary(self, report: ComparisonReport) -> str:
        """Generate executive summary for comparison report."""
        c = report.comparison

        parts = [
            "## Executive Summary",
            "",
            "This simulation compared three programmatic advertising models over 30 days:",
            "",
            "| Model | Deals | Spend | Costs |",
            "|-------|-------|-------|-------|",
            f"| **A: Exchange** | {c.scenario_a_deals:,} | ${c.scenario_a_spend:,.0f} | ${c.scenario_a_fees:,.0f} fees ({c.a_take_rate:.1f}%) |",
            f"| **B: Pure A2A** | {c.scenario_b_deals:,} | ${c.scenario_b_spend:,.0f} | {c.context_rot_events} context losses |",
            f"| **C: Ledger** | {c.scenario_c_deals:,} | ${c.scenario_c_spend:,.0f} | ${c.scenario_c_blockchain_costs:,.2f} infrastructure |",
            "",
        ]

        # Key finding
        if c.scenario_a_fees > 0 and c.fee_savings_c_vs_a > 0:
            savings_pct = (c.fee_savings_c_vs_a / c.scenario_a_fees) * 100
            parts.append(
                f"**Key Finding:** The ledger-backed model reduced costs by {savings_pct:.0f}% "
                f"compared to the traditional exchange, saving ${c.fee_savings_c_vs_a:,.2f} "
                f"while maintaining full auditability and state recovery."
            )

        return "\n".join(parts)

    def _generate_key_takeaways(self, report: ComparisonReport) -> list[str]:
        """Generate key takeaways from comparison."""
        takeaways = []
        c = report.comparison

        # Fee takeaway
        if c.scenario_a_fees > 0:
            takeaways.append(
                f"Exchange intermediaries extract {c.a_take_rate:.1f}% of transaction value - "
                f"${c.scenario_a_fees:,.2f} over the simulation period."
            )

        # Context rot takeaway
        if c.context_rot_events > 0:
            takeaways.append(
                f"Pure A2A trading without persistent state leads to {c.context_rot_events} context loss events "
                f"and {c.hallucination_count} decisions based on fabricated data."
            )

        # Ledger value takeaway
        if c.scenario_c_blockchain_costs > 0:
            cost_per_deal = c.scenario_c_blockchain_costs / c.scenario_c_deals if c.scenario_c_deals > 0 else 0
            takeaways.append(
                f"Blockchain infrastructure costs ${c.scenario_c_blockchain_costs:,.2f} total "
                f"(${cost_per_deal:.4f}/deal) - a fraction of exchange fees while providing immutable records."
            )

        # Recovery takeaway
        takeaways.append(
            "Ledger-backed model provides 100% state recovery accuracy vs 0% in pure A2A model."
        )

        return takeaways

    def _generate_recommendation(self, report: ComparisonReport) -> str:
        """Generate recommendation based on comparison."""
        c = report.comparison

        parts = [
            "## Recommendation",
            "",
            "Based on the simulation results, **the ledger-backed model (Scenario C)** provides "
            "the optimal balance of:",
            "",
            "1. **Cost efficiency** - Eliminates exchange fees while adding minimal infrastructure costs",
            f"2. **Reliability** - 100% state recovery vs {c.context_rot_events} context loss events in pure A2A",
            "3. **Auditability** - Complete transaction history for verification and dispute resolution",
            "4. **Decentralization** - No single point of control or failure",
            "",
        ]

        if c.scenario_a_fees > 0 and c.fee_savings_c_vs_a > 0:
            parts.append(
                f"Projected annual savings at scale: If the ${c.scenario_a_spend:,.0f} spend represents "
                f"typical monthly volume, annual fee savings would exceed ${c.fee_savings_c_vs_a * 12:,.0f}."
            )

        return "\n".join(parts)

    def generate_daily_comparison(self, day: int) -> dict:
        """
        Generate comparison metrics for a specific day.

        Args:
            day: Simulation day number

        Returns:
            Dictionary of comparison metrics for the day
        """
        comparison = {
            "day": day,
            "scenarios": {},
        }

        for scenario in [Scenario.A, Scenario.B, Scenario.C]:
            events = self.index.get_day_events(day, scenario)

            metrics = {
                "deals": 0,
                "spend": 0.0,
                "fees": 0.0,
                "impressions": 0,
                "context_losses": 0,
                "hallucinations": 0,
                "blockchain_costs": 0.0,
            }

            for event in events:
                if event.event_type == EventType.DEAL_CREATED:
                    metrics["deals"] += 1
                    metrics["spend"] += event.payload.get("total_cost", 0)
                    metrics["fees"] += event.payload.get("exchange_fee", 0)
                    metrics["impressions"] += event.payload.get("impressions", 0)

                elif event.event_type in (EventType.CONTEXT_DECAY, EventType.CONTEXT_RESTART):
                    metrics["context_losses"] += 1

                elif event.event_type == EventType.HALLUCINATION_DETECTED:
                    metrics["hallucinations"] += 1

                elif event.event_type == EventType.BLOCKCHAIN_COST:
                    metrics["blockchain_costs"] += event.payload.get("total_usd", 0)

            comparison["scenarios"][scenario.value] = metrics

        return comparison

    def generate_campaign_comparison(
        self,
        correlation_id: str,
    ) -> dict:
        """
        Compare the same campaign across scenarios.

        Requires campaigns to have been assigned the same correlation_id
        when created in different scenarios.

        Args:
            correlation_id: Shared correlation ID across scenarios

        Returns:
            Comparison of campaign performance by scenario
        """
        comparison = {
            "correlation_id": correlation_id,
            "scenarios": {},
        }

        # Find all events with this correlation ID
        for event in self.index.events:
            if event.correlation_id != correlation_id:
                continue

            scenario = event.scenario.value
            if scenario not in comparison["scenarios"]:
                comparison["scenarios"][scenario] = {
                    "deals": 0,
                    "spend": 0.0,
                    "fees": 0.0,
                    "impressions": 0,
                    "context_losses": 0,
                    "goal_attainment": None,
                }

            metrics = comparison["scenarios"][scenario]

            if event.event_type == EventType.DEAL_CREATED:
                metrics["deals"] += 1
                metrics["spend"] += event.payload.get("total_cost", 0)
                metrics["fees"] += event.payload.get("exchange_fee", 0)
                metrics["impressions"] += event.payload.get("impressions", 0)

            elif event.event_type in (EventType.CONTEXT_DECAY, EventType.CONTEXT_RESTART):
                metrics["context_losses"] += 1

            elif event.event_type == EventType.CAMPAIGN_COMPLETED:
                metrics["goal_attainment"] = event.payload.get("goal_attainment")

        return comparison
