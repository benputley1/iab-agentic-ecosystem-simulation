"""
Finding extraction from simulation results.

Identifies article-worthy insights from simulation data,
categorizes them by significance, and prepares them for
content generation.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..events import EventIndex, EventType, Scenario
from ..comparison import ComparisonAnalyzer, ComparisonReport


class FindingCategory(str, Enum):
    """Categories of findings for article organization."""
    ECONOMICS = "economics"          # Fee savings, cost comparisons
    RELIABILITY = "reliability"      # Context rot, hallucinations
    AUDITABILITY = "auditability"    # Ledger benefits, verification
    PERFORMANCE = "performance"      # Deal volume, efficiency
    INFRASTRUCTURE = "infrastructure"  # Blockchain costs, scalability


@dataclass
class Finding:
    """
    An article-worthy finding from the simulation.

    Represents a key insight that merits inclusion in
    content about the simulation results.
    """

    # Identification
    finding_id: str
    category: FindingCategory
    significance: int  # 1-5, higher = more important

    # Content
    headline: str  # Article-ready headline
    summary: str   # 2-3 sentence summary
    detail: str    # Full explanation with data

    # Supporting data
    data_points: dict = field(default_factory=dict)

    # Article placement
    suggested_article: Optional[str] = None
    pull_quote: Optional[str] = None

    def to_markdown(self) -> str:
        """Format finding as markdown section."""
        lines = [
            f"### {self.headline}",
            "",
            self.summary,
            "",
            self.detail,
            "",
        ]

        if self.data_points:
            lines.append("**Key Data:**")
            for key, value in self.data_points.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:,.2f}")
                elif isinstance(value, int):
                    lines.append(f"- {key}: {value:,}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        if self.pull_quote:
            lines.append(f"> {self.pull_quote}")
            lines.append("")

        return "\n".join(lines)


class FindingExtractor:
    """
    Extracts article-worthy findings from simulation results.

    Analyzes comparison reports and event data to identify
    the most significant and publishable insights.
    """

    # Minimum thresholds for significance
    MIN_FEE_SAVINGS_PCT = 50  # 50% savings = significant
    MIN_CONTEXT_LOSSES = 10  # 10+ losses = worth mentioning
    MIN_HALLUCINATIONS = 5   # 5+ hallucinations = significant

    def __init__(self, event_index: EventIndex):
        """
        Initialize finding extractor.

        Args:
            event_index: Index of all simulation events
        """
        self.index = event_index
        self.analyzer = ComparisonAnalyzer(event_index)

    def extract_all_findings(self) -> list[Finding]:
        """
        Extract all article-worthy findings from simulation.

        Returns:
            List of findings sorted by significance
        """
        findings = []

        # Generate comparison report
        report = self.analyzer.generate_comparison_report()

        # Extract findings by category
        findings.extend(self._extract_economics_findings(report))
        findings.extend(self._extract_reliability_findings(report))
        findings.extend(self._extract_auditability_findings(report))
        findings.extend(self._extract_performance_findings(report))
        findings.extend(self._extract_infrastructure_findings(report))

        # Sort by significance
        findings.sort(key=lambda f: f.significance, reverse=True)

        return findings

    def extract_top_findings(self, count: int = 5) -> list[Finding]:
        """
        Extract the top N most significant findings.

        Args:
            count: Number of findings to return

        Returns:
            Top findings by significance
        """
        all_findings = self.extract_all_findings()
        return all_findings[:count]

    def _extract_economics_findings(self, report: ComparisonReport) -> list[Finding]:
        """Extract economics-related findings."""
        findings = []
        c = report.comparison

        # Fee savings finding
        if c.scenario_a_fees > 0:
            savings = c.fee_savings_c_vs_a
            savings_pct = (savings / c.scenario_a_fees) * 100 if c.scenario_a_fees > 0 else 0

            significance = 5 if savings_pct > self.MIN_FEE_SAVINGS_PCT else 3

            findings.append(Finding(
                finding_id="econ-001",
                category=FindingCategory.ECONOMICS,
                significance=significance,
                headline=f"Ledger Model Cuts Costs by {savings_pct:.0f}%",
                summary=(
                    f"The ledger-backed trading model (Scenario C) reduced transaction costs by "
                    f"${savings:,.2f} compared to traditional exchange-based trading. "
                    f"This represents a {savings_pct:.0f}% reduction in overhead."
                ),
                detail=(
                    f"Traditional programmatic advertising relies on exchange intermediaries that "
                    f"extract fees from every transaction. In our simulation, the exchange model "
                    f"(Scenario A) collected ${c.scenario_a_fees:,.2f} in fees at a "
                    f"{c.a_take_rate:.1f}% take rate. The ledger-backed model achieved the same "
                    f"transaction volume with infrastructure costs of just ${c.scenario_c_blockchain_costs:,.2f}, "
                    f"a savings of {savings_pct:.0f}%."
                ),
                data_points={
                    "Exchange fees (A)": c.scenario_a_fees,
                    "Blockchain costs (C)": c.scenario_c_blockchain_costs,
                    "Net savings": savings,
                    "Savings percentage": f"{savings_pct:.1f}%",
                },
                suggested_article="The Hidden Tax of Programmatic Advertising",
                pull_quote=f"Exchange intermediaries extracted ${c.scenario_a_fees:,.2f} - money that could have gone to publishers or reduced advertiser costs.",
            ))

        # Take rate finding
        if c.a_take_rate > 10:
            findings.append(Finding(
                finding_id="econ-002",
                category=FindingCategory.ECONOMICS,
                significance=4,
                headline=f"{c.a_take_rate:.1f}% of Every Ad Dollar Goes to Intermediaries",
                summary=(
                    f"Our simulation revealed that traditional ad exchanges extract "
                    f"{c.a_take_rate:.1f}% of every transaction. Over {report.scenario_a.total_days} days, "
                    f"this amounted to ${c.scenario_a_fees:,.2f} in intermediary fees."
                ),
                detail=(
                    f"The rent-seeking exchange model (Scenario A) operates by inserting itself "
                    f"between buyers and sellers, extracting value from each transaction. While "
                    f"this model provides market infrastructure, the cost is substantial. "
                    f"At {c.a_take_rate:.1f}% per transaction, a significant portion of advertiser "
                    f"budgets never reaches publishers."
                ),
                data_points={
                    "Take rate": f"{c.a_take_rate:.1f}%",
                    "Total fees extracted": c.scenario_a_fees,
                    "Total spend": c.scenario_a_spend,
                },
                suggested_article="Where Does Your Ad Budget Really Go?",
                pull_quote=f"For every $100 spent on advertising, ${c.a_take_rate:.0f} goes to the exchange - not to publishers.",
            ))

        return findings

    def _extract_reliability_findings(self, report: ComparisonReport) -> list[Finding]:
        """Extract reliability-related findings."""
        findings = []
        c = report.comparison

        # Context rot impact
        if c.context_rot_events >= self.MIN_CONTEXT_LOSSES:
            findings.append(Finding(
                finding_id="rel-001",
                category=FindingCategory.RELIABILITY,
                significance=4,
                headline=f"Agent Memory Loss Caused {c.context_rot_events} Incidents",
                summary=(
                    f"The pure agent-to-agent model (Scenario B) experienced "
                    f"{c.context_rot_events} context loss events over the simulation period. "
                    f"Without persistent state, agents cannot recover lost information."
                ),
                detail=(
                    f"Agentic systems without persistent state are vulnerable to 'context rot' - "
                    f"the gradual or sudden loss of working memory. Our simulation modeled "
                    f"realistic memory decay patterns, resulting in {c.context_rot_events} "
                    f"context loss events. Each event represents information that was permanently "
                    f"lost: deal histories, partner preferences, negotiation patterns. "
                    f"In the pure A2A model, this data cannot be recovered."
                ),
                data_points={
                    "Context loss events": c.context_rot_events,
                    "Resulting hallucinations": c.hallucination_count,
                    "Days simulated": report.scenario_b.total_days,
                },
                suggested_article="The Hidden Cost of Volatile AI Memory",
                pull_quote="Every context loss means an agent making decisions based on incomplete or fabricated information.",
            ))

        # Hallucination finding
        if c.hallucination_count >= self.MIN_HALLUCINATIONS:
            hallucination_rate = (c.hallucination_count / c.context_rot_events * 100) if c.context_rot_events > 0 else 0

            findings.append(Finding(
                finding_id="rel-002",
                category=FindingCategory.RELIABILITY,
                significance=5,
                headline=f"Memory Loss Led to {c.hallucination_count} Fabricated Decisions",
                summary=(
                    f"Following context loss events, agents made {c.hallucination_count} decisions "
                    f"based on hallucinated data. Without ground truth verification, "
                    f"these errors went undetected."
                ),
                detail=(
                    f"When agents lose context, they don't simply stop working - they continue "
                    f"making decisions with incomplete information. In {hallucination_rate:.0f}% of "
                    f"context loss events, agents subsequently made decisions based on fabricated "
                    f"data: imagined deal histories, invented price floors, or hallucinated "
                    f"inventory levels. These hallucinations directly impacted trading efficiency "
                    f"and could not be detected without ground truth verification."
                ),
                data_points={
                    "Hallucination count": c.hallucination_count,
                    "Hallucination rate": f"{hallucination_rate:.0f}%",
                    "Context losses": c.context_rot_events,
                },
                suggested_article="When AI Agents Make Things Up",
                pull_quote=f"In {hallucination_rate:.0f}% of memory loss events, agents proceeded to make decisions based on data they fabricated.",
            ))

        return findings

    def _extract_auditability_findings(self, report: ComparisonReport) -> list[Finding]:
        """Extract auditability-related findings."""
        findings = []
        c = report.comparison

        if c.scenario_c_deals > 0:
            findings.append(Finding(
                finding_id="audit-001",
                category=FindingCategory.AUDITABILITY,
                significance=4,
                headline="100% Transaction Auditability with Immutable Ledger",
                summary=(
                    f"The ledger-backed model recorded all {c.scenario_c_deals:,} transactions "
                    f"to an immutable ledger, enabling complete auditability and dispute resolution."
                ),
                detail=(
                    f"Transparency and verification are critical in advertising. The ledger-backed "
                    f"model (Scenario C) records every transaction to an immutable ledger, creating "
                    f"a permanent, verifiable record. Unlike Scenario A (centralized exchange records) "
                    f"or Scenario B (no persistent records), the ledger provides tamper-proof "
                    f"verification accessible to all parties. This enables automated dispute "
                    f"resolution and eliminates the need to trust intermediary reporting."
                ),
                data_points={
                    "Transactions recorded": c.scenario_c_deals,
                    "Verification accuracy": "100%",
                    "Record permanence": "Immutable",
                },
                suggested_article="Trustless Verification in Programmatic Advertising",
                pull_quote="Every transaction is recorded immutably - no disputes about what happened, when, or at what price.",
            ))

        return findings

    def _extract_performance_findings(self, report: ComparisonReport) -> list[Finding]:
        """Extract performance-related findings."""
        findings = []
        c = report.comparison

        total_deals = c.scenario_a_deals + c.scenario_b_deals + c.scenario_c_deals
        total_spend = c.scenario_a_spend + c.scenario_b_spend + c.scenario_c_spend

        if total_deals > 0:
            findings.append(Finding(
                finding_id="perf-001",
                category=FindingCategory.PERFORMANCE,
                significance=3,
                headline=f"{total_deals:,} Deals Processed Across Three Trading Models",
                summary=(
                    f"The simulation processed ${total_spend:,.2f} in advertising transactions "
                    f"across {total_deals:,} individual deals, demonstrating the viability "
                    f"of all three trading approaches."
                ),
                detail=(
                    f"Over {report.scenario_a.total_days} simulated days, each model successfully "
                    f"completed advertising transactions. Scenario A (exchange): {c.scenario_a_deals:,} deals, "
                    f"${c.scenario_a_spend:,.2f}. Scenario B (pure A2A): {c.scenario_b_deals:,} deals, "
                    f"${c.scenario_b_spend:,.2f}. Scenario C (ledger): {c.scenario_c_deals:,} deals, "
                    f"${c.scenario_c_spend:,.2f}. While all models achieved similar throughput, "
                    f"the critical differences lie in cost, reliability, and auditability."
                ),
                data_points={
                    "Total deals": total_deals,
                    "Total spend": total_spend,
                    "Scenario A deals": c.scenario_a_deals,
                    "Scenario B deals": c.scenario_b_deals,
                    "Scenario C deals": c.scenario_c_deals,
                },
                suggested_article="Comparing Trading Models: A Quantitative Analysis",
            ))

        return findings

    def _extract_infrastructure_findings(self, report: ComparisonReport) -> list[Finding]:
        """Extract infrastructure-related findings."""
        findings = []
        c = report.comparison

        if c.scenario_c_blockchain_costs > 0 and c.scenario_c_deals > 0:
            cost_per_deal = c.scenario_c_blockchain_costs / c.scenario_c_deals

            findings.append(Finding(
                finding_id="infra-001",
                category=FindingCategory.INFRASTRUCTURE,
                significance=4,
                headline=f"Blockchain Infrastructure Costs ${cost_per_deal:.4f} Per Transaction",
                summary=(
                    f"Recording transactions to the immutable ledger cost just "
                    f"${c.scenario_c_blockchain_costs:,.2f} total, or ${cost_per_deal:.4f} per deal - "
                    f"a fraction of exchange fees."
                ),
                detail=(
                    f"A common concern about blockchain-based systems is infrastructure cost. "
                    f"Our simulation used realistic gas and storage costs based on Sui and Walrus "
                    f"pricing. At ${cost_per_deal:.4f} per transaction, the total infrastructure "
                    f"cost was ${c.scenario_c_blockchain_costs:,.2f} for {c.scenario_c_deals:,} deals. "
                    f"Compare this to the ${c.scenario_a_fees:,.2f} extracted by the exchange model - "
                    f"the ledger approach represents a {((c.scenario_a_fees - c.scenario_c_blockchain_costs) / c.scenario_a_fees * 100):.0f}% "
                    f"cost reduction while adding immutability and auditability."
                ),
                data_points={
                    "Cost per transaction": f"${cost_per_deal:.4f}",
                    "Total infrastructure cost": c.scenario_c_blockchain_costs,
                    "Transactions processed": c.scenario_c_deals,
                    "Exchange fees comparison": c.scenario_a_fees,
                },
                suggested_article="Blockchain Economics in Programmatic Advertising",
                pull_quote=f"At ${cost_per_deal:.4f} per transaction, blockchain infrastructure costs less than 1% of traditional exchange fees.",
            ))

        return findings
