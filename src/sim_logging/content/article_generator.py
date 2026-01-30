"""
Article generator for RTB simulation content series.

Generates article drafts from simulation findings and comparison reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..events import EventIndex
from ..comparison import ComparisonAnalyzer, ComparisonReport
from ..narratives import NarrativeEngine, ScenarioNarrative
from .findings import FindingExtractor, Finding, FindingCategory


@dataclass
class Article:
    """
    An article draft generated from simulation results.
    """

    # Metadata
    article_id: str
    title: str
    slug: str
    category: str

    # Content
    abstract: str
    body: str
    key_findings: list[Finding] = field(default_factory=list)

    # SEO
    keywords: list[str] = field(default_factory=list)
    meta_description: str = ""

    # Generation info
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_markdown(self) -> str:
        """Generate full markdown article."""
        lines = [
            "---",
            f'title: "{self.title}"',
            f"slug: {self.slug}",
            f"category: {self.category}",
            f'keywords: {self.keywords}',
            f'description: "{self.meta_description}"',
            f"generated: {self.generated_at.isoformat()}",
            "---",
            "",
            f"# {self.title}",
            "",
            f"*{self.abstract}*",
            "",
            self.body,
        ]

        return "\n".join(lines)


@dataclass
class ArticleSeries:
    """
    A collection of related articles forming a content series.
    """

    series_id: str
    title: str
    description: str
    articles: list[Article] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_index_markdown(self) -> str:
        """Generate index page for the series."""
        lines = [
            f"# {self.title}",
            "",
            self.description,
            "",
            "## Articles in This Series",
            "",
        ]

        for i, article in enumerate(self.articles, 1):
            lines.append(f"{i}. [{article.title}]({article.slug}.md)")
            lines.append(f"   {article.abstract[:150]}...")
            lines.append("")

        lines.extend([
            "---",
            "",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d')}*",
        ])

        return "\n".join(lines)


class ArticleGenerator:
    """
    Generates article drafts from simulation results.

    Creates a series of articles covering different aspects
    of the simulation findings, ready for editing and publication.
    """

    # Article templates
    ARTICLE_TEMPLATES = [
        {
            "id": "hidden-tax",
            "title": "The Hidden Tax of Programmatic Advertising",
            "slug": "hidden-tax-programmatic-advertising",
            "category": "Economics",
            "categories": [FindingCategory.ECONOMICS],
            "keywords": ["ad tech", "exchange fees", "programmatic advertising", "cost optimization"],
        },
        {
            "id": "volatile-memory",
            "title": "The Hidden Cost of Volatile AI Memory",
            "slug": "volatile-ai-memory-cost",
            "category": "AI/ML",
            "categories": [FindingCategory.RELIABILITY],
            "keywords": ["AI agents", "context rot", "memory loss", "hallucinations"],
        },
        {
            "id": "ledger-advantage",
            "title": "Immutable Ledgers: A New Foundation for Ad Tech",
            "slug": "immutable-ledger-ad-tech",
            "category": "Technology",
            "categories": [FindingCategory.AUDITABILITY, FindingCategory.INFRASTRUCTURE],
            "keywords": ["blockchain", "ad tech", "transparency", "verification"],
        },
        {
            "id": "scenario-comparison",
            "title": "Three Trading Models: A Quantitative Comparison",
            "slug": "three-trading-models-comparison",
            "category": "Analysis",
            "categories": [FindingCategory.PERFORMANCE],
            "keywords": ["programmatic", "A2A trading", "comparison", "simulation"],
        },
        {
            "id": "executive-summary",
            "title": "Executive Summary: The Future of Programmatic Trading",
            "slug": "executive-summary-programmatic-future",
            "category": "Summary",
            "categories": list(FindingCategory),
            "keywords": ["executive summary", "programmatic", "ad tech", "recommendations"],
        },
    ]

    def __init__(self, event_index: EventIndex):
        """
        Initialize article generator.

        Args:
            event_index: Index of all simulation events
        """
        self.index = event_index
        self.analyzer = ComparisonAnalyzer(event_index)
        self.narrative_engine = NarrativeEngine(event_index)
        self.finding_extractor = FindingExtractor(event_index)

    def generate_series(self) -> ArticleSeries:
        """
        Generate complete article series from simulation.

        Returns:
            ArticleSeries with all generated articles
        """
        # Extract findings
        findings = self.finding_extractor.extract_all_findings()

        # Generate comparison report
        report = self.analyzer.generate_comparison_report()

        # Create series
        series = ArticleSeries(
            series_id="rtb-sim-2025",
            title="Programmatic Advertising: Exchange vs A2A vs Ledger",
            description=(
                "A data-driven analysis comparing three models for programmatic advertising: "
                "traditional exchange-based trading, pure agent-to-agent (A2A) trading, "
                "and ledger-backed direct trading. Based on a 30-day simulation with "
                "5 buyers, 5 sellers, and 50 campaigns."
            ),
        )

        # Generate each article
        for template in self.ARTICLE_TEMPLATES:
            article = self._generate_article(template, findings, report)
            series.articles.append(article)

        return series

    def _generate_article(
        self,
        template: dict,
        findings: list[Finding],
        report: ComparisonReport,
    ) -> Article:
        """Generate a single article from template and findings."""

        # Filter findings for this article
        article_findings = [
            f for f in findings
            if f.category in template["categories"]
        ]

        # Generate body based on article type
        if template["id"] == "hidden-tax":
            body = self._generate_economics_article(article_findings, report)
            abstract = "How much do ad tech intermediaries really extract from programmatic transactions? Our simulation reveals the true cost of exchange-based trading."

        elif template["id"] == "volatile-memory":
            body = self._generate_reliability_article(article_findings, report)
            abstract = "When AI agents lose their memory, they don't just forget - they make things up. Here's what happens when context rot meets advertising."

        elif template["id"] == "ledger-advantage":
            body = self._generate_ledger_article(article_findings, report)
            abstract = "Blockchain isn't just about cryptocurrency. Immutable ledgers could solve ad tech's transparency and trust problems at a fraction of current costs."

        elif template["id"] == "scenario-comparison":
            body = self._generate_comparison_article(article_findings, report)
            abstract = "Exchange-based, A2A, and ledger-backed: we simulated all three trading models to measure performance, cost, and reliability."

        else:  # executive-summary
            body = self._generate_executive_article(findings, report)
            abstract = "Key findings and recommendations from a 30-day simulation comparing three models for programmatic advertising."

        meta_description = abstract[:155] + "..." if len(abstract) > 155 else abstract

        return Article(
            article_id=template["id"],
            title=template["title"],
            slug=template["slug"],
            category=template["category"],
            abstract=abstract,
            body=body,
            key_findings=article_findings,
            keywords=template["keywords"],
            meta_description=meta_description,
        )

    def _generate_economics_article(
        self,
        findings: list[Finding],
        report: ComparisonReport,
    ) -> str:
        """Generate the economics-focused article."""
        c = report.comparison

        sections = [
            "## The Exchange Tax",
            "",
            "Every programmatic advertising transaction that flows through a traditional exchange "
            "is subject to an invisible tax. Our simulation quantified this cost.",
            "",
            f"Over {report.scenario_a.total_days} simulated days of advertising activity, "
            f"the exchange-based model (Scenario A) processed ${c.scenario_a_spend:,.2f} in transactions. "
            f"Of that amount, **${c.scenario_a_fees:,.2f}** was extracted by the exchange - "
            f"a **{c.a_take_rate:.1f}% take rate** on every deal.",
            "",
            "## Where Does the Money Go?",
            "",
            "When an advertiser pays $100 for impressions in the exchange model:",
            "",
            f"- **${100 - c.a_take_rate:.2f}** goes to the publisher",
            f"- **${c.a_take_rate:.2f}** goes to the exchange",
            "",
            "This intermediary fee represents infrastructure cost, but is it justified?",
            "",
            "## The Alternative: Ledger-Backed Trading",
            "",
            f"The ledger-backed model (Scenario C) processed a similar volume of transactions "
            f"(${c.scenario_c_spend:,.2f}) with infrastructure costs of just "
            f"**${c.scenario_c_blockchain_costs:,.2f}** - that's **{c.c_infrastructure_rate:.4f}%** of transaction volume.",
            "",
        ]

        # Add findings
        for finding in findings[:2]:
            sections.append(finding.to_markdown())

        sections.extend([
            "## The Bottom Line",
            "",
            f"Over the simulation period, the ledger-backed model saved **${c.fee_savings_c_vs_a:,.2f}** "
            f"compared to traditional exchange-based trading. At scale, these savings compound dramatically.",
            "",
            "The question isn't whether intermediaries provide value - they do. "
            "The question is whether that value is worth 15% of every transaction.",
        ])

        return "\n".join(sections)

    def _generate_reliability_article(
        self,
        findings: list[Finding],
        report: ComparisonReport,
    ) -> str:
        """Generate the reliability-focused article."""
        c = report.comparison

        sections = [
            "## The Problem with Agent Memory",
            "",
            "AI agents in production systems don't have perfect memory. They experience "
            "'context rot' - the gradual or sudden loss of working memory due to token limits, "
            "session boundaries, or system restarts.",
            "",
            "Our simulation modeled realistic memory decay in the pure agent-to-agent (A2A) "
            "trading model. The results reveal a hidden cost of volatile state.",
            "",
            "## Context Loss Events",
            "",
            f"Over {report.scenario_b.total_days} days, agents experienced **{c.context_rot_events}** "
            f"context loss events. Each event represents information permanently lost:",
            "",
            "- Deal histories with trading partners",
            "- Price negotiation patterns",
            "- Partner reliability scores",
            "- Inventory availability knowledge",
            "",
            "## The Hallucination Problem",
            "",
            f"When agents lose context, they don't stop working - they continue with incomplete "
            f"information. In **{c.hallucination_count}** cases, agents made decisions based on "
            f"**fabricated data**:",
            "",
            "- Imagined deal histories ('I've done 5 deals with this seller' - actually zero)",
            "- Invented price floors ('Their minimum is $2.00 CPM' - actually $5.00)",
            "- Hallucinated inventory ('They have 1M impressions available' - actually sold out)",
            "",
        ]

        # Add findings
        for finding in findings[:2]:
            sections.append(finding.to_markdown())

        sections.extend([
            "## The Ledger Solution",
            "",
            "The ledger-backed model (Scenario C) solves this problem by maintaining "
            "an immutable record of all transactions. When an agent's memory is compromised, "
            "it can reconstruct its state from the ledger with **100% accuracy**.",
            "",
            "Context rot becomes a minor inconvenience rather than a source of accumulated errors.",
        ])

        return "\n".join(sections)

    def _generate_ledger_article(
        self,
        findings: list[Finding],
        report: ComparisonReport,
    ) -> str:
        """Generate the ledger-focused article."""
        c = report.comparison

        sections = [
            "## Beyond Cryptocurrency",
            "",
            "When people hear 'blockchain,' they think cryptocurrency. But the core innovation "
            "- an immutable, verifiable ledger - has applications far beyond digital currency.",
            "",
            "Our simulation explored what happens when you bring immutable ledgers "
            "to programmatic advertising.",
            "",
            "## The Trust Problem",
            "",
            "Traditional advertising has a trust problem:",
            "",
            "- Advertisers must trust exchange reporting",
            "- Publishers must trust payment calculations",
            "- Disputes are resolved by the intermediary who benefits from opacity",
            "",
            "## The Ledger Solution",
            "",
            f"The ledger-backed model (Scenario C) recorded all **{c.scenario_c_deals:,}** "
            f"transactions to an immutable ledger. Every deal has:",
            "",
            "- Permanent, tamper-proof record",
            "- Verification accessible to all parties",
            "- Automated dispute resolution",
            "- Complete audit trail",
            "",
        ]

        # Add findings
        for finding in findings[:2]:
            sections.append(finding.to_markdown())

        cost_per_tx = c.scenario_c_blockchain_costs / c.scenario_c_deals if c.scenario_c_deals > 0 else 0

        sections.extend([
            "## The Cost Question",
            "",
            f"Infrastructure cost for the ledger model was **${cost_per_tx:.4f}** per transaction - "
            f"total cost of **${c.scenario_c_blockchain_costs:,.2f}** for the entire simulation.",
            "",
            f"Compare this to the **${c.scenario_a_fees:,.2f}** extracted by the exchange model. "
            f"The ledger approach provides transparency and verification at a fraction of the cost.",
        ])

        return "\n".join(sections)

    def _generate_comparison_article(
        self,
        findings: list[Finding],
        report: ComparisonReport,
    ) -> str:
        """Generate the comparison-focused article."""
        c = report.comparison

        sections = [
            "## Three Models, One Question",
            "",
            "How should programmatic advertising work? We simulated three approaches:",
            "",
            "1. **Scenario A: Exchange-Based** - Traditional model with centralized intermediary",
            "2. **Scenario B: Pure A2A** - Direct agent-to-agent trading, no intermediary",
            "3. **Scenario C: Ledger-Backed** - Direct trading with immutable record-keeping",
            "",
            "## Methodology",
            "",
            f"We simulated {report.scenario_a.total_days} days of advertising activity with:",
            "",
            "- 5 buyer agents (advertisers)",
            "- 5 seller agents (publishers)",
            "- 50 campaigns total",
            "- Realistic market conditions and agent behavior",
            "",
            "## Results Summary",
            "",
            "| Metric | Exchange (A) | Pure A2A (B) | Ledger (C) |",
            "|--------|-------------|--------------|------------|",
            f"| Deals | {c.scenario_a_deals:,} | {c.scenario_b_deals:,} | {c.scenario_c_deals:,} |",
            f"| Spend | ${c.scenario_a_spend:,.0f} | ${c.scenario_b_spend:,.0f} | ${c.scenario_c_spend:,.0f} |",
            f"| Fees | ${c.scenario_a_fees:,.0f} | $0 | $0 |",
            f"| Infrastructure | $0 | $0 | ${c.scenario_c_blockchain_costs:,.2f} |",
            f"| Context Losses | 0 | {c.context_rot_events} | 0 |",
            f"| Hallucinations | 0 | {c.hallucination_count} | 0 |",
            "",
        ]

        # Add findings
        sections.append("## Key Findings")
        sections.append("")
        for finding in findings[:3]:
            sections.append(finding.to_markdown())

        sections.extend([
            "## Conclusion",
            "",
            "Each model has trade-offs:",
            "",
            "- **Exchange**: Reliable but expensive (15% fees)",
            "- **Pure A2A**: Free but unreliable (context rot, hallucinations)",
            "- **Ledger**: Best of both - reliable, auditable, and cost-effective",
        ])

        return "\n".join(sections)

    def _generate_executive_article(
        self,
        findings: list[Finding],
        report: ComparisonReport,
    ) -> str:
        """Generate the executive summary article."""
        c = report.comparison

        sections = [
            "## Executive Summary",
            "",
            "This report summarizes findings from a 30-day simulation comparing three "
            "models for programmatic advertising: traditional exchange-based trading, "
            "pure agent-to-agent (A2A) trading, and ledger-backed direct trading.",
            "",
            "## Key Numbers",
            "",
            f"- **${c.scenario_a_fees:,.2f}** extracted by exchange intermediaries",
            f"- **{c.context_rot_events}** context loss events in pure A2A model",
            f"- **{c.hallucination_count}** decisions based on fabricated data",
            f"- **${c.scenario_c_blockchain_costs:,.2f}** total ledger infrastructure cost",
            f"- **{((c.scenario_a_fees - c.scenario_c_blockchain_costs) / c.scenario_a_fees * 100):.0f}%** cost savings with ledger model",
            "",
            "## Top Findings",
            "",
        ]

        # Add top 5 findings
        for finding in findings[:5]:
            sections.append(f"### {finding.headline}")
            sections.append("")
            sections.append(finding.summary)
            sections.append("")

        sections.extend([
            "## Recommendations",
            "",
            "Based on simulation results, we recommend:",
            "",
            "1. **Evaluate ledger-backed trading** for cost savings and transparency",
            "2. **Implement state persistence** for any agent-based system",
            "3. **Deploy ground truth verification** to detect hallucinations",
            "4. **Audit current exchange fees** against delivered value",
            "",
            "## Next Steps",
            "",
            "The simulation code and full methodology are available for review. "
            "Contact us to discuss implementing ledger-backed trading in your organization.",
        ])

        return "\n".join(sections)

    def write_series(self, output_dir: Path) -> list[Path]:
        """
        Generate series and write to files.

        Args:
            output_dir: Directory to write articles

        Returns:
            List of paths to written files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        series = self.generate_series()
        written_files = []

        # Write index
        index_path = output_dir / "README.md"
        index_path.write_text(series.to_index_markdown())
        written_files.append(index_path)

        # Write articles
        for article in series.articles:
            article_path = output_dir / f"{article.slug}.md"
            article_path.write_text(article.to_markdown())
            written_files.append(article_path)

        return written_files
