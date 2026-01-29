"""
Narrative generation engine for RTB simulation.

Transforms simulation events into human-readable narratives
suitable for articles, reports, and content generation.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from .events import (
    SimulationEvent,
    EventType,
    EventIndex,
    Scenario,
)


@dataclass
class CampaignNarrative:
    """
    Complete narrative for a campaign's journey through the simulation.

    Captures the story of a campaign from creation to completion,
    including all significant events and outcomes.
    """

    campaign_id: str
    scenario: Scenario
    buyer_id: str

    # Timeline
    start_day: int = 0
    end_day: Optional[int] = None

    # Outcomes
    target_impressions: int = 0
    delivered_impressions: int = 0
    total_spend: float = 0.0
    total_fees: float = 0.0
    deals_made: int = 0

    # Context rot impact (Scenario B)
    context_losses: int = 0
    hallucinations: int = 0

    # Key events
    significant_events: list[str] = field(default_factory=list)

    # Generated narratives
    opening_summary: str = ""
    journey_narrative: str = ""
    outcome_summary: str = ""
    key_insight: str = ""

    @property
    def goal_attainment(self) -> float:
        """Percentage of goal achieved."""
        if self.target_impressions == 0:
            return 0.0
        return min(100.0, (self.delivered_impressions / self.target_impressions) * 100)

    @property
    def effective_cpm(self) -> float:
        """Effective CPM paid."""
        if self.delivered_impressions == 0:
            return 0.0
        return (self.total_spend / self.delivered_impressions) * 1000

    @property
    def fee_percentage(self) -> float:
        """Fees as percentage of spend."""
        if self.total_spend == 0:
            return 0.0
        return (self.total_fees / self.total_spend) * 100


@dataclass
class DayNarrative:
    """Narrative summary for a simulation day."""

    simulation_day: int
    scenario: Scenario

    # Metrics
    deals_count: int = 0
    total_spend: float = 0.0
    total_fees: float = 0.0
    context_losses: int = 0
    hallucinations: int = 0
    blockchain_costs: float = 0.0

    # Highlights
    notable_events: list[str] = field(default_factory=list)

    # Generated narrative
    summary: str = ""
    detail: str = ""


@dataclass
class ScenarioNarrative:
    """Complete narrative for a scenario's simulation run."""

    scenario: Scenario
    name: str
    description: str

    # Duration
    total_days: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Aggregate metrics
    total_deals: int = 0
    total_impressions: int = 0
    total_spend: float = 0.0
    total_fees: float = 0.0
    total_blockchain_costs: float = 0.0

    # Context rot (Scenario B)
    total_context_losses: int = 0
    total_hallucinations: int = 0

    # Campaign outcomes
    campaigns_started: int = 0
    campaigns_completed: int = 0
    average_goal_attainment: float = 0.0

    # Day narratives
    day_narratives: list[DayNarrative] = field(default_factory=list)

    # Generated narratives
    executive_summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    conclusion: str = ""


class NarrativeEngine:
    """
    Generates human-readable narratives from simulation events.

    Transforms raw event data into article-ready content,
    highlighting key insights and comparison points.
    """

    SCENARIO_DESCRIPTIONS = {
        Scenario.A: (
            "Current State (Rent-Seeking Exchange)",
            "Traditional programmatic advertising with exchange intermediaries extracting 15% fees.",
        ),
        Scenario.B: (
            "IAB Pure A2A (with Context Rot)",
            "Direct agent-to-agent trading with no intermediary, but vulnerable to memory loss.",
        ),
        Scenario.C: (
            "Alkimi Ledger-Backed Exchange",
            "Direct trading with immutable ledger for perfect state recovery and auditability.",
        ),
    }

    def __init__(self, event_index: EventIndex):
        """
        Initialize narrative engine.

        Args:
            event_index: Index of all simulation events
        """
        self.index = event_index

    def generate_campaign_narrative(
        self,
        campaign_id: str,
        correlation_id: Optional[str] = None,
    ) -> CampaignNarrative:
        """
        Generate complete narrative for a campaign.

        Args:
            campaign_id: Campaign identifier
            correlation_id: Optional correlation ID for cross-scenario tracking

        Returns:
            Complete campaign narrative
        """
        events = self.index.get_campaign_timeline(campaign_id)
        if not events:
            return CampaignNarrative(campaign_id=campaign_id, scenario=Scenario.A, buyer_id="unknown")

        # Extract basic info from first event
        first_event = events[0]
        scenario = first_event.scenario
        buyer_id = first_event.agent_id or "unknown"

        narrative = CampaignNarrative(
            campaign_id=campaign_id,
            scenario=scenario,
            buyer_id=buyer_id,
        )

        # Process events to build narrative
        for event in events:
            self._process_campaign_event(narrative, event)

        # Generate text narratives
        narrative.opening_summary = self._generate_campaign_opening(narrative)
        narrative.journey_narrative = self._generate_campaign_journey(narrative, events)
        narrative.outcome_summary = self._generate_campaign_outcome(narrative)
        narrative.key_insight = self._generate_campaign_insight(narrative)

        return narrative

    def _process_campaign_event(
        self,
        narrative: CampaignNarrative,
        event: SimulationEvent,
    ) -> None:
        """Process a single event for campaign narrative."""
        if event.event_type == EventType.CAMPAIGN_STARTED:
            narrative.start_day = event.simulation_day
            narrative.target_impressions = event.payload.get("target_impressions", 0)

        elif event.event_type == EventType.DEAL_CREATED:
            narrative.deals_made += 1
            narrative.delivered_impressions += event.payload.get("impressions", 0)
            narrative.total_spend += event.payload.get("total_cost", 0)
            narrative.total_fees += event.payload.get("exchange_fee", 0)

            # Track significant deals
            impressions = event.payload.get("impressions", 0)
            if impressions >= 500000:
                narrative.significant_events.append(
                    f"Day {event.simulation_day}: Major deal - {impressions:,} impressions"
                )

        elif event.event_type == EventType.CONTEXT_DECAY:
            narrative.context_losses += 1
            keys_lost = event.payload.get("keys_lost", 0)
            narrative.significant_events.append(
                f"Day {event.simulation_day}: Context decay - {keys_lost} memory keys lost"
            )

        elif event.event_type == EventType.CONTEXT_RESTART:
            narrative.context_losses += 1
            narrative.significant_events.append(
                f"Day {event.simulation_day}: CRITICAL - Agent restart, all context wiped"
            )

        elif event.event_type == EventType.HALLUCINATION_DETECTED:
            narrative.hallucinations += 1
            claim_type = event.payload.get("claim_type", "data")
            narrative.significant_events.append(
                f"Day {event.simulation_day}: Hallucination - {claim_type}"
            )

        elif event.event_type == EventType.CAMPAIGN_COMPLETED:
            narrative.end_day = event.simulation_day

    def _generate_campaign_opening(self, narrative: CampaignNarrative) -> str:
        """Generate opening summary for campaign."""
        scenario_name = self.SCENARIO_DESCRIPTIONS[narrative.scenario][0]

        return (
            f"Campaign {narrative.campaign_id} launched on day {narrative.start_day} "
            f"in {scenario_name} with a target of {narrative.target_impressions:,} impressions."
        )

    def _generate_campaign_journey(
        self,
        narrative: CampaignNarrative,
        events: list[SimulationEvent],
    ) -> str:
        """Generate journey narrative from events."""
        if not narrative.significant_events:
            return f"The campaign proceeded smoothly with {narrative.deals_made} deals completed."

        journey_parts = [
            "Key events during the campaign:",
            "",
        ]

        for event_desc in narrative.significant_events[:5]:  # Limit to 5 most significant
            journey_parts.append(f"- {event_desc}")

        return "\n".join(journey_parts)

    def _generate_campaign_outcome(self, narrative: CampaignNarrative) -> str:
        """Generate outcome summary."""
        parts = []

        # Goal attainment
        if narrative.goal_attainment >= 100:
            parts.append(f"Campaign achieved its goal with {narrative.delivered_impressions:,} impressions delivered ({narrative.goal_attainment:.1f}%).")
        else:
            parts.append(f"Campaign delivered {narrative.delivered_impressions:,} of {narrative.target_impressions:,} target impressions ({narrative.goal_attainment:.1f}%).")

        # Cost analysis
        parts.append(f"Total spend: ${narrative.total_spend:,.2f} at an effective CPM of ${narrative.effective_cpm:.2f}.")

        # Scenario-specific outcomes
        if narrative.scenario == Scenario.A:
            parts.append(f"Exchange fees: ${narrative.total_fees:,.2f} ({narrative.fee_percentage:.1f}% of spend).")
        elif narrative.scenario == Scenario.B:
            if narrative.context_losses > 0:
                parts.append(f"Context rot impact: {narrative.context_losses} memory loss events, {narrative.hallucinations} hallucinations.")
        elif narrative.scenario == Scenario.C:
            parts.append("All transactions recorded to immutable ledger with full auditability.")

        return " ".join(parts)

    def _generate_campaign_insight(self, narrative: CampaignNarrative) -> str:
        """Generate key insight for campaign."""
        if narrative.scenario == Scenario.A:
            if narrative.fee_percentage > 15:
                return f"High intermediary fees ({narrative.fee_percentage:.1f}%) significantly impacted campaign efficiency."
            return f"Exchange extracted ${narrative.total_fees:,.2f} from this campaign."

        elif narrative.scenario == Scenario.B:
            if narrative.context_losses > 0:
                return f"Without persistent state, {narrative.context_losses} context loss events degraded campaign performance."
            return "Direct A2A trading avoided fees but relied entirely on volatile agent memory."

        else:  # Scenario C
            return "Ledger-backed trading provided full auditability with no intermediary fees."

    def generate_day_narrative(
        self,
        day: int,
        scenario: Scenario,
    ) -> DayNarrative:
        """
        Generate narrative for a simulation day.

        Args:
            day: Simulation day number
            scenario: Scenario identifier

        Returns:
            Day narrative
        """
        events = self.index.get_day_events(day, scenario)

        narrative = DayNarrative(
            simulation_day=day,
            scenario=scenario,
        )

        # Aggregate metrics from events
        for event in events:
            if event.event_type == EventType.DEAL_CREATED:
                narrative.deals_count += 1
                narrative.total_spend += event.payload.get("total_cost", 0)
                narrative.total_fees += event.payload.get("exchange_fee", 0)

            elif event.event_type == EventType.CONTEXT_DECAY:
                narrative.context_losses += 1

            elif event.event_type == EventType.CONTEXT_RESTART:
                narrative.context_losses += 1
                narrative.notable_events.append(
                    f"Agent {event.agent_id} restarted - all context lost"
                )

            elif event.event_type == EventType.HALLUCINATION_DETECTED:
                narrative.hallucinations += 1
                narrative.notable_events.append(
                    f"Hallucination: {event.agent_id} - {event.payload.get('claim_type')}"
                )

            elif event.event_type == EventType.BLOCKCHAIN_COST:
                narrative.blockchain_costs += event.payload.get("total_usd", 0)

        # Generate text
        narrative.summary = self._generate_day_summary(narrative)
        narrative.detail = self._generate_day_detail(narrative)

        return narrative

    def _generate_day_summary(self, narrative: DayNarrative) -> str:
        """Generate one-line day summary."""
        parts = [f"Day {narrative.simulation_day}:"]

        parts.append(f"{narrative.deals_count} deals")
        parts.append(f"${narrative.total_spend:,.0f} spend")

        if narrative.scenario == Scenario.A:
            fee_pct = (narrative.total_fees / narrative.total_spend * 100) if narrative.total_spend > 0 else 0
            parts.append(f"${narrative.total_fees:,.0f} fees ({fee_pct:.1f}%)")

        elif narrative.scenario == Scenario.B:
            if narrative.context_losses > 0:
                parts.append(f"{narrative.context_losses} context losses")
            if narrative.hallucinations > 0:
                parts.append(f"{narrative.hallucinations} hallucinations")

        elif narrative.scenario == Scenario.C:
            parts.append(f"${narrative.blockchain_costs:.2f} ledger costs")

        return ", ".join(parts)

    def _generate_day_detail(self, narrative: DayNarrative) -> str:
        """Generate detailed day narrative."""
        lines = []

        if narrative.scenario == Scenario.A:
            lines.append(
                f"The rent-seeking exchange processed {narrative.deals_count} deals on day {narrative.simulation_day}, "
                f"generating ${narrative.total_spend:,.2f} in transaction volume. "
                f"The exchange extracted ${narrative.total_fees:,.2f} in fees."
            )

        elif narrative.scenario == Scenario.B:
            lines.append(
                f"Direct A2A trading completed {narrative.deals_count} deals worth ${narrative.total_spend:,.2f}. "
            )
            if narrative.context_losses > 0:
                lines.append(
                    f"However, {narrative.context_losses} context loss events impacted agent performance, "
                    f"with {narrative.hallucinations} decisions based on hallucinated data."
                )

        elif narrative.scenario == Scenario.C:
            lines.append(
                f"Ledger-backed trading completed {narrative.deals_count} deals worth ${narrative.total_spend:,.2f}. "
                f"All transactions were recorded to the immutable ledger at a cost of ${narrative.blockchain_costs:.2f}."
            )

        if narrative.notable_events:
            lines.append("\nNotable events:")
            for event in narrative.notable_events[:3]:
                lines.append(f"- {event}")

        return " ".join(lines)

    def generate_scenario_narrative(
        self,
        scenario: Scenario,
    ) -> ScenarioNarrative:
        """
        Generate complete scenario narrative.

        Args:
            scenario: Scenario identifier

        Returns:
            Complete scenario narrative
        """
        name, desc = self.SCENARIO_DESCRIPTIONS[scenario]

        narrative = ScenarioNarrative(
            scenario=scenario,
            name=name,
            description=desc,
        )

        # Get all scenario events
        events = self.index.get_scenario_events(scenario)
        if not events:
            return narrative

        # Track days
        days = set()
        for event in events:
            days.add(event.simulation_day)

            if event.event_type == EventType.DEAL_CREATED:
                narrative.total_deals += 1
                narrative.total_impressions += event.payload.get("impressions", 0)
                narrative.total_spend += event.payload.get("total_cost", 0)
                narrative.total_fees += event.payload.get("exchange_fee", 0)

            elif event.event_type == EventType.CONTEXT_DECAY:
                narrative.total_context_losses += 1

            elif event.event_type == EventType.CONTEXT_RESTART:
                narrative.total_context_losses += 1

            elif event.event_type == EventType.HALLUCINATION_DETECTED:
                narrative.total_hallucinations += 1

            elif event.event_type == EventType.BLOCKCHAIN_COST:
                narrative.total_blockchain_costs += event.payload.get("total_usd", 0)

            elif event.event_type == EventType.SCENARIO_STARTED:
                narrative.start_time = event.timestamp

            elif event.event_type == EventType.SCENARIO_COMPLETED:
                narrative.end_time = event.timestamp

        narrative.total_days = len(days)

        # Generate day narratives
        for day in sorted(days):
            day_narrative = self.generate_day_narrative(day, scenario)
            narrative.day_narratives.append(day_narrative)

        # Generate text
        narrative.executive_summary = self._generate_scenario_summary(narrative)
        narrative.key_findings = self._generate_scenario_findings(narrative)
        narrative.conclusion = self._generate_scenario_conclusion(narrative)

        return narrative

    def _generate_scenario_summary(self, narrative: ScenarioNarrative) -> str:
        """Generate executive summary for scenario."""
        parts = [
            f"**{narrative.name}**",
            "",
            narrative.description,
            "",
            f"Over {narrative.total_days} simulation days, {narrative.total_deals:,} deals were completed "
            f"representing {narrative.total_impressions:,} impressions and ${narrative.total_spend:,.2f} in spend.",
        ]

        if narrative.scenario == Scenario.A:
            take_rate = (narrative.total_fees / narrative.total_spend * 100) if narrative.total_spend > 0 else 0
            parts.append(
                f"\nThe exchange extracted ${narrative.total_fees:,.2f} in fees ({take_rate:.1f}% take rate)."
            )

        elif narrative.scenario == Scenario.B:
            parts.append(
                f"\n{narrative.total_context_losses} context loss events occurred, "
                f"resulting in {narrative.total_hallucinations} hallucination incidents."
            )

        elif narrative.scenario == Scenario.C:
            parts.append(
                f"\nAll transactions were recorded to the immutable ledger "
                f"at a total cost of ${narrative.total_blockchain_costs:,.2f}."
            )

        return "\n".join(parts)

    def _generate_scenario_findings(self, narrative: ScenarioNarrative) -> list[str]:
        """Generate key findings for scenario."""
        findings = []

        if narrative.scenario == Scenario.A:
            take_rate = (narrative.total_fees / narrative.total_spend * 100) if narrative.total_spend > 0 else 0
            findings.append(f"Exchange intermediary extracted {take_rate:.1f}% of all transaction value")
            findings.append(f"Total fees collected: ${narrative.total_fees:,.2f}")
            findings.append("All state maintained by centralized exchange")

        elif narrative.scenario == Scenario.B:
            findings.append("Zero intermediary fees - direct buyer-seller transactions")
            findings.append(f"Context rot caused {narrative.total_context_losses} memory loss events")
            findings.append(f"Memory loss led to {narrative.total_hallucinations} hallucinated decisions")
            findings.append("No recovery mechanism - lost context cannot be restored")

        elif narrative.scenario == Scenario.C:
            findings.append("Zero intermediary fees with immutable record keeping")
            findings.append(f"Blockchain infrastructure cost: ${narrative.total_blockchain_costs:,.2f}")
            findings.append("100% state recovery accuracy from ledger")
            findings.append("Full transaction auditability and verification")

        return findings

    def _generate_scenario_conclusion(self, narrative: ScenarioNarrative) -> str:
        """Generate conclusion for scenario."""
        if narrative.scenario == Scenario.A:
            return (
                f"The rent-seeking exchange model successfully processed {narrative.total_deals:,} deals "
                f"but extracted significant value through fees. While providing reliable infrastructure, "
                f"the intermediary take rate represents a substantial cost to market participants."
            )

        elif narrative.scenario == Scenario.B:
            return (
                f"Direct A2A trading eliminated intermediary fees but proved vulnerable to context rot. "
                f"Without persistent state, agents experienced {narrative.total_context_losses} memory loss events, "
                f"leading to {narrative.total_hallucinations} decisions based on fabricated data. "
                f"The model demonstrates the hidden costs of volatile state in agent systems."
            )

        else:  # Scenario C
            return (
                f"The ledger-backed model achieved the benefits of direct trading while maintaining "
                f"full auditability and state recovery. At a cost of ${narrative.total_blockchain_costs:,.2f}, "
                f"the immutable ledger provided complete transaction history and 100% recovery accuracy - "
                f"a fraction of the ${narrative.total_spend:,.2f} that would have been extracted by traditional exchanges."
            )
