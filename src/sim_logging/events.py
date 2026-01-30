"""
Event models and event logging infrastructure for RTB simulation.

Provides normalized event schemas that work across all scenarios,
enabling cross-scenario comparison and content generation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
import uuid
import json


class EventType(str, Enum):
    """Types of events captured by the orchestration logger."""

    # Transaction lifecycle
    BID_REQUEST = "bid_request"
    BID_RESPONSE = "bid_response"
    AUCTION_STARTED = "auction_started"
    AUCTION_COMPLETED = "auction_completed"
    DEAL_CREATED = "deal_created"
    DEAL_REJECTED = "deal_rejected"
    DELIVERY_CONFIRMED = "delivery_confirmed"

    # Campaign events
    CAMPAIGN_STARTED = "campaign_started"
    CAMPAIGN_MILESTONE = "campaign_milestone"
    CAMPAIGN_COMPLETED = "campaign_completed"
    CAMPAIGN_FAILED = "campaign_failed"

    # Context rot events (Scenario B)
    CONTEXT_DECAY = "context_decay"
    CONTEXT_RESTART = "context_restart"
    MEMORY_LOST = "memory_lost"

    # Hallucination events
    HALLUCINATION_DETECTED = "hallucination_detected"
    HALLUCINATION_IMPACT = "hallucination_impact"

    # State recovery (Scenario C)
    STATE_RECOVERY_STARTED = "state_recovery_started"
    STATE_RECOVERY_COMPLETED = "state_recovery_completed"

    # Ledger events (Scenario C)
    LEDGER_WRITE = "ledger_write"
    LEDGER_VERIFIED = "ledger_verified"

    # Fee events
    FEE_EXTRACTED = "fee_extracted"
    BLOCKCHAIN_COST = "blockchain_cost"

    # Simulation events
    DAY_STARTED = "day_started"
    DAY_COMPLETED = "day_completed"
    SCENARIO_STARTED = "scenario_started"
    SCENARIO_COMPLETED = "scenario_completed"


class Scenario(str, Enum):
    """Scenario identifiers."""
    A = "A"  # Rent-Seeking Exchange
    B = "B"  # IAB Pure A2A (with context rot)
    C = "C"  # Alkimi Ledger-Backed


@dataclass
class SimulationEvent:
    """
    Normalized event structure for cross-scenario logging.

    All events across scenarios A/B/C are converted to this format,
    enabling unified analysis and content generation.
    """

    # Core identifiers
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.BID_REQUEST
    scenario: Scenario = Scenario.A

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    simulation_day: int = 0

    # Actor identification
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None  # buyer, seller, exchange

    # Transaction context
    request_id: Optional[str] = None
    deal_id: Optional[str] = None
    campaign_id: Optional[str] = None

    # Event-specific payload
    payload: dict = field(default_factory=dict)

    # Narrative hooks
    narrative_summary: Optional[str] = None  # One-line summary
    narrative_detail: Optional[str] = None   # Multi-line detail

    # Cross-reference for comparison
    correlation_id: Optional[str] = None  # Links same campaign across scenarios

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "scenario": self.scenario.value,
            "timestamp": self.timestamp.isoformat(),
            "simulation_day": self.simulation_day,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "request_id": self.request_id,
            "deal_id": self.deal_id,
            "campaign_id": self.campaign_id,
            "payload": self.payload,
            "narrative_summary": self.narrative_summary,
            "narrative_detail": self.narrative_detail,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationEvent":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            scenario=Scenario(data["scenario"]),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.utcnow()),
            simulation_day=data.get("simulation_day", 0),
            agent_id=data.get("agent_id"),
            agent_type=data.get("agent_type"),
            request_id=data.get("request_id"),
            deal_id=data.get("deal_id"),
            campaign_id=data.get("campaign_id"),
            payload=data.get("payload", {}),
            narrative_summary=data.get("narrative_summary"),
            narrative_detail=data.get("narrative_detail"),
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class EventIndex:
    """
    In-memory index for fast event lookup during narrative generation.

    Maintains indexes by campaign, deal, agent for cross-referencing
    events during comparison analysis.
    """

    # All events in order
    events: list[SimulationEvent] = field(default_factory=list)

    # Indexes
    by_campaign: dict[str, list[SimulationEvent]] = field(default_factory=dict)
    by_deal: dict[str, list[SimulationEvent]] = field(default_factory=dict)
    by_agent: dict[str, list[SimulationEvent]] = field(default_factory=dict)
    by_scenario: dict[Scenario, list[SimulationEvent]] = field(default_factory=dict)
    by_day: dict[int, list[SimulationEvent]] = field(default_factory=dict)
    by_type: dict[EventType, list[SimulationEvent]] = field(default_factory=dict)

    def add(self, event: SimulationEvent) -> None:
        """Add event to all indexes."""
        self.events.append(event)

        # Campaign index
        if event.campaign_id:
            if event.campaign_id not in self.by_campaign:
                self.by_campaign[event.campaign_id] = []
            self.by_campaign[event.campaign_id].append(event)

        # Deal index
        if event.deal_id:
            if event.deal_id not in self.by_deal:
                self.by_deal[event.deal_id] = []
            self.by_deal[event.deal_id].append(event)

        # Agent index
        if event.agent_id:
            if event.agent_id not in self.by_agent:
                self.by_agent[event.agent_id] = []
            self.by_agent[event.agent_id].append(event)

        # Scenario index
        if event.scenario not in self.by_scenario:
            self.by_scenario[event.scenario] = []
        self.by_scenario[event.scenario].append(event)

        # Day index
        if event.simulation_day not in self.by_day:
            self.by_day[event.simulation_day] = []
        self.by_day[event.simulation_day].append(event)

        # Type index
        if event.event_type not in self.by_type:
            self.by_type[event.event_type] = []
        self.by_type[event.event_type].append(event)

    def get_campaign_timeline(self, campaign_id: str) -> list[SimulationEvent]:
        """Get all events for a campaign, sorted by time."""
        events = self.by_campaign.get(campaign_id, [])
        return sorted(events, key=lambda e: e.timestamp)

    def get_deal_events(self, deal_id: str) -> list[SimulationEvent]:
        """Get all events for a specific deal."""
        return self.by_deal.get(deal_id, [])

    def get_scenario_events(
        self,
        scenario: Scenario,
        event_type: Optional[EventType] = None,
    ) -> list[SimulationEvent]:
        """Get events for a scenario, optionally filtered by type."""
        events = self.by_scenario.get(scenario, [])
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events

    def get_day_events(
        self,
        day: int,
        scenario: Optional[Scenario] = None,
    ) -> list[SimulationEvent]:
        """Get all events for a simulation day."""
        events = self.by_day.get(day, [])
        if scenario:
            events = [e for e in events if e.scenario == scenario]
        return events

    def count_by_type(self, scenario: Scenario) -> dict[EventType, int]:
        """Count events by type for a scenario."""
        events = self.by_scenario.get(scenario, [])
        counts = {}
        for event in events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts


# =============================================================================
# Event Factory Functions
# =============================================================================

def create_bid_request_event(
    scenario: Scenario,
    buyer_id: str,
    campaign_id: str,
    request_id: str,
    impressions: int,
    max_cpm: float,
    simulation_day: int,
    correlation_id: Optional[str] = None,
) -> SimulationEvent:
    """Create a bid request event."""
    return SimulationEvent(
        event_type=EventType.BID_REQUEST,
        scenario=scenario,
        simulation_day=simulation_day,
        agent_id=buyer_id,
        agent_type="buyer",
        request_id=request_id,
        campaign_id=campaign_id,
        payload={
            "impressions_requested": impressions,
            "max_cpm": max_cpm,
        },
        narrative_summary=f"{buyer_id} requests {impressions:,} impressions at max ${max_cpm:.2f} CPM",
        correlation_id=correlation_id,
    )


def create_bid_response_event(
    scenario: Scenario,
    seller_id: str,
    request_id: str,
    offered_cpm: float,
    available_impressions: int,
    simulation_day: int,
) -> SimulationEvent:
    """Create a bid response event."""
    return SimulationEvent(
        event_type=EventType.BID_RESPONSE,
        scenario=scenario,
        simulation_day=simulation_day,
        agent_id=seller_id,
        agent_type="seller",
        request_id=request_id,
        payload={
            "offered_cpm": offered_cpm,
            "available_impressions": available_impressions,
        },
        narrative_summary=f"{seller_id} offers {available_impressions:,} impressions at ${offered_cpm:.2f} CPM",
    )


def create_deal_event(
    scenario: Scenario,
    deal_id: str,
    request_id: str,
    buyer_id: str,
    seller_id: str,
    campaign_id: str,
    impressions: int,
    cpm: float,
    total_cost: float,
    exchange_fee: float,
    simulation_day: int,
    ledger_entry_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> SimulationEvent:
    """Create a deal created event."""
    fee_pct = (exchange_fee / total_cost * 100) if total_cost > 0 else 0

    if scenario == Scenario.A:
        narrative = f"Deal {deal_id}: {impressions:,} impressions at ${cpm:.2f} CPM (exchange takes ${exchange_fee:.2f}, {fee_pct:.1f}%)"
    elif scenario == Scenario.B:
        narrative = f"Deal {deal_id}: {impressions:,} impressions at ${cpm:.2f} CPM (direct A2A, no fees)"
    else:
        narrative = f"Deal {deal_id}: {impressions:,} impressions at ${cpm:.2f} CPM (recorded to ledger)"

    return SimulationEvent(
        event_type=EventType.DEAL_CREATED,
        scenario=scenario,
        simulation_day=simulation_day,
        agent_id=buyer_id,
        agent_type="buyer",
        request_id=request_id,
        deal_id=deal_id,
        campaign_id=campaign_id,
        payload={
            "seller_id": seller_id,
            "impressions": impressions,
            "cpm": cpm,
            "total_cost": total_cost,
            "exchange_fee": exchange_fee,
            "fee_percentage": fee_pct,
            "seller_revenue": total_cost - exchange_fee,
            "ledger_entry_id": ledger_entry_id,
        },
        narrative_summary=narrative,
        correlation_id=correlation_id,
    )


def create_context_rot_event(
    agent_id: str,
    keys_lost: int,
    survival_rate: float,
    simulation_day: int,
    is_restart: bool = False,
) -> SimulationEvent:
    """Create a context rot or restart event."""
    if is_restart:
        return SimulationEvent(
            event_type=EventType.CONTEXT_RESTART,
            scenario=Scenario.B,
            simulation_day=simulation_day,
            agent_id=agent_id,
            agent_type="agent",
            payload={
                "keys_lost": keys_lost,
                "survival_rate": 0.0,
                "total_context_wiped": True,
            },
            narrative_summary=f"{agent_id} RESTARTED - all context wiped ({keys_lost} keys lost)",
            narrative_detail=f"Agent {agent_id} experienced a simulated restart on day {simulation_day}. "
                           f"All in-memory context was lost, including deal history, partner preferences, "
                           f"and negotiation patterns. In the A2A model, this data cannot be recovered.",
        )
    else:
        return SimulationEvent(
            event_type=EventType.CONTEXT_DECAY,
            scenario=Scenario.B,
            simulation_day=simulation_day,
            agent_id=agent_id,
            agent_type="agent",
            payload={
                "keys_lost": keys_lost,
                "survival_rate": survival_rate,
            },
            narrative_summary=f"{agent_id} lost {keys_lost} memory keys (survival rate: {survival_rate:.1%})",
        )


def create_hallucination_event(
    scenario: Scenario,
    agent_id: str,
    agent_type: str,
    claim_type: str,
    claimed_value: Any,
    actual_value: Any,
    simulation_day: int,
    impact_description: Optional[str] = None,
) -> SimulationEvent:
    """Create a hallucination detection event."""
    return SimulationEvent(
        event_type=EventType.HALLUCINATION_DETECTED,
        scenario=scenario,
        simulation_day=simulation_day,
        agent_id=agent_id,
        agent_type=agent_type,
        payload={
            "claim_type": claim_type,
            "claimed_value": str(claimed_value),
            "actual_value": str(actual_value),
            "impact": impact_description,
        },
        narrative_summary=f"HALLUCINATION: {agent_id} claimed {claim_type}={claimed_value}, actual={actual_value}",
        narrative_detail=f"Agent {agent_id} ({agent_type}) made a decision based on fabricated data. "
                        f"Claimed {claim_type} was {claimed_value}, but ground truth shows {actual_value}. "
                        f"{impact_description or ''}",
    )


def create_state_recovery_event(
    agent_id: str,
    agent_type: str,
    simulation_day: int,
    records_recovered: int,
    recovery_accuracy: float,
    source: str = "ledger",
) -> SimulationEvent:
    """Create a state recovery event (Scenario C)."""
    return SimulationEvent(
        event_type=EventType.STATE_RECOVERY_COMPLETED,
        scenario=Scenario.C,
        simulation_day=simulation_day,
        agent_id=agent_id,
        agent_type=agent_type,
        payload={
            "records_recovered": records_recovered,
            "recovery_accuracy": recovery_accuracy,
            "source": source,
        },
        narrative_summary=f"{agent_id} recovered {records_recovered} records from {source} ({recovery_accuracy:.1%} accuracy)",
        narrative_detail=f"Agent {agent_id} successfully recovered state from the immutable ledger. "
                        f"{records_recovered} transaction records were restored with {recovery_accuracy:.1%} fidelity. "
                        f"Unlike Scenario B, no data was permanently lost.",
    )


def create_fee_extraction_event(
    deal_id: str,
    buyer_id: str,
    seller_id: str,
    gross_amount: float,
    fee_amount: float,
    fee_percentage: float,
    simulation_day: int,
) -> SimulationEvent:
    """Create a fee extraction event (Scenario A)."""
    return SimulationEvent(
        event_type=EventType.FEE_EXTRACTED,
        scenario=Scenario.A,
        simulation_day=simulation_day,
        agent_id="exchange-001",
        agent_type="exchange",
        deal_id=deal_id,
        payload={
            "buyer_id": buyer_id,
            "seller_id": seller_id,
            "gross_amount": gross_amount,
            "fee_amount": fee_amount,
            "fee_percentage": fee_percentage,
            "net_to_seller": gross_amount - fee_amount,
        },
        narrative_summary=f"Exchange extracted ${fee_amount:.2f} ({fee_percentage:.1f}%) from deal {deal_id}",
    )


def create_blockchain_cost_event(
    transaction_id: str,
    transaction_type: str,
    payload_bytes: int,
    sui_gas: float,
    walrus_cost: float,
    total_usd: float,
    simulation_day: int,
) -> SimulationEvent:
    """Create a blockchain cost event (Scenario C)."""
    return SimulationEvent(
        event_type=EventType.BLOCKCHAIN_COST,
        scenario=Scenario.C,
        simulation_day=simulation_day,
        agent_id="ledger",
        agent_type="infrastructure",
        payload={
            "transaction_id": transaction_id,
            "transaction_type": transaction_type,
            "payload_bytes": payload_bytes,
            "sui_gas": sui_gas,
            "walrus_cost": walrus_cost,
            "total_usd": total_usd,
        },
        narrative_summary=f"Ledger: {transaction_type} ({payload_bytes} bytes) cost ${total_usd:.4f}",
    )


def create_day_summary_event(
    scenario: Scenario,
    simulation_day: int,
    deals_count: int,
    total_spend: float,
    total_fees: float,
    context_losses: int = 0,
    hallucinations: int = 0,
    blockchain_costs: float = 0.0,
) -> SimulationEvent:
    """Create a day summary event."""
    payload = {
        "deals_count": deals_count,
        "total_spend": total_spend,
        "total_fees": total_fees,
        "context_losses": context_losses,
        "hallucinations": hallucinations,
    }

    if scenario == Scenario.A:
        fee_rate = (total_fees / total_spend * 100) if total_spend > 0 else 0
        narrative = f"Day {simulation_day}: {deals_count} deals, ${total_spend:,.2f} spend, ${total_fees:,.2f} fees ({fee_rate:.1f}%)"
    elif scenario == Scenario.B:
        narrative = f"Day {simulation_day}: {deals_count} deals, ${total_spend:,.2f} spend, {context_losses} context losses, {hallucinations} hallucinations"
    else:
        payload["blockchain_costs"] = blockchain_costs
        narrative = f"Day {simulation_day}: {deals_count} deals, ${total_spend:,.2f} spend, ${blockchain_costs:.2f} ledger costs"

    return SimulationEvent(
        event_type=EventType.DAY_COMPLETED,
        scenario=scenario,
        simulation_day=simulation_day,
        payload=payload,
        narrative_summary=narrative,
    )
