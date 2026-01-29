"""
Campaign-to-Convoy synchronization for Gastown orchestration.

This module maps RTB simulation campaigns to Gastown convoys, enabling
parallel execution of campaign bidding cycles with coordinated state
management across buyer, seller, and exchange agents.

A convoy represents a collection of agents collaborating on a single
campaign execution - buyers competing for inventory, sellers responding
with offers, and (in Scenario A) exchanges mediating transactions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING
import asyncio
import structlog
import uuid

from ..infrastructure.message_schemas import DealConfirmation

# Type checking imports - these won't be loaded at runtime
if TYPE_CHECKING:
    from ..agents.buyer.wrapper import Campaign, BuyerAgentWrapper
    from ..agents.seller.adapter import SellerAgentAdapter


class CampaignProtocol(Protocol):
    """Protocol for campaign-like objects (duck typing)."""

    campaign_id: str
    name: str
    budget: float
    target_impressions: int
    target_cpm: float
    impressions_delivered: int
    spend: float

    @property
    def remaining_budget(self) -> float: ...

    @property
    def remaining_impressions(self) -> int: ...

    @property
    def is_active(self) -> bool: ...


logger = structlog.get_logger()


class ConvoyStatus(str, Enum):
    """Status of a convoy in the simulation."""

    PENDING = "pending"  # Convoy created, not yet started
    ACTIVE = "active"  # Convoy is executing bidding cycles
    PAUSED = "paused"  # Temporarily halted (can resume)
    COMPLETED = "completed"  # All campaign goals achieved
    FAILED = "failed"  # Unrecoverable error
    EXHAUSTED = "exhausted"  # Budget or impressions depleted


@dataclass
class ConvoyAgent:
    """An agent assigned to a convoy."""

    agent_id: str
    agent_type: str  # "buyer", "seller", "exchange"
    scenario: str  # "A", "B", "C"
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None
    deals_participated: int = 0

    def record_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


@dataclass
class ConvoyState:
    """State checkpoint for a convoy."""

    checkpoint_id: str
    convoy_id: str
    timestamp: datetime
    campaign_state: dict[str, Any]
    agent_states: dict[str, dict[str, Any]]
    deals: list[DealConfirmation]
    day: int
    iteration: int

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "convoy_id": self.convoy_id,
            "timestamp": self.timestamp.isoformat(),
            "campaign_state": self.campaign_state,
            "agent_states": self.agent_states,
            "deals_count": len(self.deals),
            "day": self.day,
            "iteration": self.iteration,
        }


@dataclass
class Convoy:
    """
    A Gastown convoy executing a single campaign.

    Convoys coordinate the execution of advertising campaigns across
    multiple agents (buyers, sellers, exchange). Each convoy tracks:
    - The campaign being executed
    - Participating agents
    - Transaction history
    - Sync points for coordination
    """

    convoy_id: str
    campaign: CampaignProtocol
    scenario: str

    # Agents participating in this convoy
    agents: dict[str, ConvoyAgent] = field(default_factory=dict)

    # Status tracking
    status: ConvoyStatus = ConvoyStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Progress tracking
    current_day: int = 0
    current_iteration: int = 0
    total_iterations: int = 0

    # Transaction history
    deals: list[DealConfirmation] = field(default_factory=list)

    # State checkpoints for recovery
    checkpoints: list[ConvoyState] = field(default_factory=list)

    # Sync coordination
    _sync_event: Optional[asyncio.Event] = field(default=None, repr=False)
    _sync_barrier: Optional[asyncio.Barrier] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize synchronization primitives."""
        self._sync_event = asyncio.Event()
        self._sync_event.set()  # Initially not blocked

    @property
    def total_spend(self) -> float:
        """Total buyer spend across all deals."""
        return sum(d.total_cost for d in self.deals)

    @property
    def total_impressions(self) -> int:
        """Total impressions delivered."""
        return sum(d.impressions for d in self.deals)

    @property
    def goal_progress(self) -> float:
        """Progress toward campaign goal (0.0 - 1.0)."""
        if self.campaign.target_impressions == 0:
            return 1.0
        return min(1.0, self.total_impressions / self.campaign.target_impressions)

    def add_agent(
        self,
        agent_id: str,
        agent_type: str,
    ) -> ConvoyAgent:
        """Add an agent to the convoy."""
        agent = ConvoyAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            scenario=self.scenario,
        )
        self.agents[agent_id] = agent

        logger.info(
            "convoy.agent_added",
            convoy_id=self.convoy_id,
            agent_id=agent_id,
            agent_type=agent_type,
            total_agents=len(self.agents),
        )

        return agent

    def remove_agent(self, agent_id: str) -> Optional[ConvoyAgent]:
        """Remove an agent from the convoy."""
        agent = self.agents.pop(agent_id, None)
        if agent:
            logger.info(
                "convoy.agent_removed",
                convoy_id=self.convoy_id,
                agent_id=agent_id,
            )
        return agent

    def record_deal(self, deal: DealConfirmation) -> None:
        """Record a completed deal."""
        self.deals.append(deal)
        self.total_iterations += 1

        # Update participating agents
        buyer_agent = self.agents.get(deal.buyer_id)
        if buyer_agent:
            buyer_agent.record_activity()
            buyer_agent.deals_participated += 1

        seller_agent = self.agents.get(deal.seller_id)
        if seller_agent:
            seller_agent.record_activity()
            seller_agent.deals_participated += 1

        # Update campaign state
        self.campaign.impressions_delivered += deal.impressions
        self.campaign.spend += deal.total_cost

        # Check if campaign is complete
        if not self.campaign.is_active:
            self.status = ConvoyStatus.COMPLETED
            self.completed_at = datetime.utcnow()

        logger.debug(
            "convoy.deal_recorded",
            convoy_id=self.convoy_id,
            deal_id=deal.deal_id,
            impressions=deal.impressions,
            progress=f"{self.goal_progress:.1%}",
        )

    def checkpoint(self) -> ConvoyState:
        """Create a state checkpoint for recovery."""
        checkpoint = ConvoyState(
            checkpoint_id=f"{self.convoy_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            convoy_id=self.convoy_id,
            timestamp=datetime.utcnow(),
            campaign_state={
                "campaign_id": self.campaign.campaign_id,
                "impressions_delivered": self.campaign.impressions_delivered,
                "spend": self.campaign.spend,
                "remaining_budget": self.campaign.remaining_budget,
                "remaining_impressions": self.campaign.remaining_impressions,
            },
            agent_states={
                aid: {
                    "last_activity": a.last_activity.isoformat() if a.last_activity else None,
                    "deals_participated": a.deals_participated,
                }
                for aid, a in self.agents.items()
            },
            deals=self.deals.copy(),
            day=self.current_day,
            iteration=self.current_iteration,
        )

        self.checkpoints.append(checkpoint)

        # Keep only last 5 checkpoints
        if len(self.checkpoints) > 5:
            self.checkpoints = self.checkpoints[-5:]

        logger.debug(
            "convoy.checkpoint_created",
            convoy_id=self.convoy_id,
            checkpoint_id=checkpoint.checkpoint_id,
            total_checkpoints=len(self.checkpoints),
        )

        return checkpoint

    async def sync_point(self) -> None:
        """
        Wait at a synchronization point.

        Used to coordinate agent activity at key boundaries
        (e.g., end of day, before state changes).
        """
        await self._sync_event.wait()

    def pause(self) -> None:
        """Pause the convoy at the next sync point."""
        self._sync_event.clear()
        self.status = ConvoyStatus.PAUSED
        logger.info("convoy.paused", convoy_id=self.convoy_id)

    def resume(self) -> None:
        """Resume a paused convoy."""
        self._sync_event.set()
        self.status = ConvoyStatus.ACTIVE
        logger.info("convoy.resumed", convoy_id=self.convoy_id)

    def to_dict(self) -> dict:
        """Serialize convoy for reporting."""
        return {
            "convoy_id": self.convoy_id,
            "campaign_id": self.campaign.campaign_id,
            "campaign_name": self.campaign.name,
            "scenario": self.scenario,
            "status": self.status.value,
            "agents": {aid: a.agent_type for aid, a in self.agents.items()},
            "deals_count": len(self.deals),
            "total_spend": round(self.total_spend, 2),
            "total_impressions": self.total_impressions,
            "goal_progress": round(self.goal_progress * 100, 1),
            "current_day": self.current_day,
            "created_at": self.created_at.isoformat(),
        }


class ConvoyRegistry:
    """
    Registry mapping campaigns to convoys.

    Provides centralized management of all convoys in the simulation,
    including creation, lookup, and lifecycle management.
    """

    def __init__(self):
        self._convoys: dict[str, Convoy] = {}  # convoy_id -> Convoy
        self._campaign_map: dict[str, str] = {}  # campaign_id -> convoy_id
        self._lock = asyncio.Lock()

        # Event callbacks
        self._on_convoy_complete: list[Callable[[Convoy], None]] = []
        self._on_deal_recorded: list[Callable[[Convoy, DealConfirmation], None]] = []

    @property
    def convoy_count(self) -> int:
        """Total number of convoys."""
        return len(self._convoys)

    @property
    def active_convoys(self) -> list[Convoy]:
        """List of active convoys."""
        return [c for c in self._convoys.values() if c.status == ConvoyStatus.ACTIVE]

    @property
    def completed_convoys(self) -> list[Convoy]:
        """List of completed convoys."""
        return [c for c in self._convoys.values() if c.status == ConvoyStatus.COMPLETED]

    async def create_convoy(
        self,
        campaign: CampaignProtocol,
        scenario: str,
        convoy_id: Optional[str] = None,
    ) -> Convoy:
        """
        Create a new convoy for a campaign.

        Args:
            campaign: Campaign to execute
            scenario: Simulation scenario ("A", "B", "C")
            convoy_id: Optional specific ID (generated if not provided)

        Returns:
            New Convoy instance
        """
        async with self._lock:
            # Check if campaign already has a convoy
            if campaign.campaign_id in self._campaign_map:
                existing_id = self._campaign_map[campaign.campaign_id]
                raise ValueError(
                    f"Campaign {campaign.campaign_id} already mapped to convoy {existing_id}"
                )

            # Generate convoy ID if not provided
            if convoy_id is None:
                convoy_id = f"convoy-{campaign.campaign_id}-{uuid.uuid4().hex[:8]}"

            convoy = Convoy(
                convoy_id=convoy_id,
                campaign=campaign,
                scenario=scenario,
            )

            self._convoys[convoy_id] = convoy
            self._campaign_map[campaign.campaign_id] = convoy_id

            logger.info(
                "convoy.created",
                convoy_id=convoy_id,
                campaign_id=campaign.campaign_id,
                campaign_name=campaign.name,
                scenario=scenario,
            )

            return convoy

    def get_convoy(self, convoy_id: str) -> Optional[Convoy]:
        """Get convoy by ID."""
        return self._convoys.get(convoy_id)

    def get_convoy_for_campaign(self, campaign_id: str) -> Optional[Convoy]:
        """Get convoy for a campaign."""
        convoy_id = self._campaign_map.get(campaign_id)
        if convoy_id:
            return self._convoys.get(convoy_id)
        return None

    async def start_convoy(self, convoy_id: str) -> None:
        """Start convoy execution."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            raise ValueError(f"Convoy {convoy_id} not found")

        if convoy.status != ConvoyStatus.PENDING:
            raise ValueError(f"Convoy {convoy_id} not in PENDING state")

        convoy.status = ConvoyStatus.ACTIVE
        convoy.started_at = datetime.utcnow()

        logger.info(
            "convoy.started",
            convoy_id=convoy_id,
            campaign_id=convoy.campaign.campaign_id,
        )

    async def complete_convoy(
        self,
        convoy_id: str,
        status: ConvoyStatus = ConvoyStatus.COMPLETED,
    ) -> None:
        """Mark convoy as complete."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            raise ValueError(f"Convoy {convoy_id} not found")

        convoy.status = status
        convoy.completed_at = datetime.utcnow()

        # Create final checkpoint
        convoy.checkpoint()

        logger.info(
            "convoy.completed",
            convoy_id=convoy_id,
            status=status.value,
            deals=len(convoy.deals),
            spend=round(convoy.total_spend, 2),
            impressions=convoy.total_impressions,
            goal_progress=f"{convoy.goal_progress:.1%}",
        )

        # Fire completion callbacks
        for callback in self._on_convoy_complete:
            try:
                callback(convoy)
            except Exception as e:
                logger.error("convoy.callback_error", error=str(e))

    def record_deal(
        self,
        convoy_id: str,
        deal: DealConfirmation,
    ) -> None:
        """Record a deal in a convoy."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            raise ValueError(f"Convoy {convoy_id} not found")

        convoy.record_deal(deal)

        # Fire deal callbacks
        for callback in self._on_deal_recorded:
            try:
                callback(convoy, deal)
            except Exception as e:
                logger.error("convoy.callback_error", error=str(e))

    def on_convoy_complete(
        self,
        callback: Callable[[Convoy], None],
    ) -> None:
        """Register callback for convoy completion."""
        self._on_convoy_complete.append(callback)

    def on_deal_recorded(
        self,
        callback: Callable[[Convoy, DealConfirmation], None],
    ) -> None:
        """Register callback for deal recording."""
        self._on_deal_recorded.append(callback)

    def get_metrics(self) -> dict:
        """Get registry-level metrics."""
        active = [c for c in self._convoys.values() if c.status == ConvoyStatus.ACTIVE]
        completed = [c for c in self._convoys.values() if c.status == ConvoyStatus.COMPLETED]
        failed = [c for c in self._convoys.values() if c.status in (ConvoyStatus.FAILED, ConvoyStatus.EXHAUSTED)]

        total_spend = sum(c.total_spend for c in self._convoys.values())
        total_impressions = sum(c.total_impressions for c in self._convoys.values())
        total_deals = sum(len(c.deals) for c in self._convoys.values())

        return {
            "total_convoys": len(self._convoys),
            "active_convoys": len(active),
            "completed_convoys": len(completed),
            "failed_convoys": len(failed),
            "total_deals": total_deals,
            "total_spend": round(total_spend, 2),
            "total_impressions": total_impressions,
            "average_goal_progress": round(
                sum(c.goal_progress for c in self._convoys.values()) / max(1, len(self._convoys)) * 100,
                1,
            ),
        }

    def list_convoys(
        self,
        status: Optional[ConvoyStatus] = None,
        scenario: Optional[str] = None,
    ) -> list[Convoy]:
        """
        List convoys with optional filtering.

        Args:
            status: Filter by convoy status
            scenario: Filter by scenario

        Returns:
            List of matching convoys
        """
        result = list(self._convoys.values())

        if status:
            result = [c for c in result if c.status == status]

        if scenario:
            result = [c for c in result if c.scenario == scenario]

        return result


class ConvoySyncManager:
    """
    Manages campaign-to-convoy synchronization across the simulation.

    Coordinates between the scenario engines and Gastown orchestration,
    ensuring campaigns map correctly to convoys and agents are properly
    assigned and tracked.
    """

    def __init__(
        self,
        registry: Optional[ConvoyRegistry] = None,
    ):
        self.registry = registry or ConvoyRegistry()
        self._scenario_convoys: dict[str, list[str]] = {
            "A": [],
            "B": [],
            "C": [],
        }

    async def sync_campaigns(
        self,
        campaigns: list[CampaignProtocol],
        scenario: str,
        buyers: Optional[list[Any]] = None,
        sellers: Optional[list[Any]] = None,
    ) -> list[Convoy]:
        """
        Synchronize campaigns to convoys.

        Creates convoys for new campaigns and assigns agents.

        Args:
            campaigns: List of campaigns to sync
            scenario: Simulation scenario
            buyers: Optional list of buyer agents to assign
            sellers: Optional list of seller agents to assign

        Returns:
            List of created/updated convoys
        """
        convoys = []

        for campaign in campaigns:
            # Check if convoy already exists
            existing = self.registry.get_convoy_for_campaign(campaign.campaign_id)
            if existing:
                convoys.append(existing)
                continue

            # Create new convoy
            convoy = await self.registry.create_convoy(
                campaign=campaign,
                scenario=scenario,
            )

            # Assign buyers
            if buyers:
                for buyer in buyers:
                    convoy.add_agent(
                        agent_id=buyer.buyer_id,
                        agent_type="buyer",
                    )

            # Assign sellers
            if sellers:
                for seller in sellers:
                    convoy.add_agent(
                        agent_id=seller.seller_id,
                        agent_type="seller",
                    )

            # Track by scenario
            self._scenario_convoys[scenario].append(convoy.convoy_id)
            convoys.append(convoy)

        logger.info(
            "convoy_sync.campaigns_synced",
            scenario=scenario,
            campaigns_count=len(campaigns),
            convoys_created=len([c for c in convoys if c.status == ConvoyStatus.PENDING]),
            convoys_existing=len([c for c in convoys if c.status != ConvoyStatus.PENDING]),
        )

        return convoys

    async def start_scenario_convoys(self, scenario: str) -> None:
        """Start all pending convoys for a scenario."""
        convoy_ids = self._scenario_convoys.get(scenario, [])

        for convoy_id in convoy_ids:
            convoy = self.registry.get_convoy(convoy_id)
            if convoy and convoy.status == ConvoyStatus.PENDING:
                await self.registry.start_convoy(convoy_id)

        logger.info(
            "convoy_sync.scenario_started",
            scenario=scenario,
            convoys_started=len(convoy_ids),
        )

    async def advance_day(
        self,
        scenario: str,
        day: int,
    ) -> None:
        """
        Advance all convoys in a scenario to the next day.

        Creates checkpoints at day boundaries for recovery.

        Args:
            scenario: Scenario to advance
            day: New simulation day
        """
        convoy_ids = self._scenario_convoys.get(scenario, [])

        for convoy_id in convoy_ids:
            convoy = self.registry.get_convoy(convoy_id)
            if convoy and convoy.status == ConvoyStatus.ACTIVE:
                convoy.current_day = day
                convoy.current_iteration = 0

                # Checkpoint at day boundary
                if day > 0 and day % 5 == 0:  # Every 5 days
                    convoy.checkpoint()

        logger.debug(
            "convoy_sync.day_advanced",
            scenario=scenario,
            day=day,
            active_convoys=len([
                self.registry.get_convoy(cid)
                for cid in convoy_ids
                if self.registry.get_convoy(cid) and self.registry.get_convoy(cid).status == ConvoyStatus.ACTIVE
            ]),
        )

    async def complete_scenario(self, scenario: str) -> dict:
        """
        Complete all convoys for a scenario.

        Returns summary metrics.

        Args:
            scenario: Scenario to complete

        Returns:
            Summary metrics for the scenario's convoys
        """
        convoy_ids = self._scenario_convoys.get(scenario, [])

        for convoy_id in convoy_ids:
            convoy = self.registry.get_convoy(convoy_id)
            if convoy and convoy.status == ConvoyStatus.ACTIVE:
                final_status = (
                    ConvoyStatus.COMPLETED
                    if convoy.goal_progress >= 1.0
                    else ConvoyStatus.EXHAUSTED
                )
                await self.registry.complete_convoy(convoy_id, final_status)

        # Collect metrics
        convoys = [
            self.registry.get_convoy(cid)
            for cid in convoy_ids
            if self.registry.get_convoy(cid)
        ]

        completed = [c for c in convoys if c.status == ConvoyStatus.COMPLETED]
        exhausted = [c for c in convoys if c.status == ConvoyStatus.EXHAUSTED]

        metrics = {
            "scenario": scenario,
            "total_convoys": len(convoys),
            "completed_convoys": len(completed),
            "exhausted_convoys": len(exhausted),
            "total_deals": sum(len(c.deals) for c in convoys),
            "total_spend": round(sum(c.total_spend for c in convoys), 2),
            "total_impressions": sum(c.total_impressions for c in convoys),
            "average_goal_progress": round(
                sum(c.goal_progress for c in convoys) / max(1, len(convoys)) * 100,
                1,
            ),
            "success_rate": round(len(completed) / max(1, len(convoys)) * 100, 1),
        }

        logger.info(
            "convoy_sync.scenario_completed",
            **metrics,
        )

        return metrics

    def get_convoy_status(
        self,
        campaign_id: Optional[str] = None,
        convoy_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Get status for a specific convoy.

        Args:
            campaign_id: Look up by campaign ID
            convoy_id: Look up by convoy ID

        Returns:
            Convoy status dictionary or None
        """
        convoy = None

        if convoy_id:
            convoy = self.registry.get_convoy(convoy_id)
        elif campaign_id:
            convoy = self.registry.get_convoy_for_campaign(campaign_id)

        if convoy:
            return convoy.to_dict()
        return None


# -----------------------------------------------------------------------------
# Module-level Singleton
# -----------------------------------------------------------------------------

_default_manager: Optional[ConvoySyncManager] = None


def get_convoy_manager() -> ConvoySyncManager:
    """Get the default convoy sync manager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConvoySyncManager()
    return _default_manager


def reset_convoy_manager() -> None:
    """Reset the default convoy manager (for testing)."""
    global _default_manager
    _default_manager = None
