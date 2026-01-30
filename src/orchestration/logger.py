"""
Orchestration Logger for RTB Simulation.

Central logging component that aggregates events from all scenarios,
generates narratives, and produces content-ready outputs.

This is the main entry point for the orch-logging system.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import structlog

from sim_logging.events import (
    SimulationEvent,
    EventType,
    EventIndex,
    Scenario,
    create_bid_request_event,
    create_bid_response_event,
    create_deal_event,
    create_context_rot_event,
    create_hallucination_event,
    create_state_recovery_event,
    create_fee_extraction_event,
    create_blockchain_cost_event,
    create_day_summary_event,
)
from sim_logging.narratives import NarrativeEngine
from sim_logging.comparison import ComparisonAnalyzer, ComparisonReport
from sim_logging.writer import (
    WriterConfig,
    EventWriter,
    NarrativeWriter,
    OrchestrationLogWriter,
)
from infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
)


logger = structlog.get_logger()


@dataclass
class OrchLoggerConfig:
    """Configuration for the orchestration logger."""

    # Output settings
    log_dir: Path = Path("logs")
    write_events: bool = True
    write_narratives: bool = True

    # Real-time callbacks
    enable_callbacks: bool = False

    # Content generation
    generate_daily_narratives: bool = True
    generate_campaign_narratives: bool = True
    generate_comparison_report: bool = True

    # Correlation
    auto_correlate_campaigns: bool = True


class OrchestrationLogger:
    """
    Central orchestration logger for RTB simulation.

    Collects events from all scenarios, maintains indexes,
    and generates human-readable narratives for content generation.

    Usage:
        async with OrchestrationLogger() as orch_logger:
            # Log events from scenarios
            orch_logger.log_bid_request(scenario, request)
            orch_logger.log_deal(scenario, deal)

            # At end of simulation
            report = orch_logger.generate_report()
    """

    def __init__(
        self,
        config: Optional[OrchLoggerConfig] = None,
    ):
        """
        Initialize orchestration logger.

        Args:
            config: Logger configuration
        """
        self.config = config or OrchLoggerConfig()

        # Core components
        self._index = EventIndex()
        self._writer: Optional[OrchestrationLogWriter] = None

        # Tracking
        self._current_day: dict[Scenario, int] = {}
        self._day_metrics: dict[tuple[Scenario, int], dict] = {}
        self._campaign_correlations: dict[str, str] = {}  # campaign_id -> correlation_id

        # Callbacks for real-time notifications
        self._callbacks: dict[EventType, list[Callable[[SimulationEvent], None]]] = {}

        # State
        self._started = False
        self._start_time: Optional[datetime] = None

    async def start(self) -> None:
        """Start the orchestration logger."""
        if self._started:
            return

        logger.info("orch_logger.starting", log_dir=str(self.config.log_dir))

        # Initialize writer
        writer_config = WriterConfig(base_dir=self.config.log_dir)
        self._writer = OrchestrationLogWriter(config=writer_config)
        self._writer.start()

        self._start_time = datetime.utcnow()
        self._started = True

        logger.info("orch_logger.started")

    async def stop(self) -> dict[str, Path]:
        """
        Stop the orchestration logger and generate final outputs.

        Returns:
            Dictionary mapping output type to file path
        """
        if not self._started:
            return {}

        logger.info("orch_logger.stopping", events_logged=len(self._index.events))

        outputs = {}

        # Generate narratives if configured
        if self.config.write_narratives:
            try:
                outputs = self._writer.generate_narratives()
                logger.info("orch_logger.narratives_generated", files=list(outputs.keys()))
            except Exception as e:
                logger.error("orch_logger.narrative_generation_failed", error=str(e))

        # Stop writer
        self._writer.stop()

        self._started = False

        logger.info("orch_logger.stopped", outputs=list(outputs.values()))

        return outputs

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    # =========================================================================
    # Event Logging Methods
    # =========================================================================

    def log_event(self, event: SimulationEvent) -> None:
        """
        Log a simulation event.

        Args:
            event: Event to log
        """
        # Add to index
        self._index.add(event)

        # Write to file
        if self._writer and self.config.write_events:
            self._writer.log_event(event)

        # Trigger callbacks
        if self.config.enable_callbacks:
            self._trigger_callbacks(event)

        # Update day tracking
        self._update_day_metrics(event)

    def log_bid_request(
        self,
        scenario: Scenario,
        request: BidRequest,
        simulation_day: int,
    ) -> SimulationEvent:
        """
        Log a bid request event.

        Args:
            scenario: Scenario identifier
            request: Bid request message
            simulation_day: Current simulation day

        Returns:
            Created event
        """
        correlation_id = self._get_correlation_id(request.campaign_id)

        event = create_bid_request_event(
            scenario=scenario,
            buyer_id=request.buyer_id,
            campaign_id=request.campaign_id,
            request_id=request.request_id,
            impressions=request.impressions_requested,
            max_cpm=request.max_cpm,
            simulation_day=simulation_day,
            correlation_id=correlation_id,
        )

        self.log_event(event)
        return event

    def log_bid_response(
        self,
        scenario: Scenario,
        response: BidResponse,
        simulation_day: int,
    ) -> SimulationEvent:
        """
        Log a bid response event.

        Args:
            scenario: Scenario identifier
            response: Bid response message
            simulation_day: Current simulation day

        Returns:
            Created event
        """
        event = create_bid_response_event(
            scenario=scenario,
            seller_id=response.seller_id,
            request_id=response.request_id,
            offered_cpm=response.offered_cpm,
            available_impressions=response.available_impressions,
            simulation_day=simulation_day,
        )

        self.log_event(event)
        return event

    def log_deal(
        self,
        scenario: Scenario,
        deal: DealConfirmation,
        simulation_day: int,
        campaign_id: Optional[str] = None,
    ) -> SimulationEvent:
        """
        Log a deal creation event.

        Args:
            scenario: Scenario identifier
            deal: Deal confirmation
            simulation_day: Current simulation day
            campaign_id: Campaign ID (if not in deal)

        Returns:
            Created event
        """
        cid = campaign_id or deal.request_id  # Use request_id as fallback
        correlation_id = self._get_correlation_id(cid)

        event = create_deal_event(
            scenario=scenario,
            deal_id=deal.deal_id,
            request_id=deal.request_id,
            buyer_id=deal.buyer_id,
            seller_id=deal.seller_id,
            campaign_id=cid,
            impressions=deal.impressions,
            cpm=deal.cpm,
            total_cost=deal.total_cost,
            exchange_fee=deal.exchange_fee,
            simulation_day=simulation_day,
            ledger_entry_id=deal.ledger_entry_id,
            correlation_id=correlation_id,
        )

        self.log_event(event)
        return event

    def log_context_rot(
        self,
        agent_id: str,
        keys_lost: int,
        survival_rate: float,
        simulation_day: int,
        is_restart: bool = False,
    ) -> SimulationEvent:
        """
        Log a context rot event (Scenario B only).

        Args:
            agent_id: Agent that lost context
            keys_lost: Number of memory keys lost
            survival_rate: Percentage of context surviving
            simulation_day: Current simulation day
            is_restart: Whether this was a full restart

        Returns:
            Created event
        """
        event = create_context_rot_event(
            agent_id=agent_id,
            keys_lost=keys_lost,
            survival_rate=survival_rate,
            simulation_day=simulation_day,
            is_restart=is_restart,
        )

        self.log_event(event)
        return event

    def log_hallucination(
        self,
        scenario: Scenario,
        agent_id: str,
        agent_type: str,
        claim_type: str,
        claimed_value: Any,
        actual_value: Any,
        simulation_day: int,
        impact_description: Optional[str] = None,
    ) -> SimulationEvent:
        """
        Log a hallucination detection event.

        Args:
            scenario: Scenario identifier
            agent_id: Agent that hallucinated
            agent_type: Type of agent
            claim_type: What was claimed incorrectly
            claimed_value: The hallucinated value
            actual_value: Ground truth value
            simulation_day: Current simulation day
            impact_description: Description of impact

        Returns:
            Created event
        """
        event = create_hallucination_event(
            scenario=scenario,
            agent_id=agent_id,
            agent_type=agent_type,
            claim_type=claim_type,
            claimed_value=claimed_value,
            actual_value=actual_value,
            simulation_day=simulation_day,
            impact_description=impact_description,
        )

        self.log_event(event)
        return event

    def log_state_recovery(
        self,
        agent_id: str,
        agent_type: str,
        simulation_day: int,
        records_recovered: int,
        recovery_accuracy: float = 1.0,
    ) -> SimulationEvent:
        """
        Log a state recovery event (Scenario C).

        Args:
            agent_id: Agent recovering state
            agent_type: Type of agent
            simulation_day: Current simulation day
            records_recovered: Number of records recovered
            recovery_accuracy: Fidelity of recovery (0.0-1.0)

        Returns:
            Created event
        """
        event = create_state_recovery_event(
            agent_id=agent_id,
            agent_type=agent_type,
            simulation_day=simulation_day,
            records_recovered=records_recovered,
            recovery_accuracy=recovery_accuracy,
        )

        self.log_event(event)
        return event

    def log_fee_extraction(
        self,
        deal_id: str,
        buyer_id: str,
        seller_id: str,
        gross_amount: float,
        fee_amount: float,
        simulation_day: int,
    ) -> SimulationEvent:
        """
        Log a fee extraction event (Scenario A).

        Args:
            deal_id: Deal identifier
            buyer_id: Buyer in the deal
            seller_id: Seller in the deal
            gross_amount: Total deal amount
            fee_amount: Fee extracted
            simulation_day: Current simulation day

        Returns:
            Created event
        """
        fee_percentage = (fee_amount / gross_amount * 100) if gross_amount > 0 else 0

        event = create_fee_extraction_event(
            deal_id=deal_id,
            buyer_id=buyer_id,
            seller_id=seller_id,
            gross_amount=gross_amount,
            fee_amount=fee_amount,
            fee_percentage=fee_percentage,
            simulation_day=simulation_day,
        )

        self.log_event(event)
        return event

    def log_blockchain_cost(
        self,
        transaction_id: str,
        transaction_type: str,
        payload_bytes: int,
        sui_gas: float,
        walrus_cost: float,
        total_usd: float,
        simulation_day: int,
    ) -> SimulationEvent:
        """
        Log a blockchain cost event (Scenario C).

        Args:
            transaction_id: Ledger transaction ID
            transaction_type: Type of transaction
            payload_bytes: Size of payload
            sui_gas: Sui gas cost
            walrus_cost: Walrus storage cost
            total_usd: Total USD cost
            simulation_day: Current simulation day

        Returns:
            Created event
        """
        event = create_blockchain_cost_event(
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            payload_bytes=payload_bytes,
            sui_gas=sui_gas,
            walrus_cost=walrus_cost,
            total_usd=total_usd,
            simulation_day=simulation_day,
        )

        self.log_event(event)
        return event

    def log_day_complete(
        self,
        scenario: Scenario,
        simulation_day: int,
    ) -> SimulationEvent:
        """
        Log day completion and generate day summary.

        Args:
            scenario: Scenario identifier
            simulation_day: Completed day number

        Returns:
            Day summary event
        """
        # Aggregate day metrics
        metrics = self._get_day_metrics(scenario, simulation_day)

        event = create_day_summary_event(
            scenario=scenario,
            simulation_day=simulation_day,
            deals_count=metrics.get("deals", 0),
            total_spend=metrics.get("spend", 0.0),
            total_fees=metrics.get("fees", 0.0),
            context_losses=metrics.get("context_losses", 0),
            hallucinations=metrics.get("hallucinations", 0),
            blockchain_costs=metrics.get("blockchain_costs", 0.0),
        )

        self.log_event(event)

        # Clear day metrics
        self._clear_day_metrics(scenario, simulation_day)

        return event

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(self) -> ComparisonReport:
        """
        Generate comprehensive comparison report.

        Returns:
            Full comparison report across all scenarios
        """
        analyzer = ComparisonAnalyzer(self._index)
        return analyzer.generate_comparison_report()

    def generate_scenario_narrative(self, scenario: Scenario) -> str:
        """
        Generate narrative for a specific scenario.

        Args:
            scenario: Scenario to generate narrative for

        Returns:
            Markdown-formatted narrative
        """
        engine = NarrativeEngine(self._index)
        narrative = engine.generate_scenario_narrative(scenario)
        return narrative.executive_summary

    def get_daily_comparison(self, day: int) -> dict:
        """
        Get comparison metrics for a specific day.

        Args:
            day: Simulation day

        Returns:
            Comparison data for the day
        """
        analyzer = ComparisonAnalyzer(self._index)
        return analyzer.generate_daily_comparison(day)

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_event(
        self,
        event_type: EventType,
        callback: Callable[[SimulationEvent], None],
    ) -> None:
        """
        Register callback for event type.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def _trigger_callbacks(self, event: SimulationEvent) -> None:
        """Trigger registered callbacks for event."""
        callbacks = self._callbacks.get(event.event_type, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(
                    "orch_logger.callback_error",
                    event_type=event.event_type.value,
                    error=str(e),
                )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_correlation_id(self, campaign_id: str) -> str:
        """Get or create correlation ID for campaign."""
        if not self.config.auto_correlate_campaigns:
            return campaign_id

        if campaign_id not in self._campaign_correlations:
            # Use campaign ID as correlation ID
            # In a real scenario, you'd map this to a shared ID across scenarios
            self._campaign_correlations[campaign_id] = campaign_id

        return self._campaign_correlations[campaign_id]

    def _update_day_metrics(self, event: SimulationEvent) -> None:
        """Update day metrics based on event."""
        key = (event.scenario, event.simulation_day)

        if key not in self._day_metrics:
            self._day_metrics[key] = {
                "deals": 0,
                "spend": 0.0,
                "fees": 0.0,
                "context_losses": 0,
                "hallucinations": 0,
                "blockchain_costs": 0.0,
            }

        metrics = self._day_metrics[key]

        if event.event_type == EventType.DEAL_CREATED:
            metrics["deals"] += 1
            metrics["spend"] += event.payload.get("total_cost", 0)
            metrics["fees"] += event.payload.get("exchange_fee", 0)

        elif event.event_type in (EventType.CONTEXT_DECAY, EventType.CONTEXT_RESTART):
            metrics["context_losses"] += 1

        elif event.event_type == EventType.HALLUCINATION_DETECTED:
            metrics["hallucinations"] += 1

        elif event.event_type == EventType.BLOCKCHAIN_COST:
            metrics["blockchain_costs"] += event.payload.get("total_usd", 0)

    def _get_day_metrics(self, scenario: Scenario, day: int) -> dict:
        """Get accumulated metrics for a day."""
        key = (scenario, day)
        return self._day_metrics.get(key, {
            "deals": 0,
            "spend": 0.0,
            "fees": 0.0,
            "context_losses": 0,
            "hallucinations": 0,
            "blockchain_costs": 0.0,
        })

    def _clear_day_metrics(self, scenario: Scenario, day: int) -> None:
        """Clear metrics for completed day."""
        key = (scenario, day)
        if key in self._day_metrics:
            del self._day_metrics[key]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_event_count(self) -> int:
        """Get total number of logged events."""
        return len(self._index.events)

    def get_event_counts_by_scenario(self) -> dict[Scenario, int]:
        """Get event counts by scenario."""
        counts = {}
        for scenario in Scenario:
            counts[scenario] = len(self._index.get_scenario_events(scenario))
        return counts

    def get_event_counts_by_type(self) -> dict[EventType, int]:
        """Get event counts by type."""
        counts = {}
        for event_type in EventType:
            events = self._index.by_type.get(event_type, [])
            if events:
                counts[event_type] = len(events)
        return counts


# =============================================================================
# Convenience Functions
# =============================================================================

_global_logger: Optional[OrchestrationLogger] = None


async def get_orchestration_logger() -> OrchestrationLogger:
    """
    Get the global orchestration logger instance.

    Creates one if it doesn't exist.

    Returns:
        Global orchestration logger
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = OrchestrationLogger()
        await _global_logger.start()
    return _global_logger


async def shutdown_orchestration_logger() -> dict[str, Path]:
    """
    Shutdown the global orchestration logger.

    Returns:
        Dictionary of generated output files
    """
    global _global_logger
    if _global_logger is not None:
        outputs = await _global_logger.stop()
        _global_logger = None
        return outputs
    return {}
