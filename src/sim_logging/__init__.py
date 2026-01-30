"""
Orchestration logging module for RTB simulation.

Provides comprehensive event logging and narrative generation
for content-ready output across all simulation scenarios.

Components:
- events: Event models and factory functions
- narratives: Narrative generation engine
- comparison: Cross-scenario comparison analyzer
- writer: File writers for events and narratives
"""

from .events import (
    EventType,
    Scenario,
    SimulationEvent,
    EventIndex,
    # Factory functions
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

from .narratives import (
    NarrativeEngine,
    CampaignNarrative,
    DayNarrative,
    ScenarioNarrative,
)

from .comparison import (
    ComparisonAnalyzer,
    ComparisonReport,
    ComparisonInsight,
    ScenarioComparison,
)

from .writer import (
    WriterConfig,
    EventWriter,
    NarrativeWriter,
    OrchestrationLogWriter,
)


__all__ = [
    # Event types
    "EventType",
    "Scenario",
    "SimulationEvent",
    "EventIndex",
    # Event factories
    "create_bid_request_event",
    "create_bid_response_event",
    "create_deal_event",
    "create_context_rot_event",
    "create_hallucination_event",
    "create_state_recovery_event",
    "create_fee_extraction_event",
    "create_blockchain_cost_event",
    "create_day_summary_event",
    # Narrative generation
    "NarrativeEngine",
    "CampaignNarrative",
    "DayNarrative",
    "ScenarioNarrative",
    # Comparison analysis
    "ComparisonAnalyzer",
    "ComparisonReport",
    "ComparisonInsight",
    "ScenarioComparison",
    # Writers
    "WriterConfig",
    "EventWriter",
    "NarrativeWriter",
    "OrchestrationLogWriter",
]
