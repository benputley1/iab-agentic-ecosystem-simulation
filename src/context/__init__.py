"""
Context Flow Management for Multi-Agent Hierarchy.

This module provides advanced context flow management including:
- Inter-level context passing with scenario-aware handling
- Context rot simulation (truncation, decay, restart, handoff)
- Recovery mechanisms for different scenarios
- Comprehensive metrics tracking

Works in conjunction with src/agents/base/context.py which provides
the foundational AgentContext and ContextWindow classes.

Scenarios:
- Scenario A: Exchange mediates, partial recovery via logs (~60%)
- Scenario B: Direct agent communication, no recovery (0%)
- Scenario C: Ledger-backed state, full recovery (100%)
"""

from .flow import (
    ContextFlowManager,
    ContextPassResult,
    AggregatedContext,
    FlowDirection,
)
from .window import (
    ContextWindowManager,
    ContextEntry,
    WindowState,
)
from .rot import (
    ContextRotSimulator,
    DecayResult,
    HandoffResult,
    RotType,
)
from .recovery import (
    ContextRecovery,
    RecoveryResult,
    RecoverySource,
)
from .metrics import (
    ContextMetrics,
    ContextMetricsSummary,
    MetricEvent,
)

__all__ = [
    # Flow Management
    "ContextFlowManager",
    "ContextPassResult",
    "AggregatedContext",
    "FlowDirection",
    # Window Management
    "ContextWindowManager",
    "ContextEntry",
    "WindowState",
    # Rot Simulation
    "ContextRotSimulator",
    "DecayResult",
    "HandoffResult",
    "RotType",
    # Recovery
    "ContextRecovery",
    "RecoveryResult",
    "RecoverySource",
    # Metrics
    "ContextMetrics",
    "ContextMetricsSummary",
    "MetricEvent",
]
