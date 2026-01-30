"""
IAB A2A Simulation Models.

This package contains data models for agent state tracking,
reconciliation, and campaign execution metrics.
"""

from .agent_state import (
    DealRecord,
    AgentState,
    Discrepancy,
    ReconciliationResult,
    MultiAgentMetrics,
    reconcile_deal,
)

from .campaign_execution import (
    CampaignExecution,
    BatchResult,
    RecallResult,
    PressureThreshold,
    PressureSimulationResult,
    get_pressure_threshold,
    DEFAULT_PRESSURE_THRESHOLDS,
    TOKENS_PER_IMPRESSION,
    CONTEXT_LIMIT,
)

__all__ = [
    # Agent state models
    "DealRecord",
    "AgentState",
    "Discrepancy",
    "ReconciliationResult",
    "MultiAgentMetrics",
    "reconcile_deal",
    # Campaign execution models
    "CampaignExecution",
    "BatchResult",
    "RecallResult",
    "PressureThreshold",
    "PressureSimulationResult",
    "get_pressure_threshold",
    "DEFAULT_PRESSURE_THRESHOLDS",
    "TOKENS_PER_IMPRESSION",
    "CONTEXT_LIMIT",
]
