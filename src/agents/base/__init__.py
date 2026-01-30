"""
Multi-Agent Hierarchy Base Classes.

This module provides the foundational base classes for the three-level
agent hierarchy used in the IAB Tech Lab advertising simulation:

- L1: OrchestratorAgent (Claude Opus) - Strategic decisions, portfolio management
- L2: SpecialistAgent (Claude Sonnet) - Channel expertise, specialist coordination  
- L3: FunctionalAgent (Claude Sonnet) - Tool execution, MCP integration

The hierarchy implements context passing with measurable degradation to
demonstrate the value of shared ledger state (Scenario C vs Scenario B).
"""

# Context Management
from .context import (
    AgentContext,
    ContextItem,
    ContextPriority,
    ContextWindow,
    ContextPassingProtocol,
    StandardContextPassing,
    ContextDegradationMetrics,
    measure_degradation,
)

# State Management
from .state import (
    AgentState,
    StateVersion,
    StateSnapshot,
    StateBackend,
    StateManager,
    VolatileStateBackend,
    FileStateBackend,
    LedgerStateBackend,
    VolatileState,
    LedgerBackedState,
)

# L3 Functional Agent
from .functional import (
    FunctionalAgent,
    FunctionalAgentState,
    ToolDefinition,
    ToolResult,
    L3_MODEL,
)

# L2 Specialist Agent
from .specialist import (
    SpecialistAgent,
    SpecialistAgentState,
    DelegationRequest,
    DelegationResult,
    L2_MODEL,
    # Shared models
    FitScore,
    Campaign,
    Task,
    Result,
    AvailsRequest,
    AvailsResponse,
    PricingRequest,
    PricingResponse,
)

# L1 Orchestrator Agent
from .orchestrator import (
    OrchestratorAgent,
    OrchestratorAgentState,
    CampaignState,
    StrategicDecision,
    SpecialistAssignment,
    AssignmentResult,
    L1_MODEL,
)

__all__ = [
    # Context
    "AgentContext",
    "ContextItem",
    "ContextPriority",
    "ContextWindow",
    "ContextPassingProtocol",
    "StandardContextPassing",
    "ContextDegradationMetrics",
    "measure_degradation",
    # State
    "AgentState",
    "StateVersion",
    "StateSnapshot",
    "StateBackend",
    "StateManager",
    "VolatileStateBackend",
    "FileStateBackend",
    "LedgerStateBackend",
    "VolatileState",
    "LedgerBackedState",
    # L3
    "FunctionalAgent",
    "FunctionalAgentState",
    "ToolDefinition",
    "ToolResult",
    "L3_MODEL",
    # L2
    "SpecialistAgent",
    "SpecialistAgentState",
    "DelegationRequest",
    "DelegationResult",
    "L2_MODEL",
    # Shared models
    "FitScore",
    "Campaign",
    "Task",
    "Result",
    "AvailsRequest",
    "AvailsResponse",
    "PricingRequest",
    "PricingResponse",
    # L1
    "OrchestratorAgent",
    "OrchestratorAgentState",
    "CampaignState",
    "StrategicDecision",
    "SpecialistAssignment",
    "AssignmentResult",
    "L1_MODEL",
]
