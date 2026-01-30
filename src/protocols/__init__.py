"""
IAB Simulation Protocol Handlers.

This module provides inter-agent communication protocols:
- A2A: Agent-to-Agent natural language communication
- UCP: User Context Protocol for audience embeddings
- AA: Agentic Audience protocol for targeting
- InterLevel: Hierarchy-level communication (L1 <-> L2 <-> L3)
"""

from .a2a import (
    A2AProtocol,
    A2AMessage,
    A2AResponse,
    Offer,
    NegotiationResult,
)
from .ucp import (
    UCPProtocol,
    UCPEmbedding,
    AudienceSpec,
    Inventory,
    MatchResult,
)
from .aa import (
    AAProtocol,
    AudienceSegment,
    CampaignBrief,
    ValidationResult,
    ReachEstimate,
)
from .inter_level import (
    InterLevelProtocol,
    ContextSerializer,
    AgentContext,
    Task,
    Result,
    DelegationResult,
    AckResult,
)

__all__ = [
    # A2A
    "A2AProtocol",
    "A2AMessage",
    "A2AResponse",
    "Offer",
    "NegotiationResult",
    # UCP
    "UCPProtocol",
    "UCPEmbedding",
    "AudienceSpec",
    "Inventory",
    "MatchResult",
    # AA
    "AAProtocol",
    "AudienceSegment",
    "CampaignBrief",
    "ValidationResult",
    "ReachEstimate",
    # Inter-Level
    "InterLevelProtocol",
    "ContextSerializer",
    "AgentContext",
    "Task",
    "Result",
    "DelegationResult",
    "AckResult",
]
