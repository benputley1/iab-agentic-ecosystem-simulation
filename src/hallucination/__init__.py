"""
Hallucination detection and classification for RTB simulation.

This module provides tools to detect, classify, and track hallucinations
in agent decisions - errors caused by context window limitations,
memory loss, or other AI reliability issues.
"""

from .classifier import (
    HallucinationType,
    Hallucination,
    HallucinationResult,
    HallucinationClassifier,
    AgentDecisionForCheck,
    SeverityThresholds,
)

# Re-export types from types.py for backwards compatibility
from .types import (
    AgentDecision,
    CampaignState,
    GroundTruthDBProtocol,
)

__all__ = [
    # From classifier.py
    "HallucinationType",
    "Hallucination",
    "HallucinationResult",
    "HallucinationClassifier",
    "AgentDecisionForCheck",
    "SeverityThresholds",
    # From types.py (backwards compat)
    "AgentDecision",
    "CampaignState",
    "GroundTruthDBProtocol",
]
