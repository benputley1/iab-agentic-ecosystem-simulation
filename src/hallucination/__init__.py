"""
Hallucination detection and classification for agent decisions.

This module provides tools to detect when AI agents make decisions
based on hallucinated or misremembered information during ad bidding
simulations.

Usage:
    from hallucination import HallucinationClassifier, HallucinationType
    from hallucination.types import AgentDecision, Hallucination
    
    classifier = HallucinationClassifier(ground_truth_db)
    result = classifier.check_decision(decision)
    
    if result.has_hallucinations:
        for error in result.errors:
            print(f"{error.type.value}: severity {error.severity:.2f}")
"""

from .classifier import (
    ClassifierStats,
    HallucinationClassifier,
    SeverityThresholds,
)
from .types import (
    AgentDecision,
    CampaignState,
    GroundTruthDBProtocol,
    Hallucination,
    HallucinationResult,
    HallucinationType,
)

__all__ = [
    # Enums
    "HallucinationType",
    # Dataclasses
    "AgentDecision",
    "CampaignState",
    "Hallucination",
    "HallucinationResult",
    "SeverityThresholds",
    "ClassifierStats",
    # Protocols
    "GroundTruthDBProtocol",
    # Classes
    "HallucinationClassifier",
]
