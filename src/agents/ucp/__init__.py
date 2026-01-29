"""
User Context Protocol (UCP) module for RTB Simulation.

This module provides:
- Embedding exchange for audience/targeting context between agents
- Hallucination injection for Scenario B testing
- Hallucination detection against ground truth

Components:
    embeddings: User context embedding creation and exchange
    hallucination: Injection and detection for simulation testing
"""

from .embeddings import (
    ContextType,
    EmbeddingFormat,
    UserContextEmbedding,
    UCPExchange,
    UCPExchangeRequest,
    UCPExchangeResponse,
)

from .hallucination import (
    InjectionType,
    Severity,
    InjectionRecord,
    ClaimVerification,
    HallucinationInjector,
    HallucinationDetector,
    HallucinationManager,
)

__all__ = [
    # Embeddings
    "ContextType",
    "EmbeddingFormat",
    "UserContextEmbedding",
    "UCPExchange",
    "UCPExchangeRequest",
    "UCPExchangeResponse",
    # Hallucination
    "InjectionType",
    "Severity",
    "InjectionRecord",
    "ClaimVerification",
    "HallucinationInjector",
    "HallucinationDetector",
    "HallucinationManager",
]
