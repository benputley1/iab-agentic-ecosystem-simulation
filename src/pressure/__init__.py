"""
Token Pressure Engine - Context window pressure simulation.

Simulates the effects of context window limitations on agent decisions,
tracking token accumulation, overflow events, and compression with information loss.
"""

from .token_tracker import (
    TokenPressureEngine,
    TokenPressureResult,
    CompressionEvent,
    AgentTokenState,
)

__all__ = [
    "TokenPressureEngine",
    "TokenPressureResult", 
    "CompressionEvent",
    "AgentTokenState",
]
