"""
Orchestration layer for RTB simulation.

This package provides coordination between the simulation engine,
Gastown orchestration, and the scenario-specific agent implementations.

Modules:
- convoy_sync: Campaign-to-Convoy mapping for Gastown integration
"""

from .convoy_sync import (
    Convoy,
    ConvoyAgent,
    ConvoyRegistry,
    ConvoyState,
    ConvoyStatus,
    ConvoySyncManager,
    get_convoy_manager,
    reset_convoy_manager,
)

__all__ = [
    # Convoy types
    "Convoy",
    "ConvoyAgent",
    "ConvoyState",
    "ConvoyStatus",
    # Management
    "ConvoyRegistry",
    "ConvoySyncManager",
    # Singleton access
    "get_convoy_manager",
    "reset_convoy_manager",
]
