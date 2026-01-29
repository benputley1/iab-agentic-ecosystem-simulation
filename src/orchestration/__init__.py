"""
Orchestration layer for RTB simulation.

This package provides coordination between the simulation engine,
Gastown orchestration, and the scenario-specific agent implementations.

Modules:
- convoy_sync: Campaign-to-Convoy mapping for Gastown integration
- logger: OrchestrationLogger for comprehensive event/narrative logging
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

from .logger import (
    OrchestrationLogger,
    OrchLoggerConfig,
    get_orchestration_logger,
    shutdown_orchestration_logger,
)


__all__ = [
    # Convoy types
    "Convoy",
    "ConvoyAgent",
    "ConvoyState",
    "ConvoyStatus",
    # Convoy management
    "ConvoyRegistry",
    "ConvoySyncManager",
    "get_convoy_manager",
    "reset_convoy_manager",
    # Logging
    "OrchestrationLogger",
    "OrchLoggerConfig",
    "get_orchestration_logger",
    "shutdown_orchestration_logger",
]
