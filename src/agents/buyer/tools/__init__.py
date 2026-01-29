"""Simulation tools for buyer agents.

These tools replace the IAB buyer-agent's OpenDirect tools with
Redis Streams-based communication for the RTB simulation.
"""

from .sim_client import SimulationClient
from .sim_tools import (
    SimDiscoverInventoryTool,
    SimRequestDealTool,
    SimCheckAvailsTool,
)

__all__ = [
    "SimulationClient",
    "SimDiscoverInventoryTool",
    "SimRequestDealTool",
    "SimCheckAvailsTool",
]
