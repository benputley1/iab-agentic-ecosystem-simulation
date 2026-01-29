"""
RTB Simulation Agents.

Agents wrap IAB Tech Lab reference implementations for use in the simulation.

Available agents:
- seller: Seller/Publisher agents wrapping IAB seller-agent
- buyer: Buyer/DSP agents (TODO)
- exchange: Exchange intermediary agents (TODO)
- ucp: User Context Protocol agents (TODO)
"""

from .seller import SellerAgentAdapter, SimulatedInventory, Product

__all__ = [
    "SellerAgentAdapter",
    "SimulatedInventory",
    "Product",
]
