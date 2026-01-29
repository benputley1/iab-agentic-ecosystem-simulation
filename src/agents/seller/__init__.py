"""
Seller Agent - Wraps IAB seller-agent for RTB simulation.

This module provides:
- SellerAgentAdapter: Main adapter for running seller agents
- SimulatedInventory: Synthetic inventory generator
- Product: Product model for inventory items

Usage:
    from agents.seller import SellerAgentAdapter

    async with SellerAgentAdapter(seller_id="pub-001") as adapter:
        await adapter.run()
"""

from .adapter import SellerAgentAdapter, run_seller_agent
from .inventory import SimulatedInventory, Product, InventoryType, DealType

__all__ = [
    "SellerAgentAdapter",
    "run_seller_agent",
    "SimulatedInventory",
    "Product",
    "InventoryType",
    "DealType",
]
