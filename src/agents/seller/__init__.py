"""
Seller Agent - Wraps IAB seller-agent for RTB simulation.

This module provides:
- SellerAgentAdapter: Main adapter for running seller agents
- SimulatedInventory: Synthetic inventory generator
- Product: Product model for inventory items
- InventoryManager: L1 orchestrator for yield optimization and deal decisions
- L2 Channel Inventory Specialists:
  - DisplayInventoryAgent: Banner and rich media
  - VideoInventoryAgent: Video ad inventory
  - CTVInventoryAgent: Connected TV inventory
  - MobileAppInventoryAgent: Mobile app inventory
  - NativeInventoryAgent: Native ad inventory

Usage:
    from agents.seller import SellerAgentAdapter, InventoryManager

    async with SellerAgentAdapter(seller_id="pub-001") as adapter:
        await adapter.run()
    
    # L1 Orchestrator usage
    manager = InventoryManager(seller_id="pub-001", portfolio=portfolio)
    decision = await manager.evaluate_deal_request(request)
    
    # L2 Channel Specialists
    from agents.seller import DisplayInventoryAgent, VideoInventoryAgent
    
    display_agent = DisplayInventoryAgent(seller_id="pub-001")
    avails = await display_agent.check_availability(request)
"""

from .adapter import SellerAgentAdapter, run_seller_agent
from .inventory import SimulatedInventory, Product as LegacyProduct, InventoryType, DealType
from .l1_inventory_manager import InventoryManager, create_inventory_manager

# L2 Channel Inventory Specialists
from .l2_display import DisplayInventoryAgent
from .l2_video import VideoInventoryAgent
from .l2_ctv import CTVInventoryAgent
from .l2_mobile import MobileAppInventoryAgent
from .l2_native import NativeInventoryAgent

from .models import (
    AudienceSpec,
    BuyerTier,
    ChannelType,
    CounterOffer,
    CrossSellOpportunity,
    Deal,
    DealAction,
    DealDecision,
    DealRequest,
    DealTypeEnum,
    InventoryPortfolio,
    Product,
    Task,
    TaskResult,
    YieldStrategy,
)

__all__ = [
    # Adapters
    "SellerAgentAdapter",
    "run_seller_agent",
    # Legacy inventory
    "SimulatedInventory",
    "LegacyProduct",
    "InventoryType",
    "DealType",
    # L1 Orchestrator
    "InventoryManager",
    "create_inventory_manager",
    # L2 Channel Inventory Specialists
    "DisplayInventoryAgent",
    "VideoInventoryAgent",
    "CTVInventoryAgent",
    "MobileAppInventoryAgent",
    "NativeInventoryAgent",
    # Models
    "AudienceSpec",
    "BuyerTier",
    "ChannelType",
    "CounterOffer",
    "CrossSellOpportunity",
    "Deal",
    "DealAction",
    "DealDecision",
    "DealRequest",
    "DealTypeEnum",
    "InventoryPortfolio",
    "Product",
    "Task",
    "TaskResult",
    "YieldStrategy",
]
