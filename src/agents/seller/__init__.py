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
- L3 Functional Agents: Specialized agents for pricing, avails, proposals, etc.

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
    
    # L3 Agents
    from agents.seller import PricingAgent, AvailsAgent
    from agents.seller.tools import register_tools_for_agent
    
    pricing = PricingAgent()
    register_tools_for_agent(pricing)
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

# L3 Functional Agents
from .l3_pricing import PricingAgent, BuyerContext, Price
from .l3_avails import AvailsAgent, DateRange, AvailsResult, Forecast, Allocation
from .l3_proposal_review import (
    ProposalReviewAgent, 
    Proposal, 
    ProposalStatus, 
    ReviewResult,
    CounterOffer as L3CounterOffer,
)
from .l3_upsell import UpsellAgent, Deal as L3Deal, Opportunity, Bundle
from .l3_audience_validator import (
    AudienceValidatorAgent, 
    AudienceSpec as L3AudienceSpec, 
    ValidationResult, 
    Coverage,
)

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
    # L3 Functional Agents
    "PricingAgent",
    "AvailsAgent",
    "ProposalReviewAgent",
    "UpsellAgent",
    "AudienceValidatorAgent",
    # L3 Data Types
    "BuyerContext",
    "Price",
    "DateRange",
    "AvailsResult",
    "Forecast",
    "Allocation",
    "Proposal",
    "ProposalStatus",
    "ReviewResult",
    "L3CounterOffer",
    "L3Deal",
    "Opportunity",
    "Bundle",
    "L3AudienceSpec",
    "ValidationResult",
    "Coverage",
    # Models (from models.py)
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
