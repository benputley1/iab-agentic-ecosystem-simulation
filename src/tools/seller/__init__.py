"""Seller-side MCP tools for OpenDirect.

These tools are used by seller agents to manage inventory,
pricing, deals, and ad server integration.
"""

from .pricing import (
    PriceCalculatorTool,
    FloorManagerTool,
    DiscountEngineTool,
)
from .inventory import (
    AvailsCheckerTool,
    CapacityForecasterTool,
    AllocationManagerTool,
)
from .deals import (
    ProposalGeneratorTool,
    CounterOfferBuilderTool,
    DealIDGeneratorTool,
)
from .gam import (
    ListAdUnitsTool,
    CreateOrderTool as GAMCreateOrderTool,
    CreateLineItemTool,
    BookDealTool,
    SyncInventoryTool,
)
from .audience import (
    AudienceValidationTool,
    AudienceCapabilityTool,
    CoverageCalculatorTool,
)

# All seller tools
ALL_SELLER_TOOLS = [
    # Pricing
    PriceCalculatorTool,
    FloorManagerTool,
    DiscountEngineTool,
    # Inventory
    AvailsCheckerTool,
    CapacityForecasterTool,
    AllocationManagerTool,
    # Deals
    ProposalGeneratorTool,
    CounterOfferBuilderTool,
    DealIDGeneratorTool,
    # GAM
    ListAdUnitsTool,
    GAMCreateOrderTool,
    CreateLineItemTool,
    BookDealTool,
    SyncInventoryTool,
    # Audience
    AudienceValidationTool,
    AudienceCapabilityTool,
    CoverageCalculatorTool,
]

__all__ = [
    # Pricing
    "PriceCalculatorTool",
    "FloorManagerTool",
    "DiscountEngineTool",
    # Inventory
    "AvailsCheckerTool",
    "CapacityForecasterTool",
    "AllocationManagerTool",
    # Deals
    "ProposalGeneratorTool",
    "CounterOfferBuilderTool",
    "DealIDGeneratorTool",
    # GAM
    "ListAdUnitsTool",
    "GAMCreateOrderTool",
    "CreateLineItemTool",
    "BookDealTool",
    "SyncInventoryTool",
    # Audience
    "AudienceValidationTool",
    "AudienceCapabilityTool",
    "CoverageCalculatorTool",
    # Collection
    "ALL_SELLER_TOOLS",
]
