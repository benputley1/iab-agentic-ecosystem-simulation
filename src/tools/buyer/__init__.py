"""Buyer-side MCP tools for OpenDirect.

These tools are used by buyer agents to research, plan, and execute
advertising campaigns through the OpenDirect protocol.
"""

from .research import (
    ProductSearchTool,
    AvailsCheckTool,
    PricingLookupTool,
)
from .execution import (
    CreateOrderTool,
    CreateLineTool,
    BookLineTool,
    ReserveLineTool,
)
from .dsp import (
    DiscoverInventoryTool,
    GetPricingTool,
    RequestDealTool,
    AttachDealTool,
)
from .audience import (
    AudienceDiscoveryTool,
    AudienceMatchingTool,
    CoverageEstimationTool,
)

# All buyer tools
ALL_BUYER_TOOLS = [
    # Research
    ProductSearchTool,
    AvailsCheckTool,
    PricingLookupTool,
    # Execution
    CreateOrderTool,
    CreateLineTool,
    BookLineTool,
    ReserveLineTool,
    # DSP
    DiscoverInventoryTool,
    GetPricingTool,
    RequestDealTool,
    AttachDealTool,
    # Audience
    AudienceDiscoveryTool,
    AudienceMatchingTool,
    CoverageEstimationTool,
]

__all__ = [
    # Research
    "ProductSearchTool",
    "AvailsCheckTool",
    "PricingLookupTool",
    # Execution
    "CreateOrderTool",
    "CreateLineTool",
    "BookLineTool",
    "ReserveLineTool",
    # DSP
    "DiscoverInventoryTool",
    "GetPricingTool",
    "RequestDealTool",
    "AttachDealTool",
    # Audience
    "AudienceDiscoveryTool",
    "AudienceMatchingTool",
    "CoverageEstimationTool",
    # Collection
    "ALL_BUYER_TOOLS",
]
