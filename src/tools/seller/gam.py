"""Seller GAM (Google Ad Manager) integration tools."""

from ..base import Tool, ToolResult


class ListAdUnitsTool(Tool):
    """List ad units from Google Ad Manager.
    
    Retrieves the inventory structure from GAM including
    ad units, their hierarchy, and targeting options.
    """
    
    name = "GAMListAdUnits"
    description = "List ad units from Google Ad Manager"
    parameters = {
        "parent_id": {
            "type": "string",
            "description": "Parent ad unit ID (for hierarchy)",
        },
        "status": {
            "type": "string",
            "enum": ["active", "inactive", "archived"],
            "description": "Filter by status",
        },
        "include_sizes": {
            "type": "boolean",
            "description": "Include size information",
        },
        "include_targeting": {
            "type": "boolean",
            "description": "Include targeting options",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results to return",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute ad unit listing."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "ad_units": [],
            "total_count": 0,
            "has_more": False,
        })


class CreateOrderTool(Tool):
    """Create an order in Google Ad Manager.
    
    Creates an order (IO) in GAM to traffic the
    negotiated deal.
    """
    
    name = "GAMCreateOrder"
    description = "Create an order in Google Ad Manager"
    parameters = {
        "name": {
            "type": "string",
            "description": "Order name",
            "required": True,
        },
        "advertiser_id": {
            "type": "string",
            "description": "GAM advertiser ID",
            "required": True,
        },
        "trafficker_id": {
            "type": "string",
            "description": "GAM trafficker user ID",
        },
        "salesperson_id": {
            "type": "string",
            "description": "GAM salesperson user ID",
        },
        "po_number": {
            "type": "string",
            "description": "Purchase order number",
        },
        "notes": {
            "type": "string",
            "description": "Order notes",
        },
        "start_date": {
            "type": "string",
            "description": "Order start date",
        },
        "end_date": {
            "type": "string",
            "description": "Order end date",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute order creation."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "order_id": None,
            "name": kwargs.get("name"),
            "status": "draft",
        })


class CreateLineItemTool(Tool):
    """Create a line item in Google Ad Manager.
    
    Creates a line item within a GAM order with targeting,
    pricing, and delivery settings.
    """
    
    name = "GAMCreateLineItem"
    description = "Create a line item in Google Ad Manager"
    parameters = {
        "order_id": {
            "type": "string",
            "description": "GAM order ID",
            "required": True,
        },
        "name": {
            "type": "string",
            "description": "Line item name",
            "required": True,
        },
        "ad_unit_ids": {
            "type": "array",
            "description": "Target ad unit IDs",
            "required": True,
        },
        "line_item_type": {
            "type": "string",
            "enum": ["standard", "sponsorship", "network", "bulk", "price_priority", "house"],
            "description": "Line item type",
        },
        "creative_sizes": {
            "type": "array",
            "description": "Creative sizes",
        },
        "start_date": {
            "type": "string",
            "description": "Start date",
        },
        "end_date": {
            "type": "string",
            "description": "End date",
        },
        "cost_type": {
            "type": "string",
            "enum": ["cpm", "cpc", "cpd", "flat_rate"],
            "description": "Cost type",
        },
        "cost_per_unit": {
            "type": "number",
            "description": "Cost per unit (CPM, CPC, etc.)",
        },
        "units_bought": {
            "type": "integer",
            "description": "Units purchased (impressions, clicks)",
        },
        "targeting": {
            "type": "object",
            "description": "Targeting configuration",
        },
        "frequency_caps": {
            "type": "array",
            "description": "Frequency capping rules",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute line item creation."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "line_item_id": None,
            "order_id": kwargs.get("order_id"),
            "name": kwargs.get("name"),
            "status": "draft",
        })


class BookDealTool(Tool):
    """Book/approve a deal in Google Ad Manager.
    
    Transitions an order from draft to approved state,
    making it ready to serve.
    """
    
    name = "GAMBookDeal"
    description = "Book/approve a deal in Google Ad Manager"
    parameters = {
        "order_id": {
            "type": "string",
            "description": "GAM order ID to book",
            "required": True,
        },
        "skip_approval": {
            "type": "boolean",
            "description": "Skip approval workflow if permitted",
        },
        "notes": {
            "type": "string",
            "description": "Booking notes",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute deal booking."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "order_id": kwargs.get("order_id"),
            "status": "approved",
            "booked_at": None,
        })


class SyncInventoryTool(Tool):
    """Sync inventory between OpenDirect and GAM.
    
    Synchronizes inventory catalog between the OpenDirect
    system and Google Ad Manager.
    """
    
    name = "GAMSyncInventory"
    description = "Sync inventory between OpenDirect and Google Ad Manager"
    parameters = {
        "direction": {
            "type": "string",
            "enum": ["from_gam", "to_gam", "bidirectional"],
            "description": "Sync direction",
            "required": True,
        },
        "inventory_ids": {
            "type": "array",
            "description": "Specific inventory IDs to sync (or all if empty)",
        },
        "include_pricing": {
            "type": "boolean",
            "description": "Include pricing data in sync",
        },
        "dry_run": {
            "type": "boolean",
            "description": "Preview changes without applying",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute inventory sync."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "synced_count": 0,
            "errors": [],
            "direction": kwargs.get("direction"),
            "dry_run": kwargs.get("dry_run", False),
        })
