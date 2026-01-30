"""Buyer execution tools for orders and line items."""

from ..base import Tool, ToolResult


class CreateOrderTool(Tool):
    """Create a new advertising order.
    
    Creates an order container that will hold line items
    for a specific campaign or advertiser.
    """
    
    name = "CreateOrder"
    description = "Create a new advertising order/IO (insertion order)"
    parameters = {
        "advertiser_id": {
            "type": "string",
            "description": "Advertiser ID this order is for",
            "required": True,
        },
        "name": {
            "type": "string",
            "description": "Order name",
            "required": True,
        },
        "budget": {
            "type": "number",
            "description": "Total order budget in dollars",
        },
        "currency": {
            "type": "string",
            "description": "Budget currency (ISO 4217)",
        },
        "start_date": {
            "type": "string",
            "description": "Order start date (ISO 8601)",
        },
        "end_date": {
            "type": "string",
            "description": "Order end date (ISO 8601)",
        },
        "contacts": {
            "type": "array",
            "description": "Contact information for order",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute order creation."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "order_id": None,
            "status": "draft",
            "created_at": None,
        })


class CreateLineTool(Tool):
    """Create a line item within an order.
    
    Line items define the specific inventory, targeting,
    and budget allocation for a campaign.
    """
    
    name = "CreateLine"
    description = "Create a line item within an existing order"
    parameters = {
        "order_id": {
            "type": "string",
            "description": "Order ID to add line to",
            "required": True,
        },
        "product_id": {
            "type": "string",
            "description": "Product ID for this line",
            "required": True,
        },
        "name": {
            "type": "string",
            "description": "Line item name",
            "required": True,
        },
        "impressions": {
            "type": "integer",
            "description": "Number of impressions to book",
        },
        "budget": {
            "type": "number",
            "description": "Line item budget",
        },
        "cpm": {
            "type": "number",
            "description": "Agreed CPM rate",
        },
        "start_date": {
            "type": "string",
            "description": "Line start date",
        },
        "end_date": {
            "type": "string",
            "description": "Line end date",
        },
        "targeting": {
            "type": "object",
            "description": "Targeting criteria",
        },
        "frequency_cap": {
            "type": "object",
            "description": "Frequency capping rules",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute line item creation."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "line_id": None,
            "order_id": kwargs.get("order_id"),
            "status": "draft",
        })


class BookLineTool(Tool):
    """Book a line item to confirm the buy.
    
    Transitions a line item from draft/reserved status
    to booked, committing to the purchase.
    """
    
    name = "BookLine"
    description = "Book a line item to confirm and commit to the purchase"
    parameters = {
        "line_id": {
            "type": "string",
            "description": "Line item ID to book",
            "required": True,
        },
        "confirm_price": {
            "type": "boolean",
            "description": "Confirm acceptance of current pricing",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute line booking."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "line_id": kwargs.get("line_id"),
            "status": "booked",
            "booked_at": None,
        })


class ReserveLineTool(Tool):
    """Reserve a line item temporarily.
    
    Places a temporary hold on inventory without
    full commitment, allowing for approval workflows.
    """
    
    name = "ReserveLine"
    description = "Reserve a line item temporarily (hold without booking)"
    parameters = {
        "line_id": {
            "type": "string",
            "description": "Line item ID to reserve",
            "required": True,
        },
        "hold_duration_hours": {
            "type": "integer",
            "description": "Hours to hold the reservation",
        },
        "notes": {
            "type": "string",
            "description": "Notes about the reservation",
        },
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute line reservation."""
        errors = self.validate_args(**kwargs)
        if errors:
            return ToolResult.fail("; ".join(errors))
        
        return ToolResult.ok({
            "line_id": kwargs.get("line_id"),
            "status": "reserved",
            "expires_at": None,
        })
