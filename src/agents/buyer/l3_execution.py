"""L3 Execution Agent for Buyer Agent System.

The Execution Agent handles order management and deal booking:
- CreateOrder: Create new advertising orders
- CreateLine: Create line items within orders
- BookLine: Book inventory (finalize deals)
- ReserveLine: Reserve inventory temporarily
"""

from typing import Any, Optional
from datetime import datetime
import uuid

from .l3_base import (
    FunctionalAgent,
    ToolResult,
    ToolExecutionStatus,
    AgentContext,
)
from .l3_tools import (
    ToolSchema,
    OrderSpec,
    Order,
    Deal,
    BookingConfirmation,
)

# Optional import - may not be available in test environments
try:
    from .tools.sim_client import SimulationClient
except ImportError:
    SimulationClient = None  # type: ignore


class ExecutionAgent(FunctionalAgent[BookingConfirmation]):
    """Order and booking execution functional agent.
    
    This L3 agent specializes in creating orders, managing line items,
    and booking inventory. It handles the complete deal execution lifecycle.
    
    Tools:
        - CreateOrder: Create new order
        - CreateLine: Create line item
        - BookLine: Book inventory
        - ReserveLine: Reserve inventory
    
    Example:
        ```python
        agent = ExecutionAgent(context)
        
        # Create order
        order = await agent.create_order(OrderSpec(
            campaign_id="camp-001",
            buyer_id="buyer-001",
            name="Q1 Campaign",
            budget=50000.0,
        ))
        
        # Book deal
        confirmation = await agent.book_deal(Deal(
            order_id=order.order_id,
            seller_id="seller-001",
            product_id="prod-001",
            impressions=1000000,
            cpm=20.0,
            deal_type="PD",
        ))
        ```
    """
    
    TOOLS = ["CreateOrder", "CreateLine", "BookLine", "ReserveLine"]
    
    def __init__(
        self,
        context: AgentContext,
        sim_client: Optional[SimulationClient] = None,
        **kwargs,
    ):
        """Initialize execution agent.
        
        Args:
            context: Agent context with buyer/scenario info
            sim_client: Optional simulation client for deal operations
            **kwargs: Additional args passed to FunctionalAgent
        """
        super().__init__(context, **kwargs)
        self._sim_client = sim_client
        self._orders: dict[str, Order] = {}
        self._lines: dict[str, dict] = {}
        self._reservations: dict[str, dict] = {}
    
    @property
    def system_prompt(self) -> str:
        """System prompt for execution agent."""
        return f"""You are an Execution Specialist for a programmatic advertising buyer.
Your role is to execute orders and book advertising inventory.

Buyer ID: {self.context.buyer_id}
Scenario: {self.context.scenario}
Campaign: {self.context.campaign_id or "not specified"}

Your capabilities:
1. CreateOrder - Create new advertising orders
2. CreateLine - Add line items to orders
3. BookLine - Finalize deals and book inventory
4. ReserveLine - Temporarily reserve inventory

Execution workflow:
1. Create an order with budget and targeting
2. Add line items specifying inventory and pricing
3. Book lines to finalize deals OR reserve to hold inventory

Important considerations:
- Verify budget availability before booking
- Confirm pricing is within acceptable range
- Ensure targeting aligns with campaign goals
- Use ReserveLine for uncertain deals

Always confirm successful execution with deal details."""
    
    @property
    def available_tools(self) -> list[dict]:
        """Tools available to this agent."""
        return [
            ToolSchema.create_order(),
            ToolSchema.create_line(),
            ToolSchema.book_line(),
            ToolSchema.reserve_line(),
        ]
    
    async def _execute_tool(self, name: str, params: dict) -> ToolResult:
        """Execute an execution tool.
        
        Args:
            name: Tool name
            params: Tool parameters from LLM
            
        Returns:
            ToolResult with execution outcome
        """
        if name == "CreateOrder":
            return await self._tool_create_order(params)
        elif name == "CreateLine":
            return await self._tool_create_line(params)
        elif name == "BookLine":
            return await self._tool_book_line(params)
        elif name == "ReserveLine":
            return await self._tool_reserve_line(params)
        else:
            return ToolResult(
                tool_name=name,
                status=ToolExecutionStatus.FAILED,
                error=f"Unknown tool: {name}",
            )
    
    # -------------------------------------------------------------------------
    # High-Level Methods
    # -------------------------------------------------------------------------
    
    async def create_order(self, order_spec: OrderSpec) -> Order:
        """Create a new advertising order.
        
        Args:
            order_spec: Order specification
            
        Returns:
            Created order
        """
        result = await self._tool_create_order({
            "campaign_id": order_spec.campaign_id,
            "name": order_spec.name,
            "budget": order_spec.budget,
            "channel": order_spec.channel,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to create order: {result.error}")
    
    async def create_line(
        self,
        order_id: str,
        name: str,
        product_id: str,
        impressions: int,
        cpm: float,
        targeting: Optional[dict] = None,
    ) -> dict:
        """Create a line item within an order.
        
        Args:
            order_id: Order to add line to
            name: Line item name
            product_id: Product/inventory to book
            impressions: Number of impressions
            cpm: CPM bid price
            targeting: Optional targeting parameters
            
        Returns:
            Created line item
        """
        result = await self._tool_create_line({
            "order_id": order_id,
            "name": name,
            "product_id": product_id,
            "impressions": impressions,
            "cpm": cpm,
            "targeting": targeting,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to create line: {result.error}")
    
    async def book_deal(self, deal: Deal) -> BookingConfirmation:
        """Book a deal to commit inventory.
        
        Args:
            deal: Deal to book
            
        Returns:
            Booking confirmation
        """
        # First ensure we have a line for this deal
        line_id = f"LINE-{deal.order_id}-{uuid.uuid4().hex[:8]}"
        
        result = await self._tool_book_line({
            "line_id": line_id,
            "order_id": deal.order_id,
            "seller_id": deal.seller_id,
            "deal_type": deal.deal_type,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to book deal: {result.error}")
    
    async def reserve_inventory(
        self,
        order_id: str,
        line_id: str,
        hold_hours: int = 24,
    ) -> dict:
        """Reserve inventory without full booking.
        
        Args:
            order_id: Order ID
            line_id: Line item ID
            hold_hours: Hours to hold reservation
            
        Returns:
            Reservation details
        """
        result = await self._tool_reserve_line({
            "order_id": order_id,
            "line_id": line_id,
            "hold_duration_hours": hold_hours,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to reserve: {result.error}")
    
    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------
    
    async def _tool_create_order(self, params: dict) -> ToolResult:
        """Execute CreateOrder tool."""
        try:
            order_id = f"ORD-{uuid.uuid4().hex[:8]}"
            
            order = Order(
                order_id=order_id,
                campaign_id=params.get("campaign_id", "unknown"),
                buyer_id=self.context.buyer_id,
                name=params.get("name", f"Order {order_id}"),
                budget=params.get("budget", 10000.0),
                status="created",
                created_at=datetime.utcnow().isoformat(),
            )
            
            # Store order
            self._orders[order_id] = order
            
            return ToolResult(
                tool_name="CreateOrder",
                status=ToolExecutionStatus.SUCCESS,
                data=order,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="CreateOrder",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_create_line(self, params: dict) -> ToolResult:
        """Execute CreateLine tool."""
        try:
            order_id = params.get("order_id")
            if not order_id:
                return ToolResult(
                    tool_name="CreateLine",
                    status=ToolExecutionStatus.FAILED,
                    error="order_id is required",
                )
            
            line_id = f"LINE-{uuid.uuid4().hex[:8]}"
            
            line = {
                "line_id": line_id,
                "order_id": order_id,
                "name": params.get("name", f"Line {line_id}"),
                "product_id": params.get("product_id"),
                "impressions": params.get("impressions", 100000),
                "cpm": params.get("cpm", 15.0),
                "targeting": params.get("targeting", {}),
                "status": "draft",
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Calculate cost
            line["estimated_cost"] = (
                line["impressions"] / 1000 * line["cpm"]
            )
            
            # Store line
            self._lines[line_id] = line
            
            return ToolResult(
                tool_name="CreateLine",
                status=ToolExecutionStatus.SUCCESS,
                data=line,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="CreateLine",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_book_line(self, params: dict) -> ToolResult:
        """Execute BookLine tool."""
        try:
            line_id = params.get("line_id")
            order_id = params.get("order_id")
            seller_id = params.get("seller_id")
            deal_type = params.get("deal_type", "OA")
            
            if not all([line_id, order_id, seller_id]):
                return ToolResult(
                    tool_name="BookLine",
                    status=ToolExecutionStatus.FAILED,
                    error="line_id, order_id, and seller_id are required",
                )
            
            # Get line details (or use defaults)
            line = self._lines.get(line_id, {
                "impressions": 100000,
                "cpm": 15.0,
            })
            
            deal_id = f"DEAL-{uuid.uuid4().hex[:8]}"
            impressions = line.get("impressions", 100000)
            cpm = line.get("cpm", 15.0)
            total_cost = impressions / 1000 * cpm
            
            # Apply exchange fee for Scenario A
            if self.context.scenario == "A":
                exchange_fee = total_cost * 0.15
                total_cost += exchange_fee
            
            # Use simulation client if available
            if self._sim_client:
                result = await self._sim_client.request_deal(
                    seller_id=seller_id,
                    campaign_id=self.context.campaign_id or order_id,
                    impressions=impressions,
                    max_cpm=cpm * 1.1,  # Allow 10% overage
                    channel=self.context.channel or "display",
                )
                
                if result.success and result.data:
                    deal = result.data
                    confirmation = BookingConfirmation(
                        deal_id=deal.deal_id,
                        order_id=order_id,
                        seller_id=deal.seller_id,
                        impressions=deal.impressions,
                        cpm=deal.cpm,
                        total_cost=deal.total_cost,
                        status="booked",
                        booked_at=datetime.utcnow().isoformat(),
                    )
                    return ToolResult(
                        tool_name="BookLine",
                        status=ToolExecutionStatus.SUCCESS,
                        data=confirmation,
                    )
            
            # Mock response
            confirmation = BookingConfirmation(
                deal_id=deal_id,
                order_id=order_id,
                seller_id=seller_id,
                impressions=impressions,
                cpm=cpm,
                total_cost=total_cost,
                status="booked",
                booked_at=datetime.utcnow().isoformat(),
            )
            
            # Update line status
            if line_id in self._lines:
                self._lines[line_id]["status"] = "booked"
                self._lines[line_id]["deal_id"] = deal_id
            
            return ToolResult(
                tool_name="BookLine",
                status=ToolExecutionStatus.SUCCESS,
                data=confirmation,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="BookLine",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_reserve_line(self, params: dict) -> ToolResult:
        """Execute ReserveLine tool."""
        try:
            line_id = params.get("line_id")
            order_id = params.get("order_id")
            hold_hours = params.get("hold_duration_hours", 24)
            
            if not all([line_id, order_id]):
                return ToolResult(
                    tool_name="ReserveLine",
                    status=ToolExecutionStatus.FAILED,
                    error="line_id and order_id are required",
                )
            
            reservation_id = f"RES-{uuid.uuid4().hex[:8]}"
            
            reservation = {
                "reservation_id": reservation_id,
                "line_id": line_id,
                "order_id": order_id,
                "hold_duration_hours": hold_hours,
                "status": "reserved",
                "expires_at": datetime.utcnow().isoformat(),  # Would add hours
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Store reservation
            self._reservations[reservation_id] = reservation
            
            # Update line status
            if line_id in self._lines:
                self._lines[line_id]["status"] = "reserved"
                self._lines[line_id]["reservation_id"] = reservation_id
            
            return ToolResult(
                tool_name="ReserveLine",
                status=ToolExecutionStatus.SUCCESS,
                data=reservation,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="ReserveLine",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
