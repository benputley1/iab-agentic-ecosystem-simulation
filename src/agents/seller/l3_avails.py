"""
L3 Avails Agent - Inventory availability management.

Handles checking current availability, forecasting future capacity,
and managing inventory allocation.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, date
import uuid

from ..base import FunctionalAgent, ToolDefinition


@dataclass
class DateRange:
    """Date range for availability queries."""
    
    start_date: date
    end_date: date
    
    @property
    def days(self) -> int:
        """Number of days in range."""
        return (self.end_date - self.start_date).days + 1
    
    def contains(self, d: date) -> bool:
        """Check if date is within range."""
        return self.start_date <= d <= self.end_date


@dataclass
class AvailsResult:
    """Result of availability check."""
    
    product_id: str
    available: bool
    available_impressions: int
    requested_impressions: int = 0
    utilization_pct: float = 0.0
    date_range: Optional[DateRange] = None
    daily_breakdown: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Forecast:
    """Capacity forecast result."""
    
    product_id: str
    forecast_days: int
    daily_capacity: list[int] = field(default_factory=list)
    total_capacity: int = 0
    confidence: float = 0.95
    trend: str = "stable"  # increasing, decreasing, stable
    metadata: dict = field(default_factory=dict)


@dataclass
class Allocation:
    """Inventory allocation result."""
    
    allocation_id: str
    product_id: str
    deal_id: str
    impressions_allocated: int
    date_range: DateRange
    status: str = "pending"  # pending, confirmed, released
    created_at: datetime = field(default_factory=datetime.utcnow)


class AvailsAgent(FunctionalAgent):
    """
    Inventory availability management.
    
    Tools:
    - AvailsChecker: Check current availability
    - CapacityForecaster: Forecast future capacity
    - AllocationManager: Manage inventory allocation
    
    This agent handles:
    - Real-time availability checks
    - Capacity forecasting based on historical data
    - Allocation management for deals
    - Utilization tracking
    """
    
    def __init__(self, **kwargs):
        """Initialize AvailsAgent."""
        kwargs.setdefault("name", "AvailsAgent")
        super().__init__(**kwargs)
        self._inventory: dict[str, dict] = {}
        self._allocations: dict[str, dict] = {}
    
    def _register_tools(self) -> None:
        """Register avails tools."""
        self.register_tool(
            ToolDefinition(
                name="AvailsChecker",
                description="Check current availability for a product in a date range",
                parameters={
                    "product_id": {"type": "string"},
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"}
                },
                required_params=["product_id", "start_date", "end_date"]
            ),
            handler=self._handle_avails_checker
        )
        
        self.register_tool(
            ToolDefinition(
                name="CapacityForecaster",
                description="Forecast future capacity for a product",
                parameters={
                    "product_id": {"type": "string"},
                    "days": {"type": "integer"},
                    "from_date": {"type": "string", "format": "date"}
                },
                required_params=["product_id", "days", "from_date"]
            ),
            handler=self._handle_capacity_forecaster
        )
        
        self.register_tool(
            ToolDefinition(
                name="AllocationManager",
                description="Allocate or release inventory",
                parameters={
                    "action": {"type": "string", "enum": ["allocate", "release"]},
                    "product_id": {"type": "string"},
                    "deal_id": {"type": "string"},
                    "impressions": {"type": "integer"},
                    "allocation_id": {"type": "string"}
                },
                required_params=["action"]
            ),
            handler=self._handle_allocation_manager
        )
    
    def get_system_prompt(self) -> str:
        """Get system prompt for avails operations."""
        return """You are an Avails Agent responsible for managing inventory availability.

Your responsibilities:
1. Check real-time availability for advertising products
2. Forecast future capacity based on historical patterns
3. Manage inventory allocations for deals
4. Track utilization and prevent overbooking

Available tools:
- AvailsChecker: Check availability for date ranges
- CapacityForecaster: Predict future capacity
- AllocationManager: Allocate and release inventory

Always ensure accurate availability data and prevent overselling."""
    
    def _handle_avails_checker(
        self, 
        product_id: str, 
        start_date: str, 
        end_date: str
    ) -> dict:
        """Handle AvailsChecker tool."""
        inv = self._inventory.get(product_id, {})
        daily_capacity = inv.get("daily_impressions", 100000)
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        days = (end - start).days + 1
        
        total_capacity = daily_capacity * days
        allocated = inv.get("allocated", 0)
        available = total_capacity - allocated
        
        return {
            "available_impressions": max(0, available),
            "total_capacity": total_capacity,
            "already_allocated": allocated,
            "days": days,
        }
    
    def _handle_capacity_forecaster(
        self, 
        product_id: str, 
        days: int, 
        from_date: str
    ) -> dict:
        """Handle CapacityForecaster tool."""
        daily_capacity = 100000
        daily_forecast = [daily_capacity] * days
        
        return {
            "daily_capacity": daily_forecast,
            "total_capacity": sum(daily_forecast),
            "confidence": 0.85,
            "trend": "stable",
            "model": "baseline",
        }
    
    def _handle_allocation_manager(
        self,
        action: str,
        product_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        impressions: Optional[int] = None,
        allocation_id: Optional[str] = None,
        **kwargs
    ) -> dict:
        """Handle AllocationManager tool."""
        if action == "allocate":
            alloc_id = f"alloc-{uuid.uuid4().hex[:8]}"
            self._allocations[alloc_id] = {
                "product_id": product_id,
                "deal_id": deal_id,
                "impressions": impressions,
                "status": "confirmed",
            }
            return {"allocation_id": alloc_id, "status": "confirmed"}
        
        elif action == "release":
            if allocation_id in self._allocations:
                self._allocations[allocation_id]["status"] = "released"
                return {"success": True}
            return {"success": False, "error": "Allocation not found"}
        
        return {"success": False, "error": f"Unknown action: {action}"}
    
    async def check_avails(
        self,
        product_id: str,
        dates: DateRange,
        impressions_requested: int = 0,
    ) -> AvailsResult:
        """Check availability for date range."""
        data = self._handle_avails_checker(
            product_id,
            dates.start_date.isoformat(),
            dates.end_date.isoformat()
        )
        
        available_imps = data.get("available_impressions", 0)
        total_capacity = data.get("total_capacity", available_imps)
        
        allocated = total_capacity - available_imps
        utilization = (allocated / total_capacity * 100) if total_capacity > 0 else 0
        
        is_available = impressions_requested <= available_imps if impressions_requested > 0 else available_imps > 0
        
        return AvailsResult(
            product_id=product_id,
            available=is_available,
            available_impressions=available_imps,
            requested_impressions=impressions_requested,
            utilization_pct=round(utilization, 1),
            date_range=dates,
            metadata={
                "total_capacity": total_capacity,
                "already_allocated": allocated,
            },
        )
    
    async def forecast_capacity(
        self,
        product_id: str,
        days: int,
        from_date: Optional[date] = None,
    ) -> Forecast:
        """Forecast future capacity."""
        if from_date is None:
            from_date = date.today()
        
        data = self._handle_capacity_forecaster(
            product_id, days, from_date.isoformat()
        )
        
        daily_capacity = data.get("daily_capacity", [])
        
        return Forecast(
            product_id=product_id,
            forecast_days=days,
            daily_capacity=daily_capacity,
            total_capacity=sum(daily_capacity),
            confidence=data.get("confidence", 0.95),
            trend=data.get("trend", "stable"),
            metadata={"from_date": from_date.isoformat()},
        )
    
    async def allocate_inventory(
        self,
        product_id: str,
        deal_id: str,
        impressions: int,
        dates: DateRange,
    ) -> Allocation:
        """Allocate inventory for a deal."""
        result = self._handle_allocation_manager(
            action="allocate",
            product_id=product_id,
            deal_id=deal_id,
            impressions=impressions,
        )
        
        if not result.get("allocation_id"):
            raise ValueError("Allocation failed")
        
        return Allocation(
            allocation_id=result.get("allocation_id"),
            product_id=product_id,
            deal_id=deal_id,
            impressions_allocated=impressions,
            date_range=dates,
            status="confirmed",
        )
    
    async def release_allocation(self, allocation_id: str) -> bool:
        """Release a previously made allocation."""
        result = self._handle_allocation_manager(
            action="release",
            allocation_id=allocation_id,
        )
        return result.get("success", False)
    
    async def get_utilization(self, product_id: str, dates: DateRange) -> float:
        """Get current utilization percentage."""
        avails = await self.check_avails(product_id, dates)
        return avails.utilization_pct
