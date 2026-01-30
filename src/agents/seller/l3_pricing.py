"""
L3 Pricing Agent - Pricing calculation and management.

Handles price calculations, floor management, and discount application
for seller inventory.
"""

from dataclasses import dataclass, field
from typing import Optional, Any

from ..base import FunctionalAgent, ToolDefinition, ToolResult


@dataclass
class BuyerContext:
    """Context about the buyer for pricing decisions."""
    
    buyer_id: str
    tier: str = "standard"  # premium, standard, new
    historical_spend: float = 0.0
    relationship_months: int = 0
    preferred_deal_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Price:
    """Calculated price result."""
    
    base_cpm: float
    final_cpm: float
    floor_cpm: float
    discount_applied: float = 0.0
    discount_reason: Optional[str] = None
    currency: str = "USD"
    metadata: dict = field(default_factory=dict)


class PricingAgent(FunctionalAgent):
    """
    Pricing calculation and management.
    
    Tools:
    - PriceCalculator: Calculate price based on rules
    - FloorManager: Manage floor prices
    - DiscountEngine: Apply tiered discounts
    
    This agent handles:
    - Base price calculation from product rules
    - Floor price enforcement
    - Tiered discount application based on buyer tier/volume
    - Dynamic pricing based on demand
    """
    
    # Default discount tiers
    TIER_DISCOUNTS = {
        "premium": 0.15,    # 15% discount for premium buyers
        "standard": 0.05,   # 5% for standard
        "new": 0.0,         # No discount for new buyers
    }
    
    # Volume discount thresholds
    VOLUME_DISCOUNTS = [
        (1000000, 0.10),   # 10% for 1M+ impressions
        (500000, 0.07),    # 7% for 500K+
        (100000, 0.03),    # 3% for 100K+
        (0, 0.0),          # No volume discount below 100K
    ]
    
    def __init__(self, **kwargs):
        """Initialize PricingAgent."""
        kwargs.setdefault("name", "PricingAgent")
        super().__init__(**kwargs)
        self._floors: dict[str, float] = {}
        self._price_rules: dict[str, dict] = {}
    
    def _register_tools(self) -> None:
        """Register pricing tools."""
        self.register_tool(
            ToolDefinition(
                name="PriceCalculator",
                description="Calculate base price for a product based on pricing rules",
                parameters={
                    "product_id": {"type": "string", "description": "Product identifier"}
                },
                required_params=["product_id"]
            ),
            handler=self._handle_price_calculator
        )
        
        self.register_tool(
            ToolDefinition(
                name="FloorManager",
                description="Get or set floor price for a product",
                parameters={
                    "product_id": {"type": "string", "description": "Product identifier"},
                    "action": {"type": "string", "enum": ["get_floor", "set_floor"]},
                    "floor_cpm": {"type": "number", "description": "Floor CPM to set"}
                },
                required_params=["product_id", "action"]
            ),
            handler=self._handle_floor_manager
        )
        
        self.register_tool(
            ToolDefinition(
                name="DiscountEngine",
                description="Calculate discount based on buyer tier and volume",
                parameters={
                    "tier": {"type": "string", "description": "Buyer tier"},
                    "volume": {"type": "integer", "description": "Requested impressions"},
                    "base_price": {"type": "number", "description": "Base CPM"}
                },
                required_params=["tier", "volume", "base_price"]
            ),
            handler=self._handle_discount_engine
        )
    
    def get_system_prompt(self) -> str:
        """Get system prompt for pricing decisions."""
        return """You are a Pricing Agent responsible for calculating optimal prices for advertising inventory.

Your responsibilities:
1. Calculate base prices using product pricing rules
2. Apply appropriate discounts based on buyer tier and volume
3. Enforce floor prices to protect revenue
4. Provide clear reasoning for pricing decisions

Available tools:
- PriceCalculator: Get base price for a product
- FloorManager: Manage floor prices
- DiscountEngine: Calculate tiered discounts

Always ensure prices are above floor and provide transparent discount breakdowns."""
    
    def _handle_price_calculator(self, product_id: str) -> dict:
        """Handle PriceCalculator tool execution."""
        rule = self._price_rules.get(product_id, {})
        return {
            "base_cpm": rule.get("base_cpm", 15.0),
            "currency": rule.get("currency", "USD"),
            "pricing_model": rule.get("model", "cpm"),
        }
    
    def _handle_floor_manager(
        self, 
        product_id: str, 
        action: str, 
        floor_cpm: Optional[float] = None
    ) -> dict:
        """Handle FloorManager tool execution."""
        if action == "set_floor" and floor_cpm is not None:
            self._floors[product_id] = floor_cpm
            return {"success": True, "floor_cpm": floor_cpm}
        return {"floor_cpm": self._floors.get(product_id, 10.0)}
    
    def _handle_discount_engine(
        self, 
        tier: str, 
        volume: int, 
        base_price: float
    ) -> dict:
        """Handle DiscountEngine tool execution."""
        tier_discount = self.TIER_DISCOUNTS.get(tier, 0.0)
        
        volume_discount = 0.0
        for threshold, discount in self.VOLUME_DISCOUNTS:
            if volume >= threshold:
                volume_discount = discount
                break
        
        total = min(tier_discount + volume_discount, 0.25)
        
        return {
            "discount_pct": total,
            "tier_discount": tier_discount,
            "volume_discount": volume_discount,
            "reason": f"{tier} tier + {volume:,} volume",
        }
    
    async def calculate_price(
        self,
        product_id: str,
        buyer_context: BuyerContext,
        requested_impressions: int = 0,
    ) -> Price:
        """Calculate tiered price for a product.
        
        Args:
            product_id: ID of the product to price
            buyer_context: Context about the buyer
            requested_impressions: Number of impressions (for volume discount)
            
        Returns:
            Price with base, final, and discount information
        """
        # Get base price
        base_result = self._handle_price_calculator(product_id)
        base_cpm = base_result.get("base_cpm", 15.0)
        
        # Get floor
        floor_result = self._handle_floor_manager(product_id, "get_floor")
        floor_cpm = floor_result.get("floor_cpm", base_cpm * 0.7)
        
        # Calculate discount
        discount_result = self._handle_discount_engine(
            buyer_context.tier, 
            requested_impressions, 
            base_cpm
        )
        discount_pct = discount_result.get("discount_pct", 0.0)
        discount_reason = discount_result.get("reason", "")
        
        final_cpm = base_cpm * (1 - discount_pct)
        
        # Enforce floor
        if final_cpm < floor_cpm:
            final_cpm = floor_cpm
            discount_reason = f"{discount_reason} (capped at floor)"
        
        return Price(
            base_cpm=base_cpm,
            final_cpm=round(final_cpm, 2),
            floor_cpm=floor_cpm,
            discount_applied=round(discount_pct * 100, 1),
            discount_reason=discount_reason,
            metadata={
                "product_id": product_id,
                "buyer_tier": buyer_context.tier,
                "requested_impressions": requested_impressions,
            },
        )
    
    async def apply_discount(
        self,
        base_price: float,
        tier: str,
        volume: int,
    ) -> tuple[float, str]:
        """Apply discount based on buyer tier and volume."""
        result = self._handle_discount_engine(tier, volume, base_price)
        return result["discount_pct"], result["reason"]
    
    async def get_floor_price(self, product_id: str) -> float:
        """Get the floor price for a product."""
        result = self._handle_floor_manager(product_id, "get_floor")
        return result.get("floor_cpm", 0.0)
    
    async def set_floor_price(self, product_id: str, floor_cpm: float) -> bool:
        """Set the floor price for a product."""
        result = self._handle_floor_manager(product_id, "set_floor", floor_cpm)
        return result.get("success", False)
