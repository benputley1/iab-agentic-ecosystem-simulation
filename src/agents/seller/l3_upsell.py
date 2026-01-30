"""
L3 Upsell Agent - Cross-sell and upsell opportunities.

Handles finding cross-sell opportunities, creating inventory bundles,
and calculating bundle values.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from ..base import FunctionalAgent, ToolDefinition


@dataclass
class Deal:
    """Current deal for cross-sell analysis."""
    
    deal_id: str
    buyer_id: str
    product_id: str
    channel: str
    impressions: int
    cpm: float
    start_date: str
    end_date: str
    audience_segments: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Opportunity:
    """Cross-sell or upsell opportunity."""
    
    opportunity_id: str
    opportunity_type: str  # cross_sell, upsell, bundle
    recommended_product_id: str
    recommended_product_name: str
    rationale: str
    estimated_value: float
    confidence: float
    synergy_score: float = 0.0
    audience_overlap: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Bundle:
    """Inventory bundle."""
    
    bundle_id: str
    name: str
    products: list[str]
    total_impressions: int
    bundle_cpm: float
    discount_pct: float
    total_value: float
    savings: float
    valid_until: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


class UpsellAgent(FunctionalAgent):
    """
    Cross-sell and upsell opportunities.
    
    Tools:
    - CrossSellAnalyzer: Find cross-sell opportunities
    - BundleBuilder: Create inventory bundles
    - ValueCalculator: Calculate bundle value
    
    This agent handles:
    - Analyzing current deals for cross-sell potential
    - Identifying complementary inventory
    - Creating value-added bundles
    - Calculating optimal bundle pricing
    """
    
    # Cross-sell rules: channel -> recommended channels
    CROSS_SELL_RULES = {
        "display": ["video", "native"],
        "video": ["ctv", "display"],
        "ctv": ["video", "display"],
        "native": ["display", "video"],
        "mobile": ["display", "video", "native"],
    }
    
    # Bundle discount tiers
    BUNDLE_DISCOUNTS = {
        2: 0.05,   # 5% for 2 products
        3: 0.10,   # 10% for 3 products
        4: 0.12,   # 12% for 4 products
        5: 0.15,   # 15% for 5+ products
    }
    
    def __init__(self, **kwargs):
        """Initialize UpsellAgent."""
        kwargs.setdefault("name", "UpsellAgent")
        super().__init__(**kwargs)
    
    def _register_tools(self) -> None:
        """Register upsell tools."""
        self.register_tool(
            ToolDefinition(
                name="CrossSellAnalyzer",
                description="Analyze current deal for cross-sell opportunities",
                parameters={
                    "deal_id": {"type": "string"},
                    "buyer_id": {"type": "string"},
                    "current_channel": {"type": "string"},
                    "audience_segments": {"type": "array", "items": {"type": "string"}}
                },
                required_params=["deal_id", "current_channel"]
            ),
            handler=self._handle_cross_sell_analyzer
        )
        
        self.register_tool(
            ToolDefinition(
                name="BundleBuilder",
                description="Create an inventory bundle from products",
                parameters={
                    "product_ids": {"type": "array", "items": {"type": "string"}}
                },
                required_params=["product_ids"]
            ),
            handler=self._handle_bundle_builder
        )
        
        self.register_tool(
            ToolDefinition(
                name="ValueCalculator",
                description="Calculate value for products or bundle",
                parameters={
                    "product_ids": {"type": "array", "items": {"type": "string"}},
                    "impressions": {"type": "object"}
                },
                required_params=["product_ids"]
            ),
            handler=self._handle_value_calculator
        )
    
    def get_system_prompt(self) -> str:
        """Get system prompt for upsell operations."""
        return """You are an Upsell Agent responsible for identifying revenue opportunities.

Your responsibilities:
1. Analyze current deals for cross-sell potential
2. Identify complementary inventory that adds value
3. Create compelling bundles with appropriate discounts
4. Calculate optimal bundle pricing

Available tools:
- CrossSellAnalyzer: Find cross-sell opportunities
- BundleBuilder: Create product bundles
- ValueCalculator: Calculate bundle value

Focus on value creation for both buyer and seller."""
    
    def _handle_cross_sell_analyzer(
        self,
        deal_id: str,
        current_channel: str,
        buyer_id: Optional[str] = None,
        audience_segments: Optional[list] = None,
    ) -> dict:
        """Handle CrossSellAnalyzer tool."""
        channel_recs = self.CROSS_SELL_RULES.get(current_channel, ["display"])
        
        opportunities = []
        for i, channel in enumerate(channel_recs):
            opportunities.append({
                "id": f"opp-{deal_id}-{i}",
                "product_id": f"recommended-{channel}",
                "product_name": f"{channel.upper()} Inventory",
                "rationale": f"Complementary {channel} for {current_channel} campaign",
                "estimated_value": 5000.0,
                "confidence": 0.8 - (i * 0.1),
                "synergy_score": 0.85 - (i * 0.1),
                "audience_overlap": 0.6,
            })
        
        return {"opportunities": opportunities}
    
    def _handle_bundle_builder(self, product_ids: list) -> dict:
        """Handle BundleBuilder tool."""
        num_products = len(product_ids)
        total_imps = 100000 * num_products
        
        discounts = {2: 0.05, 3: 0.10, 4: 0.12, 5: 0.15}
        discount = discounts.get(min(num_products, 5), 0.15)
        
        base_cpm = 15.0
        bundle_cpm = base_cpm * (1 - discount)
        
        return {
            "total_impressions": total_imps,
            "individual_value": total_imps * base_cpm / 1000,
            "bundle_cpm": bundle_cpm,
            "discount_pct": discount,
        }
    
    def _handle_value_calculator(
        self,
        product_ids: list,
        impressions: Optional[dict] = None,
    ) -> dict:
        """Handle ValueCalculator tool."""
        total_imps = sum((impressions or {}).values()) or 100000 * len(product_ids)
        base_cpm = 15.0
        individual_value = total_imps * base_cpm / 1000
        
        num_products = len(product_ids)
        discount = min(0.05 * (num_products - 1), 0.15)
        bundle_value = individual_value * (1 - discount)
        
        return {
            "total_impressions": total_imps,
            "total_value": individual_value,
            "bundle_value": bundle_value,
        }
    
    async def find_cross_sell(
        self,
        current_deal: Deal,
        available_products: Optional[list[dict]] = None,
    ) -> list[Opportunity]:
        """Find cross-sell opportunities based on current deal."""
        result = self._handle_cross_sell_analyzer(
            deal_id=current_deal.deal_id,
            buyer_id=current_deal.buyer_id,
            current_channel=current_deal.channel,
            audience_segments=current_deal.audience_segments,
        )
        
        opportunities = []
        for opp_data in result.get("opportunities", []):
            opportunities.append(Opportunity(
                opportunity_id=opp_data.get("id", f"opp-{len(opportunities)}"),
                opportunity_type="cross_sell",
                recommended_product_id=opp_data.get("product_id", ""),
                recommended_product_name=opp_data.get("product_name", ""),
                rationale=opp_data.get("rationale", ""),
                estimated_value=opp_data.get("estimated_value", 0.0),
                confidence=opp_data.get("confidence", 0.5),
                synergy_score=opp_data.get("synergy_score", 0.0),
                audience_overlap=opp_data.get("audience_overlap", 0.0),
            ))
        
        return opportunities
    
    async def create_bundle(
        self,
        products: list[str],
        product_details: Optional[list[dict]] = None,
        bundle_name: Optional[str] = None,
    ) -> Bundle:
        """Create product bundle with pricing."""
        if not products:
            raise ValueError("Cannot create bundle with no products")
        
        result = self._handle_bundle_builder(products)
        
        total_imps = result.get("total_impressions", 0)
        individual_value = result.get("individual_value", 0.0)
        bundle_cpm = result.get("bundle_cpm", 0.0)
        discount = result.get("discount_pct", 0.0)
        
        bundle_value = total_imps * bundle_cpm / 1000
        savings = individual_value - bundle_value
        
        return Bundle(
            bundle_id=f"bundle-{'-'.join(p[:4] for p in products[:3])}",
            name=bundle_name or f"Multi-Channel Bundle ({len(products)} products)",
            products=products,
            total_impressions=total_imps,
            bundle_cpm=round(bundle_cpm, 2),
            discount_pct=round(discount * 100, 1),
            total_value=round(bundle_value, 2),
            savings=round(savings, 2),
            metadata={
                "individual_value": individual_value,
                "num_products": len(products),
            },
        )
    
    async def calculate_bundle_value(
        self,
        products: list[str],
        impressions_per_product: Optional[dict[str, int]] = None,
    ) -> dict:
        """Calculate value for a potential bundle."""
        result = self._handle_value_calculator(products, impressions_per_product)
        
        total_imps = result.get("total_impressions", 0)
        individual_value = result.get("total_value", 0.0)
        bundle_value = result.get("bundle_value", 0.0)
        
        num_products = len(products)
        discount = min(0.05 * (num_products - 1), 0.15)
        
        return {
            "total_impressions": total_imps,
            "individual_value": round(individual_value, 2),
            "bundle_value": round(bundle_value, 2),
            "savings": round(individual_value - bundle_value, 2),
            "discount_pct": discount * 100,
        }
    
    async def get_upsell_recommendations(
        self,
        current_deal: Deal,
        budget_headroom: float = 0.0,
    ) -> list[Opportunity]:
        """Get upsell recommendations for existing deal."""
        opportunities = []
        
        # Volume upsell
        if budget_headroom > 0:
            additional_imps = int(budget_headroom / current_deal.cpm * 1000)
            if additional_imps > current_deal.impressions * 0.1:
                opportunities.append(Opportunity(
                    opportunity_id=f"upsell-volume-{current_deal.deal_id}",
                    opportunity_type="upsell",
                    recommended_product_id=current_deal.product_id,
                    recommended_product_name="Volume Increase",
                    rationale=f"Increase impressions by {additional_imps:,} with remaining budget",
                    estimated_value=budget_headroom,
                    confidence=0.9,
                ))
        
        # Premium placement upsell
        opportunities.append(Opportunity(
            opportunity_id=f"upsell-premium-{current_deal.deal_id}",
            opportunity_type="upsell",
            recommended_product_id=f"{current_deal.product_id}-premium",
            recommended_product_name="Premium Placement Upgrade",
            rationale="Upgrade to premium placements for better viewability",
            estimated_value=current_deal.cpm * current_deal.impressions * 0.2 / 1000,
            confidence=0.7,
        ))
        
        return opportunities
