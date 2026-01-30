"""L3 Research Agent for Buyer Agent System.

The Research Agent handles inventory discovery and market analysis:
- ProductSearch: Find available inventory across sellers
- AvailsCheck: Check availability with specific sellers
- PricingLookup: Get current pricing information
- CompetitiveIntel: Analyze market trends and competition
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
    SearchCriteria,
    Product,
    AvailsResult,
)

# Optional import - may not be available in test environments
try:
    from .tools.sim_client import SimulationClient
except ImportError:
    SimulationClient = None  # type: ignore


class ResearchAgent(FunctionalAgent[list[Product]]):
    """Research and discovery functional agent.
    
    This L3 agent specializes in inventory research and market analysis.
    It uses Claude Sonnet to interpret research requests and select
    appropriate tools.
    
    Tools:
        - ProductSearch: Find available inventory
        - AvailsCheck: Check availability
        - PricingLookup: Get current pricing
        - CompetitiveIntel: Market analysis
    
    Example:
        ```python
        agent = ResearchAgent(context)
        
        # Natural language research
        result = await agent.execute(
            "Find CTV inventory under $25 CPM with at least 100k impressions"
        )
        
        # Direct method call
        products = await agent.search_products(SearchCriteria(
            channel="ctv",
            max_cpm=25.0,
            min_impressions=100000,
        ))
        ```
    """
    
    TOOLS = ["ProductSearch", "AvailsCheck", "PricingLookup", "CompetitiveIntel"]
    
    def __init__(
        self,
        context: AgentContext,
        sim_client: Optional[SimulationClient] = None,
        **kwargs,
    ):
        """Initialize research agent.
        
        Args:
            context: Agent context with buyer/scenario info
            sim_client: Optional simulation client for inventory operations
            **kwargs: Additional args passed to FunctionalAgent
        """
        super().__init__(context, **kwargs)
        self._sim_client = sim_client
    
    @property
    def system_prompt(self) -> str:
        """System prompt for research agent."""
        return f"""You are a Research Analyst for a programmatic advertising buyer.
Your role is to discover and evaluate advertising inventory.

Buyer ID: {self.context.buyer_id}
Scenario: {self.context.scenario}
Channel: {self.context.channel or "all"}

Your capabilities:
1. ProductSearch - Find available inventory matching criteria
2. AvailsCheck - Verify availability with specific sellers
3. PricingLookup - Get current market pricing
4. CompetitiveIntel - Analyze market trends

When researching inventory:
- Consider CPM pricing relative to market rates
- Evaluate available impression volume
- Check targeting capabilities
- Assess deal type options (OA, PD, PG)

Provide clear recommendations with supporting data.
Always use tools to gather real data rather than making assumptions."""
    
    @property
    def available_tools(self) -> list[dict]:
        """Tools available to this agent."""
        return [
            ToolSchema.product_search(),
            ToolSchema.avails_check(),
            ToolSchema.pricing_lookup(),
            ToolSchema.competitive_intel(),
        ]
    
    async def _execute_tool(self, name: str, params: dict) -> ToolResult:
        """Execute a research tool.
        
        Args:
            name: Tool name
            params: Tool parameters from LLM
            
        Returns:
            ToolResult with execution outcome
        """
        if name == "ProductSearch":
            return await self._tool_product_search(params)
        elif name == "AvailsCheck":
            return await self._tool_avails_check(params)
        elif name == "PricingLookup":
            return await self._tool_pricing_lookup(params)
        elif name == "CompetitiveIntel":
            return await self._tool_competitive_intel(params)
        else:
            return ToolResult(
                tool_name=name,
                status=ToolExecutionStatus.FAILED,
                error=f"Unknown tool: {name}",
            )
    
    # -------------------------------------------------------------------------
    # High-Level Methods
    # -------------------------------------------------------------------------
    
    async def search_products(
        self,
        criteria: SearchCriteria,
    ) -> list[Product]:
        """Search for available inventory.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching products
        """
        result = await self._tool_product_search({
            "channel": criteria.channel,
            "max_cpm": criteria.max_cpm,
            "min_impressions": criteria.min_impressions,
            "query": criteria.query,
            "targeting": criteria.targeting,
        })
        
        if result.success and result.data:
            return result.data
        return []
    
    async def check_availability(
        self,
        product_id: str,
        impressions: int,
        seller_id: Optional[str] = None,
    ) -> AvailsResult:
        """Check availability for a product.
        
        Args:
            product_id: Product to check
            impressions: Number of impressions needed
            seller_id: Optional seller ID
            
        Returns:
            Availability result
        """
        result = await self._tool_avails_check({
            "product_id": product_id,
            "impressions": impressions,
            "seller_id": seller_id,
        })
        
        if result.success and result.data:
            return result.data
        
        return AvailsResult(
            available=False,
            impressions=0,
            cpm=None,
            deal_type=None,
            seller_id=seller_id or "unknown",
        )
    
    async def get_pricing(
        self,
        product_id: Optional[str] = None,
        channel: Optional[str] = None,
        deal_type: str = "OA",
    ) -> dict:
        """Get pricing information.
        
        Args:
            product_id: Optional product ID
            channel: Optional channel filter
            deal_type: Deal type (OA, PD, PG)
            
        Returns:
            Pricing data dict
        """
        result = await self._tool_pricing_lookup({
            "product_id": product_id,
            "channel": channel,
            "deal_type": deal_type,
        })
        
        return result.data if result.success else {}
    
    async def analyze_market(
        self,
        channel: Optional[str] = None,
        timeframe: str = "30d",
    ) -> dict:
        """Get competitive intelligence and market analysis.
        
        Args:
            channel: Channel to analyze
            timeframe: Analysis timeframe
            
        Returns:
            Market analysis data
        """
        result = await self._tool_competitive_intel({
            "channel": channel,
            "timeframe": timeframe,
        })
        
        return result.data if result.success else {}
    
    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------
    
    async def _tool_product_search(self, params: dict) -> ToolResult:
        """Execute ProductSearch tool."""
        try:
            # Use simulation client if available
            if self._sim_client:
                filters = {}
                if params.get("channel"):
                    filters["channel"] = params["channel"]
                if params.get("max_cpm"):
                    filters["maxPrice"] = params["max_cpm"]
                if params.get("min_impressions"):
                    filters["minImpressions"] = params["min_impressions"]
                
                result = await self._sim_client.search_products(
                    query=params.get("query"),
                    filters=filters if filters else None,
                )
                
                if result.success:
                    products = [
                        Product(
                            product_id=item.product_id,
                            seller_id=item.seller_id,
                            name=item.name,
                            channel=item.channel,
                            base_cpm=item.base_cpm,
                            floor_price=item.floor_price,
                            available_impressions=item.available_impressions,
                            targeting=item.targeting,
                            deal_type=item.deal_type.value,
                        )
                        for item in result.data
                    ]
                    return ToolResult(
                        tool_name="ProductSearch",
                        status=ToolExecutionStatus.SUCCESS,
                        data=products,
                    )
            
            # Mock response for testing without sim_client
            mock_products = [
                Product(
                    product_id=f"PROD-{uuid.uuid4().hex[:8]}",
                    seller_id=f"seller-{i:03d}",
                    name=f"Premium {params.get('channel', 'display').upper()} Inventory",
                    channel=params.get("channel", "display"),
                    base_cpm=15.0 + (i * 2),
                    floor_price=12.0 + (i * 1.5),
                    available_impressions=500000 * (i + 1),
                    targeting=["geo", "demographic"],
                    deal_type="OA",
                )
                for i in range(3)
            ]
            
            # Apply filters
            if params.get("max_cpm"):
                mock_products = [
                    p for p in mock_products
                    if p.base_cpm <= params["max_cpm"]
                ]
            
            if params.get("min_impressions"):
                mock_products = [
                    p for p in mock_products
                    if p.available_impressions >= params["min_impressions"]
                ]
            
            return ToolResult(
                tool_name="ProductSearch",
                status=ToolExecutionStatus.SUCCESS,
                data=mock_products,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="ProductSearch",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_avails_check(self, params: dict) -> ToolResult:
        """Execute AvailsCheck tool."""
        try:
            seller_id = params.get("seller_id", "unknown")
            impressions = params.get("impressions", 100000)
            
            # Use simulation client if available
            if self._sim_client and seller_id != "unknown":
                result = await self._sim_client.check_avails(
                    seller_id=seller_id,
                    channel=params.get("channel", "display"),
                    impressions=impressions,
                )
                
                if result.success:
                    data = result.data
                    avails = AvailsResult(
                        available=data.get("available", False),
                        impressions=data.get("impressions", 0),
                        cpm=data.get("cpm"),
                        deal_type=data.get("deal_type"),
                        seller_id=seller_id,
                    )
                    return ToolResult(
                        tool_name="AvailsCheck",
                        status=ToolExecutionStatus.SUCCESS,
                        data=avails,
                    )
            
            # Mock response
            avails = AvailsResult(
                available=True,
                impressions=min(impressions, 1000000),
                cpm=18.50,
                deal_type="OA",
                seller_id=seller_id,
            )
            
            return ToolResult(
                tool_name="AvailsCheck",
                status=ToolExecutionStatus.SUCCESS,
                data=avails,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="AvailsCheck",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_pricing_lookup(self, params: dict) -> ToolResult:
        """Execute PricingLookup tool."""
        try:
            channel = params.get("channel", "display")
            deal_type = params.get("deal_type", "OA")
            
            # Pricing varies by channel and deal type
            base_prices = {
                "display": {"OA": 15.0, "PD": 18.0, "PG": 22.0},
                "video": {"OA": 25.0, "PD": 30.0, "PG": 35.0},
                "ctv": {"OA": 35.0, "PD": 42.0, "PG": 50.0},
                "native": {"OA": 12.0, "PD": 15.0, "PG": 18.0},
                "mobile": {"OA": 10.0, "PD": 13.0, "PG": 16.0},
            }
            
            channel_prices = base_prices.get(channel, base_prices["display"])
            avg_cpm = channel_prices.get(deal_type, channel_prices["OA"])
            
            pricing_data = {
                "channel": channel,
                "deal_type": deal_type,
                "average_cpm": avg_cpm,
                "floor_cpm": avg_cpm * 0.7,
                "ceiling_cpm": avg_cpm * 1.4,
                "trend": "stable",
                "last_updated": datetime.utcnow().isoformat(),
            }
            
            return ToolResult(
                tool_name="PricingLookup",
                status=ToolExecutionStatus.SUCCESS,
                data=pricing_data,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="PricingLookup",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_competitive_intel(self, params: dict) -> ToolResult:
        """Execute CompetitiveIntel tool."""
        try:
            channel = params.get("channel", "display")
            timeframe = params.get("timeframe", "30d")
            
            intel_data = {
                "channel": channel,
                "timeframe": timeframe,
                "market_trends": {
                    "cpm_trend": "+3.2%",
                    "inventory_availability": "high",
                    "competition_level": "moderate",
                },
                "top_categories": [
                    {"name": "Auto", "share": 0.18},
                    {"name": "Finance", "share": 0.15},
                    {"name": "Retail", "share": 0.12},
                ],
                "recommendations": [
                    "Consider PG deals for premium inventory",
                    "Video inventory showing higher engagement",
                    "Mobile CPMs trending lower - opportunity",
                ],
                "generated_at": datetime.utcnow().isoformat(),
            }
            
            return ToolResult(
                tool_name="CompetitiveIntel",
                status=ToolExecutionStatus.SUCCESS,
                data=intel_data,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="CompetitiveIntel",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
