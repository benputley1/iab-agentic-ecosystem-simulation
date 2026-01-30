"""
L2 Native Inventory Agent - Native Ad Management.

Manages native advertising inventory including:
- Content recommendations
- In-feed ads
- Sponsored content
"""

from typing import Any
import structlog

from anthropic import AsyncAnthropic

from ..base.specialist import (
    SpecialistAgent,
    AvailsRequest,
    AvailsResponse,
    PricingRequest,
    PricingResponse,
    Campaign,
    FitScore,
    DelegationRequest,
)
from ..base.context import AgentContext, ContextPriority
from ..base.state import StateBackend

logger = structlog.get_logger()


# Native ad formats
NATIVE_FORMATS = {
    "in_feed": {
        "name": "In-Feed",
        "tier": "standard",
        "avg_ctr": 0.012,
        "avg_engagement": 0.25,
    },
    "content_recommendation": {
        "name": "Content Recommendation",
        "tier": "standard",
        "avg_ctr": 0.008,
        "avg_engagement": 0.18,
    },
    "sponsored_content": {
        "name": "Sponsored Content",
        "tier": "premium",
        "avg_ctr": 0.015,
        "avg_engagement": 0.35,
    },
    "branded_content": {
        "name": "Branded Content",
        "tier": "premium",
        "avg_ctr": 0.020,
        "avg_engagement": 0.40,
    },
    "search_native": {
        "name": "Search Native",
        "tier": "premium",
        "avg_ctr": 0.025,
        "avg_engagement": 0.30,
    },
}

# Publisher verticals
PUBLISHER_VERTICALS = {
    "news": {"native_lift": 1.2, "content_quality": "high"},
    "lifestyle": {"native_lift": 1.3, "content_quality": "high"},
    "technology": {"native_lift": 1.1, "content_quality": "high"},
    "entertainment": {"native_lift": 1.4, "content_quality": "medium"},
    "finance": {"native_lift": 1.0, "content_quality": "high"},
}


class NativeInventoryAgent(SpecialistAgent):
    """
    L2 Native Inventory Specialist Agent.
    
    Manages native advertising inventory with focus on:
    - Seamless content integration
    - Context matching
    - Brand storytelling
    """
    
    # Native-specific pricing
    DEFAULT_FLOOR_CPM = 3.0
    DEFAULT_BASE_CPM = 10.0
    DEFAULT_MAX_CPM = 30.0
    
    # Format multipliers
    FORMAT_MULTIPLIERS = {
        "branded_content": 1.8,
        "sponsored_content": 1.5,
        "search_native": 1.4,
        "in_feed": 1.0,
        "content_recommendation": 0.8,
    }
    
    # Context matching multipliers
    CONTEXT_MULTIPLIERS = {
        "exact": 1.3,
        "related": 1.15,
        "broad": 1.0,
        "none": 0.8,
    }
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "NativeInventoryAgent",
        anthropic_client: AsyncAnthropic | None = None,
        state_backend: StateBackend | None = None,
        supported_formats: list[str] | None = None,
    ):
        """Initialize native inventory agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            channel="native",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
        )
        
        self.supported_formats = supported_formats or list(NATIVE_FORMATS.keys())
        
        # Daily avails by format
        self._daily_avails: dict[str, int] = {
            "in_feed": 500_000,
            "content_recommendation": 800_000,
            "sponsored_content": 50_000,
            "branded_content": 20_000,
            "search_native": 200_000,
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for native inventory specialist."""
        return f"""You are an L2 Native Inventory Specialist Agent managing native advertising.

Channel: NATIVE
Supported Formats: {', '.join(self.supported_formats)}

Your Responsibilities:
1. Evaluate native inventory availability
2. Calculate pricing with context matching premiums
3. Optimize for engagement and brand storytelling
4. Ensure seamless content integration

Native-Specific Knowledge:
- Premium: Branded content, sponsored articles
- Standard: In-feed, content recommendations
- Context matching is critical for performance
- Non-disruptive user experience

Pricing Guidelines:
- Floor CPM: ${self.DEFAULT_FLOOR_CPM:.2f}
- Base CPM: ${self.DEFAULT_BASE_CPM:.2f}
- Max CPM: ${self.DEFAULT_MAX_CPM:.2f}

Pricing Factors:
- Branded content: +80% premium
- Sponsored content: +50% premium
- Exact context match: +30% premium
- Content recommendations: -20% discount"""
    
    def get_channel_capabilities(self) -> list[str]:
        """Return list of native channel capabilities."""
        return [
            "in_feed_ads",
            "content_recommendations",
            "sponsored_content",
            "branded_content",
            "context_matching",
            "brand_storytelling",
            "editorial_integration",
            "native_video",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str
    ) -> list[DelegationRequest]:
        """Plan delegations to L3 agents for native objectives."""
        delegations = []
        objective_lower = objective.lower()
        
        if "avail" in objective_lower or "availability" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="avails_agent",
                task=f"Check native inventory availability: {objective}",
                context_additions={
                    "channel": "native",
                    "formats": self.supported_formats,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "price" in objective_lower or "pricing" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"Calculate native pricing: {objective}",
                context_additions={
                    "channel": "native",
                    "floor_cpm": self.DEFAULT_FLOOR_CPM,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "context" in objective_lower or "content" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="content_analyzer",
                task=f"Analyze content context match: {objective}",
                context_additions={"channel": "native"},
                priority=ContextPriority.MEDIUM,
            ))
        
        if not delegations:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"General native inventory analysis: {objective}",
                context_additions={"channel": "native"},
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations
    
    async def check_availability(self, request: AvailsRequest) -> AvailsResponse:
        """Check native inventory availability."""
        format_type = request.targeting.get("format", "in_feed")
        formats = [format_type] if format_type in self._daily_avails else self.supported_formats
        
        total_available = sum(
            self._daily_avails.get(f, 0) * 30 for f in formats
        )
        
        # Apply context matching reduction
        context_match = request.targeting.get("context_match", "broad")
        if context_match == "exact":
            total_available = int(total_available * 0.3)
        elif context_match == "related":
            total_available = int(total_available * 0.5)
        
        available = total_available >= request.impressions_needed
        fill_rate = min(1.0, total_available / max(1, request.impressions_needed))
        
        format_info = NATIVE_FORMATS.get(format_type, {})
        
        return AvailsResponse(
            available=available,
            impressions_available=total_available,
            recommended_allocation=min(request.impressions_needed, total_available),
            constraints_met=fill_rate >= 0.8,
            notes=f"Native {format_type}. Avg CTR: {format_info.get('avg_ctr', 0.01):.2%}",
            details={
                "formats": formats,
                "context_match": context_match,
                "avg_ctr": format_info.get("avg_ctr", 0.01),
            },
        )
    
    async def calculate_price(self, request: PricingRequest) -> PricingResponse:
        """Calculate price for native inventory."""
        base_cpm = self.DEFAULT_BASE_CPM
        
        # Format adjustment
        native_format = request.targeting.get("format", "in_feed")
        if native_format in self.FORMAT_MULTIPLIERS:
            base_cpm *= self.FORMAT_MULTIPLIERS[native_format]
        
        # Context matching quality
        context_quality = request.targeting.get("context_match", "broad")
        if context_quality in self.CONTEXT_MULTIPLIERS:
            base_cpm *= self.CONTEXT_MULTIPLIERS[context_quality]
        
        # Vertical premium
        vertical = request.targeting.get("vertical")
        if vertical and vertical in PUBLISHER_VERTICALS:
            lift = PUBLISHER_VERTICALS[vertical]["native_lift"]
            base_cpm *= lift
        
        # Deal type
        deal_modifiers = {"pg": 1.20, "pmp": 1.10, "open": 1.0}
        base_cpm *= deal_modifiers.get(request.deal_type.lower(), 1.0)
        
        # Volume discount
        discount = 0.10 if request.impressions >= 2_000_000 else 0.05 if request.impressions >= 500_000 else 0.0
        final_cpm = base_cpm * (1 - discount)
        
        return PricingResponse(
            base_cpm=round(self.DEFAULT_BASE_CPM, 2),
            floor_cpm=self.DEFAULT_FLOOR_CPM,
            recommended_cpm=round(final_cpm, 2),
            discount_applied=discount,
            pricing_factors={
                "format": native_format,
                "context_match": context_quality,
                "vertical": vertical,
            },
            notes=f"Native {native_format} pricing",
        )
    
    async def evaluate_fit(self, campaign: Campaign) -> FitScore:
        """Evaluate campaign fit for native inventory."""
        factors = {}
        
        # Channel fit
        channel_fit = 1.0 if "native" in campaign.objective.lower() or "content" in campaign.objective.lower() else 0.5
        factors["channel_match"] = channel_fit
        
        # Budget fit
        min_spend = self.DEFAULT_FLOOR_CPM * (campaign.target_impressions / 1000)
        budget_fit = min(1.0, campaign.budget / max(1, min_spend))
        factors["budget_adequacy"] = budget_fit
        
        # Price fit
        if campaign.target_cpm >= self.DEFAULT_BASE_CPM:
            price_fit = 1.0
        elif campaign.target_cpm >= self.DEFAULT_FLOOR_CPM:
            price_fit = campaign.target_cpm / self.DEFAULT_BASE_CPM
        else:
            price_fit = 0.4
        factors["price_alignment"] = price_fit
        
        # Native is inherently brand-safe
        factors["brand_safety"] = 0.95
        
        overall = (channel_fit * 0.25 + budget_fit * 0.30 + price_fit * 0.25 + factors["brand_safety"] * 0.20)
        
        return FitScore(
            score=round(overall, 3),
            confidence=0.85,
            factors=factors,
            notes="Native provides premium brand-safe environment",
        )
