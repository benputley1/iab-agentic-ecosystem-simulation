"""
L2 Mobile App Inventory Agent - Mobile App Ad Management.

Manages mobile app advertising inventory including:
- In-app placements
- Rewarded video
- Interstitials
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


# Mobile ad formats
MOBILE_FORMATS = {
    "rewarded_video": {
        "name": "Rewarded Video",
        "tier": "premium",
        "avg_engagement": 0.85,
        "completion_rate": 0.92,
    },
    "interstitial": {
        "name": "Interstitial",
        "tier": "premium",
        "avg_engagement": 0.65,
        "completion_rate": 0.70,
    },
    "interstitial_video": {
        "name": "Video Interstitial",
        "tier": "premium",
        "avg_engagement": 0.70,
        "completion_rate": 0.75,
    },
    "banner": {
        "name": "Banner",
        "tier": "standard",
        "avg_engagement": 0.15,
    },
    "native": {
        "name": "Native In-App",
        "tier": "standard",
        "avg_engagement": 0.35,
    },
    "playable": {
        "name": "Playable Ad",
        "tier": "premium",
        "avg_engagement": 0.75,
    },
}

# App categories
APP_CATEGORIES = {
    "gaming": {"avg_cpm": 15.0, "volume": "high"},
    "social": {"avg_cpm": 12.0, "volume": "very_high"},
    "utility": {"avg_cpm": 8.0, "volume": "high"},
    "entertainment": {"avg_cpm": 10.0, "volume": "high"},
    "finance": {"avg_cpm": 18.0, "volume": "medium"},
}


class MobileAppInventoryAgent(SpecialistAgent):
    """
    L2 Mobile App Inventory Specialist Agent.
    
    Manages mobile app advertising inventory with focus on:
    - User engagement optimization
    - App category targeting
    - Privacy-compliant targeting
    """
    
    # Mobile-specific pricing
    DEFAULT_FLOOR_CPM = 4.0
    DEFAULT_BASE_CPM = 12.0
    DEFAULT_MAX_CPM = 35.0
    
    # Format multipliers
    FORMAT_MULTIPLIERS = {
        "rewarded_video": 1.5,
        "interstitial_video": 1.3,
        "interstitial": 1.2,
        "playable": 1.4,
        "native": 1.0,
        "banner": 0.6,
    }
    
    # OS multipliers
    OS_MULTIPLIERS = {
        "ios": 1.2,
        "android": 1.0,
        "both": 1.1,
    }
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "MobileAppInventoryAgent",
        anthropic_client: AsyncAnthropic | None = None,
        state_backend: StateBackend | None = None,
        supported_formats: list[str] | None = None,
    ):
        """Initialize mobile app inventory agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            channel="mobile",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
        )
        
        self.supported_formats = supported_formats or list(MOBILE_FORMATS.keys())
        
        # Daily avails by format
        self._daily_avails: dict[str, int] = {
            "rewarded_video": 200_000,
            "interstitial": 300_000,
            "interstitial_video": 150_000,
            "banner": 1_000_000,
            "native": 400_000,
            "playable": 50_000,
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for mobile inventory specialist."""
        return f"""You are an L2 Mobile App Inventory Specialist Agent managing mobile app advertising.

Channel: MOBILE
Supported Formats: {', '.join(self.supported_formats)}

Your Responsibilities:
1. Evaluate mobile app inventory availability
2. Calculate pricing for various ad formats
3. Optimize for user engagement
4. Handle privacy-compliant targeting (ATT/GAID)

Mobile-Specific Knowledge:
- Premium formats: Rewarded video (92% completion), playable ads
- High engagement in gaming and social apps
- iOS users typically higher LTV

Pricing Guidelines:
- Floor CPM: ${self.DEFAULT_FLOOR_CPM:.2f}
- Base CPM: ${self.DEFAULT_BASE_CPM:.2f}
- Max CPM: ${self.DEFAULT_MAX_CPM:.2f}

Pricing Factors:
- Rewarded video: +50% premium (user opt-in)
- Playable ads: +40% premium
- iOS: +20% premium
- Banner: -40% discount"""
    
    def get_channel_capabilities(self) -> list[str]:
        """Return list of mobile channel capabilities."""
        return [
            "rewarded_video",
            "interstitial_ads",
            "playable_ads",
            "in_app_banner",
            "native_in_app",
            "ios_targeting",
            "android_targeting",
            "app_category_targeting",
            "device_id_targeting",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str
    ) -> list[DelegationRequest]:
        """Plan delegations to L3 agents for mobile objectives."""
        delegations = []
        objective_lower = objective.lower()
        
        if "avail" in objective_lower or "availability" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="avails_agent",
                task=f"Check mobile inventory availability: {objective}",
                context_additions={
                    "channel": "mobile",
                    "formats": self.supported_formats,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "price" in objective_lower or "pricing" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"Calculate mobile pricing: {objective}",
                context_additions={
                    "channel": "mobile",
                    "floor_cpm": self.DEFAULT_FLOOR_CPM,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "install" in objective_lower or "cpi" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="performance_forecaster",
                task=f"Forecast app install performance: {objective}",
                context_additions={"channel": "mobile", "kpi": "installs"},
                priority=ContextPriority.MEDIUM,
            ))
        
        if not delegations:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"General mobile inventory analysis: {objective}",
                context_additions={"channel": "mobile"},
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations
    
    async def check_availability(self, request: AvailsRequest) -> AvailsResponse:
        """Check mobile app inventory availability."""
        format_type = request.targeting.get("format", "rewarded_video")
        formats = [format_type] if format_type in self._daily_avails else self.supported_formats
        
        total_available = sum(
            self._daily_avails.get(f, 0) * 30 for f in formats
        )
        
        # Apply OS filter
        target_os = request.targeting.get("os", "both")
        if target_os in ["ios", "android"]:
            total_available = int(total_available * 0.5)
        
        available = total_available >= request.impressions_needed
        fill_rate = min(1.0, total_available / max(1, request.impressions_needed))
        
        format_info = MOBILE_FORMATS.get(format_type, {})
        
        return AvailsResponse(
            available=available,
            impressions_available=total_available,
            recommended_allocation=min(request.impressions_needed, total_available),
            constraints_met=fill_rate >= 0.8,
            notes=f"Mobile {format_type}. Engagement: {format_info.get('avg_engagement', 0.5):.0%}",
            details={
                "formats": formats,
                "target_os": target_os,
                "avg_engagement": format_info.get("avg_engagement", 0.5),
            },
        )
    
    async def calculate_price(self, request: PricingRequest) -> PricingResponse:
        """Calculate price for mobile inventory."""
        base_cpm = self.DEFAULT_BASE_CPM
        
        # Format adjustment
        ad_format = request.targeting.get("format", "banner")
        if ad_format in self.FORMAT_MULTIPLIERS:
            base_cpm *= self.FORMAT_MULTIPLIERS[ad_format]
        
        # OS adjustment
        target_os = request.targeting.get("os", "both")
        if target_os in self.OS_MULTIPLIERS:
            base_cpm *= self.OS_MULTIPLIERS[target_os]
        
        # App category adjustment
        app_category = request.targeting.get("app_category")
        if app_category and app_category in APP_CATEGORIES:
            category_cpm = APP_CATEGORIES[app_category]["avg_cpm"]
            base_cpm = (base_cpm + category_cpm) / 2
        
        # Deal type
        deal_modifiers = {"pg": 1.15, "pmp": 1.10, "open": 1.0}
        base_cpm *= deal_modifiers.get(request.deal_type.lower(), 1.0)
        
        # Volume discount
        discount = 0.12 if request.impressions >= 5_000_000 else 0.08 if request.impressions >= 1_000_000 else 0.0
        final_cpm = base_cpm * (1 - discount)
        
        return PricingResponse(
            base_cpm=round(self.DEFAULT_BASE_CPM, 2),
            floor_cpm=self.DEFAULT_FLOOR_CPM,
            recommended_cpm=round(final_cpm, 2),
            discount_applied=discount,
            pricing_factors={
                "format": ad_format,
                "os": target_os,
                "app_category": app_category,
            },
            notes=f"Mobile {ad_format} pricing",
        )
    
    async def evaluate_fit(self, campaign: Campaign) -> FitScore:
        """Evaluate campaign fit for mobile inventory."""
        factors = {}
        
        # Channel fit
        channel_fit = 1.0 if "mobile" in campaign.objective.lower() or "app" in campaign.objective.lower() else 0.4
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
            price_fit = 0.3
        factors["price_alignment"] = price_fit
        
        overall = (channel_fit * 0.30 + budget_fit * 0.35 + price_fit * 0.35)
        
        return FitScore(
            score=round(overall, 3),
            confidence=0.8,
            factors=factors,
            notes="Mobile app inventory fit evaluation",
        )
