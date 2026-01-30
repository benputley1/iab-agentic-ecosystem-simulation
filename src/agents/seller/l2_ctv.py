"""
L2 CTV Inventory Agent - Connected TV Management.

Manages Connected TV advertising inventory including:
- Streaming app inventory
- Household targeting
- Premium placement management
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


# CTV platform characteristics
CTV_PLATFORMS = {
    "roku": {"name": "Roku", "tier": "premium", "household_reach": 0.35},
    "fire_tv": {"name": "Amazon Fire TV", "tier": "premium", "household_reach": 0.25},
    "apple_tv": {"name": "Apple TV", "tier": "premium", "household_reach": 0.10},
    "samsung_tv": {"name": "Samsung Smart TV", "tier": "standard", "household_reach": 0.20},
    "lg_tv": {"name": "LG Smart TV", "tier": "standard", "household_reach": 0.12},
    "chromecast": {"name": "Chromecast", "tier": "standard", "household_reach": 0.15},
}

# Content types
CTV_CONTENT_TYPES = {
    "live_sports": {"premium": True, "avg_cpm": 45.0, "completion_rate": 0.92},
    "premium_entertainment": {"premium": True, "avg_cpm": 35.0, "completion_rate": 0.88},
    "news": {"premium": False, "avg_cpm": 25.0, "completion_rate": 0.80},
    "movies": {"premium": True, "avg_cpm": 30.0, "completion_rate": 0.85},
    "avod": {"premium": False, "avg_cpm": 20.0, "completion_rate": 0.75},
}


class CTVInventoryAgent(SpecialistAgent):
    """
    L2 CTV (Connected TV) Inventory Specialist Agent.
    
    Manages premium CTV advertising inventory with focus on:
    - Household-level targeting
    - Cross-device reach
    - Premium content adjacency
    """
    
    # CTV-specific pricing (premium channel)
    DEFAULT_FLOOR_CPM = 15.0
    DEFAULT_BASE_CPM = 30.0
    DEFAULT_MAX_CPM = 75.0
    
    # Content type multipliers
    CONTENT_MULTIPLIERS = {
        "live_sports": 1.5,
        "premium_entertainment": 1.3,
        "movies": 1.2,
        "news": 1.0,
        "avod": 0.9,
    }
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "CTVInventoryAgent",
        anthropic_client: AsyncAnthropic | None = None,
        state_backend: StateBackend | None = None,
        supported_platforms: list[str] | None = None,
    ):
        """Initialize CTV inventory agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            channel="ctv",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
        )
        
        self.supported_platforms = supported_platforms or list(CTV_PLATFORMS.keys())
        
        # Daily avails by platform
        self._daily_avails: dict[str, int] = {
            platform: int(CTV_PLATFORMS[platform]["household_reach"] * 100_000)
            for platform in self.supported_platforms
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for CTV inventory specialist."""
        return f"""You are an L2 CTV Inventory Specialist Agent managing Connected TV advertising inventory.

Channel: CTV (Connected TV)
Supported Platforms: {', '.join(self.supported_platforms)}

Your Responsibilities:
1. Evaluate CTV inventory availability across streaming platforms
2. Calculate premium pricing for big-screen experience
3. Optimize for household-level targeting
4. Ensure brand-safe, premium content placement

CTV-Specific Knowledge:
- Non-skippable by default (95%+ completion)
- Co-viewing multiplier (avg 2.1 viewers per impression)
- Household graph for targeting
- Premium, brand-safe environment

Pricing Guidelines:
- Floor CPM: ${self.DEFAULT_FLOOR_CPM:.2f}
- Base CPM: ${self.DEFAULT_BASE_CPM:.2f}
- Max CPM: ${self.DEFAULT_MAX_CPM:.2f}

Pricing Factors:
- Live sports: +50% premium
- Premium entertainment: +30% premium
- Household targeting: +20% premium
- Cross-device reach: +15% premium"""
    
    def get_channel_capabilities(self) -> list[str]:
        """Return list of CTV channel capabilities."""
        return [
            "streaming_app_inventory",
            "household_targeting",
            "cross_device_reach",
            "premium_content_adjacency",
            "non_skippable_ads",
            "live_sports_inventory",
            "co_viewing_attribution",
            "brand_safe_environment",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str
    ) -> list[DelegationRequest]:
        """Plan delegations to L3 agents for CTV objectives."""
        delegations = []
        objective_lower = objective.lower()
        
        if "avail" in objective_lower or "availability" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="avails_agent",
                task=f"Check CTV inventory availability: {objective}",
                context_additions={
                    "channel": "ctv",
                    "platforms": self.supported_platforms,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "price" in objective_lower or "pricing" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"Calculate CTV pricing: {objective}",
                context_additions={
                    "channel": "ctv",
                    "floor_cpm": self.DEFAULT_FLOOR_CPM,
                    "premium_channel": True,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "household" in objective_lower or "targeting" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="audience_validator",
                task=f"Validate household targeting: {objective}",
                context_additions={"channel": "ctv"},
                priority=ContextPriority.MEDIUM,
            ))
        
        if not delegations:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"General CTV inventory analysis: {objective}",
                context_additions={"channel": "ctv"},
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations
    
    async def check_availability(self, request: AvailsRequest) -> AvailsResponse:
        """Check CTV inventory availability."""
        platforms = request.targeting.get("platforms", self.supported_platforms)
        platforms = [p for p in platforms if p in self._daily_avails]
        
        total_available = sum(
            self._daily_avails.get(p, 0) * 30 for p in platforms
        )
        
        # Household targeting reduces available but increases quality
        if request.targeting.get("household_targeting"):
            total_available = int(total_available * 0.6)
        
        available = total_available >= request.impressions_needed
        fill_rate = min(1.0, total_available / max(1, request.impressions_needed))
        
        # Calculate household reach
        household_reach = sum(
            CTV_PLATFORMS[p]["household_reach"] for p in platforms if p in CTV_PLATFORMS
        )
        
        return AvailsResponse(
            available=available,
            impressions_available=total_available,
            recommended_allocation=min(request.impressions_needed, total_available),
            constraints_met=fill_rate >= 0.8,
            notes=f"CTV inventory. Household reach: {household_reach:.0%}",
            details={
                "platforms": platforms,
                "household_reach": household_reach,
                "completion_rate": 0.95,  # CTV is non-skippable
            },
        )
    
    async def calculate_price(self, request: PricingRequest) -> PricingResponse:
        """Calculate price for CTV inventory."""
        base_cpm = self.DEFAULT_BASE_CPM
        
        # Content type adjustment
        content_type = request.targeting.get("content_type", "avod")
        if content_type in self.CONTENT_MULTIPLIERS:
            base_cpm *= self.CONTENT_MULTIPLIERS[content_type]
        
        # Household targeting premium
        if request.targeting.get("household_targeting"):
            base_cpm *= 1.20
        
        # Cross-device premium
        if request.targeting.get("cross_device"):
            base_cpm *= 1.15
        
        # Deal type modifier (smaller for premium CTV)
        deal_modifiers = {"pg": 1.20, "pmp": 1.10, "open": 1.0}
        base_cpm *= deal_modifiers.get(request.deal_type.lower(), 1.0)
        
        # Volume discount (smaller for CTV)
        discount = 0.05 if request.impressions >= 1_000_000 else 0.0
        final_cpm = base_cpm * (1 - discount)
        
        return PricingResponse(
            base_cpm=round(self.DEFAULT_BASE_CPM, 2),
            floor_cpm=self.DEFAULT_FLOOR_CPM,
            recommended_cpm=round(final_cpm, 2),
            discount_applied=discount,
            pricing_factors={
                "content_type": content_type,
                "household_targeting": request.targeting.get("household_targeting", False),
                "cross_device": request.targeting.get("cross_device", False),
            },
            notes="CTV premium inventory pricing",
        )
    
    async def evaluate_fit(self, campaign: Campaign) -> FitScore:
        """Evaluate campaign fit for CTV inventory."""
        factors = {}
        
        # Channel fit
        channel_fit = 1.0 if "ctv" in campaign.objective.lower() or "tv" in campaign.objective.lower() else 0.3
        factors["channel_match"] = channel_fit
        
        # Budget fit - CTV requires significant budget
        min_spend = self.DEFAULT_FLOOR_CPM * (campaign.target_impressions / 1000)
        budget_fit = min(1.0, campaign.budget / max(1, min_spend))
        factors["budget_adequacy"] = budget_fit
        
        # Price fit - stricter for CTV
        if campaign.target_cpm >= self.DEFAULT_BASE_CPM:
            price_fit = 1.0
        elif campaign.target_cpm >= self.DEFAULT_FLOOR_CPM:
            price_fit = campaign.target_cpm / self.DEFAULT_BASE_CPM
        else:
            price_fit = 0.1  # Below floor is very poor fit
        factors["price_alignment"] = price_fit
        
        overall = (channel_fit * 0.25 + budget_fit * 0.35 + price_fit * 0.40)
        
        return FitScore(
            score=round(overall, 3),
            confidence=0.85,
            factors=factors,
            notes="CTV inventory requires premium budget",
        )
