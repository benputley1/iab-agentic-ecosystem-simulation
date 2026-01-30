"""
L2 Video Inventory Agent - Video Ad Management.

Manages video advertising inventory including:
- Pre-roll, mid-roll, post-roll placements
- In-stream vs out-stream formats
- Completion rate optimization
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


# Video placement types and characteristics
VIDEO_PLACEMENTS = {
    "pre_roll": {
        "name": "Pre-Roll",
        "tier": "premium",
        "avg_completion_rate": 0.75,
        "skip_rate": 0.25,
    },
    "mid_roll": {
        "name": "Mid-Roll",
        "tier": "premium",
        "avg_completion_rate": 0.85,
        "skip_rate": 0.15,
    },
    "post_roll": {
        "name": "Post-Roll",
        "tier": "standard",
        "avg_completion_rate": 0.60,
        "skip_rate": 0.10,
    },
    "in_stream": {
        "name": "In-Stream",
        "tier": "premium",
        "avg_completion_rate": 0.70,
        "skip_rate": 0.30,
    },
    "out_stream": {
        "name": "Out-Stream",
        "tier": "standard",
        "avg_completion_rate": 0.45,
        "skip_rate": 0.50,
    },
}


class VideoInventoryAgent(SpecialistAgent):
    """
    L2 Video Inventory Specialist Agent.
    
    Manages video advertising inventory with focus on:
    - Maximizing completion rates
    - Optimizing for viewability
    - Balancing user experience with monetization
    """
    
    # Video-specific pricing (higher than display)
    DEFAULT_FLOOR_CPM = 8.0
    DEFAULT_BASE_CPM = 18.0
    DEFAULT_MAX_CPM = 45.0
    
    # Placement multipliers
    PLACEMENT_MULTIPLIERS = {
        "pre_roll": 1.3,
        "mid_roll": 1.4,
        "post_roll": 0.8,
        "in_stream": 1.2,
        "out_stream": 0.9,
    }
    
    # Duration multipliers
    DURATION_MULTIPLIERS = {
        6: 0.7,   # 6-second bumper
        15: 1.0,  # Standard
        30: 1.2,  # Full spot
        60: 1.0,  # Long form
    }
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "VideoInventoryAgent",
        anthropic_client: AsyncAnthropic | None = None,
        state_backend: StateBackend | None = None,
        supported_placements: list[str] | None = None,
    ):
        """Initialize video inventory agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            channel="video",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
        )
        
        self.supported_placements = supported_placements or list(VIDEO_PLACEMENTS.keys())
        
        # Daily avails by placement type
        self._daily_avails: dict[str, int] = {
            "pre_roll": 50_000,
            "mid_roll": 30_000,
            "post_roll": 40_000,
            "in_stream": 60_000,
            "out_stream": 100_000,
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for video inventory specialist."""
        return f"""You are an L2 Video Inventory Specialist Agent managing video advertising inventory.

Channel: VIDEO
Supported Placements: {', '.join(self.supported_placements)}

Your Responsibilities:
1. Evaluate video inventory availability
2. Calculate optimal pricing based on placement and duration
3. Assess campaign fit for video inventory
4. Optimize for completion rates and viewability

Video-Specific Knowledge:
- Pre-roll: 75% avg completion, premium pricing
- Mid-roll: 85% avg completion, highest engagement
- Post-roll: 60% avg completion, lower pricing
- In-stream: Native video experience
- Out-stream: Autoplay in non-video content

Pricing Guidelines:
- Floor CPM: ${self.DEFAULT_FLOOR_CPM:.2f}
- Base CPM: ${self.DEFAULT_BASE_CPM:.2f}
- Max CPM: ${self.DEFAULT_MAX_CPM:.2f}

Pricing Factors:
- Mid-roll: +40% premium (highest completion)
- Pre-roll: +30% premium
- Non-skippable: +25% premium
- Post-roll: -20% discount

Completion Rate Benchmarks:
- Pre-roll: 75%
- Mid-roll: 85%
- Post-roll: 60%
- Out-stream: 45%"""
    
    def get_channel_capabilities(self) -> list[str]:
        """Return list of video channel capabilities."""
        return [
            "pre_roll_placement",
            "mid_roll_placement",
            "post_roll_placement",
            "in_stream_video",
            "out_stream_video",
            "non_skippable_ads",
            "completion_rate_optimization",
            "sound_on_targeting",
            "vast_vpaid_support",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str
    ) -> list[DelegationRequest]:
        """Plan delegations to L3 agents for video objectives."""
        delegations = []
        objective_lower = objective.lower()
        
        if "avail" in objective_lower or "availability" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="avails_agent",
                task=f"Check video inventory availability: {objective}",
                context_additions={
                    "channel": "video",
                    "placements": self.supported_placements,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "price" in objective_lower or "pricing" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"Calculate video pricing: {objective}",
                context_additions={
                    "channel": "video",
                    "floor_cpm": self.DEFAULT_FLOOR_CPM,
                    "base_cpm": self.DEFAULT_BASE_CPM,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "completion" in objective_lower or "performance" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="performance_forecaster",
                task=f"Forecast video completion rates: {objective}",
                context_additions={"channel": "video"},
                priority=ContextPriority.MEDIUM,
            ))
        
        if not delegations:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"General video inventory analysis: {objective}",
                context_additions={"channel": "video"},
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations
    
    async def check_availability(self, request: AvailsRequest) -> AvailsResponse:
        """Check video inventory availability."""
        placement = request.targeting.get("placement", "pre_roll")
        placements = [placement] if placement in self._daily_avails else self.supported_placements
        
        total_available = sum(
            self._daily_avails.get(p, 0) * 30 for p in placements
        )
        
        available = total_available >= request.impressions_needed
        fill_rate = min(1.0, total_available / max(1, request.impressions_needed))
        
        placement_info = VIDEO_PLACEMENTS.get(placement, {})
        quality = placement_info.get("tier", "standard")
        
        return AvailsResponse(
            available=available,
            impressions_available=total_available,
            recommended_allocation=min(request.impressions_needed, total_available),
            constraints_met=fill_rate >= 0.8,
            notes=f"Video inventory. Est completion: {placement_info.get('avg_completion_rate', 0.7):.0%}",
            details={
                "placements": placements,
                "quality_tier": quality,
                "avg_completion_rate": placement_info.get("avg_completion_rate", 0.7),
            },
        )
    
    async def calculate_price(self, request: PricingRequest) -> PricingResponse:
        """Calculate price for video inventory."""
        base_cpm = self.DEFAULT_BASE_CPM
        
        # Deal type modifier
        deal_modifiers = {"pg": 1.25, "pmp": 1.15, "open": 1.0}
        base_cpm *= deal_modifiers.get(request.deal_type.lower(), 1.0)
        
        # Placement modifier
        placement = request.targeting.get("placement", "pre_roll")
        if placement in self.PLACEMENT_MULTIPLIERS:
            base_cpm *= self.PLACEMENT_MULTIPLIERS[placement]
        
        # Duration modifier
        duration = request.targeting.get("duration", 15)
        if duration in self.DURATION_MULTIPLIERS:
            base_cpm *= self.DURATION_MULTIPLIERS[duration]
        
        # Non-skippable premium
        if request.targeting.get("non_skippable", False):
            base_cpm *= 1.25
        
        # Volume discount
        discount = 0.08 if request.impressions >= 500_000 else 0.05 if request.impressions >= 250_000 else 0.0
        final_cpm = base_cpm * (1 - discount)
        
        return PricingResponse(
            base_cpm=round(self.DEFAULT_BASE_CPM, 2),
            floor_cpm=self.DEFAULT_FLOOR_CPM,
            recommended_cpm=round(final_cpm, 2),
            discount_applied=discount,
            pricing_factors={
                "placement": placement,
                "duration": duration,
                "non_skippable": request.targeting.get("non_skippable", False),
            },
            notes=f"Video CPM for {placement}",
        )
    
    async def evaluate_fit(self, campaign: Campaign) -> FitScore:
        """Evaluate campaign fit for video inventory."""
        factors = {}
        
        # Channel fit
        channel_fit = 1.0 if "video" in campaign.objective.lower() else 0.4
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
            price_fit = 0.2
        factors["price_alignment"] = price_fit
        
        overall = (channel_fit * 0.35 + budget_fit * 0.30 + price_fit * 0.35)
        
        return FitScore(
            score=round(overall, 3),
            confidence=0.8,
            factors=factors,
            notes="Video inventory fit evaluation",
        )
