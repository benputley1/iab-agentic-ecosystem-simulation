"""
L2 Display Inventory Agent - Banner and Rich Media Management.

Manages display advertising inventory including:
- Banner placements (standard IAB sizes)
- Rich media units
- Interactive display formats
"""

from datetime import datetime, timedelta
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


# Standard IAB Display Ad Sizes
IAB_DISPLAY_SIZES = {
    # Desktop
    "300x250": {"name": "Medium Rectangle", "tier": "premium"},
    "728x90": {"name": "Leaderboard", "tier": "premium"},
    "160x600": {"name": "Wide Skyscraper", "tier": "standard"},
    "300x600": {"name": "Half Page", "tier": "premium"},
    "970x250": {"name": "Billboard", "tier": "premium"},
    "320x50": {"name": "Mobile Leaderboard", "tier": "standard"},
    # Mobile
    "300x50": {"name": "Mobile Banner", "tier": "standard"},
    "320x100": {"name": "Large Mobile Banner", "tier": "standard"},
    "320x480": {"name": "Mobile Interstitial", "tier": "premium"},
    # Rising Stars
    "970x90": {"name": "Super Leaderboard", "tier": "premium"},
}


class DisplayInventoryAgent(SpecialistAgent):
    """
    L2 Display Inventory Specialist Agent.
    
    Manages display advertising inventory including banners, rich media,
    and standard IAB display formats. Optimizes for viewability and
    engagement metrics.
    """
    
    # Display-specific pricing
    DEFAULT_FLOOR_CPM = 2.0
    DEFAULT_BASE_CPM = 8.0
    DEFAULT_MAX_CPM = 25.0
    
    # Premium placement multipliers
    PLACEMENT_MULTIPLIERS = {
        "above_fold": 1.4,
        "in_view": 1.2,
        "sidebar": 0.9,
        "below_fold": 0.7,
        "interstitial": 1.5,
    }
    
    # Size tier pricing
    SIZE_TIER_MULTIPLIERS = {
        "premium": 1.3,
        "standard": 1.0,
        "remnant": 0.7,
    }
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "DisplayInventoryAgent",
        anthropic_client: AsyncAnthropic | None = None,
        state_backend: StateBackend | None = None,
        supported_sizes: list[str] | None = None,
    ):
        """
        Initialize display inventory agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Agent name
            anthropic_client: Anthropic client for LLM calls
            state_backend: State persistence backend
            supported_sizes: List of supported ad sizes (default: all IAB sizes)
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            channel="display",
            anthropic_client=anthropic_client,
            state_backend=state_backend,
        )
        
        # Supported sizes for this inventory
        self.supported_sizes = supported_sizes or list(IAB_DISPLAY_SIZES.keys())
        
        # Track available impressions by size
        self._daily_avails: dict[str, int] = {
            size: 100_000 for size in self.supported_sizes
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for display inventory specialist."""
        return f"""You are an L2 Display Inventory Specialist Agent managing display advertising inventory.

Channel: DISPLAY
Supported Formats: Banner ads, rich media, interactive display

Your Responsibilities:
1. Evaluate display inventory availability requests
2. Calculate optimal pricing based on placement and size
3. Assess campaign fit for display inventory
4. Coordinate L3 functional agents for detailed analysis

Display-Specific Knowledge:
- Supported IAB sizes: {', '.join(self.supported_sizes[:5])}...
- Premium placements: above-fold, interstitial, high-viewability
- Key metrics: viewability rate, CTR, engagement rate
- Rich media options: expandable, video-in-banner, interactive

Pricing Guidelines:
- Floor CPM: ${self.DEFAULT_FLOOR_CPM:.2f}
- Base CPM: ${self.DEFAULT_BASE_CPM:.2f}
- Max CPM: ${self.DEFAULT_MAX_CPM:.2f}

Pricing Factors:
- Above-fold placement: +40% premium
- Premium sizes (300x250, 728x90): +30% premium
- Interstitial: +50% premium
- Below-fold: -30% discount

Always provide actionable recommendations based on inventory capabilities."""
    
    def get_channel_capabilities(self) -> list[str]:
        """Return list of display channel capabilities."""
        return [
            "banner_placement",
            "rich_media",
            "standard_iab_sizes",
            "viewability_optimization",
            "above_fold_premium",
            "interstitial_ads",
            "expandable_units",
            "video_in_banner",
        ]
    
    async def plan_execution(
        self,
        context: AgentContext,
        objective: str
    ) -> list[DelegationRequest]:
        """
        Plan delegations to L3 functional agents for display objectives.
        
        Args:
            context: Context from L1 orchestrator
            objective: Display-specific objective
            
        Returns:
            List of delegation requests for L3 agents
        """
        delegations = []
        objective_lower = objective.lower()
        
        # Determine which L3 agents to invoke based on objective
        if "avail" in objective_lower or "availability" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="avails_agent",
                task=f"Check display inventory availability: {objective}",
                context_additions={"channel": "display", "sizes": self.supported_sizes},
                priority=ContextPriority.HIGH,
            ))
        
        if "price" in objective_lower or "pricing" in objective_lower or "cpm" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"Calculate display pricing: {objective}",
                context_additions={
                    "channel": "display",
                    "floor_cpm": self.DEFAULT_FLOOR_CPM,
                    "base_cpm": self.DEFAULT_BASE_CPM,
                },
                priority=ContextPriority.HIGH,
            ))
        
        if "fit" in objective_lower or "evaluate" in objective_lower or "campaign" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="proposal_review_agent",
                task=f"Evaluate campaign fit for display: {objective}",
                context_additions={"channel": "display"},
                priority=ContextPriority.MEDIUM,
            ))
        
        if "creative" in objective_lower or "compliance" in objective_lower:
            delegations.append(DelegationRequest(
                functional_agent_id="creative_validator",
                task=f"Validate display creative: {objective}",
                context_additions={"supported_sizes": self.supported_sizes},
                priority=ContextPriority.MEDIUM,
            ))
        
        # If no specific delegations, do general analysis
        if not delegations:
            delegations.append(DelegationRequest(
                functional_agent_id="pricing_agent",
                task=f"General display inventory analysis: {objective}",
                context_additions={"channel": "display"},
                priority=ContextPriority.MEDIUM,
            ))
        
        return delegations
    
    # -------------------------------------------------------------------------
    # Display-Specific Business Logic Methods
    # -------------------------------------------------------------------------
    
    async def check_availability(self, request: AvailsRequest) -> AvailsResponse:
        """
        Check display inventory availability.
        
        Args:
            request: Availability check request
            
        Returns:
            Availability response with details
        """
        # Get sizes from targeting or use defaults
        sizes = request.targeting.get("sizes", self.supported_sizes[:3])
        placement = request.targeting.get("placement", "standard")
        
        # Calculate available impressions
        total_available = 0
        for size in sizes:
            if size in self._daily_avails:
                # Assume 30-day window if not specified
                days = 30
                total_available += self._daily_avails[size] * days
        
        # Check if request can be fulfilled
        impressions_needed = request.impressions_needed
        available = total_available >= impressions_needed
        
        # Calculate fill rate
        fill_rate = min(1.0, total_available / max(1, impressions_needed))
        
        # Determine quality tier
        if placement in ["above_fold", "interstitial"]:
            quality = "premium"
        elif placement in ["sidebar", "below_fold"]:
            quality = "remnant"
        else:
            quality = "standard"
        
        return AvailsResponse(
            available=available,
            impressions_available=total_available,
            recommended_allocation=min(impressions_needed, total_available),
            constraints_met=fill_rate >= 0.8,
            notes=f"Display inventory at {fill_rate:.0%} fill rate. Quality: {quality}",
            details={
                "sizes_checked": sizes,
                "placement": placement,
                "quality_tier": quality,
                "fill_rate": fill_rate,
            },
        )
    
    async def calculate_price(self, request: PricingRequest) -> PricingResponse:
        """
        Calculate price for display inventory.
        
        Args:
            request: Pricing calculation request
            
        Returns:
            Pricing response with CPM details
        """
        base_cpm = self.DEFAULT_BASE_CPM
        
        # Apply deal type modifier
        deal_modifiers = {
            "pg": 1.3,
            "pmp": 1.15,
            "open": 1.0,
        }
        base_cpm *= deal_modifiers.get(request.deal_type.lower(), 1.0)
        
        # Apply placement modifier
        placement = request.targeting.get("placement", "standard")
        if placement in self.PLACEMENT_MULTIPLIERS:
            base_cpm *= self.PLACEMENT_MULTIPLIERS[placement]
        
        # Apply volume discount
        discount = 0.0
        if request.impressions >= 1_000_000:
            discount = 0.10
        elif request.impressions >= 500_000:
            discount = 0.05
        
        final_cpm = base_cpm * (1 - discount)
        
        return PricingResponse(
            base_cpm=round(self.DEFAULT_BASE_CPM, 2),
            floor_cpm=self.DEFAULT_FLOOR_CPM,
            recommended_cpm=round(final_cpm, 2),
            discount_applied=discount,
            pricing_factors={
                "deal_type": request.deal_type,
                "placement": placement,
                "volume_discount": discount,
            },
            notes=f"Display CPM calculated for {request.impressions:,} impressions",
        )
    
    async def evaluate_fit(self, campaign: Campaign) -> FitScore:
        """
        Evaluate how well a campaign fits display inventory.
        
        Args:
            campaign: Campaign to evaluate
            
        Returns:
            Fit score with breakdown
        """
        factors = {}
        
        # Channel fit
        channel_fit = 1.0 if "display" in campaign.objective.lower() else 0.5
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
        
        # Overall score
        overall = (channel_fit * 0.3 + budget_fit * 0.35 + price_fit * 0.35)
        
        return FitScore(
            score=round(overall, 3),
            confidence=0.85,
            factors=factors,
            notes="Display inventory fit evaluation complete",
        )
