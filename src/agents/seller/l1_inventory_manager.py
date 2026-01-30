"""Level 1 Inventory Manager for the Seller Agent System.

The Inventory Manager is the strategic orchestrator for seller-side operations,
using Claude Opus for high-stakes decisions about yield optimization,
deal acceptance, and portfolio strategy.
"""

import json
import re
import uuid
from datetime import date, datetime
from typing import Any, Optional

from anthropic import AsyncAnthropic

from ..base.orchestrator import (
    OrchestratorAgent,
    CampaignState,
    StrategicDecision,
    SpecialistAssignment,
    L1_MODEL,
)
from ..base.context import AgentContext, ContextPriority
from ..base.state import StateBackend, VolatileStateBackend

from .models import (
    AudienceSpec,
    BuyerTier,
    ChannelType,
    CounterOffer,
    CrossSellOpportunity,
    Deal,
    DealAction,
    DealDecision,
    DealRequest,
    DealTypeEnum,
    InventoryPortfolio,
    Product,
    Task,
    TaskResult,
    YieldStrategy,
)
from .prompts import (
    INVENTORY_MANAGER_SYSTEM_PROMPT,
    DEAL_EVALUATION_PROMPT,
    YIELD_OPTIMIZATION_PROMPT,
    CROSS_SELL_PROMPT,
)


# Tier discount guidelines (for context in prompts)
TIER_DISCOUNTS = {
    BuyerTier.PUBLIC: 0,
    BuyerTier.SEAT: 5,
    BuyerTier.AGENCY: 10,
    BuyerTier.ADVERTISER: 15,
}


class InventoryManager(OrchestratorAgent):
    """Level 1 Seller Orchestrator using Claude Opus.
    
    Responsibilities:
    - Yield optimization across inventory
    - Deal acceptance/rejection decisions
    - Portfolio strategy
    - Cross-sell opportunity identification
    - Coordination of L2 channel inventory specialists
    
    This agent makes strategic decisions that affect the entire
    seller's inventory portfolio.
    """
    
    def __init__(
        self,
        seller_id: str,
        portfolio: Optional[InventoryPortfolio] = None,
        anthropic_client: Optional[AsyncAnthropic] = None,
        model: str = L1_MODEL,
        state_backend: Optional[StateBackend] = None,
        mock_llm: bool = False,
        **kwargs,
    ):
        """Initialize the Inventory Manager.
        
        Args:
            seller_id: Unique identifier for this seller
            portfolio: Initial inventory portfolio (can be set later)
            anthropic_client: Anthropic client (creates one if not provided)
            model: LLM model to use (defaults to Opus)
            state_backend: State storage backend
            mock_llm: Use mock LLM (skip actual API calls)
            **kwargs: Additional args passed to OrchestratorAgent
        """
        super().__init__(
            agent_id=f"inventory-manager-{seller_id}",
            name=f"InventoryManager-{seller_id}",
            anthropic_client=anthropic_client,
            model=model,
            state_backend=state_backend,
            **kwargs,
        )
        
        self.seller_id = seller_id
        self.mock_llm = mock_llm
        self._portfolio = portfolio or InventoryPortfolio(seller_id=seller_id)
        
        # Track active deals
        self._active_deals: dict[str, Deal] = {}
        
        # Decision history for context tracking
        self._decision_history: list[dict] = []
    
    @property
    def portfolio(self) -> InventoryPortfolio:
        """Get the current inventory portfolio."""
        return self._portfolio
    
    @portfolio.setter
    def portfolio(self, value: InventoryPortfolio) -> None:
        """Set the inventory portfolio."""
        self._portfolio = value
    
    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this orchestrator agent."""
        return INVENTORY_MANAGER_SYSTEM_PROMPT
    
    async def make_strategic_decision(
        self,
        context: AgentContext,
        decision_request: str,
    ) -> StrategicDecision:
        """Make a strategic decision using Opus.
        
        Args:
            context: Current agent context
            decision_request: Description of the decision needed
            
        Returns:
            StrategicDecision with action and reasoning
        """
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": decision_request}],
        )
        
        raw_response = response.content[0].text
        
        # Parse response
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                data = json.loads(json_match.group())
                return StrategicDecision(
                    decision_type=data.get("decision_type", "general"),
                    description=data.get("description", decision_request),
                    rationale=data.get("rationale", raw_response),
                    impact=data.get("impact", {}),
                )
        except (json.JSONDecodeError, KeyError):
            pass
        
        return StrategicDecision(
            decision_type="general",
            description=decision_request,
            rationale=raw_response,
            impact={},
        )
    
    async def plan_specialist_assignments(
        self,
        context: AgentContext,
        campaign: CampaignState,
    ) -> list[SpecialistAssignment]:
        """Plan assignments to L2 channel specialists for inventory management.
        
        Args:
            context: Current agent context
            campaign: Campaign state (maps to deal/inventory request)
            
        Returns:
            List of specialist assignments
        """
        assignments = []
        
        # For seller agents, we assign based on channel inventory needs
        for channel, specialist_id in self._channel_specialists.items():
            if campaign.budget_remaining > 0:
                allocation = campaign.budget_remaining * 0.2  # Distribute across channels
                assignments.append(
                    SpecialistAssignment(
                        specialist_id=specialist_id,
                        channel=channel,
                        objective=f"Manage {channel} inventory for request {campaign.campaign_id}",
                        context_items={
                            "campaign_id": campaign.campaign_id,
                            "budget": allocation,
                        },
                        priority=ContextPriority.HIGH,
                        budget_allocation=allocation,
                    )
                )
        
        return assignments
    
    # -------------------------------------------------------------------------
    # Deal Evaluation
    # -------------------------------------------------------------------------
    
    async def evaluate_deal_request(self, request: DealRequest) -> DealDecision:
        """Strategic decision: accept, reject, or counter a deal request.
        
        Uses Claude Opus to evaluate the deal considering:
        - Yield impact on the portfolio
        - Inventory availability and commitments
        - Buyer relationship value
        - Market conditions
        
        Args:
            request: The deal request to evaluate
            
        Returns:
            DealDecision with action, pricing, and reasoning
        """
        # Use mock decision in mock mode (skip LLM call)
        if self.mock_llm:
            return self._mock_deal_decision(request)
        
        # Get product info
        product = self._portfolio.get_product_by_id(request.product_id)
        if not product:
            return DealDecision(
                request_id=request.request_id,
                action=DealAction.REJECT,
                price=0.0,
                impressions=0,
                reasoning=f"Product {request.product_id} not found in portfolio",
                confidence=1.0,
            )
        
        # Calculate context values
        channel_str = product.channel.value if isinstance(product.channel, ChannelType) else product.channel
        fill_rate = self._portfolio.fill_rate.get(channel_str, 0.5)
        committed_imps = self._get_committed_impressions(product.product_id)
        
        # Format the prompt
        prompt = DEAL_EVALUATION_PROMPT.format(
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            buyer_tier=request.buyer_tier.value if isinstance(request.buyer_tier, BuyerTier) else request.buyer_tier,
            product_id=request.product_id,
            impressions=request.impressions,
            max_cpm=request.max_cpm,
            deal_type=request.deal_type.value if isinstance(request.deal_type, DealTypeEnum) else request.deal_type,
            start_date=request.flight_dates[0].isoformat(),
            end_date=request.flight_dates[1].isoformat(),
            duration=request.duration_days,
            audience_spec=json.dumps(request.audience_spec.to_dict(), indent=2),
            total_value=request.total_value,
            floor_cpm=product.floor_cpm,
            base_cpm=product.base_cpm,
            fill_rate=fill_rate,
            daily_avails=product.daily_impressions,
            committed_impressions=committed_imps,
            tier_discount=TIER_DISCOUNTS.get(request.buyer_tier, 0),
            market_conditions="Normal demand",
            revenue_ytd=self._portfolio.revenue_ytd,
        )
        
        # Call LLM
        import time
        start = time.time()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.time() - start) * 1000
        
        # Parse response
        decision = self._parse_deal_decision(request.request_id, response.content[0].text)
        decision.model_used = self.model
        decision.latency_ms = latency_ms
        
        # Record decision
        self._decision_history.append({
            "action": f"deal_evaluation:{decision.action.value}",
            "reasoning": decision.reasoning,
            "data": {
                "request_id": request.request_id,
                "buyer_id": request.buyer_id,
                "offered_cpm": request.max_cpm,
                "decided_cpm": decision.price,
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return decision
    
    # -------------------------------------------------------------------------
    # Yield Optimization
    # -------------------------------------------------------------------------
    
    async def optimize_yield(self, portfolio: Optional[InventoryPortfolio] = None) -> YieldStrategy:
        """Optimize pricing and allocation across the inventory portfolio.
        
        Balances fill rate vs CPM to maximize overall revenue.
        
        Args:
            portfolio: Portfolio to optimize (defaults to self._portfolio)
            
        Returns:
            YieldStrategy with floor adjustments and recommendations
        """
        portfolio = portfolio or self._portfolio
        
        # Build portfolio summary
        portfolio_summary = self._build_portfolio_summary(portfolio)
        channel_breakdown = self._build_channel_breakdown(portfolio)
        
        # Calculate metrics
        avg_fill = sum(portfolio.fill_rate.values()) / max(len(portfolio.fill_rate), 1)
        avg_cpm = sum(portfolio.avg_cpm.values()) / max(len(portfolio.avg_cpm), 1)
        
        prompt = YIELD_OPTIMIZATION_PROMPT.format(
            portfolio_summary=portfolio_summary,
            channel_breakdown=channel_breakdown,
            avg_fill_rate=avg_fill,
            avg_cpm=avg_cpm,
            revenue_mtd=portfolio.revenue_ytd * 0.1,
            revenue_target=portfolio.revenue_ytd * 0.12,
            variance=(0.1 / 0.12 - 1) * 100,
            recent_deals=self._format_recent_deals(),
            seasonal_context="Q4 peak season",
            competitive_context="Moderate competition",
            demand_signals="Steady demand across channels",
        )
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        
        strategy = self._parse_yield_strategy(response.content[0].text)
        strategy.model_used = self.model
        
        # Record decision
        self._decision_history.append({
            "action": "yield_optimization",
            "reasoning": f"Generated yield strategy with {len(strategy.floor_adjustments)} adjustments",
            "data": {
                "expected_lift": strategy.expected_revenue_lift,
                "adjustments_count": len(strategy.floor_adjustments),
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return strategy
    
    # -------------------------------------------------------------------------
    # Cross-Sell
    # -------------------------------------------------------------------------
    
    async def identify_cross_sell(self, deal: Deal) -> list[CrossSellOpportunity]:
        """Find upsell/cross-sell opportunities from an existing deal.
        
        E.g., a CTV buyer might also want mobile inventory,
        or a display buyer might benefit from video.
        
        Args:
            deal: Active deal to analyze
            
        Returns:
            List of cross-sell opportunities with confidence scores
        """
        # Get buyer history
        buyer_deals = [d for d in self._active_deals.values() if d.buyer_id == deal.buyer_id]
        channels_used = list(set(
            d.product_id.split("-")[1] if "-" in d.product_id else "unknown"
            for d in buyer_deals
        ))
        avg_cpm = sum(d.agreed_cpm for d in buyer_deals) / max(len(buyer_deals), 1)
        lifetime_value = sum(d.agreed_cpm * d.impressions / 1000 for d in buyer_deals)
        
        # Get available inventory
        available_inventory = self._format_available_inventory()
        
        prompt = CROSS_SELL_PROMPT.format(
            deal_id=deal.deal_id,
            buyer_id=deal.buyer_id,
            buyer_tier=deal.buyer_tier.value if isinstance(deal.buyer_tier, BuyerTier) else deal.buyer_tier,
            product_id=deal.product_id,
            channel=deal.product_id.split("-")[1] if "-" in deal.product_id else "unknown",
            agreed_cpm=deal.agreed_cpm,
            impressions=deal.impressions,
            start_date=deal.flight_dates[0].isoformat(),
            end_date=deal.flight_dates[1].isoformat(),
            audience_spec="{}",
            deal_count=len(buyer_deals),
            channels_used=", ".join(channels_used) or "None",
            avg_cpm_paid=avg_cpm,
            lifetime_value=lifetime_value,
            available_inventory=available_inventory,
        )
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        
        opportunities = self._parse_cross_sell_opportunities(deal.deal_id, response.content[0].text)
        
        # Record
        self._decision_history.append({
            "action": "cross_sell_identification",
            "reasoning": f"Found {len(opportunities)} cross-sell opportunities",
            "data": {
                "source_deal": deal.deal_id,
                "opportunities_count": len(opportunities),
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return opportunities
    
    # -------------------------------------------------------------------------
    # Delegation
    # -------------------------------------------------------------------------
    
    async def delegate_to_channel(self, channel: str, task: Task) -> TaskResult:
        """Pass work to the appropriate L2 channel inventory agent.
        
        Args:
            channel: Channel type (display, video, ctv, mobile_app, native)
            task: Task to execute
            
        Returns:
            Result from the channel specialist
        """
        specialist = self.get_specialist_for_channel(channel)
        
        if specialist is None:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=f"No L2 specialist registered for channel: {channel}",
            )
        
        try:
            if hasattr(specialist, "process_task"):
                result = await specialist.process_task(task)
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    data=result if isinstance(result, dict) else {"result": result},
                )
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Specialist {channel} does not implement process_task",
                )
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
            )
    
    # -------------------------------------------------------------------------
    # Deal Management
    # -------------------------------------------------------------------------
    
    def add_deal(self, deal: Deal) -> None:
        """Add an active deal to tracking."""
        self._active_deals[deal.deal_id] = deal
    
    def get_active_deals(self) -> list[Deal]:
        """Get all active deals."""
        return list(self._active_deals.values())
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _get_committed_impressions(self, product_id: str) -> int:
        """Get total committed impressions for a product."""
        return sum(
            deal.remaining_impressions
            for deal in self._active_deals.values()
            if deal.product_id == product_id
        )
    
    def _mock_deal_decision(self, request: DealRequest) -> DealDecision:
        """Generate a mock deal decision without LLM call.
        
        Args:
            request: The deal request to evaluate
            
        Returns:
            DealDecision with mock accept action
        """
        # Simple mock logic: accept if CPM is reasonable
        # Accept deals where offered CPM >= $5 (a reasonable floor)
        min_acceptable_cpm = 5.0
        
        if request.max_cpm >= min_acceptable_cpm:
            return DealDecision(
                request_id=request.request_id,
                action=DealAction.ACCEPT,
                price=request.max_cpm,
                impressions=request.impressions,
                reasoning="Mock mode: Accepted deal at offered CPM",
                confidence=0.9,
            )
        else:
            return DealDecision(
                request_id=request.request_id,
                action=DealAction.COUNTER,
                price=min_acceptable_cpm,
                impressions=request.impressions,
                reasoning=f"Mock mode: Counter-offered at ${min_acceptable_cpm} CPM floor",
                confidence=0.8,
                counter_offer=CounterOffer(
                    suggested_cpm=min_acceptable_cpm,
                    suggested_impressions=request.impressions,
                    reasoning="Below minimum CPM threshold",
                ),
            )
    
    def _build_portfolio_summary(self, portfolio: InventoryPortfolio) -> str:
        """Build a text summary of the portfolio."""
        lines = [
            f"Seller: {portfolio.seller_id}",
            f"Total Products: {len(portfolio.products)}",
            f"Total Daily Impressions: {portfolio.get_total_daily_impressions():,}",
            f"Revenue YTD: ${portfolio.revenue_ytd:,.2f}",
        ]
        return "\n".join(lines)
    
    def _build_channel_breakdown(self, portfolio: InventoryPortfolio) -> str:
        """Build channel-by-channel breakdown."""
        lines = []
        for channel in ChannelType:
            products = portfolio.get_channel_products(channel)
            if products:
                imps = sum(p.daily_impressions for p in products)
                avg_floor = sum(p.floor_cpm for p in products) / len(products)
                fill = portfolio.fill_rate.get(channel.value, 0)
                lines.append(
                    f"- {channel.value.upper()}: {len(products)} products, "
                    f"{imps:,} daily imps, ${avg_floor:.2f} avg floor, {fill:.0%} fill"
                )
        return "\n".join(lines) if lines else "No channel data available"
    
    def _format_recent_deals(self) -> str:
        """Format recent deals for prompts."""
        deals = sorted(
            self._active_deals.values(),
            key=lambda d: d.created_at,
            reverse=True,
        )[:5]
        
        if not deals:
            return "No recent deals"
        
        lines = []
        for d in deals:
            lines.append(
                f"- {d.deal_id}: {d.buyer_id}, ${d.agreed_cpm:.2f} CPM, "
                f"{d.impressions:,} imps, {d.deal_type.value if isinstance(d.deal_type, DealTypeEnum) else d.deal_type}"
            )
        return "\n".join(lines)
    
    def _format_available_inventory(self) -> str:
        """Format available inventory for cross-sell prompts."""
        lines = []
        for product in self._portfolio.products[:10]:
            lines.append(
                f"- {product.product_id}: {product.name} "
                f"({product.channel.value if isinstance(product.channel, ChannelType) else product.channel}), "
                f"${product.base_cpm:.2f} CPM, {product.daily_impressions:,} daily imps"
            )
        return "\n".join(lines) if lines else "No inventory available"
    
    def _parse_deal_decision(self, request_id: str, llm_response: str) -> DealDecision:
        """Parse LLM response into a DealDecision."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                data = json.loads(json_match.group())
                
                action_str = data.get("action", "reject").lower()
                action = DealAction(action_str) if action_str in [a.value for a in DealAction] else DealAction.REJECT
                
                counter_offer = None
                if data.get("counter_offer"):
                    co = data["counter_offer"]
                    counter_offer = CounterOffer(
                        suggested_cpm=co.get("suggested_cpm", 0.0),
                        suggested_impressions=co.get("suggested_impressions", 0),
                        alternative_products=co.get("alternative_products", []),
                        reasoning=co.get("reasoning", ""),
                    )
                
                return DealDecision(
                    request_id=request_id,
                    action=action,
                    price=data.get("recommended_cpm", 0.0),
                    impressions=data.get("recommended_impressions", 0),
                    reasoning=data.get("reasoning", "No reasoning provided"),
                    counter_offer=counter_offer,
                    confidence=data.get("confidence", 0.5),
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        return DealDecision(
            request_id=request_id,
            action=DealAction.REJECT,
            price=0.0,
            impressions=0,
            reasoning=f"Failed to parse LLM response: {llm_response[:200]}...",
            confidence=0.0,
        )
    
    def _parse_yield_strategy(self, llm_response: str) -> YieldStrategy:
        """Parse LLM response into a YieldStrategy."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                data = json.loads(json_match.group())
                
                return YieldStrategy(
                    floor_adjustments=data.get("floor_adjustments", {}),
                    allocation_priorities=data.get("allocation_priorities", []),
                    pacing_recommendations=data.get("pacing_recommendations", {}),
                    insights=data.get("insights", []),
                    expected_revenue_lift=data.get("expected_revenue_lift", 0.0),
                    expected_fill_rate_change=data.get("expected_fill_rate_change", 0.0),
                    confidence=data.get("confidence", 0.5),
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        return YieldStrategy(
            insights=["Failed to parse optimization response"],
            confidence=0.0,
        )
    
    def _parse_cross_sell_opportunities(
        self,
        source_deal_id: str,
        llm_response: str,
    ) -> list[CrossSellOpportunity]:
        """Parse LLM response into CrossSellOpportunity list."""
        opportunities = []
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                data = json.loads(json_match.group())
                
                for opp in data.get("opportunities", []):
                    channel_str = opp.get("recommended_channel", "display")
                    try:
                        channel = ChannelType(channel_str)
                    except ValueError:
                        channel = ChannelType.DISPLAY
                    
                    opportunities.append(
                        CrossSellOpportunity(
                            source_deal_id=source_deal_id,
                            source_product_id="",
                            recommended_product_id=opp.get("recommended_product_id", ""),
                            recommended_channel=channel,
                            estimated_value=opp.get("estimated_value", 0.0),
                            confidence=opp.get("confidence", 0.5),
                            reasoning=opp.get("reasoning", ""),
                            suggested_impressions=opp.get("suggested_impressions", 0),
                            suggested_cpm=opp.get("suggested_cpm", 0.0),
                        )
                    )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        return opportunities


# Factory function
async def create_inventory_manager(
    seller_id: str,
    portfolio: Optional[InventoryPortfolio] = None,
    anthropic_client: Optional[AsyncAnthropic] = None,
    model: str = L1_MODEL,
) -> InventoryManager:
    """Create an Inventory Manager instance.
    
    Args:
        seller_id: Unique seller identifier
        portfolio: Initial inventory portfolio
        anthropic_client: Anthropic client (optional)
        model: LLM model to use
        
    Returns:
        Initialized InventoryManager
    """
    manager = InventoryManager(
        seller_id=seller_id,
        portfolio=portfolio,
        anthropic_client=anthropic_client,
        model=model,
    )
    await manager.initialize()
    return manager
