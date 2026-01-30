"""
Seller Agent System - Complete hierarchy with all levels.

Provides a unified interface for the seller-side agent hierarchy:
- L1: Inventory Manager (yield optimization, deal decisions)
- L2: Channel Inventory Specialists (display, video, CTV, mobile, native)
- L3: Functional Agents (pricing, avails, audience validation, upsell)

This system coordinates the full hierarchy for inventory management
and deal evaluation across multiple buyers.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from agents.seller.l1_inventory_manager import InventoryManager
from agents.seller.l2_display import DisplayInventoryAgent
from agents.seller.l2_video import VideoInventoryAgent
from agents.seller.l2_ctv import CTVInventoryAgent
from agents.seller.l2_mobile import MobileAppInventoryAgent
from agents.seller.l2_native import NativeInventoryAgent
from agents.seller.l3_pricing import PricingAgent
from agents.seller.l3_avails import AvailsAgent
from agents.seller.l3_audience_validator import AudienceValidatorAgent
from agents.seller.l3_upsell import UpsellAgent
from agents.seller.l3_proposal_review import ProposalReviewAgent
from agents.seller.models import (
    DealRequest,
    DealDecision,
    DealAction,
    InventoryPortfolio,
    Product,
    ChannelType,
    BuyerTier,
    Deal,
    Task,
    TaskResult,
    YieldStrategy,
)
from protocols.inter_level import (
    InterLevelProtocol,
    AgentContext,
    Task as ProtocolTask,
    Result,
    ResultStatus,
    ContextSerializer,
)

logger = logging.getLogger(__name__)


@dataclass
class SellerContextConfig:
    """Configuration for context flow in seller hierarchy."""
    
    # Token limits
    l1_to_l2_limit: int = 8000
    l2_to_l3_limit: int = 4000
    l3_to_l2_limit: int = 2000
    l2_to_l1_limit: int = 4000
    
    # Context rot
    enable_context_rot: bool = False
    rot_rate_per_level: float = 0.05
    
    # Recovery
    recovery_enabled: bool = False
    recovery_accuracy: float = 0.0


@dataclass
class SellerHierarchyMetrics:
    """Metrics for seller hierarchy execution."""
    
    total_l1_decisions: int = 0
    total_l2_tasks: int = 0
    total_l3_operations: int = 0
    
    # Deal metrics
    deals_evaluated: int = 0
    deals_accepted: int = 0
    deals_rejected: int = 0
    deals_countered: int = 0
    
    # Token tracking
    l1_tokens_used: int = 0
    l2_tokens_used: int = 0
    l3_tokens_used: int = 0
    
    # Context tracking
    context_handoffs: int = 0
    context_rot_events: int = 0
    
    # Revenue
    total_revenue: float = 0.0
    total_impressions_sold: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "l1_decisions": self.total_l1_decisions,
            "l2_tasks": self.total_l2_tasks,
            "l3_operations": self.total_l3_operations,
            "deals_evaluated": self.deals_evaluated,
            "deals_accepted": self.deals_accepted,
            "deals_rejected": self.deals_rejected,
            "deals_countered": self.deals_countered,
            "l1_tokens": self.l1_tokens_used,
            "l2_tokens": self.l2_tokens_used,
            "l3_tokens": self.l3_tokens_used,
            "context_handoffs": self.context_handoffs,
            "context_rot_events": self.context_rot_events,
            "total_revenue": self.total_revenue,
            "total_impressions_sold": self.total_impressions_sold,
        }


@dataclass
class DealEvaluationResult:
    """Result of evaluating a deal through the hierarchy."""
    
    deal_id: str
    request: DealRequest
    decision: DealDecision
    
    # Hierarchy tracking
    l1_decision: Optional[str] = None
    l2_channel_check: Optional[dict] = None
    l3_pricing: Optional[dict] = None
    l3_avails: Optional[dict] = None
    
    # Context preservation
    context_preserved_pct: float = 100.0
    
    # Timing
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "deal_id": self.deal_id,
            "decision": self.decision.action.value if self.decision else None,
            "l1_decision": self.l1_decision,
            "context_preserved_pct": self.context_preserved_pct,
            "execution_time_ms": self.execution_time_ms,
        }


class SellerContextFlowManager:
    """Manages context flow in the seller hierarchy."""
    
    def __init__(self, config: Optional[SellerContextConfig] = None):
        """Initialize the context flow manager."""
        self.config = config or SellerContextConfig()
        self._serializer = ContextSerializer()
        self._context_history: list[dict] = []
        self._tokens_original: int = 0
        self._tokens_current: int = 0
    
    def create_context_for_deal(
        self,
        deal_request: DealRequest,
        agent_id: str,
    ) -> AgentContext:
        """Create context for deal evaluation."""
        context = AgentContext.create(
            agent_id=agent_id,
            level=1,
            working_memory={
                "deal_id": deal_request.deal_id,
                "buyer_id": deal_request.buyer_id,
                "buyer_tier": deal_request.buyer_tier.value if deal_request.buyer_tier else "public",
                "requested_products": [p.product_id for p in deal_request.products] if deal_request.products else [],
                "requested_impressions": deal_request.impressions,
                "offered_cpm": deal_request.offered_cpm,
                "audience_spec": deal_request.audience.to_dict() if deal_request.audience else {},
            },
            constraints={
                "min_cpm_threshold": 5.0,  # Minimum acceptable CPM
                "max_discount_pct": 20.0,  # Maximum discount allowed
            },
            metadata={
                "evaluation_started": datetime.utcnow().isoformat(),
            },
        )
        
        self._tokens_original = self._serializer.to_tokens(context)
        self._tokens_current = self._tokens_original
        
        return context
    
    def pass_down(
        self,
        context: AgentContext,
        from_level: int,
        to_level: int,
    ) -> AgentContext:
        """Pass context to lower level."""
        if from_level == 1 and to_level == 2:
            limit = self.config.l1_to_l2_limit
        elif from_level == 2 and to_level == 3:
            limit = self.config.l2_to_l3_limit
        else:
            limit = 4000
        
        truncated = self._serializer.truncate_to_limit(context, limit)
        
        if self.config.enable_context_rot:
            truncated = self._apply_rot(truncated)
        
        self._context_history.append({
            "action": "pass_down",
            "from": from_level,
            "to": to_level,
            "tokens": truncated.token_count,
        })
        
        self._tokens_current = truncated.token_count
        return truncated
    
    def pass_up(
        self,
        context: AgentContext,
        result: dict,
        from_level: int,
        to_level: int,
    ) -> AgentContext:
        """Pass context and result to higher level."""
        if from_level == 3 and to_level == 2:
            limit = self.config.l3_to_l2_limit
        elif from_level == 2 and to_level == 1:
            limit = self.config.l2_to_l1_limit
        else:
            limit = 4000
        
        updated = AgentContext(
            context_id=context.context_id,
            agent_id=context.agent_id,
            level=to_level,
            conversation_history=context.conversation_history.copy(),
            working_memory={
                **context.working_memory,
                f"l{from_level}_result": result,
            },
            constraints=context.constraints.copy(),
            metadata=context.metadata.copy(),
        )
        
        truncated = self._serializer.truncate_to_limit(updated, limit)
        self._tokens_current = truncated.token_count
        
        return truncated
    
    def _apply_rot(self, context: AgentContext) -> AgentContext:
        """Apply context rot."""
        import random
        
        rate = self.config.rot_rate_per_level
        
        rotted_memory = {
            k: v for k, v in context.working_memory.items()
            if random.random() > rate
        }
        
        return AgentContext(
            context_id=context.context_id,
            agent_id=context.agent_id,
            level=context.level,
            conversation_history=context.conversation_history,
            working_memory=rotted_memory,
            constraints=context.constraints,
            metadata={**context.metadata, "rot_applied": True},
        )
    
    def get_preservation(self) -> float:
        """Get context preservation percentage."""
        if self._tokens_original == 0:
            return 100.0
        return (self._tokens_current / self._tokens_original) * 100


class SellerAgentSystem:
    """Complete seller agent system with all hierarchy levels.
    
    Coordinates:
    - L1: Inventory Manager (strategic yield optimization)
    - L2: Channel Inventory Specialists (display, video, CTV, mobile, native)
    - L3: Functional Agents (pricing, avails, audience, upsell, proposal)
    
    Example:
        ```python
        system = SellerAgentSystem(seller_id="pub-001")
        await system.initialize()
        
        decision = await system.evaluate_deal(deal_request)
        ```
    """
    
    def __init__(
        self,
        seller_id: Optional[str] = None,
        scenario: str = "A",
        context_config: Optional[SellerContextConfig] = None,
        mock_llm: bool = False,
    ):
        """Initialize the seller agent system.
        
        Args:
            seller_id: Unique identifier for this seller
            scenario: Simulation scenario (A, B, or C)
            context_config: Context flow configuration
            mock_llm: Use mock LLM responses
        """
        self.seller_id = seller_id or f"pub-{uuid.uuid4().hex[:8]}"
        self.scenario = scenario
        self.mock_llm = mock_llm
        
        # Context management
        self.context_manager = SellerContextFlowManager(context_config)
        
        # Hierarchy components (lazy initialized)
        self._l1_inventory_manager: Optional[InventoryManager] = None
        self._l2_inventory: dict[str, Any] = {}
        self._l3_functional: dict[str, Any] = {}
        
        # Metrics
        self.metrics = SellerHierarchyMetrics()
        
        # State
        self._initialized = False
        self._portfolio: Optional[InventoryPortfolio] = None
        
        logger.info(f"SellerAgentSystem created: {self.seller_id}")
    
    @property
    def l1_inventory_manager(self) -> InventoryManager:
        """Get L1 Inventory Manager."""
        if self._l1_inventory_manager is None:
            raise RuntimeError("System not initialized")
        return self._l1_inventory_manager
    
    @property
    def l2_inventory(self) -> dict[str, Any]:
        """Get L2 inventory specialists."""
        return self._l2_inventory
    
    @property
    def l3_functional(self) -> dict[str, Any]:
        """Get L3 functional agents."""
        return self._l3_functional
    
    @property
    def portfolio(self) -> InventoryPortfolio:
        """Get current inventory portfolio."""
        if self._portfolio is None:
            self._portfolio = InventoryPortfolio(seller_id=self.seller_id)
        return self._portfolio
    
    async def initialize(self) -> None:
        """Initialize all hierarchy components."""
        if self._initialized:
            return
        
        logger.info(f"Initializing SellerAgentSystem {self.seller_id}")
        
        # Initialize portfolio
        self._portfolio = InventoryPortfolio(seller_id=self.seller_id)
        
        # Create L1 Inventory Manager
        self._l1_inventory_manager = InventoryManager(
            seller_id=self.seller_id,
            portfolio=self._portfolio,
        )
        await self._l1_inventory_manager.initialize()
        
        # Create L2 Channel Inventory Specialists
        self._l2_inventory = {
            ChannelType.DISPLAY.value: DisplayInventoryAgent(
                agent_id=f"{self.seller_id}-l2-display",
            ),
            ChannelType.VIDEO.value: VideoInventoryAgent(
                agent_id=f"{self.seller_id}-l2-video",
            ),
            ChannelType.CTV.value: CTVInventoryAgent(
                agent_id=f"{self.seller_id}-l2-ctv",
            ),
            ChannelType.MOBILE_APP.value: MobileAppInventoryAgent(
                agent_id=f"{self.seller_id}-l2-mobile",
            ),
            ChannelType.NATIVE.value: NativeInventoryAgent(
                agent_id=f"{self.seller_id}-l2-native",
            ),
        }
        
        # Initialize L2 specialists
        for specialist in self._l2_inventory.values():
            await specialist.initialize()
        
        # Create L3 Functional Agents
        self._l3_functional = {
            "pricing": PricingAgent(
                agent_id=f"{self.seller_id}-l3-pricing",
            ),
            "avails": AvailsAgent(
                agent_id=f"{self.seller_id}-l3-avails",
            ),
            "audience": AudienceValidatorAgent(
                agent_id=f"{self.seller_id}-l3-audience",
            ),
            "upsell": UpsellAgent(
                agent_id=f"{self.seller_id}-l3-upsell",
            ),
            "proposal": ProposalReviewAgent(
                agent_id=f"{self.seller_id}-l3-proposal",
            ),
        }
        
        # Initialize L3 agents
        for agent in self._l3_functional.values():
            await agent.initialize()
        
        self._initialized = True
        logger.info(f"SellerAgentSystem {self.seller_id} initialized")
    
    async def evaluate_deal(self, request: DealRequest) -> DealDecision:
        """Evaluate a deal request through the full hierarchy.
        
        Flow:
        1. L1 makes strategic decision (accept/reject/counter)
        2. L2 checks channel-specific availability
        3. L3 calculates pricing and validates audience
        4. Decision flows back up
        
        Args:
            request: Deal request to evaluate
            
        Returns:
            DealDecision with action and terms
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        start_time = datetime.utcnow()
        self.metrics.deals_evaluated += 1
        
        try:
            # Create context for this deal
            context = self.context_manager.create_context_for_deal(
                deal_request=request,
                agent_id=self._l1_inventory_manager.agent_id,
            )
            
            # L1: Strategic evaluation
            logger.info(f"L1 evaluating deal {request.deal_id}")
            self.metrics.total_l1_decisions += 1
            
            decision = await self._l1_inventory_manager.evaluate_deal(request)
            
            # L2: Channel-specific checks
            channel = request.products[0].channel if request.products else ChannelType.DISPLAY
            specialist = self._l2_inventory.get(channel.value if hasattr(channel, 'value') else str(channel))
            
            if specialist:
                l2_context = self.context_manager.pass_down(context, 1, 2)
                self.metrics.context_handoffs += 1
                self.metrics.total_l2_tasks += 1
                
                # L2 checks availability
                logger.info(f"L2 checking {channel} availability")
                availability = await specialist.check_availability(
                    impressions=request.impressions,
                    date_range=(request.start_date, request.end_date) if hasattr(request, 'start_date') else None,
                )
                
                # L3: Pricing calculation
                pricing_agent = self._l3_functional.get("pricing")
                if pricing_agent:
                    l3_context = self.context_manager.pass_down(l2_context, 2, 3)
                    self.metrics.context_handoffs += 1
                    self.metrics.total_l3_operations += 1
                    
                    logger.info(f"L3 calculating price for deal {request.deal_id}")
                    pricing_result = await pricing_agent.calculate_price(
                        impressions=request.impressions,
                        buyer_tier=request.buyer_tier,
                        floor_cpm=decision.counter_cpm if decision.counter_cpm else 10.0,
                    )
                    
                    # Update decision with L3 pricing
                    if pricing_result.get("recommended_cpm"):
                        decision.counter_cpm = pricing_result["recommended_cpm"]
            
            # Track decision
            if decision.action == DealAction.ACCEPT:
                self.metrics.deals_accepted += 1
                self.metrics.total_revenue += request.offered_cpm * request.impressions / 1000
                self.metrics.total_impressions_sold += request.impressions
            elif decision.action == DealAction.REJECT:
                self.metrics.deals_rejected += 1
            elif decision.action == DealAction.COUNTER:
                self.metrics.deals_countered += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"Deal evaluation failed: {e}", exc_info=True)
            return DealDecision(
                deal_id=request.deal_id,
                action=DealAction.REJECT,
                reasoning=f"Evaluation error: {str(e)}",
            )
    
    async def handle_buyer_request(
        self,
        buyer_id: str,
        channel: str,
        impressions: int,
        offered_cpm: float,
        audience_spec: Optional[dict] = None,
    ) -> DealDecision:
        """Handle an incoming buyer request.
        
        Creates a DealRequest and processes it through the hierarchy.
        
        Args:
            buyer_id: Buyer's identifier
            channel: Requested channel
            impressions: Requested impressions
            offered_cpm: Offered CPM
            audience_spec: Optional audience specification
            
        Returns:
            DealDecision
        """
        from agents.seller.models import AudienceSpec as SellerAudienceSpec
        
        # Create deal request
        request = DealRequest(
            deal_id=f"deal-{uuid.uuid4().hex[:8]}",
            buyer_id=buyer_id,
            buyer_tier=BuyerTier.PUBLIC,
            impressions=impressions,
            offered_cpm=offered_cpm,
            products=[],  # Will be determined by L2
            audience=SellerAudienceSpec(**audience_spec) if audience_spec else None,
        )
        
        return await self.evaluate_deal(request)
    
    def get_metrics(self) -> dict:
        """Get hierarchy execution metrics."""
        return self.metrics.to_dict()
    
    def get_inventory_summary(self) -> dict:
        """Get inventory portfolio summary."""
        return self.portfolio.to_dict() if self.portfolio else {}
    
    async def shutdown(self) -> None:
        """Shutdown the seller agent system."""
        logger.info(f"Shutting down SellerAgentSystem {self.seller_id}")
        
        # Shutdown L3 agents
        for agent in self._l3_functional.values():
            if hasattr(agent, "shutdown"):
                await agent.shutdown()
        
        # Shutdown L2 specialists
        for specialist in self._l2_inventory.values():
            if hasattr(specialist, "shutdown"):
                await specialist.shutdown()
        
        # Shutdown L1
        if self._l1_inventory_manager and hasattr(self._l1_inventory_manager, "shutdown"):
            await self._l1_inventory_manager.shutdown()
        
        self._initialized = False
        logger.info(f"SellerAgentSystem {self.seller_id} shutdown complete")


async def create_seller_system(
    seller_id: Optional[str] = None,
    scenario: str = "A",
    context_config: Optional[SellerContextConfig] = None,
    mock_llm: bool = False,
) -> SellerAgentSystem:
    """Create and initialize a seller agent system.
    
    Args:
        seller_id: Optional seller ID
        scenario: Simulation scenario
        context_config: Context flow configuration
        mock_llm: Use mock LLM
        
    Returns:
        Initialized SellerAgentSystem
    """
    system = SellerAgentSystem(
        seller_id=seller_id,
        scenario=scenario,
        context_config=context_config,
        mock_llm=mock_llm,
    )
    await system.initialize()
    return system
