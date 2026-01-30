"""
Scenario B: Direct A2A with Multi-Agent Hierarchy.

Both buyer and seller use full agent hierarchies (L1 -> L2 -> L3).
No exchange mediation - agents communicate directly.
Context rot at each level with NO recovery mechanism.
0% fees but accumulating errors.

Key Characteristics:
- Full hierarchy on both sides (6 agent levels total)
- Direct agent-to-agent communication
- Context rot occurs at EVERY handoff
- No verification layer = errors compound
- Zero intermediary fees

This represents a direct A2A scenario where agents negotiate directly
but suffer from accumulating context degradation with no recovery.
"""

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from .base import BaseScenario, ScenarioConfig, ScenarioMetrics
from .context_rot import (
    ContextRotSimulator,
    ContextRotConfig,
    RecoverySource,
    SCENARIO_B_ROT_CONFIG,
)
from orchestration.buyer_system import (
    BuyerAgentSystem,
    ContextFlowConfig,
    CampaignResult,
)
from orchestration.seller_system import (
    SellerAgentSystem,
    SellerContextConfig,
)
from agents.buyer.models import (
    Campaign,
    CampaignObjectives,
    CampaignStatus,
    AudienceSpec,
    Channel,
)
from agents.seller.models import (
    DealRequest,
    DealDecision,
    DealAction,
    BuyerTier,
    DealTypeEnum,
    AudienceSpec as SellerAudienceSpec,
)
from infrastructure.message_schemas import DealConfirmation
from protocols.a2a import A2AProtocol, A2AMessage, Offer

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentScenarioBConfig(ScenarioConfig):
    """Configuration specific to multi-agent Scenario B."""
    
    scenario_code: str = "B"
    name: str = "Multi-Agent Direct A2A (No Exchange)"
    description: str = "Full hierarchy with direct communication, no recovery"
    
    # No exchange = no fees
    exchange_fee_pct: float = 0.0
    
    # No recovery
    recovery_enabled: bool = False
    recovery_accuracy: float = 0.0
    
    # Context rot - higher rate since no recovery
    context_rot_rate: float = 0.08  # 8% rot per level
    
    # Compounding errors
    error_accumulation_factor: float = 1.2  # Errors compound by 20%
    
    # Multi-agent specific
    enable_hierarchy: bool = True
    l1_model: str = "claude-sonnet-4-20250514"
    l2_model: str = "claude-sonnet-4-20250514"
    l3_model: str = "claude-sonnet-4-20250514"


@dataclass
class DirectDealResult:
    """Result of a direct A2A deal cycle."""
    
    deal_id: str
    campaign_id: str
    success: bool
    
    # Financial (no exchange fees)
    buyer_cost: float = 0.0
    seller_revenue: float = 0.0  # Same as buyer cost in direct deals
    impressions: int = 0
    
    # Hierarchy tracking
    buyer_context_hops: int = 0  # L1 -> L2 -> L3 = 2 hops
    seller_context_hops: int = 0
    a2a_hops: int = 0  # Direct communication hops
    
    # Context rot tracking
    buyer_context_preserved_pct: float = 100.0
    seller_context_preserved_pct: float = 100.0
    total_context_rot_events: int = 0
    
    # Error tracking
    errors_detected: int = 0
    errors_undetected: int = 0  # Estimated errors that slipped through
    
    # Timing
    execution_time_ms: float = 0.0
    
    # Messages
    errors: list[str] = field(default_factory=list)
    
    def to_deal_confirmation(self) -> DealConfirmation:
        """Convert to standard DealConfirmation."""
        return DealConfirmation(
            deal_id=self.deal_id,
            request_id=self.deal_id,  # Use deal_id as request_id for direct deals
            buyer_id="",
            seller_id="",
            impressions=self.impressions,
            cpm=self.buyer_cost / self.impressions * 1000 if self.impressions > 0 else 0,
            total_cost=self.buyer_cost,
            exchange_fee=0.0,  # No exchange fees
            scenario="B",  # Scenario B - direct A2A
            timestamp=datetime.utcnow(),
        )


class MultiAgentScenarioB(BaseScenario):
    """
    Scenario B: Direct A2A with Multi-Agent Hierarchy.
    
    No exchange. Agents communicate directly.
    Context rot at each level with no recovery.
    0% fees but accumulating errors.
    
    This scenario demonstrates:
    - Full context rot without mitigation
    - Error compounding across hierarchy levels
    - Direct negotiation without verification
    - The risks of pure decentralized operation
    """
    
    def __init__(
        self,
        config: Optional[MultiAgentScenarioBConfig] = None,
        num_buyers: int = 3,
        num_sellers: int = 3,
        mock_llm: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize Multi-Agent Scenario B.
        
        Args:
            config: Scenario configuration
            num_buyers: Number of buyer systems
            num_sellers: Number of seller systems
            mock_llm: Use mock LLM responses
            seed: Random seed
        """
        config = config or MultiAgentScenarioBConfig()
        super().__init__(
            scenario_id="B-MultiAgent",
            scenario_name=config.name,
            config=config,
        )
        
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.mock_llm = mock_llm
        self.seed = seed
        
        # Set random seed
        if seed:
            random.seed(seed)
        
        # Systems (initialized in setup)
        self._buyer_systems: dict[str, BuyerAgentSystem] = {}
        self._seller_systems: dict[str, SellerAgentSystem] = {}
        
        # A2A Protocol for direct communication
        self._a2a_protocols: dict[str, A2AProtocol] = {}
        
        # Context rot simulation - NO recovery
        self._context_rot_config = ContextRotConfig(
            decay_rate=config.context_rot_rate,
            recovery_source=RecoverySource.NONE,  # No recovery!
            recovery_accuracy=0.0,
        )
        self._context_rot_simulator = ContextRotSimulator(
            config=self._context_rot_config,
            seed=seed,
        )
        
        # Active campaigns
        self._active_campaigns: dict[str, Campaign] = {}
        
        # Error tracking
        self._accumulated_errors: int = 0
        self._error_rate_multiplier: float = 1.0
        
        logger.info(
            f"MultiAgentScenarioB initialized: "
            f"{num_buyers} buyers, {num_sellers} sellers, "
            f"NO exchange, {config.context_rot_rate*100}% rot rate"
        )
    
    @property
    def buyer_systems(self) -> dict[str, BuyerAgentSystem]:
        """Get all buyer systems."""
        return self._buyer_systems
    
    @property
    def seller_systems(self) -> dict[str, SellerAgentSystem]:
        """Get all seller systems."""
        return self._seller_systems
    
    async def setup(self) -> None:
        """Set up all scenario components."""
        logger.info("Setting up MultiAgentScenarioB (Direct A2A)")
        
        # Create context configuration - rot enabled, NO recovery
        buyer_context_config = ContextFlowConfig(
            enable_context_rot=True,
            rot_rate_per_level=self.config.context_decay_rate if hasattr(self.config, 'context_decay_rate') else 0.08,
            recovery_enabled=False,
            recovery_accuracy=0.0,
        )
        
        seller_context_config = SellerContextConfig(
            enable_context_rot=True,
            rot_rate_per_level=self.config.context_decay_rate if hasattr(self.config, 'context_decay_rate') else 0.08,
            recovery_enabled=False,
            recovery_accuracy=0.0,
        )
        
        # Create buyer systems
        for i in range(self.num_buyers):
            buyer_id = f"buyer-{i:03d}"
            system = BuyerAgentSystem(
                buyer_id=buyer_id,
                scenario="B",
                context_config=buyer_context_config,
                mock_llm=self.mock_llm,
            )
            await system.initialize()
            self._buyer_systems[buyer_id] = system
            
            # Create A2A protocol for this buyer
            self._a2a_protocols[buyer_id] = A2AProtocol(agent_id=buyer_id)
            
            logger.info(f"Buyer system {buyer_id} initialized (no exchange)")
        
        # Create seller systems
        for i in range(self.num_sellers):
            seller_id = f"pub-{i:03d}"
            system = SellerAgentSystem(
                seller_id=seller_id,
                scenario="B",
                context_config=seller_context_config,
                mock_llm=self.mock_llm,
            )
            await system.initialize()
            self._seller_systems[seller_id] = system
            
            # Create A2A protocol for this seller
            self._a2a_protocols[seller_id] = A2AProtocol(agent_id=seller_id)
            
            logger.info(f"Seller system {seller_id} initialized (no exchange)")
        
        logger.info("MultiAgentScenarioB setup complete")
    
    async def run_day(self, day: int) -> list[DealConfirmation]:
        """Run one simulation day.
        
        Args:
            day: Simulation day number
            
        Returns:
            List of deals completed this day
        """
        self._current_day = day
        deals = []
        
        # Update error multiplier for accumulated errors
        self._error_rate_multiplier = 1.0 + (self._accumulated_errors * 0.01)
        
        logger.info(
            f"Day {day}: Processing {len(self._active_campaigns)} campaigns, "
            f"error multiplier: {self._error_rate_multiplier:.2f}x"
        )
        
        # Process each active campaign
        for campaign_id, campaign in self._active_campaigns.items():
            if not campaign.is_active:
                continue
            
            campaign_deals = await self._run_campaign_day(campaign, day)
            deals.extend(campaign_deals)
        
        # Apply daily context decay - MORE severe without recovery
        await self._apply_daily_context_decay(day)
        
        return deals
    
    async def _run_campaign_day(
        self,
        campaign: Campaign,
        day: int,
    ) -> list[DealConfirmation]:
        """Run deal cycles for a campaign on a given day."""
        deals = []
        
        buyer_id = f"buyer-{hash(campaign.advertiser) % self.num_buyers:03d}"
        buyer_system = self._buyer_systems.get(buyer_id)
        
        if not buyer_system:
            return deals
        
        daily_budget = campaign.daily_budget
        if daily_budget <= 0:
            return deals
        
        remaining_budget = daily_budget
        max_cycles = 10
        
        for cycle in range(max_cycles):
            if remaining_budget <= 0:
                break
            
            seller_id = f"pub-{cycle % self.num_sellers:03d}"
            seller_system = self._seller_systems.get(seller_id)
            
            if not seller_system:
                continue
            
            # Run direct A2A deal cycle (no exchange)
            result = await self.run_deal_cycle(
                campaign=campaign,
                buyer_system=buyer_system,
                seller_system=seller_system,
            )
            
            if result.success:
                confirmation = result.to_deal_confirmation()
                confirmation.buyer_id = buyer_id
                confirmation.seller_id = seller_id
                deals.append(confirmation)
                
                remaining_budget -= result.buyer_cost
                campaign.spend += result.buyer_cost
                campaign.impressions_delivered += result.impressions
                
                self.metrics.record_deal(confirmation)
            
            # Track accumulated errors
            self._accumulated_errors += result.errors_undetected
        
        return deals
    
    async def run_deal_cycle(
        self,
        campaign: Campaign,
        buyer_system: BuyerAgentSystem,
        seller_system: SellerAgentSystem,
    ) -> DirectDealResult:
        """
        Direct A2A deal cycle with full hierarchy on both sides.
        
        Flow:
        - Full hierarchy on both sides
        - Context rot at each handoff
        - No external verification
        - Errors compound
        
        Args:
            campaign: Campaign being executed
            buyer_system: Buyer's agent hierarchy
            seller_system: Seller's agent hierarchy
            
        Returns:
            DirectDealResult with execution details
        """
        start_time = datetime.utcnow()
        deal_id = f"deal-{uuid.uuid4().hex[:8]}"
        
        result = DirectDealResult(
            deal_id=deal_id,
            campaign_id=campaign.campaign_id,
            success=False,
        )
        
        context_rot_events = 0
        
        try:
            # === BUYER HIERARCHY (L1 -> L2 -> L3) ===
            
            # Context rot at each level
            result.buyer_context_hops = 2  # L1->L2, L2->L3
            
            # L1: Portfolio Manager
            logger.debug(f"Deal {deal_id}: Buyer L1 (no exchange verification)")
            allocation = await buyer_system.l1_portfolio_manager.allocate_budget([campaign])
            context_rot_events += self._simulate_rot_event()
            
            # L2: Channel Specialist
            logger.debug(f"Deal {deal_id}: Buyer L2 selecting channel")
            channel_selections = await buyer_system.l1_portfolio_manager.select_channels(campaign)
            selected_channel = channel_selections[0].channel if channel_selections else Channel.DISPLAY.value
            context_rot_events += self._simulate_rot_event()
            
            # L3: Functional Execution
            logger.debug(f"Deal {deal_id}: Buyer L3 preparing request")
            target_cpm = campaign.objectives.cpm_target
            target_impressions = min(
                campaign.objectives.reach_target // 10,
                int(campaign.remaining_budget / target_cpm * 1000),
            )
            context_rot_events += self._simulate_rot_event()
            
            # Track buyer context preservation
            result.buyer_context_preserved_pct = buyer_system.context_manager.get_context_preservation()
            
            # === DIRECT A2A COMMUNICATION (NO EXCHANGE) ===
            
            # Create offer via A2A protocol
            buyer_protocol = self._a2a_protocols.get(buyer_system.buyer_id)
            seller_protocol = self._a2a_protocols.get(seller_system.seller_id)
            
            if buyer_protocol and seller_protocol:
                # Buyer sends offer directly to seller
                offer = Offer(
                    offer_id=deal_id,
                    offerer=buyer_system.buyer_id,
                    recipient=seller_system.seller_id,
                    terms={
                        "impressions": target_impressions,
                        "cpm": target_cpm,
                        "channel": selected_channel,
                    },
                    description=f"Offer for {target_impressions:,} impressions at ${target_cpm:.2f} CPM",
                )
                
                # A2A hop - context rot!
                result.a2a_hops = 1
                context_rot_events += self._simulate_rot_event()
            
            # === SELLER HIERARCHY (L1 -> L2 -> L3) ===
            
            result.seller_context_hops = 2  # L1->L2, L2->L3
            
            # Create deal request (potentially with rotted context)
            deal_request = DealRequest(
                request_id=deal_id,
                buyer_id=buyer_system.buyer_id,
                buyer_tier=BuyerTier.AGENCY,
                product_id="default-product",
                impressions=target_impressions,
                max_cpm=target_cpm,
                deal_type=DealTypeEnum.PREFERRED_DEAL,
                flight_dates=(date.today(), date.today() + timedelta(days=30)),
                audience_spec=SellerAudienceSpec(),
            )
            
            # Apply context rot to request data (simulating errors)
            if random.random() < self._context_rot_config.decay_rate * self._error_rate_multiplier:
                # Corrupt some data
                deal_request.impressions = int(deal_request.impressions * random.uniform(0.8, 1.2))
                result.errors_undetected += 1
            
            # L1: Inventory Manager evaluates
            logger.debug(f"Deal {deal_id}: Seller L1 evaluating")
            decision = await seller_system.evaluate_deal(deal_request)
            context_rot_events += self._simulate_rot_event()
            
            # L2 & L3 called within evaluate_deal, add their rot events
            context_rot_events += self._simulate_rot_event() * 2
            
            # Track seller context preservation
            result.seller_context_preserved_pct = seller_system.context_manager.get_preservation()
            
            # === DIRECT SETTLEMENT (NO EXCHANGE VERIFICATION) ===
            
            if decision.action == DealAction.ACCEPT:
                final_cpm = decision.price or target_cpm
                total_cost = final_cpm * target_impressions / 1000
                
                # No exchange fees! But also no verification...
                result.buyer_cost = total_cost
                result.seller_revenue = total_cost  # Full amount goes to seller
                result.impressions = target_impressions
                result.success = True
                
                # But errors may have accumulated undetected
                if result.errors_undetected > 0:
                    logger.warning(
                        f"Deal {deal_id}: {result.errors_undetected} undetected errors "
                        f"(no exchange verification)"
                    )
                
                logger.info(
                    f"Deal {deal_id} accepted (DIRECT): {target_impressions:,} impressions "
                    f"@ ${final_cpm:.2f} CPM, NO fees"
                )
            else:
                result.errors.append(f"Rejected: {decision.reasoning}")
            
            result.total_context_rot_events = context_rot_events
            
        except Exception as e:
            logger.error(f"Direct deal cycle failed: {e}", exc_info=True)
            result.errors.append(str(e))
        
        end_time = datetime.utcnow()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return result
    
    def _simulate_rot_event(self) -> int:
        """Simulate a context rot event.
        
        Returns:
            1 if rot occurred, 0 otherwise
        """
        effective_rate = self._context_rot_config.decay_rate * self._error_rate_multiplier
        if random.random() < effective_rate:
            self.metrics.context_rot_events += 1
            return 1
        return 0
    
    async def _apply_daily_context_decay(self, day: int) -> None:
        """Apply daily context decay to all agents.
        
        In Scenario B, decay is more severe because there's no recovery.
        
        Args:
            day: Current simulation day
        """
        # Context decay accumulates over time
        decay_factor = 1.0 + (day * 0.01)  # 1% additional decay per day
        
        # Record accumulated decay
        self.metrics.keys_lost_total += int(decay_factor * len(self._buyer_systems))
        self.metrics.keys_lost_total += int(decay_factor * len(self._seller_systems))
    
    def add_campaign(self, campaign: Campaign) -> None:
        """Add a campaign to be processed."""
        self._active_campaigns[campaign.campaign_id] = campaign
        campaign.status = CampaignStatus.ACTIVE
        self.metrics.campaigns_started += 1
        logger.info(f"Added campaign {campaign.campaign_id} to scenario B")
    
    def add_campaigns(self, campaigns: list[Campaign]) -> None:
        """Add multiple campaigns."""
        for campaign in campaigns:
            self.add_campaign(campaign)
    
    async def teardown(self) -> None:
        """Clean up scenario resources."""
        logger.info("Tearing down MultiAgentScenarioB")
        
        for system in self._buyer_systems.values():
            await system.shutdown()
        self._buyer_systems.clear()
        
        for system in self._seller_systems.values():
            await system.shutdown()
        self._seller_systems.clear()
        
        self._active_campaigns.clear()
        self._a2a_protocols.clear()
        
        await self.disconnect_ground_truth()
        
        logger.info("MultiAgentScenarioB teardown complete")
    
    def get_hierarchy_metrics(self) -> dict:
        """Get metrics from all agent hierarchies."""
        return {
            "buyers": {bid: s.get_metrics() for bid, s in self._buyer_systems.items()},
            "sellers": {sid: s.get_metrics() for sid, s in self._seller_systems.items()},
            "accumulated_errors": self._accumulated_errors,
            "error_rate_multiplier": self._error_rate_multiplier,
            "total_context_rot_events": self.metrics.context_rot_events,
            "recovery_rate": 0.0,  # No recovery in Scenario B!
        }
    
    def get_error_summary(self) -> dict:
        """Get summary of accumulated errors."""
        return {
            "total_errors": self._accumulated_errors,
            "error_multiplier": self._error_rate_multiplier,
            "context_rot_events": self.metrics.context_rot_events,
            "recovery_attempts": 0,
            "recovery_successes": 0,
            "unverified_deals": self.metrics.total_deals,  # All deals unverified!
        }
