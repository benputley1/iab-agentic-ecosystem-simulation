"""
Scenario A: Traditional Exchange with Multi-Agent Hierarchy.

Both buyer and seller use full agent hierarchies (L1 -> L2 -> L3).
Exchange mediates all transactions and provides ~60% context recovery.
15% exchange fees are charged on all deals.

Key Characteristics:
- Full hierarchy on both sides (6 agent levels total)
- Exchange provides transaction verification
- Context rot occurs at each level but exchange catches ~60% of errors
- Standard ad tech fees (15%)

This represents the current state of programmatic advertising where
exchanges add value through verification but extract significant fees.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional

from .base import BaseScenario, ScenarioConfig, ScenarioMetrics
from .context_rot import (
    ContextRotSimulator,
    ContextRotConfig,
    RecoverySource,
    SCENARIO_A_ROT_CONFIG,
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
from agents.seller.models import DealRequest, DealDecision, DealAction, BuyerTier, DealTypeEnum, AudienceSpec
from agents.exchange.auction import RentSeekingExchange
from agents.exchange.fees import FeeConfig
from infrastructure.message_schemas import DealConfirmation

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentScenarioAConfig(ScenarioConfig):
    """Configuration specific to multi-agent Scenario A."""
    
    scenario_code: str = "A"
    name: str = "Multi-Agent Traditional Exchange"
    description: str = "Full hierarchy with exchange mediation and 15% fees"
    
    # Exchange configuration
    exchange_fee_pct: float = 0.15  # 15% fees
    exchange_recovery_rate: float = 0.60  # 60% context recovery
    
    # Multi-agent specific
    enable_hierarchy: bool = True
    l1_model: str = "claude-sonnet-4-20250514"  # L1 uses Opus
    l2_model: str = "claude-sonnet-4-20250514"  # L2 uses Sonnet
    l3_model: str = "claude-sonnet-4-20250514"  # L3 uses Haiku
    
    # Context configuration
    context_rot_rate: float = 0.05  # 5% rot per level
    max_context_tokens: int = 8000


@dataclass
class DealCycleResult:
    """Result of a single deal cycle through the hierarchy."""
    
    deal_id: str
    campaign_id: str
    success: bool
    
    # Financial
    buyer_cost: float = 0.0
    seller_revenue: float = 0.0
    exchange_fee: float = 0.0
    impressions: int = 0
    
    # Hierarchy tracking
    buyer_l1_decision: Optional[str] = None
    buyer_l2_channel: Optional[str] = None
    buyer_l3_execution: Optional[dict] = None
    seller_l1_decision: Optional[str] = None
    seller_l2_availability: Optional[dict] = None
    seller_l3_pricing: Optional[dict] = None
    
    # Context preservation
    buyer_context_preserved_pct: float = 100.0
    seller_context_preserved_pct: float = 100.0
    exchange_recovery_applied: bool = False
    
    # Timing
    execution_time_ms: float = 0.0
    
    # Errors
    errors: list[str] = field(default_factory=list)
    
    def to_deal_confirmation(self) -> DealConfirmation:
        """Convert to standard DealConfirmation."""
        return DealConfirmation(
            deal_id=self.deal_id,
            buyer_id="",  # Set by caller
            seller_id="",
            campaign_id=self.campaign_id,
            impressions=self.impressions,
            cpm=self.buyer_cost / self.impressions * 1000 if self.impressions > 0 else 0,
            total_cost=self.buyer_cost,
            seller_revenue=self.seller_revenue,
            exchange_fee=self.exchange_fee,
            timestamp=datetime.utcnow(),
        )


class MultiAgentScenarioA(BaseScenario):
    """
    Scenario A: Traditional Exchange with Multi-Agent Hierarchy.
    
    Both buyer and seller use full agent hierarchies:
    - Buyer: L1 Portfolio Manager -> L2 Channel Specialists -> L3 Functional
    - Seller: L1 Inventory Manager -> L2 Inventory Specialists -> L3 Functional
    
    Exchange mediates and provides ~60% context recovery.
    15% exchange fees are charged on all deals.
    
    This scenario demonstrates:
    - Context rot across 6 agent levels
    - Exchange verification catching errors
    - Fee extraction by intermediaries
    """
    
    def __init__(
        self,
        config: Optional[MultiAgentScenarioAConfig] = None,
        num_buyers: int = 3,
        num_sellers: int = 3,
        mock_llm: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize Multi-Agent Scenario A.
        
        Args:
            config: Scenario configuration
            num_buyers: Number of buyer systems
            num_sellers: Number of seller systems
            mock_llm: Use mock LLM responses
            seed: Random seed for reproducibility
        """
        config = config or MultiAgentScenarioAConfig()
        super().__init__(
            scenario_id="A-MultiAgent",
            scenario_name=config.name,
            config=config,
        )
        
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.mock_llm = mock_llm
        self.seed = seed
        
        # Systems (initialized in setup)
        self._buyer_systems: dict[str, BuyerAgentSystem] = {}
        self._seller_systems: dict[str, SellerAgentSystem] = {}
        self._exchange: Optional[RentSeekingExchange] = None
        
        # Context rot simulation
        self._context_rot_config = ContextRotConfig(
            decay_rate=config.context_rot_rate,
            recovery_source=RecoverySource.EXCHANGE,
            recovery_accuracy=config.exchange_recovery_rate,
        )
        self._context_rot_simulator = ContextRotSimulator(
            config=self._context_rot_config,
            seed=seed,
        )
        
        # Campaigns being processed
        self._active_campaigns: dict[str, Campaign] = {}
        
        logger.info(
            f"MultiAgentScenarioA initialized: "
            f"{num_buyers} buyers, {num_sellers} sellers, "
            f"{config.exchange_fee_pct*100}% fees"
        )
    
    @property
    def buyer_systems(self) -> dict[str, BuyerAgentSystem]:
        """Get all buyer systems."""
        return self._buyer_systems
    
    @property
    def seller_systems(self) -> dict[str, SellerAgentSystem]:
        """Get all seller systems."""
        return self._seller_systems
    
    @property
    def exchange(self) -> RentSeekingExchange:
        """Get the exchange."""
        if self._exchange is None:
            raise RuntimeError("Scenario not set up. Call setup() first.")
        return self._exchange
    
    async def setup(self) -> None:
        """Set up all scenario components."""
        logger.info("Setting up MultiAgentScenarioA")
        
        # Create context configuration with rot enabled
        buyer_context_config = ContextFlowConfig(
            enable_context_rot=True,
            rot_rate_per_level=self.config.context_decay_rate if hasattr(self.config, 'context_decay_rate') else 0.05,
            recovery_enabled=True,
            recovery_accuracy=self._context_rot_config.recovery_accuracy,
        )
        
        seller_context_config = SellerContextConfig(
            enable_context_rot=True,
            rot_rate_per_level=self.config.context_decay_rate if hasattr(self.config, 'context_decay_rate') else 0.05,
            recovery_enabled=True,
            recovery_accuracy=self._context_rot_config.recovery_accuracy,
        )
        
        # Create buyer systems
        for i in range(self.num_buyers):
            buyer_id = f"buyer-{i:03d}"
            system = BuyerAgentSystem(
                buyer_id=buyer_id,
                scenario="A",
                context_config=buyer_context_config,
                mock_llm=self.mock_llm,
            )
            await system.initialize()
            self._buyer_systems[buyer_id] = system
            logger.info(f"Buyer system {buyer_id} initialized")
        
        # Create seller systems
        for i in range(self.num_sellers):
            seller_id = f"pub-{i:03d}"
            system = SellerAgentSystem(
                seller_id=seller_id,
                scenario="A",
                context_config=seller_context_config,
                mock_llm=self.mock_llm,
            )
            await system.initialize()
            self._seller_systems[seller_id] = system
            logger.info(f"Seller system {seller_id} initialized")
        
        # Connect to Redis bus
        bus = await self.connect_bus()
        
        # Ensure consumer groups exist
        await bus.ensure_consumer_group("rtb:requests", "sellers-group")
        await bus.ensure_consumer_group("rtb:requests", "exchange-group")
        await bus.ensure_consumer_group("rtb:responses", "exchange-group")
        await bus.ensure_consumer_group("rtb:deals", "buyers-group")
        await bus.ensure_consumer_group("rtb:deals", "sellers-group")
        logger.info("Redis bus connected with consumer groups")
        
        # Create exchange with bus
        self._exchange = RentSeekingExchange(
            bus=bus,
            fee_config=FeeConfig(base_fee_pct=self.config.exchange_fee_pct),
        )
        
        # Connect to ground truth if available
        if self._ground_truth_repo:
            logger.info("Ground truth repository connected")
        
        logger.info("MultiAgentScenarioA setup complete")
    
    async def run_day(self, day: int) -> list[DealConfirmation]:
        """Run one simulation day.
        
        Args:
            day: Simulation day number
            
        Returns:
            List of deals completed this day
        """
        self._current_day = day
        deals = []
        
        logger.info(f"Running day {day} with {len(self._active_campaigns)} campaigns")
        
        # Process each active campaign
        for campaign_id, campaign in self._active_campaigns.items():
            if not campaign.is_active:
                continue
            
            # Run deal cycles for this campaign
            campaign_deals = await self._run_campaign_day(campaign, day)
            deals.extend(campaign_deals)
        
        # Apply daily context decay to all agents
        await self._apply_daily_context_decay(day)
        
        return deals
    
    async def _run_campaign_day(
        self,
        campaign: Campaign,
        day: int,
    ) -> list[DealConfirmation]:
        """Run deal cycles for a campaign on a given day.
        
        Args:
            campaign: Campaign to process
            day: Simulation day
            
        Returns:
            List of deals completed
        """
        deals = []
        
        # Get buyer system for this campaign
        buyer_id = f"buyer-{hash(campaign.advertiser) % self.num_buyers:03d}"
        buyer_system = self._buyer_systems.get(buyer_id)
        
        if not buyer_system:
            logger.warning(f"No buyer system found for {buyer_id}")
            return deals
        
        # Calculate daily budget
        daily_budget = campaign.daily_budget
        if daily_budget <= 0:
            return deals
        
        # Run deal cycles until daily budget exhausted
        remaining_budget = daily_budget
        max_cycles = 10  # Limit cycles per day
        
        for cycle in range(max_cycles):
            if remaining_budget <= 0:
                break
            
            # Select a seller to negotiate with
            seller_id = f"pub-{cycle % self.num_sellers:03d}"
            seller_system = self._seller_systems.get(seller_id)
            
            if not seller_system:
                continue
            
            # Run full deal cycle
            result = await self.run_deal_cycle(
                campaign=campaign,
                buyer_system=buyer_system,
                seller_system=seller_system,
            )
            
            if result.success:
                # Record deal
                confirmation = result.to_deal_confirmation()
                confirmation.buyer_id = buyer_id
                confirmation.seller_id = seller_id
                deals.append(confirmation)
                
                # Update budget
                remaining_budget -= result.buyer_cost
                campaign.spend += result.buyer_cost
                campaign.impressions_delivered += result.impressions
                
                # Record to metrics
                self.metrics.record_deal(confirmation)
        
        return deals
    
    async def run_deal_cycle(
        self,
        campaign: Campaign,
        buyer_system: BuyerAgentSystem,
        seller_system: SellerAgentSystem,
    ) -> DealCycleResult:
        """
        Full deal cycle with multi-agent hierarchy.
        
        Flow:
        1. Buyer L1 allocates budget for this deal
        2. Buyer L2 creates channel plan
        3. Buyer L3 searches inventory and creates request
        4. Request sent to seller via exchange
        5. Seller L1 evaluates deal strategically
        6. Seller L2 checks availability
        7. Seller L3 calculates price
        8. Exchange mediates (applies fees, verifies)
        9. Deal confirmed or rejected
        
        Args:
            campaign: Campaign being executed
            buyer_system: Buyer's agent hierarchy
            seller_system: Seller's agent hierarchy
            
        Returns:
            DealCycleResult with full execution details
        """
        start_time = datetime.utcnow()
        deal_id = f"deal-{uuid.uuid4().hex[:8]}"
        
        result = DealCycleResult(
            deal_id=deal_id,
            campaign_id=campaign.campaign_id,
            success=False,
        )
        
        try:
            # === BUYER SIDE ===
            
            # Step 1: Buyer L1 - Strategic budget allocation
            logger.debug(f"Deal {deal_id}: Buyer L1 allocating budget")
            l1_allocation = await buyer_system.l1_portfolio_manager.allocate_budget([campaign])
            result.buyer_l1_decision = "allocated"
            
            # Step 2: Buyer L2 - Channel selection
            logger.debug(f"Deal {deal_id}: Buyer L2 selecting channels")
            channel_selections = await buyer_system.l1_portfolio_manager.select_channels(campaign)
            selected_channel = channel_selections[0].channel if channel_selections else Channel.DISPLAY.value
            result.buyer_l2_channel = selected_channel
            
            # Step 3: Buyer L3 - Search inventory and create request
            logger.debug(f"Deal {deal_id}: Buyer L3 searching inventory")
            # Use target CPM from campaign objectives
            target_cpm = campaign.objectives.cpm_target
            target_impressions = min(
                campaign.objectives.reach_target // 10,  # Portion per deal
                int(campaign.remaining_budget / target_cpm * 1000),
            )
            
            result.buyer_l3_execution = {
                "channel": selected_channel,
                "target_cpm": target_cpm,
                "target_impressions": target_impressions,
            }
            
            # Track buyer context preservation
            result.buyer_context_preserved_pct = buyer_system.context_manager.get_context_preservation()
            
            # === EXCHANGE MEDIATION ===
            
            # Create deal request
            deal_request = DealRequest(
                request_id=deal_id,
                buyer_id=buyer_system.buyer_id,
                buyer_tier=BuyerTier.AGENCY,  # Default tier
                product_id="default-product",
                impressions=target_impressions,
                max_cpm=target_cpm,
                deal_type=DealTypeEnum.PREFERRED_DEAL,
                flight_dates=(date.today(), date.today() + timedelta(days=30)),
                audience_spec=AudienceSpec(),
            )
            
            # === SELLER SIDE ===
            
            # Step 4: Seller L1 - Strategic deal evaluation
            logger.debug(f"Deal {deal_id}: Seller L1 evaluating")
            decision = await seller_system.evaluate_deal(deal_request)
            result.seller_l1_decision = decision.action.value
            
            # Step 5-6: Seller L2 & L3 are called within evaluate_deal
            result.seller_l2_availability = {"checked": True}
            result.seller_l3_pricing = {
                "counter_cpm": decision.counter_cpm,
                "accepted_cpm": decision.accepted_cpm,
            }
            
            # Track seller context preservation
            result.seller_context_preserved_pct = seller_system.context_manager.get_preservation()
            
            # === EXCHANGE SETTLEMENT ===
            
            if decision.action == DealAction.ACCEPT:
                # Calculate financials
                final_cpm = decision.accepted_cpm or target_cpm
                gross_cost = final_cpm * target_impressions / 1000
                
                # Exchange applies fees
                exchange_fee = gross_cost * self.config.exchange_fee_pct
                seller_revenue = gross_cost - exchange_fee
                
                # Exchange verifies and may recover context errors
                recovery_result = await self._exchange_verification(result)
                result.exchange_recovery_applied = recovery_result.get("recovery_applied", False)
                
                result.buyer_cost = gross_cost
                result.seller_revenue = seller_revenue
                result.exchange_fee = exchange_fee
                result.impressions = target_impressions
                result.success = True
                
                logger.info(
                    f"Deal {deal_id} accepted: {target_impressions:,} impressions "
                    f"@ ${final_cpm:.2f} CPM, fee: ${exchange_fee:.2f}"
                )
                
            elif decision.action == DealAction.COUNTER:
                # Counter-offer - could implement negotiation loop
                logger.info(f"Deal {deal_id} countered at ${decision.counter_cpm:.2f} CPM")
                result.errors.append(f"Counter-offer: ${decision.counter_cpm:.2f} CPM")
                
            else:
                # Rejected
                logger.info(f"Deal {deal_id} rejected: {decision.reasoning}")
                result.errors.append(f"Rejected: {decision.reasoning}")
            
        except Exception as e:
            logger.error(f"Deal cycle failed: {e}", exc_info=True)
            result.errors.append(str(e))
        
        # Calculate execution time
        end_time = datetime.utcnow()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return result
    
    async def _exchange_verification(self, result: DealCycleResult) -> dict:
        """Apply exchange verification and potential context recovery.
        
        The exchange catches ~60% of context errors through transaction
        record verification.
        
        Args:
            result: Current deal result
            
        Returns:
            Dict with verification results
        """
        import random
        
        # Check if context was significantly degraded
        avg_preservation = (
            result.buyer_context_preserved_pct +
            result.seller_context_preserved_pct
        ) / 2
        
        recovery_applied = False
        
        if avg_preservation < 80:  # Significant context loss
            # Exchange attempts recovery with 60% success rate
            if random.random() < self._context_rot_config.recovery_accuracy:
                recovery_applied = True
                logger.debug(f"Exchange recovered context for deal {result.deal_id}")
        
        return {
            "verification_passed": True,
            "recovery_applied": recovery_applied,
            "avg_context_preservation": avg_preservation,
        }
    
    async def _apply_daily_context_decay(self, day: int) -> None:
        """Apply daily context decay to all agent memories.
        
        Args:
            day: Current simulation day
        """
        # Apply decay to buyer systems
        for buyer_id, system in self._buyer_systems.items():
            # Context rot is applied during hierarchy traversal
            # This method could be extended to apply additional decay
            pass
        
        # Apply decay to seller systems
        for seller_id, system in self._seller_systems.items():
            pass
    
    def add_campaign(self, campaign: Campaign) -> None:
        """Add a campaign to be processed.
        
        Args:
            campaign: Campaign to add
        """
        self._active_campaigns[campaign.campaign_id] = campaign
        campaign.status = CampaignStatus.ACTIVE
        self.metrics.campaigns_started += 1
        logger.info(f"Added campaign {campaign.campaign_id} to scenario")
    
    def add_campaigns(self, campaigns: list[Campaign]) -> None:
        """Add multiple campaigns.
        
        Args:
            campaigns: Campaigns to add
        """
        for campaign in campaigns:
            self.add_campaign(campaign)
    
    async def teardown(self) -> None:
        """Clean up scenario resources."""
        logger.info("Tearing down MultiAgentScenarioA")
        
        # Shutdown buyer systems
        for system in self._buyer_systems.values():
            await system.shutdown()
        self._buyer_systems.clear()
        
        # Shutdown seller systems
        for system in self._seller_systems.values():
            await system.shutdown()
        self._seller_systems.clear()
        
        # Clear campaigns
        self._active_campaigns.clear()
        
        # Disconnect ground truth
        await self.disconnect_ground_truth()
        
        # Disconnect Redis bus
        await self.disconnect_bus()
        
        logger.info("MultiAgentScenarioA teardown complete")
    
    def get_hierarchy_metrics(self) -> dict:
        """Get metrics from all agent hierarchies."""
        buyer_metrics = {}
        for buyer_id, system in self._buyer_systems.items():
            buyer_metrics[buyer_id] = system.get_metrics()
        
        seller_metrics = {}
        for seller_id, system in self._seller_systems.items():
            seller_metrics[seller_id] = system.get_metrics()
        
        return {
            "buyers": buyer_metrics,
            "sellers": seller_metrics,
            "exchange_recovery_rate": self._context_rot_config.recovery_accuracy,
        }
