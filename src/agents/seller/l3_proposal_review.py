"""
L3 Proposal Review Agent - Proposal and counter-offer handling.

Handles reviewing incoming proposals, generating deal IDs, and building
counter-offers when terms need adjustment.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid

from ..base import FunctionalAgent, ToolDefinition


class ProposalStatus(str, Enum):
    """Status of a proposal review."""
    ACCEPTABLE = "acceptable"
    NEEDS_ADJUSTMENT = "needs_adjustment"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class Proposal:
    """Incoming deal proposal."""
    
    proposal_id: str
    buyer_id: str
    product_id: str
    requested_impressions: int
    proposed_cpm: float
    start_date: str
    end_date: str
    deal_type: str = "preferred_deal"
    targeting: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ReviewResult:
    """Result of proposal review."""
    
    proposal_id: str
    status: ProposalStatus
    reasons: list[str] = field(default_factory=list)
    recommended_action: str = ""
    price_acceptable: bool = True
    avails_acceptable: bool = True
    targeting_achievable: bool = True
    suggested_adjustments: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class CounterOffer:
    """Counter-offer to a proposal."""
    
    counter_offer_id: str
    original_proposal_id: str
    deal_id: str
    seller_id: str
    offered_cpm: float
    offered_impressions: int
    start_date: str
    end_date: str
    deal_type: str
    adjustments_made: list[str] = field(default_factory=list)
    valid_until: Optional[datetime] = None
    terms: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class ProposalReviewAgent(FunctionalAgent):
    """
    Proposal and counter-offer handling.
    
    Tools:
    - ProposalGenerator: Generate deal proposals
    - CounterOfferBuilder: Build counter-offers
    - DealIDGenerator: Generate deal IDs
    
    This agent handles:
    - Reviewing incoming proposals against constraints
    - Validating pricing, availability, and targeting
    - Building counter-offers when adjustments needed
    - Generating unique deal IDs
    """
    
    def __init__(self, **kwargs):
        """Initialize ProposalReviewAgent."""
        kwargs.setdefault("name", "ProposalReviewAgent")
        super().__init__(**kwargs)
    
    def _register_tools(self) -> None:
        """Register proposal tools."""
        self.register_tool(
            ToolDefinition(
                name="ProposalGenerator",
                description="Generate a new deal proposal",
                parameters={
                    "buyer_id": {"type": "string"},
                    "product_id": {"type": "string"},
                    "impressions": {"type": "integer"},
                    "cpm": {"type": "number"}
                },
                required_params=["buyer_id", "product_id"]
            ),
            handler=self._handle_proposal_generator
        )
        
        self.register_tool(
            ToolDefinition(
                name="CounterOfferBuilder",
                description="Build a counter-offer for a proposal",
                parameters={
                    "original_proposal_id": {"type": "string"},
                    "offered_cpm": {"type": "number"},
                    "offered_impressions": {"type": "integer"},
                    "constraints": {"type": "object"}
                },
                required_params=["original_proposal_id", "offered_cpm", "offered_impressions"]
            ),
            handler=self._handle_counter_offer_builder
        )
        
        self.register_tool(
            ToolDefinition(
                name="DealIDGenerator",
                description="Generate a unique deal ID",
                parameters={
                    "buyer_id": {"type": "string"},
                    "product_id": {"type": "string"}
                },
                required_params=["buyer_id", "product_id"]
            ),
            handler=self._handle_deal_id_generator
        )
    
    def get_system_prompt(self) -> str:
        """Get system prompt for proposal review."""
        return """You are a Proposal Review Agent responsible for evaluating and responding to deal proposals.

Your responsibilities:
1. Review incoming proposals against pricing and availability constraints
2. Identify issues with proposals (price too low, volume too high, etc.)
3. Build counter-offers when proposals need adjustment
4. Generate unique deal IDs for accepted deals

Available tools:
- ProposalGenerator: Create new proposals
- CounterOfferBuilder: Build counter-offers
- DealIDGenerator: Generate deal IDs

Always provide clear reasons for any rejections or counter-offers."""
    
    def _handle_proposal_generator(self, **kwargs) -> dict:
        """Handle ProposalGenerator tool."""
        return {
            "proposal_id": f"prop-{uuid.uuid4().hex[:8]}",
            "generated": True,
        }
    
    def _handle_counter_offer_builder(
        self,
        original_proposal_id: str,
        offered_cpm: float,
        offered_impressions: int,
        constraints: Optional[dict] = None,
    ) -> dict:
        """Handle CounterOfferBuilder tool."""
        return {
            "terms": {
                "payment_terms": "net_30",
                "cancellation_window": 7,
            },
            "built": True,
        }
    
    def _handle_deal_id_generator(self, buyer_id: str, product_id: str) -> dict:
        """Handle DealIDGenerator tool."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return {
            "deal_id": f"deal-{buyer_id[:4]}-{product_id[:4]}-{timestamp}-{uuid.uuid4().hex[:4]}",
        }
    
    async def review_proposal(
        self,
        proposal: Proposal,
        floor_cpm: Optional[float] = None,
        max_impressions: Optional[int] = None,
    ) -> ReviewResult:
        """Review incoming proposal against constraints."""
        reasons = []
        suggested_adjustments = {}
        
        # Check price
        price_acceptable = True
        if floor_cpm is not None and proposal.proposed_cpm < floor_cpm:
            price_acceptable = False
            reasons.append(f"Proposed CPM ${proposal.proposed_cpm} below floor ${floor_cpm}")
            suggested_adjustments["cpm"] = floor_cpm
        
        # Check availability
        avails_acceptable = True
        if max_impressions is not None and proposal.requested_impressions > max_impressions:
            avails_acceptable = False
            reasons.append(
                f"Requested {proposal.requested_impressions:,} impressions exceeds "
                f"available {max_impressions:,}"
            )
            suggested_adjustments["impressions"] = max_impressions
        
        targeting_achievable = True
        
        # Determine status
        if price_acceptable and avails_acceptable and targeting_achievable:
            status = ProposalStatus.ACCEPTABLE
            recommended_action = "accept"
        elif not price_acceptable and not avails_acceptable:
            status = ProposalStatus.REJECTED
            recommended_action = "reject"
        else:
            status = ProposalStatus.NEEDS_ADJUSTMENT
            recommended_action = "counter"
        
        return ReviewResult(
            proposal_id=proposal.proposal_id,
            status=status,
            reasons=reasons,
            recommended_action=recommended_action,
            price_acceptable=price_acceptable,
            avails_acceptable=avails_acceptable,
            targeting_achievable=targeting_achievable,
            suggested_adjustments=suggested_adjustments,
            metadata={
                "buyer_id": proposal.buyer_id,
                "product_id": proposal.product_id,
            },
        )
    
    async def build_counter_offer(
        self,
        original: Proposal,
        constraints: dict,
        seller_id: Optional[str] = None,
    ) -> CounterOffer:
        """Build counter-offer based on constraints."""
        adjustments = []
        
        # Get deal ID
        deal_id = await self.generate_deal_id(original.buyer_id, original.product_id)
        
        # Determine offered CPM
        min_cpm = constraints.get("min_cpm", original.proposed_cpm)
        offered_cpm = max(original.proposed_cpm, min_cpm)
        if offered_cpm > original.proposed_cpm:
            adjustments.append(f"Increased CPM from ${original.proposed_cpm} to ${offered_cpm}")
        
        # Determine offered impressions
        max_impressions = constraints.get("max_impressions", original.requested_impressions)
        offered_impressions = min(original.requested_impressions, max_impressions)
        if offered_impressions < original.requested_impressions:
            adjustments.append(
                f"Reduced impressions from {original.requested_impressions:,} to {offered_impressions:,}"
            )
        
        # Build counter offer
        result = self._handle_counter_offer_builder(
            original.proposal_id,
            offered_cpm,
            offered_impressions,
            constraints,
        )
        
        return CounterOffer(
            counter_offer_id=f"counter-{uuid.uuid4().hex[:8]}",
            original_proposal_id=original.proposal_id,
            deal_id=deal_id,
            seller_id=seller_id or "unknown",
            offered_cpm=offered_cpm,
            offered_impressions=offered_impressions,
            start_date=original.start_date,
            end_date=original.end_date,
            deal_type=original.deal_type,
            adjustments_made=adjustments,
            terms=result.get("terms", {}),
            metadata={
                "original_cpm": original.proposed_cpm,
                "original_impressions": original.requested_impressions,
            },
        )
    
    async def generate_deal_id(self, buyer_id: str, product_id: str) -> str:
        """Generate a unique deal ID."""
        result = self._handle_deal_id_generator(buyer_id, product_id)
        return result.get("deal_id", f"deal-{uuid.uuid4().hex[:12]}")
    
    async def accept_proposal(self, proposal: Proposal) -> dict:
        """Accept a proposal and generate deal confirmation."""
        deal_id = await self.generate_deal_id(proposal.buyer_id, proposal.product_id)
        
        return {
            "deal_id": deal_id,
            "proposal_id": proposal.proposal_id,
            "status": "accepted",
            "agreed_cpm": proposal.proposed_cpm,
            "agreed_impressions": proposal.requested_impressions,
            "start_date": proposal.start_date,
            "end_date": proposal.end_date,
            "accepted_at": datetime.utcnow().isoformat(),
        }
