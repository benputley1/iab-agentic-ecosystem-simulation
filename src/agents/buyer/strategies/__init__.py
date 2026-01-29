"""Bidding strategies for buyer agents.

Strategies determine how buyers calculate bids based on campaign
goals and market conditions.
"""

from enum import Enum
from typing import Protocol
from dataclasses import dataclass


class BidStrategy(str, Enum):
    """Bidding strategy types."""

    TARGET_CPM = "target_cpm"
    """Bid at or below target CPM. Balanced approach."""

    MAXIMIZE_REACH = "maximize_reach"
    """Bid aggressively to win maximum impressions."""

    FLOOR_PLUS = "floor_plus"
    """Bid just above floor price. Cost-efficient approach."""

    PACING = "pacing"
    """Adjust bids based on campaign delivery pace."""

    BUDGET_AWARE = "budget_aware"
    """Reduce bids as budget depletes."""


@dataclass
class BidDecision:
    """Result of a bid calculation."""

    bid_cpm: float
    should_bid: bool
    rationale: str


class BidCalculator(Protocol):
    """Protocol for bid calculation strategies."""

    def calculate(
        self,
        target_cpm: float,
        offer_cpm: float,
        floor_price: float,
        budget_remaining: float,
        impressions_remaining: int,
        pacing_factor: float,
    ) -> BidDecision:
        """Calculate a bid.

        Args:
            target_cpm: Campaign's target CPM
            offer_cpm: Seller's offered CPM
            floor_price: Minimum acceptable CPM
            budget_remaining: Remaining campaign budget
            impressions_remaining: Impressions left to deliver
            pacing_factor: 1.0 = on pace, >1 = behind, <1 = ahead

        Returns:
            BidDecision with bid amount and rationale
        """
        ...


class TargetCPMStrategy:
    """Bid at target CPM, accept offers at or below."""

    def calculate(
        self,
        target_cpm: float,
        offer_cpm: float,
        floor_price: float,
        budget_remaining: float,
        impressions_remaining: int,
        pacing_factor: float,
    ) -> BidDecision:
        if offer_cpm <= target_cpm:
            return BidDecision(
                bid_cpm=offer_cpm,
                should_bid=True,
                rationale=f"Offer ${offer_cpm:.2f} at/below target ${target_cpm:.2f}",
            )
        return BidDecision(
            bid_cpm=target_cpm,
            should_bid=False,
            rationale=f"Offer ${offer_cpm:.2f} exceeds target ${target_cpm:.2f}",
        )


class MaximizeReachStrategy:
    """Bid aggressively to win impressions."""

    def __init__(self, max_multiplier: float = 1.5):
        self.max_multiplier = max_multiplier

    def calculate(
        self,
        target_cpm: float,
        offer_cpm: float,
        floor_price: float,
        budget_remaining: float,
        impressions_remaining: int,
        pacing_factor: float,
    ) -> BidDecision:
        max_bid = target_cpm * self.max_multiplier
        if offer_cpm <= max_bid:
            return BidDecision(
                bid_cpm=offer_cpm,
                should_bid=True,
                rationale=f"Maximize reach: accepting ${offer_cpm:.2f}",
            )
        return BidDecision(
            bid_cpm=max_bid,
            should_bid=False,
            rationale=f"Offer ${offer_cpm:.2f} exceeds max ${max_bid:.2f}",
        )


class FloorPlusStrategy:
    """Bid just above floor price."""

    def __init__(self, floor_increment: float = 0.05):
        self.floor_increment = floor_increment

    def calculate(
        self,
        target_cpm: float,
        offer_cpm: float,
        floor_price: float,
        budget_remaining: float,
        impressions_remaining: int,
        pacing_factor: float,
    ) -> BidDecision:
        bid = floor_price * (1 + self.floor_increment)
        if bid >= offer_cpm:
            return BidDecision(
                bid_cpm=bid,
                should_bid=True,
                rationale=f"Floor+{self.floor_increment*100:.0f}%: ${bid:.2f}",
            )
        return BidDecision(
            bid_cpm=bid,
            should_bid=False,
            rationale=f"Floor+ bid ${bid:.2f} below offer ${offer_cpm:.2f}",
        )


class PacingStrategy:
    """Adjust bids based on delivery pace."""

    def calculate(
        self,
        target_cpm: float,
        offer_cpm: float,
        floor_price: float,
        budget_remaining: float,
        impressions_remaining: int,
        pacing_factor: float,
    ) -> BidDecision:
        # Adjust target based on pacing
        # Behind pace (>1) = bid higher
        # Ahead of pace (<1) = bid lower
        adjusted_target = target_cpm * min(max(pacing_factor, 0.8), 1.5)

        if offer_cpm <= adjusted_target:
            return BidDecision(
                bid_cpm=offer_cpm,
                should_bid=True,
                rationale=f"Paced bid: ${offer_cpm:.2f} (factor={pacing_factor:.2f})",
            )
        return BidDecision(
            bid_cpm=adjusted_target,
            should_bid=False,
            rationale=f"Paced target ${adjusted_target:.2f} below offer",
        )


class BudgetAwareStrategy:
    """Reduce bids as budget depletes."""

    def calculate(
        self,
        target_cpm: float,
        offer_cpm: float,
        floor_price: float,
        budget_remaining: float,
        impressions_remaining: int,
        pacing_factor: float,
    ) -> BidDecision:
        # Calculate effective CPM needed to fulfill remaining impressions
        if impressions_remaining > 0:
            max_affordable_cpm = (budget_remaining / impressions_remaining) * 1000
        else:
            max_affordable_cpm = target_cpm

        effective_max = min(target_cpm, max_affordable_cpm)

        if offer_cpm <= effective_max:
            return BidDecision(
                bid_cpm=offer_cpm,
                should_bid=True,
                rationale=f"Budget-aware: ${offer_cpm:.2f} within ${effective_max:.2f} cap",
            )
        return BidDecision(
            bid_cpm=effective_max,
            should_bid=False,
            rationale=f"Budget cap ${effective_max:.2f} below offer ${offer_cpm:.2f}",
        )


def get_strategy(strategy_type: BidStrategy) -> BidCalculator:
    """Get a strategy calculator by type.

    Args:
        strategy_type: Strategy to use

    Returns:
        BidCalculator instance
    """
    strategies: dict[BidStrategy, BidCalculator] = {
        BidStrategy.TARGET_CPM: TargetCPMStrategy(),
        BidStrategy.MAXIMIZE_REACH: MaximizeReachStrategy(),
        BidStrategy.FLOOR_PLUS: FloorPlusStrategy(),
        BidStrategy.PACING: PacingStrategy(),
        BidStrategy.BUDGET_AWARE: BudgetAwareStrategy(),
    }
    return strategies.get(strategy_type, TargetCPMStrategy())


__all__ = [
    "BidStrategy",
    "BidDecision",
    "BidCalculator",
    "TargetCPMStrategy",
    "MaximizeReachStrategy",
    "FloorPlusStrategy",
    "PacingStrategy",
    "BudgetAwareStrategy",
    "get_strategy",
]
