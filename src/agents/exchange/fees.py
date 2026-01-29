"""
Fee configuration and extraction logic for Scenario A rent-seeking exchange.

The exchange extracts fees (10-20%) from every transaction as a "tech tax".
"""

from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger()


# Default fee range for rent-seeking exchange (Scenario A)
DEFAULT_MIN_FEE_PCT = 0.10  # 10%
DEFAULT_MAX_FEE_PCT = 0.20  # 20%
DEFAULT_FEE_PCT = 0.15      # 15% default


@dataclass
class FeeConfig:
    """Configuration for exchange fee extraction."""

    base_fee_pct: float = DEFAULT_FEE_PCT
    min_fee_pct: float = DEFAULT_MIN_FEE_PCT
    max_fee_pct: float = DEFAULT_MAX_FEE_PCT

    # Optional: buyer/seller favoritism adjustments
    preferred_buyers: Optional[dict[str, float]] = None   # buyer_id -> fee discount
    preferred_sellers: Optional[dict[str, float]] = None  # seller_id -> fee discount

    def __post_init__(self):
        """Validate fee configuration."""
        if not (0 <= self.min_fee_pct <= self.max_fee_pct <= 1.0):
            raise ValueError(
                f"Invalid fee range: min={self.min_fee_pct}, max={self.max_fee_pct}"
            )
        if not (self.min_fee_pct <= self.base_fee_pct <= self.max_fee_pct):
            raise ValueError(
                f"Base fee {self.base_fee_pct} outside range "
                f"[{self.min_fee_pct}, {self.max_fee_pct}]"
            )
        self.preferred_buyers = self.preferred_buyers or {}
        self.preferred_sellers = self.preferred_sellers or {}

    def get_effective_fee(
        self,
        buyer_id: Optional[str] = None,
        seller_id: Optional[str] = None,
        deal_value: Optional[float] = None,
    ) -> float:
        """
        Calculate effective fee percentage for a transaction.

        Applies any preferential discounts for specific buyers/sellers.

        Args:
            buyer_id: Buyer involved in transaction
            seller_id: Seller involved in transaction
            deal_value: Total deal value (for potential volume discounts)

        Returns:
            Fee as decimal (e.g., 0.15 for 15%)
        """
        fee = self.base_fee_pct

        # Apply buyer discount if preferred
        if buyer_id and buyer_id in self.preferred_buyers:
            buyer_discount = self.preferred_buyers[buyer_id]
            fee = fee * (1 - buyer_discount)

        # Apply seller discount if preferred
        if seller_id and seller_id in self.preferred_sellers:
            seller_discount = self.preferred_sellers[seller_id]
            fee = fee * (1 - seller_discount)

        # Clamp to configured range
        fee = max(self.min_fee_pct, min(self.max_fee_pct, fee))

        logger.debug(
            "fee.calculated",
            buyer_id=buyer_id,
            seller_id=seller_id,
            base_fee=self.base_fee_pct,
            effective_fee=fee,
        )

        return fee


def calculate_exchange_fee(
    total_cost: float,
    fee_pct: float,
) -> float:
    """
    Calculate exchange fee amount.

    Args:
        total_cost: Total transaction value
        fee_pct: Fee percentage as decimal (e.g., 0.15)

    Returns:
        Fee amount in currency units
    """
    return total_cost * fee_pct


def calculate_markup_cpm(
    original_cpm: float,
    fee_pct: float,
) -> float:
    """
    Calculate marked-up CPM to pass to buyer.

    The exchange adds its fee on top of the seller's offered price.

    Args:
        original_cpm: Seller's original offered CPM
        fee_pct: Exchange fee percentage as decimal

    Returns:
        Marked-up CPM that buyer sees
    """
    return original_cpm * (1 + fee_pct)


def calculate_seller_revenue(
    buyer_pays: float,
    fee_pct: float,
) -> float:
    """
    Calculate what seller receives after exchange fee.

    Args:
        buyer_pays: Total amount buyer paid
        fee_pct: Exchange fee percentage

    Returns:
        Amount seller receives
    """
    return buyer_pays / (1 + fee_pct)
