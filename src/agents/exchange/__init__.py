"""
Exchange agent implementations for RTB simulation.

Scenario A: Rent-seeking exchange with second-price auction and fee extraction.
"""

from .auction import (
    AuctionBid,
    AuctionResult,
    SecondPriceAuction,
    RentSeekingExchange,
)
from .fees import (
    FeeConfig,
    calculate_exchange_fee,
    calculate_markup_cpm,
    calculate_seller_revenue,
    DEFAULT_FEE_PCT,
    DEFAULT_MIN_FEE_PCT,
    DEFAULT_MAX_FEE_PCT,
)

__all__ = [
    # Auction
    "AuctionBid",
    "AuctionResult",
    "SecondPriceAuction",
    "RentSeekingExchange",
    # Fees
    "FeeConfig",
    "calculate_exchange_fee",
    "calculate_markup_cpm",
    "calculate_seller_revenue",
    "DEFAULT_FEE_PCT",
    "DEFAULT_MIN_FEE_PCT",
    "DEFAULT_MAX_FEE_PCT",
]
