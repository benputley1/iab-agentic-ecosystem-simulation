"""
Simulated inventory for RTB seller agents.

Creates synthetic publisher inventory for the simulation environment.
"""

import random
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class InventoryType(str, Enum):
    """Types of ad inventory."""
    DISPLAY = "display"
    VIDEO = "video"
    CTV = "ctv"
    NATIVE = "native"


class DealType(str, Enum):
    """Types of programmatic deals."""
    PROGRAMMATIC_GUARANTEED = "PG"
    PREFERRED_DEAL = "PD"
    PRIVATE_AUCTION = "PA"
    OPEN_AUCTION = "OA"


class Product(BaseModel):
    """A publisher's ad product/placement."""
    product_id: str
    name: str
    description: str
    inventory_type: str
    base_cpm: float
    floor_cpm: float
    currency: str = "USD"
    audience_targeting: list[str] = Field(default_factory=list)
    content_targeting: list[str] = Field(default_factory=list)
    supported_deal_types: list[DealType] = Field(
        default_factory=lambda: [DealType.OPEN_AUCTION, DealType.PREFERRED_DEAL]
    )
    daily_impressions: int = 100000


class SimulatedInventory:
    """Generates and manages simulated inventory for sellers.

    Creates realistic inventory that varies by:
    - Channel type (display, video, CTV, native)
    - Quality tier (premium, standard, remnant)
    - Audience segments available
    """

    # Base CPM ranges by inventory type
    BASE_CPM_RANGES = {
        InventoryType.DISPLAY: (5.0, 15.0),
        InventoryType.VIDEO: (15.0, 35.0),
        InventoryType.CTV: (25.0, 55.0),
        InventoryType.NATIVE: (8.0, 20.0),
    }

    # Floor CPM is typically 60-80% of base
    FLOOR_CPM_RATIO = (0.6, 0.8)

    # Common audience segments
    AUDIENCE_SEGMENTS = [
        "auto_intenders", "tech_enthusiasts", "parents",
        "sports_fans", "travel_intenders", "home_improvers",
        "business_professionals", "entertainment_seekers",
        "health_wellness", "fashion_beauty",
    ]

    # Content categories
    CONTENT_CATEGORIES = [
        "news", "sports", "entertainment", "technology",
        "lifestyle", "business", "health", "food",
        "travel", "automotive",
    ]

    def __init__(self, seller_id: str, seed: Optional[int] = None):
        """Initialize inventory for a seller.

        Args:
            seller_id: Unique seller identifier
            seed: Random seed for reproducible inventory
        """
        self.seller_id = seller_id
        self._rng = random.Random(seed)
        self._products: dict[str, Product] = {}

    def generate_catalog(
        self,
        num_products: int = 5,
        inventory_types: Optional[list[InventoryType]] = None,
    ) -> dict[str, Product]:
        """Generate a product catalog for this seller.

        Args:
            num_products: Number of products to generate
            inventory_types: Types to include (default: all types)

        Returns:
            Dict of product_id to Product
        """
        if inventory_types is None:
            inventory_types = list(InventoryType)

        for i in range(num_products):
            inv_type = self._rng.choice(inventory_types)
            product = self._generate_product(inv_type, i)
            self._products[product.product_id] = product

        return self._products

    def _generate_product(self, inv_type: InventoryType, index: int) -> Product:
        """Generate a single product."""
        base_low, base_high = self.BASE_CPM_RANGES[inv_type]
        base_cpm = round(self._rng.uniform(base_low, base_high), 2)
        floor_ratio = self._rng.uniform(*self.FLOOR_CPM_RATIO)
        floor_cpm = round(base_cpm * floor_ratio, 2)

        # Select random audience segments
        num_segments = self._rng.randint(2, 5)
        segments = self._rng.sample(self.AUDIENCE_SEGMENTS, num_segments)

        # Select content categories
        num_categories = self._rng.randint(1, 3)
        categories = self._rng.sample(self.CONTENT_CATEGORIES, num_categories)

        # Product naming
        quality_tier = self._rng.choice(["Premium", "Standard", "Performance"])
        product_name = f"{quality_tier} {inv_type.value.title()}"

        # Daily impressions based on quality
        if quality_tier == "Premium":
            daily_imps = self._rng.randint(50000, 200000)
        elif quality_tier == "Standard":
            daily_imps = self._rng.randint(200000, 1000000)
        else:
            daily_imps = self._rng.randint(500000, 5000000)

        # Deal types based on quality
        if quality_tier == "Premium":
            deal_types = [
                DealType.PROGRAMMATIC_GUARANTEED,
                DealType.PREFERRED_DEAL,
                DealType.PRIVATE_AUCTION,
            ]
        else:
            deal_types = [
                DealType.PREFERRED_DEAL,
                DealType.PRIVATE_AUCTION,
                DealType.OPEN_AUCTION,
            ]

        return Product(
            product_id=f"{self.seller_id}-{inv_type.value}-{index:03d}",
            name=product_name,
            description=f"{quality_tier} {inv_type.value} inventory with {', '.join(categories)} content",
            inventory_type=inv_type.value,
            base_cpm=base_cpm,
            floor_cpm=floor_cpm,
            audience_targeting=segments,
            content_targeting=categories,
            supported_deal_types=deal_types,
            daily_impressions=daily_imps,
        )

    def get_product(self, product_id: str) -> Optional[Product]:
        """Get a product by ID."""
        return self._products.get(product_id)

    def get_products_for_channel(self, channel: str) -> list[Product]:
        """Get all products matching a channel type."""
        return [
            p for p in self._products.values()
            if p.inventory_type == channel
        ]

    def check_availability(
        self,
        product_id: str,
        impressions_requested: int,
        days: int = 30,
    ) -> tuple[bool, int]:
        """Check if requested impressions are available.

        Args:
            product_id: Product to check
            impressions_requested: Number of impressions requested
            days: Campaign duration in days

        Returns:
            Tuple of (available, max_impressions)
        """
        product = self._products.get(product_id)
        if not product:
            return False, 0

        max_available = product.daily_impressions * days
        is_available = impressions_requested <= max_available

        return is_available, max_available

    @property
    def products(self) -> dict[str, Product]:
        """Get all products."""
        return self._products
