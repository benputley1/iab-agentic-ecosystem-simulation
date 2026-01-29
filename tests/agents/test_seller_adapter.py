"""Tests for the seller agent adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.agents.seller import (
    SellerAgentAdapter,
    SimulatedInventory,
    Product,
    InventoryType,
)
from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealType,
)


class TestSimulatedInventory:
    """Tests for SimulatedInventory."""

    def test_generate_catalog(self):
        """Test catalog generation."""
        inventory = SimulatedInventory("seller-001", seed=42)
        catalog = inventory.generate_catalog(num_products=5)

        assert len(catalog) == 5
        assert all(isinstance(p, Product) for p in catalog.values())

    def test_reproducible_with_seed(self):
        """Test that same seed produces same inventory."""
        inv1 = SimulatedInventory("seller-001", seed=42)
        inv2 = SimulatedInventory("seller-001", seed=42)

        cat1 = inv1.generate_catalog(num_products=3)
        cat2 = inv2.generate_catalog(num_products=3)

        for pid in cat1:
            assert cat1[pid].base_cpm == cat2[pid].base_cpm
            assert cat1[pid].floor_cpm == cat2[pid].floor_cpm

    def test_get_products_for_channel(self):
        """Test filtering products by channel."""
        inventory = SimulatedInventory("seller-001", seed=42)
        inventory.generate_catalog(
            num_products=10,
            inventory_types=[InventoryType.DISPLAY, InventoryType.VIDEO],
        )

        display = inventory.get_products_for_channel("display")
        video = inventory.get_products_for_channel("video")

        for p in display:
            assert p.inventory_type == "display"
        for p in video:
            assert p.inventory_type == "video"

    def test_check_availability(self):
        """Test availability checking."""
        inventory = SimulatedInventory("seller-001", seed=42)
        catalog = inventory.generate_catalog(num_products=1)
        product_id = list(catalog.keys())[0]
        product = catalog[product_id]

        # Request within daily capacity
        available, max_imps = inventory.check_availability(
            product_id,
            impressions_requested=product.daily_impressions * 10,
            days=30,
        )
        assert available

        # Request exceeds capacity
        available, max_imps = inventory.check_availability(
            product_id,
            impressions_requested=product.daily_impressions * 100,
            days=30,
        )
        assert not available

    def test_product_has_deal_types(self):
        """Test that products have supported deal types."""
        inventory = SimulatedInventory("seller-001", seed=42)
        catalog = inventory.generate_catalog(num_products=5)

        for product in catalog.values():
            assert len(product.supported_deal_types) > 0


class TestSellerAgentAdapter:
    """Tests for SellerAgentAdapter."""

    @pytest.fixture
    def mock_bus(self):
        """Mock Redis bus."""
        bus = AsyncMock()
        bus.connect = AsyncMock(return_value=bus)
        bus.disconnect = AsyncMock()
        bus.ensure_consumer_group = AsyncMock()
        bus.read_bid_requests = AsyncMock(return_value=[])
        bus.ack_bid_requests = AsyncMock()
        bus.publish_bid_response = AsyncMock()
        return bus

    @pytest.fixture
    def adapter(self, mock_bus):
        """Create adapter with mocked bus."""
        adapter = SellerAgentAdapter(
            seller_id="test-seller",
            scenario="A",
            mock_llm=True,
            inventory_seed=42,
        )
        adapter._bus = mock_bus
        return adapter

    @pytest.mark.asyncio
    async def test_connect_generates_inventory(self, adapter, mock_bus):
        """Test that connect generates inventory."""
        await adapter.connect()

        assert len(adapter.products) == 5
        mock_bus.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_bid_request_success(self, adapter, mock_bus):
        """Test handling a bid request that should be accepted."""
        await adapter.connect()

        # Get a product for this channel
        products = adapter._inventory.get_products_for_channel("display")
        if not products:
            pytest.skip("No display products generated")

        product = products[0]

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=product.base_cpm * 1.5,  # Above base CPM
            targeting={"segments": product.audience_targeting[:1]},
        )

        await adapter._handle_bid_request(request, "msg-001")

        # Should have published a response
        mock_bus.publish_bid_response.assert_called_once()
        response = mock_bus.publish_bid_response.call_args[0][0]

        assert isinstance(response, BidResponse)
        assert response.request_id == request.request_id
        assert response.seller_id == adapter.seller_id
        assert response.offered_cpm >= product.floor_cpm

    @pytest.mark.asyncio
    async def test_handle_bid_request_below_floor(self, adapter, mock_bus):
        """Test handling a bid request below floor price."""
        await adapter.connect()

        # Get a product
        products = adapter._inventory.get_products_for_channel("display")
        if not products:
            pytest.skip("No display products generated")

        product = products[0]

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=product.floor_cpm * 0.5,  # Below floor
            targeting={},
        )

        await adapter._handle_bid_request(request, "msg-001")

        # Should NOT have published a response (rejected)
        mock_bus.publish_bid_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_bid_request_no_inventory(self, adapter, mock_bus):
        """Test handling a bid request for unavailable channel."""
        await adapter.connect()

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="nonexistent_channel",
            impressions_requested=10000,
            max_cpm=50.0,
            targeting={},
        )

        await adapter._handle_bid_request(request, "msg-001")

        # Should NOT have published a response (no inventory)
        mock_bus.publish_bid_response.assert_not_called()

    def test_mock_evaluate_accepts_above_floor(self, adapter):
        """Test mock evaluation accepts prices above floor."""
        product = Product(
            product_id="test-prod",
            name="Test Product",
            description="Test",
            inventory_type="display",
            base_cpm=15.0,
            floor_cpm=10.0,
            daily_impressions=100000,
        )

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=12.0,  # Above floor
            targeting={},
        )

        adapter._products = {product.product_id: product}
        adapter._inventory._products = adapter._products

        result = adapter._mock_evaluate(request, product)

        assert result["accept"] is True
        assert result["offer_cpm"] >= product.floor_cpm
        assert result["offer_cpm"] <= request.max_cpm

    def test_mock_evaluate_rejects_below_floor(self, adapter):
        """Test mock evaluation rejects prices below floor."""
        product = Product(
            product_id="test-prod",
            name="Test Product",
            description="Test",
            inventory_type="display",
            base_cpm=15.0,
            floor_cpm=10.0,
            daily_impressions=100000,
        )

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=8.0,  # Below floor
            targeting={},
        )

        result = adapter._mock_evaluate(request, product)

        assert result["accept"] is False
        assert "floor" in result["reason"].lower()

    def test_select_product_prefers_price_match(self, adapter):
        """Test product selection prefers price-compatible products."""
        cheap_product = Product(
            product_id="cheap",
            name="Cheap",
            description="Low price",
            inventory_type="display",
            base_cpm=5.0,
            floor_cpm=3.0,
            daily_impressions=100000,
        )

        expensive_product = Product(
            product_id="expensive",
            name="Expensive",
            description="High price",
            inventory_type="display",
            base_cpm=50.0,
            floor_cpm=40.0,
            daily_impressions=100000,
        )

        adapter._inventory._products = {
            "cheap": cheap_product,
            "expensive": expensive_product,
        }

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=10.0,  # Can only afford cheap
            targeting={},
        )

        selected = adapter._select_product(
            [cheap_product, expensive_product],
            request,
        )

        assert selected.product_id == "cheap"

    def test_create_response_has_deal_id(self, adapter):
        """Test that responses include deal IDs."""
        product = Product(
            product_id="test-prod",
            name="Test",
            description="Test",
            inventory_type="display",
            base_cpm=10.0,
            floor_cpm=8.0,
            daily_impressions=100000,
        )

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=15.0,
            targeting={},
        )

        adapter.seller_id = "TEST"
        evaluation = {
            "offer_cpm": 9.0,
            "offer_impressions": 10000,
            "deal_type": DealType.PREFERRED_DEAL,
        }

        response = adapter._create_response(request, product, evaluation)

        assert response.deal_id is not None
        assert response.deal_id.startswith("TEST")
        assert response.valid_until is not None


class TestDealTypeSelection:
    """Tests for deal type selection logic."""

    @pytest.fixture
    def adapter(self):
        return SellerAgentAdapter(
            seller_id="test",
            mock_llm=True,
        )

    def test_selects_pg_for_large_volume(self, adapter):
        """Test PG selection for large volume requests."""
        from src.agents.seller.inventory import DealType as InvDealType

        product = Product(
            product_id="test",
            name="Test",
            description="Test",
            inventory_type="display",
            base_cpm=10.0,
            floor_cpm=8.0,
            daily_impressions=100000,
            supported_deal_types=[
                InvDealType.PROGRAMMATIC_GUARANTEED,
                InvDealType.PREFERRED_DEAL,
            ],
        )

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=2000000,  # Large volume
            max_cpm=15.0,
            targeting={},
        )

        deal_type = adapter._select_deal_type(request, product)

        assert deal_type == DealType.PROGRAMMATIC_GUARANTEED

    def test_selects_pd_for_small_volume(self, adapter):
        """Test PD selection for smaller volume."""
        from src.agents.seller.inventory import DealType as InvDealType

        product = Product(
            product_id="test",
            name="Test",
            description="Test",
            inventory_type="display",
            base_cpm=10.0,
            floor_cpm=8.0,
            daily_impressions=100000,
            supported_deal_types=[
                InvDealType.PROGRAMMATIC_GUARANTEED,
                InvDealType.PREFERRED_DEAL,
            ],
        )

        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=50000,  # Small volume
            max_cpm=15.0,
            targeting={},
        )

        deal_type = adapter._select_deal_type(request, product)

        assert deal_type == DealType.PREFERRED_DEAL
