"""Tests for exchange fee configuration and calculations."""

import pytest
from src.agents.exchange.fees import (
    FeeConfig,
    calculate_exchange_fee,
    calculate_markup_cpm,
    calculate_seller_revenue,
    DEFAULT_FEE_PCT,
    DEFAULT_MIN_FEE_PCT,
    DEFAULT_MAX_FEE_PCT,
)


class TestFeeConfig:
    """Tests for FeeConfig class."""

    def test_default_values(self):
        """Default config uses 15% base fee."""
        config = FeeConfig()
        assert config.base_fee_pct == DEFAULT_FEE_PCT
        assert config.min_fee_pct == DEFAULT_MIN_FEE_PCT
        assert config.max_fee_pct == DEFAULT_MAX_FEE_PCT

    def test_custom_fee_range(self):
        """Can configure custom fee range within 10-20%."""
        config = FeeConfig(
            base_fee_pct=0.12,
            min_fee_pct=0.10,
            max_fee_pct=0.18,
        )
        assert config.base_fee_pct == 0.12
        assert config.min_fee_pct == 0.10
        assert config.max_fee_pct == 0.18

    def test_invalid_range_raises(self):
        """Invalid fee range raises ValueError."""
        with pytest.raises(ValueError):
            FeeConfig(min_fee_pct=0.20, max_fee_pct=0.10)

    def test_base_outside_range_raises(self):
        """Base fee outside range raises ValueError."""
        with pytest.raises(ValueError):
            FeeConfig(base_fee_pct=0.25, min_fee_pct=0.10, max_fee_pct=0.20)

    def test_effective_fee_no_discounts(self):
        """Without discounts, effective fee equals base fee."""
        config = FeeConfig(base_fee_pct=0.15)
        fee = config.get_effective_fee(buyer_id="buyer-001", seller_id="seller-001")
        assert fee == 0.15

    def test_buyer_discount(self):
        """Preferred buyer gets fee discount."""
        config = FeeConfig(
            base_fee_pct=0.15,
            preferred_buyers={"vip-buyer": 0.20},  # 20% discount
        )
        fee = config.get_effective_fee(buyer_id="vip-buyer")
        assert fee == 0.12  # 0.15 * 0.80 = 0.12

    def test_seller_discount(self):
        """Preferred seller gets fee discount."""
        config = FeeConfig(
            base_fee_pct=0.15,
            preferred_sellers={"premium-seller": 0.10},  # 10% discount
        )
        fee = config.get_effective_fee(seller_id="premium-seller")
        assert fee == 0.135  # 0.15 * 0.90 = 0.135

    def test_combined_discounts(self):
        """Both buyer and seller discounts stack multiplicatively."""
        config = FeeConfig(
            base_fee_pct=0.20,
            min_fee_pct=0.10,
            preferred_buyers={"vip-buyer": 0.25},
            preferred_sellers={"premium-seller": 0.20},
        )
        fee = config.get_effective_fee(
            buyer_id="vip-buyer",
            seller_id="premium-seller",
        )
        # 0.20 * 0.75 * 0.80 = 0.12
        assert fee == pytest.approx(0.12)

    def test_fee_clamped_to_minimum(self):
        """Fee is clamped to minimum if discounts would go below."""
        config = FeeConfig(
            base_fee_pct=0.12,
            min_fee_pct=0.10,
            max_fee_pct=0.20,
            preferred_buyers={"mega-vip": 0.50},  # 50% discount would give 0.06
        )
        fee = config.get_effective_fee(buyer_id="mega-vip")
        assert fee == 0.10  # Clamped to minimum


class TestFeeCalculations:
    """Tests for fee calculation functions."""

    def test_calculate_exchange_fee(self):
        """Exchange fee is percentage of total cost."""
        fee = calculate_exchange_fee(total_cost=100.0, fee_pct=0.15)
        assert fee == 15.0

    def test_calculate_exchange_fee_zero(self):
        """Zero fee percentage gives zero fee."""
        fee = calculate_exchange_fee(total_cost=100.0, fee_pct=0.0)
        assert fee == 0.0

    def test_calculate_markup_cpm(self):
        """Markup adds fee on top of original CPM."""
        marked_up = calculate_markup_cpm(original_cpm=10.0, fee_pct=0.15)
        assert marked_up == 11.5  # 10 * 1.15

    def test_calculate_markup_cpm_at_boundaries(self):
        """Markup at 10% and 20% boundaries."""
        assert calculate_markup_cpm(10.0, 0.10) == 11.0
        assert calculate_markup_cpm(10.0, 0.20) == 12.0

    def test_calculate_seller_revenue(self):
        """Seller revenue is buyer payment minus exchange fee."""
        revenue = calculate_seller_revenue(buyer_pays=115.0, fee_pct=0.15)
        assert revenue == pytest.approx(100.0)  # 115 / 1.15 = 100

    def test_fee_roundtrip(self):
        """Markup then revenue calculation gives original value."""
        original_cpm = 12.50
        fee_pct = 0.15
        marked_up = calculate_markup_cpm(original_cpm, fee_pct)
        recovered = calculate_seller_revenue(marked_up, fee_pct)
        assert abs(recovered - original_cpm) < 0.001


class TestFeeScenarios:
    """Test realistic fee scenarios."""

    def test_scenario_a_typical_deal(self):
        """Typical Scenario A deal with 15% fee."""
        impressions = 100000
        cpm = 10.0
        fee_pct = 0.15

        total_cost = (impressions / 1000) * cpm  # $1000
        exchange_fee = calculate_exchange_fee(total_cost, fee_pct)  # $150
        seller_revenue = total_cost - exchange_fee  # $850

        assert total_cost == 1000.0
        assert exchange_fee == 150.0
        assert seller_revenue == 850.0

    def test_minimum_fee_extraction(self):
        """At 10% minimum fee."""
        config = FeeConfig(base_fee_pct=0.10)
        fee = config.get_effective_fee()

        total_cost = 1000.0
        exchange_fee = calculate_exchange_fee(total_cost, fee)

        assert exchange_fee == 100.0

    def test_maximum_fee_extraction(self):
        """At 20% maximum fee."""
        config = FeeConfig(base_fee_pct=0.20)
        fee = config.get_effective_fee()

        total_cost = 1000.0
        exchange_fee = calculate_exchange_fee(total_cost, fee)

        assert exchange_fee == 200.0
