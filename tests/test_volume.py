"""
Comprehensive tests for the Realistic Volume Generator.
"""

import pytest
from datetime import datetime, timedelta
from collections import Counter

from src.volume import (
    BidRequest,
    VolumeProfile,
    RealisticVolumeGenerator,
    AdFormat,
)


class TestVolumeProfile:
    """Tests for VolumeProfile dataclass."""
    
    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = VolumeProfile(
            name="test",
            daily_requests=50_000,
            bid_rate=0.25,
        )
        assert profile.name == "test"
        assert profile.daily_requests == 50_000
        assert profile.bid_rate == 0.25
    
    def test_profile_default_format_weights(self):
        """Test that default format weights are provided."""
        profile = VolumeProfile(name="test", daily_requests=1000)
        assert len(profile.format_weights) > 0
        assert AdFormat.BANNER_300x250 in profile.format_weights
        # Weights should sum to approximately 1.0
        total_weight = sum(profile.format_weights.values())
        assert 0.99 <= total_weight <= 1.01
    
    def test_profile_custom_floor_prices(self):
        """Test custom floor price range."""
        profile = VolumeProfile(
            name="premium",
            daily_requests=1000,
            floor_price_min=1.00,
            floor_price_max=10.00,
        )
        assert profile.floor_price_min == 1.00
        assert profile.floor_price_max == 10.00


class TestBidRequest:
    """Tests for BidRequest dataclass."""
    
    def test_bid_request_creation(self):
        """Test basic bid request creation."""
        timestamp = datetime.now()
        request = BidRequest(
            request_id="test-123",
            timestamp=timestamp,
            publisher_id="pub_001",
            user_id="user_001",
            floor_price=1.50,
            ad_format=AdFormat.BANNER_300x250,
        )
        assert request.request_id == "test-123"
        assert request.timestamp == timestamp
        assert request.publisher_id == "pub_001"
        assert request.floor_price == 1.50
        assert request.ad_format == AdFormat.BANNER_300x250
    
    def test_bid_request_defaults(self):
        """Test default values."""
        request = BidRequest(
            request_id="test",
            timestamp=datetime.now(),
            publisher_id="pub",
            user_id="user",
            floor_price=1.0,
            ad_format=AdFormat.NATIVE,
        )
        assert request.device_type == "desktop"
        assert request.geo_country == "US"
        assert request.site_domain is None
    
    def test_bid_request_to_dict(self):
        """Test serialization to dictionary."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        request = BidRequest(
            request_id="test-456",
            timestamp=timestamp,
            publisher_id="pub_002",
            user_id="user_002",
            floor_price=2.25,
            ad_format=AdFormat.VIDEO_INSTREAM,
            device_type="mobile",
            geo_country="GB",
            site_domain="example.com",
        )
        
        d = request.to_dict()
        assert d["request_id"] == "test-456"
        assert d["timestamp"] == "2024-01-15T10:30:00"
        assert d["ad_format"] == "video_instream"
        assert d["device_type"] == "mobile"
        assert d["geo_country"] == "GB"


class TestRealisticVolumeGenerator:
    """Tests for RealisticVolumeGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return RealisticVolumeGenerator(
            num_publishers=50,
            num_users=1000,
            seed=42,
            base_date=datetime(2024, 1, 15),  # Monday
        )
    
    def test_predefined_profiles_exist(self, generator):
        """Test that all predefined profiles are available."""
        expected_profiles = ["small", "medium", "large", "enterprise"]
        for profile_name in expected_profiles:
            profile = generator.get_profile(profile_name)
            assert profile is not None
            assert profile.name == profile_name
    
    def test_profile_volumes(self, generator):
        """Test that profiles have correct daily volumes."""
        assert generator.get_profile("small").daily_requests == 10_000
        assert generator.get_profile("medium").daily_requests == 100_000
        assert generator.get_profile("large").daily_requests == 1_000_000
        assert generator.get_profile("enterprise").daily_requests == 10_000_000
    
    def test_invalid_profile_raises_error(self, generator):
        """Test that invalid profile name raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            generator.get_profile("invalid_profile")
        assert "Unknown profile" in str(excinfo.value)
        assert "invalid_profile" in str(excinfo.value)
    
    def test_generate_day_returns_list(self, generator):
        """Test that generate_day returns a list of BidRequests."""
        requests = generator.generate_day("small", day_number=0)
        assert isinstance(requests, list)
        assert len(requests) > 0
        assert all(isinstance(r, BidRequest) for r in requests)
    
    def test_generate_day_volume_approximate(self, generator):
        """Test that daily volume is approximately correct."""
        # Small profile = 10K requests
        requests = generator.generate_day("small", day_number=0)
        # Allow 20% variance due to hour-by-hour rounding
        assert 8_000 <= len(requests) <= 12_000
    
    def test_generate_day_sorted_by_timestamp(self, generator):
        """Test that requests are sorted by timestamp."""
        requests = generator.generate_day("small", day_number=0)
        timestamps = [r.timestamp for r in requests]
        assert timestamps == sorted(timestamps)
    
    def test_generate_day_timestamps_within_day(self, generator):
        """Test that all timestamps fall within the correct day."""
        day_number = 2
        requests = generator.generate_day("small", day_number=day_number)
        
        expected_start = generator.base_date + timedelta(days=day_number)
        expected_end = expected_start + timedelta(days=1)
        
        for request in requests:
            assert expected_start <= request.timestamp < expected_end
    
    def test_weekend_detection(self, generator):
        """Test weekend detection (base_date is Monday Jan 15)."""
        # Day 0 = Monday, Day 5 = Saturday, Day 6 = Sunday
        assert not generator.is_weekend(0)  # Monday
        assert not generator.is_weekend(4)  # Friday
        assert generator.is_weekend(5)       # Saturday
        assert generator.is_weekend(6)       # Sunday
        assert not generator.is_weekend(7)  # Monday
    
    def test_weekend_volume_reduced(self, generator):
        """Test that weekend volume is ~70% of weekday."""
        # Generate weekday (day 0 = Monday) and weekend (day 5 = Saturday)
        weekday_requests = generator.generate_day("small", day_number=0)
        weekend_requests = generator.generate_day("small", day_number=5)
        
        ratio = len(weekend_requests) / len(weekday_requests)
        # Should be approximately 0.70 (allow some variance)
        assert 0.60 <= ratio <= 0.80
    
    def test_hourly_distribution_has_peaks(self, generator):
        """Test that hourly distribution has expected peaks."""
        hourly_weights = generator.get_hourly_distribution(day_number=0)
        
        # Peak morning hours (9-11) should be higher than night (1-4)
        morning_peak = sum(hourly_weights[9:12])
        night_low = sum(hourly_weights[1:5])
        assert morning_peak > night_low * 2
        
        # Peak evening hours (19-21) should be high
        evening_peak = sum(hourly_weights[19:22])
        assert evening_peak > night_low * 2
    
    def test_hourly_distribution_weekend_scaled(self, generator):
        """Test that weekend hourly weights are scaled down."""
        weekday_weights = generator.get_hourly_distribution(day_number=0)
        weekend_weights = generator.get_hourly_distribution(day_number=5)
        
        weekday_total = sum(weekday_weights)
        weekend_total = sum(weekend_weights)
        
        ratio = weekend_total / weekday_total
        assert 0.69 <= ratio <= 0.71  # Should be exactly 0.70
    
    def test_unique_request_ids(self, generator):
        """Test that all request IDs are unique."""
        requests = generator.generate_day("small", day_number=0)
        request_ids = [r.request_id for r in requests]
        assert len(request_ids) == len(set(request_ids))
    
    def test_publisher_distribution(self, generator):
        """Test that requests are distributed across publishers."""
        requests = generator.generate_day("small", day_number=0)
        publishers = set(r.publisher_id for r in requests)
        
        # Should use a significant portion of the publisher pool
        assert len(publishers) >= generator.num_publishers * 0.5
    
    def test_user_distribution(self, generator):
        """Test that requests are distributed across users."""
        requests = generator.generate_day("small", day_number=0)
        users = set(r.user_id for r in requests)
        
        # Should have significant user diversity
        # With 10K requests and 1000 users, most users should appear
        assert len(users) >= generator.num_users * 0.8
    
    def test_ad_format_distribution(self, generator):
        """Test that ad formats follow expected distribution."""
        requests = generator.generate_day("small", day_number=0)
        format_counts = Counter(r.ad_format for r in requests)
        
        # Banner 300x250 should be most common (weight 0.35)
        assert format_counts[AdFormat.BANNER_300x250] > format_counts[AdFormat.INTERSTITIAL]
        
        # All formats should appear
        for ad_format in AdFormat:
            assert ad_format in format_counts
    
    def test_floor_price_range(self, generator):
        """Test that floor prices are within expected range."""
        requests = generator.generate_day("small", day_number=0)
        
        for request in requests:
            # Base range is 0.10 - 5.00, with multipliers for video
            assert 0.10 <= request.floor_price <= 15.00  # Allow for video premium
    
    def test_video_floor_price_premium(self, generator):
        """Test that video formats have higher floor prices on average."""
        requests = generator.generate_day("small", day_number=0)
        
        banner_prices = [
            r.floor_price for r in requests 
            if r.ad_format == AdFormat.BANNER_300x250
        ]
        video_prices = [
            r.floor_price for r in requests 
            if r.ad_format == AdFormat.VIDEO_INSTREAM
        ]
        
        if banner_prices and video_prices:
            avg_banner = sum(banner_prices) / len(banner_prices)
            avg_video = sum(video_prices) / len(video_prices)
            assert avg_video > avg_banner
    
    def test_device_type_distribution(self, generator):
        """Test device type distribution."""
        requests = generator.generate_day("small", day_number=0)
        device_counts = Counter(r.device_type for r in requests)
        
        # Mobile should be most common (weight 0.55)
        assert device_counts["mobile"] > device_counts["desktop"]
        assert device_counts["desktop"] > device_counts["tablet"]
    
    def test_geo_distribution(self, generator):
        """Test geographic distribution."""
        requests = generator.generate_day("small", day_number=0)
        geo_counts = Counter(r.geo_country for r in requests)
        
        # US should be most common (weight 0.40)
        assert geo_counts["US"] > geo_counts["GB"]
    
    def test_site_domain_populated(self, generator):
        """Test that site domain is populated for all requests."""
        requests = generator.generate_day("small", day_number=0)
        
        for request in requests:
            assert request.site_domain is not None
            assert request.site_domain.endswith(".example.com")
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        gen1 = RealisticVolumeGenerator(seed=12345)
        gen2 = RealisticVolumeGenerator(seed=12345)
        
        requests1 = gen1.generate_day("small", day_number=0)
        requests2 = gen2.generate_day("small", day_number=0)
        
        # First request should be identical
        assert requests1[0].request_id == requests2[0].request_id
        assert requests1[0].publisher_id == requests2[0].publisher_id
    
    def test_generate_campaign(self, generator):
        """Test campaign generation across multiple days."""
        campaign = generator.generate_campaign("small", num_days=7)
        
        assert len(campaign) == 7
        assert all(isinstance(campaign[d], list) for d in range(7))
        
        # Each day should have requests
        for day_requests in campaign.values():
            assert len(day_requests) > 0
    
    def test_volume_summary(self, generator):
        """Test volume summary calculation."""
        summary = generator.get_volume_summary("medium", num_days=30)
        
        assert summary["profile"] == "medium"
        assert summary["daily_requests_weekday"] == 100_000
        assert summary["daily_requests_weekend"] == 70_000  # 70% of weekday
        assert summary["num_days"] == 30
        assert "weekdays" in summary
        assert "weekends" in summary
        assert summary["weekdays"] + summary["weekends"] == 30
        assert "total_requests" in summary
        assert "expected_bids" in summary
        assert "context_pressure" in summary
    
    def test_context_pressure_estimation(self, generator):
        """Test context pressure level estimation."""
        small_summary = generator.get_volume_summary("small", num_days=30)
        large_summary = generator.get_volume_summary("large", num_days=30)
        enterprise_summary = generator.get_volume_summary("enterprise", num_days=30)
        
        # Small campaign should have low pressure
        assert small_summary["context_pressure"] == "low"
        
        # Large campaign should have high pressure
        assert large_summary["context_pressure"] == "high"
        
        # Enterprise should have extreme pressure
        assert enterprise_summary["context_pressure"] == "extreme"
    
    def test_custom_profile(self, generator):
        """Test using a custom VolumeProfile."""
        custom_profile = VolumeProfile(
            name="custom",
            daily_requests=5_000,
            bid_rate=0.5,
            floor_price_min=2.00,
            floor_price_max=8.00,
        )
        
        requests = generator.generate_day(custom_profile, day_number=0)
        
        # Should generate approximately 5K requests
        assert 4_000 <= len(requests) <= 6_000
        
        # Floor prices should be in custom range (accounting for video premium)
        for request in requests:
            if request.ad_format not in (AdFormat.VIDEO_INSTREAM, AdFormat.VIDEO_OUTSTREAM, AdFormat.INTERSTITIAL):
                assert 2.00 <= request.floor_price <= 8.00


class TestHourlyGeneration:
    """Tests specifically for hourly generation patterns."""
    
    @pytest.fixture
    def generator(self):
        return RealisticVolumeGenerator(seed=42)
    
    def test_generate_hour_correct_volume(self, generator):
        """Test that generate_hour produces correct number of requests."""
        profile = generator.get_profile("small")
        requests = generator.generate_hour(
            volume=100,
            hour=10,
            day_number=0,
            profile=profile,
        )
        assert len(requests) == 100
    
    def test_generate_hour_timestamps_within_hour(self, generator):
        """Test that all timestamps fall within the specified hour."""
        profile = generator.get_profile("small")
        hour = 14
        day_number = 0
        
        requests = generator.generate_hour(
            volume=100,
            hour=hour,
            day_number=day_number,
            profile=profile,
        )
        
        expected_start = generator.base_date + timedelta(days=day_number, hours=hour)
        expected_end = expected_start + timedelta(hours=1)
        
        for request in requests:
            assert expected_start <= request.timestamp < expected_end
    
    def test_peak_hours_have_more_volume(self, generator):
        """Test that peak hours have proportionally more requests."""
        requests = generator.generate_day("small", day_number=0)
        
        # Count requests per hour
        hour_counts = Counter(r.timestamp.hour for r in requests)
        
        # Peak hours (9-11, 19-21) should have more than night hours (1-4)
        peak_count = sum(hour_counts[h] for h in [9, 10, 11, 19, 20, 21])
        night_count = sum(hour_counts[h] for h in [1, 2, 3, 4])
        
        # Peak should have significantly more
        assert peak_count > night_count * 3


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_publishers(self):
        """Test behavior with minimal publisher pool."""
        generator = RealisticVolumeGenerator(num_publishers=1, num_users=100, seed=42)
        requests = generator.generate_day("small", day_number=0)
        
        # All requests should be from the single publisher
        publishers = set(r.publisher_id for r in requests)
        assert len(publishers) == 1
    
    def test_zero_day_number(self):
        """Test that day 0 works correctly."""
        generator = RealisticVolumeGenerator(seed=42)
        requests = generator.generate_day("small", day_number=0)
        assert len(requests) > 0
    
    def test_large_day_number(self):
        """Test behavior with large day numbers."""
        generator = RealisticVolumeGenerator(seed=42)
        requests = generator.generate_day("small", day_number=365)
        assert len(requests) > 0
        
        # Verify timestamp is correct
        expected_date = generator.base_date + timedelta(days=365)
        assert requests[0].timestamp.date() == expected_date.date()
    
    def test_profile_as_string_or_object(self):
        """Test that both string and VolumeProfile work."""
        generator = RealisticVolumeGenerator(seed=42)
        
        # Using string
        requests1 = generator.generate_day("small", day_number=0)
        
        # Reset seed for comparison
        generator = RealisticVolumeGenerator(seed=42)
        
        # Using VolumeProfile object
        profile = generator.get_profile("small")
        requests2 = generator.generate_day(profile, day_number=0)
        
        assert len(requests1) == len(requests2)


class TestAdFormat:
    """Tests for AdFormat enum."""
    
    def test_all_formats_have_values(self):
        """Test that all formats have string values."""
        for ad_format in AdFormat:
            assert isinstance(ad_format.value, str)
            assert len(ad_format.value) > 0
    
    def test_format_value_lowercase(self):
        """Test that format values are lowercase."""
        for ad_format in AdFormat:
            assert ad_format.value == ad_format.value.lower()
