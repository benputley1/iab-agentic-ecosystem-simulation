"""
Realistic Volume Generator for RTB Simulation.

Generates realistic bid request volumes that stress context windows,
with temporal patterns matching real-world ad traffic.
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class AdFormat(Enum):
    """Standard IAB ad formats."""
    BANNER_300x250 = "banner_300x250"
    BANNER_728x90 = "banner_728x90"
    BANNER_320x50 = "banner_320x50"
    VIDEO_INSTREAM = "video_instream"
    VIDEO_OUTSTREAM = "video_outstream"
    NATIVE = "native"
    INTERSTITIAL = "interstitial"


@dataclass
class VolumeProfile:
    """Configuration for volume generation profile."""
    name: str
    daily_requests: int
    bid_rate: float = 0.2  # Probability that a request gets bid on
    description: str = ""
    
    # Distribution weights for ad formats
    format_weights: Dict[AdFormat, float] = field(default_factory=lambda: {
        AdFormat.BANNER_300x250: 0.35,
        AdFormat.BANNER_728x90: 0.20,
        AdFormat.BANNER_320x50: 0.15,
        AdFormat.VIDEO_INSTREAM: 0.10,
        AdFormat.VIDEO_OUTSTREAM: 0.08,
        AdFormat.NATIVE: 0.07,
        AdFormat.INTERSTITIAL: 0.05,
    })
    
    # Floor price range (USD)
    floor_price_min: float = 0.10
    floor_price_max: float = 5.00


@dataclass
class BidRequest:
    """Represents a single bid request in the RTB ecosystem."""
    request_id: str
    timestamp: datetime
    publisher_id: str
    user_id: str
    floor_price: float
    ad_format: AdFormat
    
    # Optional metadata
    device_type: str = "desktop"
    geo_country: str = "US"
    site_domain: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "publisher_id": self.publisher_id,
            "user_id": self.user_id,
            "floor_price": self.floor_price,
            "ad_format": self.ad_format.value,
            "device_type": self.device_type,
            "geo_country": self.geo_country,
            "site_domain": self.site_domain,
        }


class RealisticVolumeGenerator:
    """
    Generate realistic ad request volumes with temporal patterns.
    
    Simulates real-world ad traffic patterns including:
    - Hourly distribution (peak hours 9-11am, 7-10pm)
    - Weekday/weekend patterns (weekend = 70% of weekday)
    - Publisher and user diversity
    """
    
    # Predefined volume profiles
    VOLUME_PROFILES: Dict[str, VolumeProfile] = {
        "small": VolumeProfile(
            name="small",
            daily_requests=10_000,
            bid_rate=0.30,
            description="Small campaign - 10K daily requests"
        ),
        "medium": VolumeProfile(
            name="medium",
            daily_requests=100_000,
            bid_rate=0.20,
            description="Medium campaign - 100K daily requests"
        ),
        "large": VolumeProfile(
            name="large",
            daily_requests=1_000_000,
            bid_rate=0.10,
            description="Large campaign - 1M daily requests"
        ),
        "enterprise": VolumeProfile(
            name="enterprise",
            daily_requests=10_000_000,
            bid_rate=0.05,
            description="Enterprise campaign - 10M daily requests"
        ),
    }
    
    # Hourly distribution weights (0-23 hours)
    # Peak hours: 9-11am (work morning), 7-10pm (evening leisure)
    HOURLY_WEIGHTS = [
        0.2,   # 00:00 - very low
        0.1,   # 01:00 - minimal
        0.1,   # 02:00 - minimal
        0.1,   # 03:00 - minimal
        0.1,   # 04:00 - minimal
        0.2,   # 05:00 - starting to wake
        0.4,   # 06:00 - early risers
        0.7,   # 07:00 - commute starts
        0.9,   # 08:00 - work begins
        1.3,   # 09:00 - PEAK morning
        1.4,   # 10:00 - PEAK morning
        1.3,   # 11:00 - PEAK morning
        1.1,   # 12:00 - lunch
        1.0,   # 13:00 - afternoon
        0.9,   # 14:00 - afternoon
        0.9,   # 15:00 - afternoon
        1.0,   # 16:00 - end of work
        1.1,   # 17:00 - commute home
        1.2,   # 18:00 - evening
        1.5,   # 19:00 - PEAK evening
        1.6,   # 20:00 - PEAK evening
        1.5,   # 21:00 - PEAK evening
        1.0,   # 22:00 - winding down
        0.5,   # 23:00 - late night
    ]
    
    # Weekend multiplier (weekend traffic is 70% of weekday)
    WEEKEND_MULTIPLIER = 0.70
    
    # Device type distribution
    DEVICE_WEIGHTS = {
        "mobile": 0.55,
        "desktop": 0.35,
        "tablet": 0.08,
        "ctv": 0.02,
    }
    
    # Geo distribution (simplified)
    GEO_WEIGHTS = {
        "US": 0.40,
        "GB": 0.12,
        "DE": 0.10,
        "FR": 0.08,
        "CA": 0.06,
        "AU": 0.05,
        "JP": 0.05,
        "BR": 0.04,
        "IN": 0.05,
        "OTHER": 0.05,
    }
    
    def __init__(
        self,
        num_publishers: int = 100,
        num_users: int = 10_000,
        seed: Optional[int] = None,
        base_date: Optional[datetime] = None,
    ):
        """
        Initialize the volume generator.
        
        Args:
            num_publishers: Number of unique publishers in the pool
            num_users: Number of unique users in the pool
            seed: Random seed for reproducibility
            base_date: Starting date for the simulation (default: today)
        """
        self.num_publishers = num_publishers
        self.num_users = num_users
        self.seed = seed
        self.base_date = base_date or datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        # Use dedicated Random instance for reproducibility
        self._rng = random.Random(seed)
        
        # Pre-generate publisher and user pools
        self._publishers = [f"pub_{i:05d}" for i in range(num_publishers)]
        self._users = [f"user_{i:08d}" for i in range(num_users)]
        
        # Publisher site domains
        self._publisher_domains = {
            pub: f"site{i}.example.com" 
            for i, pub in enumerate(self._publishers)
        }
    
    def get_profile(self, profile_name: str) -> VolumeProfile:
        """
        Get a volume profile by name.
        
        Args:
            profile_name: One of 'small', 'medium', 'large', 'enterprise'
            
        Returns:
            VolumeProfile configuration
            
        Raises:
            ValueError: If profile name is not recognized
        """
        if profile_name not in self.VOLUME_PROFILES:
            valid = ", ".join(self.VOLUME_PROFILES.keys())
            raise ValueError(f"Unknown profile '{profile_name}'. Valid profiles: {valid}")
        return self.VOLUME_PROFILES[profile_name]
    
    def get_hourly_distribution(self, day_number: int) -> List[float]:
        """
        Get hourly weights adjusted for day of week.
        
        Args:
            day_number: Day number in simulation (0 = first day)
            
        Returns:
            List of 24 hourly weights
        """
        # Determine day of week (0 = Monday, 6 = Sunday)
        actual_date = self.base_date + timedelta(days=day_number)
        day_of_week = actual_date.weekday()
        
        # Apply weekend multiplier
        is_weekend = day_of_week >= 5  # Saturday or Sunday
        multiplier = self.WEEKEND_MULTIPLIER if is_weekend else 1.0
        
        return [w * multiplier for w in self.HOURLY_WEIGHTS]
    
    def is_weekend(self, day_number: int) -> bool:
        """Check if a day number falls on a weekend."""
        actual_date = self.base_date + timedelta(days=day_number)
        return actual_date.weekday() >= 5
    
    def _select_weighted(self, weights: Dict[str, float]) -> str:
        """Select a key based on weights."""
        items = list(weights.keys())
        probs = list(weights.values())
        return self._rng.choices(items, weights=probs, k=1)[0]
    
    def _select_ad_format(self, profile: VolumeProfile) -> AdFormat:
        """Select an ad format based on profile weights."""
        formats = list(profile.format_weights.keys())
        weights = list(profile.format_weights.values())
        return self._rng.choices(formats, weights=weights, k=1)[0]
    
    def _generate_request_id(self) -> str:
        """Generate a deterministic request ID using internal RNG."""
        hex_chars = '0123456789abcdef'
        parts = [
            ''.join(self._rng.choices(hex_chars, k=8)),
            ''.join(self._rng.choices(hex_chars, k=4)),
            '4' + ''.join(self._rng.choices(hex_chars, k=3)),  # Version 4
            self._rng.choice('89ab') + ''.join(self._rng.choices(hex_chars, k=3)),  # Variant
            ''.join(self._rng.choices(hex_chars, k=12)),
        ]
        return '-'.join(parts)
    
    def _generate_floor_price(self, profile: VolumeProfile, ad_format: AdFormat) -> float:
        """
        Generate a floor price based on profile and ad format.
        
        Video formats typically have higher floor prices.
        """
        base_min = profile.floor_price_min
        base_max = profile.floor_price_max
        
        # Video premium
        if ad_format in (AdFormat.VIDEO_INSTREAM, AdFormat.VIDEO_OUTSTREAM):
            base_min *= 2.0
            base_max *= 2.5
        elif ad_format == AdFormat.INTERSTITIAL:
            base_min *= 1.5
            base_max *= 2.0
        
        return round(self._rng.uniform(base_min, base_max), 4)
    
    def generate_hour(
        self,
        volume: int,
        hour: int,
        day_number: int,
        profile: VolumeProfile,
    ) -> List[BidRequest]:
        """
        Generate bid requests for a single hour.
        
        Args:
            volume: Number of requests to generate
            hour: Hour of day (0-23)
            day_number: Day number in simulation
            profile: Volume profile configuration
            
        Returns:
            List of BidRequest objects
        """
        requests = []
        base_datetime = self.base_date + timedelta(days=day_number, hours=hour)
        
        for i in range(volume):
            # Distribute requests throughout the hour
            seconds_offset = self._rng.randint(0, 3599)
            timestamp = base_datetime + timedelta(seconds=seconds_offset)
            
            # Select attributes
            publisher_id = self._rng.choice(self._publishers)
            user_id = self._rng.choice(self._users)
            ad_format = self._select_ad_format(profile)
            device_type = self._select_weighted(self.DEVICE_WEIGHTS)
            geo_country = self._select_weighted(self.GEO_WEIGHTS)
            floor_price = self._generate_floor_price(profile, ad_format)
            
            request = BidRequest(
                request_id=self._generate_request_id(),
                timestamp=timestamp,
                publisher_id=publisher_id,
                user_id=user_id,
                floor_price=floor_price,
                ad_format=ad_format,
                device_type=device_type,
                geo_country=geo_country,
                site_domain=self._publisher_domains[publisher_id],
            )
            requests.append(request)
        
        return requests
    
    def generate_day(
        self,
        profile: str | VolumeProfile,
        day_number: int,
    ) -> List[BidRequest]:
        """
        Generate one day of bid requests.
        
        Args:
            profile: Profile name ('small', 'medium', 'large', 'enterprise')
                    or VolumeProfile instance
            day_number: Day number in simulation (0 = first day)
            
        Returns:
            List of BidRequest objects for the entire day
        """
        # Resolve profile
        if isinstance(profile, str):
            volume_profile = self.get_profile(profile)
        else:
            volume_profile = profile
        
        # Get hourly distribution (base weights, not adjusted for weekend)
        hourly_weights = self.HOURLY_WEIGHTS.copy()
        total_weight = sum(hourly_weights)
        
        # Calculate base daily volume with weekend adjustment
        daily_volume = volume_profile.daily_requests
        if self.is_weekend(day_number):
            daily_volume = int(daily_volume * self.WEEKEND_MULTIPLIER)
        
        requests = []
        for hour in range(24):
            # Calculate volume for this hour based on weight
            hour_fraction = hourly_weights[hour] / total_weight
            hour_volume = int(daily_volume * hour_fraction)
            
            # Add some variance (+/- 10%)
            variance = self._rng.uniform(0.9, 1.1)
            hour_volume = int(hour_volume * variance)
            
            hour_requests = self.generate_hour(
                volume=hour_volume,
                hour=hour,
                day_number=day_number,
                profile=volume_profile,
            )
            requests.extend(hour_requests)
        
        # Sort by timestamp
        requests.sort(key=lambda r: r.timestamp)
        
        return requests
    
    def generate_campaign(
        self,
        profile: str | VolumeProfile,
        num_days: int,
    ) -> Dict[int, List[BidRequest]]:
        """
        Generate bid requests for an entire campaign.
        
        Args:
            profile: Profile name or VolumeProfile instance
            num_days: Number of days to simulate
            
        Returns:
            Dictionary mapping day number to list of BidRequests
        """
        campaign_requests = {}
        for day in range(num_days):
            campaign_requests[day] = self.generate_day(profile, day)
        return campaign_requests
    
    def get_volume_summary(
        self,
        profile: str | VolumeProfile,
        num_days: int = 30,
    ) -> Dict:
        """
        Get a summary of expected volume for a profile.
        
        Args:
            profile: Profile name or VolumeProfile instance
            num_days: Number of days in campaign
            
        Returns:
            Dictionary with volume statistics
        """
        if isinstance(profile, str):
            volume_profile = self.get_profile(profile)
        else:
            volume_profile = profile
        
        # Count weekdays and weekends
        weekdays = sum(
            1 for d in range(num_days) if not self.is_weekend(d)
        )
        weekends = num_days - weekdays
        
        # Calculate expected volumes
        weekday_volume = volume_profile.daily_requests
        weekend_volume = int(weekday_volume * self.WEEKEND_MULTIPLIER)
        
        total_requests = (weekdays * weekday_volume) + (weekends * weekend_volume)
        
        return {
            "profile": volume_profile.name,
            "daily_requests_weekday": weekday_volume,
            "daily_requests_weekend": weekend_volume,
            "num_days": num_days,
            "weekdays": weekdays,
            "weekends": weekends,
            "total_requests": total_requests,
            "bid_rate": volume_profile.bid_rate,
            "expected_bids": int(total_requests * volume_profile.bid_rate),
            "context_pressure": self._estimate_context_pressure(total_requests),
        }
    
    def _estimate_context_pressure(self, total_requests: int) -> str:
        """Estimate context pressure level based on volume."""
        if total_requests < 500_000:
            return "low"
        elif total_requests < 5_000_000:
            return "medium"
        elif total_requests < 50_000_000:
            return "high"
        else:
            return "extreme"
