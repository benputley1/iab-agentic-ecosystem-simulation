"""
Campaign Brief Loader.

Loads realistic campaign briefs from JSON and converts them to simulation-ready formats.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random


@dataclass
class ChannelConfig:
    """Configuration for a single channel within a campaign."""
    channel_type: str
    allocation_pct: float
    cpm_floor: float
    cpm_max: float
    
    @property
    def cpm_target(self) -> float:
        """Calculate target CPM (midpoint of range)."""
        return (self.cpm_floor + self.cpm_max) / 2


@dataclass
class CampaignBrief:
    """Loaded campaign brief ready for simulation."""
    id: str
    name: str
    advertiser: str
    agency: str
    objective: str
    category: str
    budget_usd: float
    flight_start: datetime
    flight_end: datetime
    channels: Dict[str, ChannelConfig]
    targeting: Dict[str, Any]
    kpis: Dict[str, Any]
    creative_specs: Dict[str, Any]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def flight_days(self) -> int:
        """Number of days in the campaign flight."""
        return (self.flight_end - self.flight_start).days
    
    @property
    def daily_budget(self) -> float:
        """Average daily budget."""
        return self.budget_usd / max(self.flight_days, 1)
    
    def get_channel_budget(self, channel: str) -> float:
        """Get budget allocated to a specific channel."""
        if channel in self.channels:
            return self.budget_usd * self.channels[channel].allocation_pct
        return 0.0
    
    def to_deal_opportunity(self, channel: str, impressions: int, seller_id: str, buyer_id: str) -> Dict[str, Any]:
        """Convert to a deal opportunity dict for the pricing agent."""
        ch_config = self.channels.get(channel)
        if not ch_config:
            # Default if channel not configured
            return {
                "channel": channel,
                "impressions": impressions,
                "cpm_floor": 5.0,
                "cpm_max": 25.0,
                "seller_id": seller_id,
                "buyer_id": buyer_id,
                "campaign_id": self.id,
                "campaign_name": self.name,
                "category": self.category,
                "objective": self.objective,
            }
        
        return {
            "channel": channel,
            "impressions": impressions,
            "cpm_floor": ch_config.cpm_floor,
            "cpm_max": ch_config.cpm_max,
            "seller_id": seller_id,
            "buyer_id": buyer_id,
            "campaign_id": self.id,
            "campaign_name": self.name,
            "category": self.category,
            "objective": self.objective,
            "advertiser": self.advertiser,
            "targeting": self.targeting,
        }


@dataclass
class Seller:
    """Seller/publisher configuration."""
    id: str
    name: str
    seller_type: str
    inventory: List[str]
    quality_score: float
    avg_viewability: Optional[float]
    floor_premium: float
    
    def applies_to_channel(self, channel: str) -> bool:
        """Check if this seller has inventory for the given channel."""
        return channel in self.inventory


@dataclass
class MarketConditions:
    """Market conditions affecting the simulation."""
    seasonality: Dict[str, Dict[str, Any]]
    competition: Dict[str, Dict[str, Any]]
    context_events: List[Dict[str, Any]]
    
    def get_competition_premium(self, category: str) -> float:
        """Get competitive premium for a category."""
        if category in self.competition:
            return self.competition[category].get("avg_cpm_premium", 1.0)
        return 1.0
    
    def get_events_for_day(self, day: int) -> List[Dict[str, Any]]:
        """Get context events scheduled for a specific day."""
        return [e for e in self.context_events if e.get("day") == day]


class CampaignBriefLoader:
    """
    Loads campaign briefs from JSON and provides them to the simulation.
    
    Usage:
        loader = CampaignBriefLoader("data/campaign_briefs.json")
        campaigns = loader.get_campaigns()
        sellers = loader.get_sellers()
        
        for campaign in campaigns:
            deal = campaign.to_deal_opportunity("video", 50000, "SELL-001", "buyer-001")
    """
    
    def __init__(self, briefs_path: str = "data/campaign_briefs.json"):
        """
        Initialize the loader.
        
        Args:
            briefs_path: Path to the campaign briefs JSON file
        """
        self.briefs_path = Path(briefs_path)
        self._campaigns: List[CampaignBrief] = []
        self._sellers: List[Seller] = []
        self._market_conditions: Optional[MarketConditions] = None
        self._simulation_params: Dict[str, Any] = {}
        self._loaded = False
    
    def load(self) -> None:
        """Load and parse the briefs file."""
        if not self.briefs_path.exists():
            raise FileNotFoundError(f"Campaign briefs file not found: {self.briefs_path}")
        
        with open(self.briefs_path) as f:
            data = json.load(f)
        
        # Parse campaigns
        for camp_data in data.get("campaigns", []):
            campaign = self._parse_campaign(camp_data)
            self._campaigns.append(campaign)
        
        # Parse sellers
        for seller_data in data.get("sellers", []):
            seller = self._parse_seller(seller_data)
            self._sellers.append(seller)
        
        # Parse market conditions
        market_data = data.get("market_conditions", {})
        sim_params = data.get("simulation_params", {})
        
        self._market_conditions = MarketConditions(
            seasonality=market_data.get("seasonality", {}),
            competition=market_data.get("competitive_landscape", {}),
            context_events=sim_params.get("context_pressure_events", []),
        )
        
        self._simulation_params = sim_params
        self._loaded = True
    
    def _parse_campaign(self, data: Dict[str, Any]) -> CampaignBrief:
        """Parse a single campaign from JSON data."""
        # Parse channels
        channels = {}
        for ch_name, ch_data in data.get("channels", {}).items():
            channels[ch_name] = ChannelConfig(
                channel_type=ch_name,
                allocation_pct=ch_data.get("allocation_pct", 0.25),
                cpm_floor=ch_data.get("cpm_floor", 5.0),
                cpm_max=ch_data.get("cpm_max", 25.0),
            )
        
        # Parse flight dates
        flight_dates = data.get("flight_dates", {})
        start_str = flight_dates.get("start", "2025-02-01")
        end_str = flight_dates.get("end", "2025-03-02")
        
        return CampaignBrief(
            id=data.get("id", f"CAMP-{random.randint(1000, 9999)}"),
            name=data.get("name", "Unnamed Campaign"),
            advertiser=data.get("advertiser", "Unknown Advertiser"),
            agency=data.get("agency", "Unknown Agency"),
            objective=data.get("objective", "awareness"),
            category=data.get("category", "general"),
            budget_usd=data.get("budget_usd", 50000),
            flight_start=datetime.strptime(start_str, "%Y-%m-%d"),
            flight_end=datetime.strptime(end_str, "%Y-%m-%d"),
            channels=channels,
            targeting=data.get("targeting", {}),
            kpis=data.get("kpis", {}),
            creative_specs=data.get("creative_specs", {}),
            raw_data=data,
        )
    
    def _parse_seller(self, data: Dict[str, Any]) -> Seller:
        """Parse a single seller from JSON data."""
        return Seller(
            id=data.get("id", f"SELL-{random.randint(100, 999)}"),
            name=data.get("name", "Unknown Seller"),
            seller_type=data.get("type", "ssp"),
            inventory=data.get("inventory", ["display"]),
            quality_score=data.get("quality_score", 0.80),
            avg_viewability=data.get("avg_viewability"),
            floor_premium=data.get("floor_premium", 1.0),
        )
    
    def get_campaigns(self) -> List[CampaignBrief]:
        """Get all loaded campaigns."""
        if not self._loaded:
            self.load()
        return self._campaigns
    
    def get_campaign(self, campaign_id: str) -> Optional[CampaignBrief]:
        """Get a specific campaign by ID."""
        if not self._loaded:
            self.load()
        for camp in self._campaigns:
            if camp.id == campaign_id:
                return camp
        return None
    
    def get_sellers(self) -> List[Seller]:
        """Get all loaded sellers."""
        if not self._loaded:
            self.load()
        return self._sellers
    
    def get_sellers_for_channel(self, channel: str) -> List[Seller]:
        """Get sellers that have inventory for a specific channel."""
        if not self._loaded:
            self.load()
        return [s for s in self._sellers if s.applies_to_channel(channel)]
    
    def get_market_conditions(self) -> MarketConditions:
        """Get market conditions."""
        if not self._loaded:
            self.load()
        return self._market_conditions
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Get simulation parameters."""
        if not self._loaded:
            self.load()
        return self._simulation_params
    
    def generate_deal_stream(
        self,
        day: int,
        requests_per_day: int = 10000,
        buyer_id: str = "buyer-001",
    ) -> List[Dict[str, Any]]:
        """
        Generate a stream of deal opportunities for a simulation day.
        
        This creates realistic bid request scenarios by:
        - Distributing requests across campaigns based on budget
        - Matching sellers to channels
        - Applying market condition premiums
        
        Args:
            day: Simulation day (1-30)
            requests_per_day: Total bid requests to generate
            buyer_id: ID of the buyer agent
        
        Returns:
            List of deal opportunity dicts ready for the pricing agent
        """
        if not self._loaded:
            self.load()
        
        deals = []
        campaigns = self.get_campaigns()
        sellers = self.get_sellers()
        market = self.get_market_conditions()
        
        if not campaigns or not sellers:
            return deals
        
        # Distribute requests proportionally by budget
        total_budget = sum(c.budget_usd for c in campaigns)
        
        for campaign in campaigns:
            # Requests for this campaign
            camp_pct = campaign.budget_usd / total_budget
            camp_requests = max(1, int(requests_per_day * camp_pct))
            
            # Distribute across channels
            for channel, ch_config in campaign.channels.items():
                channel_requests = max(1, int(camp_requests * ch_config.allocation_pct))
                
                # Find eligible sellers
                eligible_sellers = self.get_sellers_for_channel(channel)
                if not eligible_sellers:
                    continue
                
                for _ in range(channel_requests):
                    # Pick a random seller
                    seller = random.choice(eligible_sellers)
                    
                    # Generate impression count (varies by channel)
                    if channel == "ctv":
                        impressions = random.randint(1000, 10000)
                    elif channel == "video":
                        impressions = random.randint(5000, 50000)
                    else:
                        impressions = random.randint(10000, 100000)
                    
                    # Apply market conditions
                    competition_premium = market.get_competition_premium(campaign.category)
                    adjusted_floor = ch_config.cpm_floor * seller.floor_premium * competition_premium
                    adjusted_max = ch_config.cpm_max * seller.floor_premium * competition_premium
                    
                    deal = {
                        "channel": channel,
                        "impressions": impressions,
                        "cpm_floor": round(adjusted_floor, 2),
                        "cpm_max": round(adjusted_max, 2),
                        "seller_id": seller.id,
                        "seller_name": seller.name,
                        "buyer_id": buyer_id,
                        "campaign_id": campaign.id,
                        "campaign_name": campaign.name,
                        "advertiser": campaign.advertiser,
                        "category": campaign.category,
                        "objective": campaign.objective,
                        "quality_score": seller.quality_score,
                        "simulation_day": day,
                    }
                    deals.append(deal)
        
        # Shuffle to simulate real market randomness
        random.shuffle(deals)
        
        return deals


# Quick test
if __name__ == "__main__":
    loader = CampaignBriefLoader()
    loader.load()
    
    print(f"Loaded {len(loader.get_campaigns())} campaigns:")
    for camp in loader.get_campaigns():
        print(f"  - {camp.id}: {camp.name} (${camp.budget_usd:,.0f})")
        print(f"    Channels: {list(camp.channels.keys())}")
    
    print(f"\nLoaded {len(loader.get_sellers())} sellers")
    
    print("\nGenerating deal stream for day 1...")
    deals = loader.generate_deal_stream(day=1, requests_per_day=100)
    print(f"Generated {len(deals)} deal opportunities")
    
    if deals:
        print("\nSample deal:")
        print(json.dumps(deals[0], indent=2))
