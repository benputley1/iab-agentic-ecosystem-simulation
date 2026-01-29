"""
Message schemas for A2A communication in RTB simulation.

These schemas define the messages exchanged between buyer, seller, and exchange
agents via Redis Streams.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class DealType(str, Enum):
    """Types of programmatic deals."""
    PROGRAMMATIC_GUARANTEED = "PG"
    PREFERRED_DEAL = "PD"
    PRIVATE_AUCTION = "PA"
    OPEN_AUCTION = "OA"


class MessageType(str, Enum):
    """Types of messages on the bus."""
    BID_REQUEST = "bid_request"
    BID_RESPONSE = "bid_response"
    DEAL_CONFIRMATION = "deal_confirmation"
    DEAL_REJECTION = "deal_rejection"
    HEARTBEAT = "heartbeat"


class BidRequest(BaseModel):
    """
    Buyer -> Seller/Exchange bid request.

    Represents a buyer's intent to purchase ad inventory.
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.BID_REQUEST
    buyer_id: str
    campaign_id: str
    channel: str  # "display", "video", "ctv", "native"
    impressions_requested: int
    max_cpm: float  # Maximum cost per mille willing to pay
    targeting: dict = Field(default_factory=dict)  # Audience segments, geo, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional fields for enhanced targeting
    floor_price: Optional[float] = None
    publisher_allowlist: Optional[list[str]] = None
    publisher_blocklist: Optional[list[str]] = None

    def to_stream_data(self) -> dict:
        """Convert to Redis Stream compatible dict (all string values)."""
        return {
            "request_id": self.request_id,
            "message_type": self.message_type.value,
            "buyer_id": self.buyer_id,
            "campaign_id": self.campaign_id,
            "channel": self.channel,
            "impressions_requested": str(self.impressions_requested),
            "max_cpm": str(self.max_cpm),
            "targeting": self.model_dump_json(include={"targeting"}),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_stream_data(cls, data: dict) -> "BidRequest":
        """Parse from Redis Stream data."""
        import json
        targeting = json.loads(data.get("targeting", "{}"))
        if isinstance(targeting, dict) and "targeting" in targeting:
            targeting = targeting["targeting"]
        return cls(
            request_id=data["request_id"],
            buyer_id=data["buyer_id"],
            campaign_id=data["campaign_id"],
            channel=data["channel"],
            impressions_requested=int(data["impressions_requested"]),
            max_cpm=float(data["max_cpm"]),
            targeting=targeting,
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class BidResponse(BaseModel):
    """
    Seller -> Buyer/Exchange bid response.

    Represents a seller's response to a bid request.
    """
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str  # Reference to the original request
    message_type: MessageType = MessageType.BID_RESPONSE
    seller_id: str
    offered_cpm: float
    available_impressions: int
    deal_type: DealType = DealType.OPEN_AUCTION
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Additional response data
    deal_id: Optional[str] = None  # Pre-generated deal ID if accepted
    valid_until: Optional[datetime] = None  # Offer expiration
    inventory_details: Optional[dict] = None

    def to_stream_data(self) -> dict:
        """Convert to Redis Stream compatible dict."""
        data = {
            "response_id": self.response_id,
            "request_id": self.request_id,
            "message_type": self.message_type.value,
            "seller_id": self.seller_id,
            "offered_cpm": str(self.offered_cpm),
            "available_impressions": str(self.available_impressions),
            "deal_type": self.deal_type.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.deal_id:
            data["deal_id"] = self.deal_id
        if self.valid_until:
            data["valid_until"] = self.valid_until.isoformat()
        return data

    @classmethod
    def from_stream_data(cls, data: dict) -> "BidResponse":
        """Parse from Redis Stream data."""
        return cls(
            response_id=data["response_id"],
            request_id=data["request_id"],
            seller_id=data["seller_id"],
            offered_cpm=float(data["offered_cpm"]),
            available_impressions=int(data["available_impressions"]),
            deal_type=DealType(data["deal_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            deal_id=data.get("deal_id"),
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
        )


class DealConfirmation(BaseModel):
    """
    Final deal record confirming a transaction.

    Created when buyer and seller agree on terms.
    """
    deal_id: str = Field(default_factory=lambda: f"DEAL-{uuid.uuid4().hex[:12].upper()}")
    request_id: str
    message_type: MessageType = MessageType.DEAL_CONFIRMATION
    buyer_id: str
    seller_id: str
    impressions: int
    cpm: float
    total_cost: float  # impressions/1000 * cpm
    exchange_fee: float = 0.0  # 0 for Scenarios B and C
    scenario: str  # "A", "B", or "C"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Ledger tracking (Scenario C)
    ledger_entry_id: Optional[str] = None
    blob_id: Optional[str] = None

    @property
    def seller_revenue(self) -> float:
        """Net revenue to seller after fees."""
        return self.total_cost - self.exchange_fee

    @property
    def fee_percentage(self) -> float:
        """Fee as percentage of total cost."""
        if self.total_cost == 0:
            return 0.0
        return (self.exchange_fee / self.total_cost) * 100

    def to_stream_data(self) -> dict:
        """Convert to Redis Stream compatible dict."""
        data = {
            "deal_id": self.deal_id,
            "request_id": self.request_id,
            "message_type": self.message_type.value,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "impressions": str(self.impressions),
            "cpm": str(self.cpm),
            "total_cost": str(self.total_cost),
            "exchange_fee": str(self.exchange_fee),
            "scenario": self.scenario,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.ledger_entry_id:
            data["ledger_entry_id"] = self.ledger_entry_id
        if self.blob_id:
            data["blob_id"] = self.blob_id
        return data

    @classmethod
    def from_stream_data(cls, data: dict) -> "DealConfirmation":
        """Parse from Redis Stream data."""
        return cls(
            deal_id=data["deal_id"],
            request_id=data["request_id"],
            buyer_id=data["buyer_id"],
            seller_id=data["seller_id"],
            impressions=int(data["impressions"]),
            cpm=float(data["cpm"]),
            total_cost=float(data["total_cost"]),
            exchange_fee=float(data["exchange_fee"]),
            scenario=data["scenario"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ledger_entry_id=data.get("ledger_entry_id"),
            blob_id=data.get("blob_id"),
        )

    @classmethod
    def from_deal(
        cls,
        request: BidRequest,
        response: BidResponse,
        scenario: str,
        exchange_fee_pct: float = 0.0,
    ) -> "DealConfirmation":
        """Create a deal confirmation from request and response."""
        impressions = min(request.impressions_requested, response.available_impressions)
        total_cost = (impressions / 1000) * response.offered_cpm
        exchange_fee = total_cost * exchange_fee_pct

        return cls(
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            seller_id=response.seller_id,
            impressions=impressions,
            cpm=response.offered_cpm,
            total_cost=total_cost,
            exchange_fee=exchange_fee,
            scenario=scenario,
            deal_id=response.deal_id or f"DEAL-{uuid.uuid4().hex[:12].upper()}",
        )


# Redis Stream names for message routing
STREAMS = {
    "bid_requests": "rtb:requests",
    "bid_responses": "rtb:responses",
    "deals": "rtb:deals",
    "events": "rtb:events",
}

# Consumer group names
CONSUMER_GROUPS = {
    "buyers": "buyers-group",
    "sellers": "sellers-group",
    "exchange": "exchange-group",
    "analytics": "analytics-group",
}
