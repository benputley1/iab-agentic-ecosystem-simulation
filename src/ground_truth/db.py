"""
Ground Truth Database - The authoritative source of truth for all events.

This module provides an immutable record of actual events that occurred
during simulation. It serves as the reference point for detecting
hallucinations and measuring agent accuracy.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Types of events that can be recorded in ground truth."""
    SPEND = "spend"
    IMPRESSION = "impression"
    BID = "bid"
    WIN = "win"
    CLICK = "click"


@dataclass
class Event:
    """An immutable event record in the ground truth database."""
    event_id: str
    timestamp: datetime
    event_type: EventType
    campaign_id: str
    amount: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        event_type: EventType,
        campaign_id: str,
        amount: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> "Event":
        """Factory method to create a new event with auto-generated ID."""
        return cls(
            event_id=event_id or str(uuid.uuid4()),
            timestamp=timestamp or datetime.utcnow(),
            event_type=event_type,
            campaign_id=campaign_id,
            amount=amount,
            metadata=metadata or {},
        )
    
    def to_tuple(self) -> tuple:
        """Convert event to tuple for database insertion."""
        return (
            self.event_id,
            self.timestamp.isoformat(),
            self.event_type.value,
            self.campaign_id,
            self.amount,
            json.dumps(self.metadata),
        )
    
    @classmethod
    def from_row(cls, row: tuple) -> "Event":
        """Create event from database row."""
        return cls(
            event_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            event_type=EventType(row[2]),
            campaign_id=row[3],
            amount=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )


@dataclass
class CampaignState:
    """The state of a campaign at a specific point in time."""
    campaign_id: str
    total_spend: float
    impressions: int
    wins: int
    clicks: int
    bids: int
    last_event: Optional[datetime]
    
    @classmethod
    def empty(cls, campaign_id: str) -> "CampaignState":
        """Create an empty campaign state."""
        return cls(
            campaign_id=campaign_id,
            total_spend=0.0,
            impressions=0,
            wins=0,
            clicks=0,
            bids=0,
            last_event=None,
        )
    
    @classmethod
    def from_row(cls, campaign_id: str, row: tuple) -> "CampaignState":
        """Create campaign state from database aggregation row."""
        return cls(
            campaign_id=campaign_id,
            total_spend=row[0] or 0.0,
            impressions=row[1] or 0,
            wins=row[2] or 0,
            clicks=row[3] or 0,
            bids=row[4] or 0,
            last_event=datetime.fromisoformat(row[5]) if row[5] else None,
        )


class GroundTruthDB:
    """
    Immutable record of actual events - the source of truth.
    
    This database stores every event that occurs during simulation
    and provides methods to query the true state at any point in time.
    Events cannot be modified once recorded.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize the ground truth database.
        
        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory DB.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize the database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                amount REAL NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create indexes for efficient querying
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_campaign_time 
            ON events(campaign_id, timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type 
            ON events(event_type)
        """)
        
        self.conn.commit()
    
    def record_event(self, event: Event) -> None:
        """
        Record an event to ground truth (immutable).
        
        Once recorded, events cannot be modified or deleted.
        
        Args:
            event: The event to record.
            
        Raises:
            sqlite3.IntegrityError: If event_id already exists.
        """
        self.conn.execute("""
            INSERT INTO events (
                event_id, timestamp, event_type, 
                campaign_id, amount, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, event.to_tuple())
        self.conn.commit()
    
    def record_events(self, events: List[Event]) -> None:
        """
        Record multiple events in a single transaction.
        
        Args:
            events: List of events to record.
        """
        self.conn.executemany("""
            INSERT INTO events (
                event_id, timestamp, event_type, 
                campaign_id, amount, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, [e.to_tuple() for e in events])
        self.conn.commit()
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """
        Retrieve a specific event by ID.
        
        Args:
            event_id: The unique event identifier.
            
        Returns:
            The event if found, None otherwise.
        """
        cursor = self.conn.execute("""
            SELECT event_id, timestamp, event_type, campaign_id, amount, metadata
            FROM events
            WHERE event_id = ?
        """, (event_id,))
        row = cursor.fetchone()
        return Event.from_row(row) if row else None
    
    def get_campaign_state(
        self, 
        campaign_id: str, 
        at_time: Optional[datetime] = None
    ) -> CampaignState:
        """
        Get the true campaign state at a specific point in time.
        
        Args:
            campaign_id: The campaign to query.
            at_time: Point in time to query state. If None, returns current state.
            
        Returns:
            CampaignState with aggregated metrics up to the specified time.
        """
        if at_time is None:
            at_time = datetime.utcnow()
        
        cursor = self.conn.execute("""
            SELECT 
                SUM(CASE WHEN event_type = 'spend' THEN amount ELSE 0 END) as total_spend,
                COUNT(CASE WHEN event_type = 'impression' THEN 1 END) as impressions,
                COUNT(CASE WHEN event_type = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN event_type = 'click' THEN 1 END) as clicks,
                COUNT(CASE WHEN event_type = 'bid' THEN 1 END) as bids,
                MAX(timestamp) as last_event
            FROM events
            WHERE campaign_id = ? AND timestamp <= ?
        """, (campaign_id, at_time.isoformat()))
        
        row = cursor.fetchone()
        return CampaignState.from_row(campaign_id, row)
    
    def get_campaign_events(
        self,
        campaign_id: str,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """
        Get events for a campaign with optional filtering.
        
        Args:
            campaign_id: The campaign to query.
            event_type: Filter by event type.
            start_time: Start of time range.
            end_time: End of time range.
            limit: Maximum number of events to return.
            
        Returns:
            List of matching events, ordered by timestamp.
        """
        query = """
            SELECT event_id, timestamp, event_type, campaign_id, amount, metadata
            FROM events
            WHERE campaign_id = ?
        """
        params: List[Any] = [campaign_id]
        
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.conn.execute(query, params)
        return [Event.from_row(row) for row in cursor.fetchall()]
    
    def get_all_campaigns(self) -> List[str]:
        """
        Get list of all campaign IDs in the database.
        
        Returns:
            List of unique campaign IDs.
        """
        cursor = self.conn.execute("""
            SELECT DISTINCT campaign_id FROM events ORDER BY campaign_id
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def count_events(
        self,
        campaign_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
    ) -> int:
        """
        Count events with optional filtering.
        
        Args:
            campaign_id: Filter by campaign.
            event_type: Filter by event type.
            
        Returns:
            Count of matching events.
        """
        query = "SELECT COUNT(*) FROM events WHERE 1=1"
        params: List[Any] = []
        
        if campaign_id is not None:
            query += " AND campaign_id = ?"
            params.append(campaign_id)
        
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self) -> "GroundTruthDB":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
