"""
Comprehensive tests for the Ground Truth Database component.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime, timedelta

from src.ground_truth import Event, EventType, CampaignState, GroundTruthDB


class TestEventType:
    """Tests for EventType enum."""
    
    def test_all_event_types_exist(self):
        """Verify all required event types are defined."""
        assert EventType.SPEND.value == "spend"
        assert EventType.IMPRESSION.value == "impression"
        assert EventType.BID.value == "bid"
        assert EventType.WIN.value == "win"
        assert EventType.CLICK.value == "click"
    
    def test_event_type_is_string_enum(self):
        """EventType should be usable as string."""
        assert str(EventType.SPEND) == "EventType.SPEND"
        assert EventType.SPEND.value == "spend"


class TestEvent:
    """Tests for Event dataclass."""
    
    def test_create_event_with_factory(self):
        """Event.create() should generate ID and timestamp."""
        event = Event.create(
            event_type=EventType.SPEND,
            campaign_id="camp-001",
            amount=10.50,
        )
        
        assert event.event_id is not None
        assert len(event.event_id) == 36  # UUID format
        assert event.timestamp is not None
        assert event.event_type == EventType.SPEND
        assert event.campaign_id == "camp-001"
        assert event.amount == 10.50
        assert event.metadata == {}
    
    def test_create_event_with_custom_values(self):
        """Event.create() should accept custom ID, timestamp, and metadata."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        metadata = {"user_id": "u123", "placement": "header"}
        
        event = Event.create(
            event_type=EventType.IMPRESSION,
            campaign_id="camp-002",
            amount=0.005,
            timestamp=ts,
            metadata=metadata,
            event_id="custom-id-001",
        )
        
        assert event.event_id == "custom-id-001"
        assert event.timestamp == ts
        assert event.metadata == metadata
    
    def test_event_to_tuple(self):
        """Event.to_tuple() should produce correct DB insertion format."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        event = Event(
            event_id="evt-001",
            timestamp=ts,
            event_type=EventType.WIN,
            campaign_id="camp-003",
            amount=2.50,
            metadata={"bid_id": "bid-123"},
        )
        
        t = event.to_tuple()
        
        assert t[0] == "evt-001"
        assert t[1] == "2024-01-15T10:30:00"
        assert t[2] == "win"
        assert t[3] == "camp-003"
        assert t[4] == 2.50
        assert json.loads(t[5]) == {"bid_id": "bid-123"}
    
    def test_event_from_row(self):
        """Event.from_row() should reconstruct event from DB row."""
        row = (
            "evt-002",
            "2024-01-15T14:45:30",
            "click",
            "camp-004",
            0.15,
            '{"source": "mobile"}',
        )
        
        event = Event.from_row(row)
        
        assert event.event_id == "evt-002"
        assert event.timestamp == datetime(2024, 1, 15, 14, 45, 30)
        assert event.event_type == EventType.CLICK
        assert event.campaign_id == "camp-004"
        assert event.amount == 0.15
        assert event.metadata == {"source": "mobile"}
    
    def test_event_from_row_with_null_metadata(self):
        """Event.from_row() should handle null metadata gracefully."""
        row = ("evt-003", "2024-01-15T14:45:30", "bid", "camp-005", 1.0, None)
        
        event = Event.from_row(row)
        
        assert event.metadata == {}


class TestCampaignState:
    """Tests for CampaignState dataclass."""
    
    def test_empty_campaign_state(self):
        """CampaignState.empty() should create zeroed state."""
        state = CampaignState.empty("camp-001")
        
        assert state.campaign_id == "camp-001"
        assert state.total_spend == 0.0
        assert state.impressions == 0
        assert state.wins == 0
        assert state.clicks == 0
        assert state.bids == 0
        assert state.last_event is None
    
    def test_campaign_state_from_row(self):
        """CampaignState.from_row() should parse DB aggregation result."""
        row = (150.75, 1000, 800, 50, 1200, "2024-01-15T18:00:00")
        
        state = CampaignState.from_row("camp-002", row)
        
        assert state.campaign_id == "camp-002"
        assert state.total_spend == 150.75
        assert state.impressions == 1000
        assert state.wins == 800
        assert state.clicks == 50
        assert state.bids == 1200
        assert state.last_event == datetime(2024, 1, 15, 18, 0, 0)
    
    def test_campaign_state_from_row_with_nulls(self):
        """CampaignState.from_row() should handle null values."""
        row = (None, None, None, None, None, None)
        
        state = CampaignState.from_row("camp-003", row)
        
        assert state.total_spend == 0.0
        assert state.impressions == 0
        assert state.wins == 0
        assert state.clicks == 0
        assert state.bids == 0
        assert state.last_event is None


class TestGroundTruthDB:
    """Tests for GroundTruthDB class."""
    
    @pytest.fixture
    def db(self):
        """Create an in-memory database for testing."""
        db = GroundTruthDB(":memory:")
        yield db
        db.close()
    
    @pytest.fixture
    def db_file(self):
        """Create a file-based database for persistence tests."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    def test_init_creates_schema(self, db):
        """Database should have events table after init."""
        cursor = db.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='events'
        """)
        assert cursor.fetchone() is not None
    
    def test_record_single_event(self, db):
        """Should record a single event successfully."""
        event = Event.create(
            event_type=EventType.BID,
            campaign_id="camp-001",
            amount=1.50,
        )
        
        db.record_event(event)
        
        retrieved = db.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.event_id == event.event_id
        assert retrieved.campaign_id == "camp-001"
        assert retrieved.amount == 1.50
    
    def test_record_events_batch(self, db):
        """Should record multiple events in batch."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.WIN, "camp-001", 1.0),
            Event.create(EventType.IMPRESSION, "camp-001", 0.0),
        ]
        
        db.record_events(events)
        
        assert db.count_events(campaign_id="camp-001") == 3
    
    def test_event_immutability(self, db):
        """Should not allow duplicate event IDs."""
        event = Event.create(
            event_type=EventType.SPEND,
            campaign_id="camp-001",
            amount=5.0,
            event_id="fixed-id-001",
        )
        
        db.record_event(event)
        
        # Attempting to record same event ID should fail
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            db.record_event(event)
    
    def test_get_campaign_state_empty(self, db):
        """Should return empty state for non-existent campaign."""
        state = db.get_campaign_state("non-existent-campaign")
        
        assert state.campaign_id == "non-existent-campaign"
        assert state.total_spend == 0.0
        assert state.impressions == 0
    
    def test_get_campaign_state_aggregates_correctly(self, db):
        """Should correctly aggregate campaign metrics."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        events = [
            Event.create(EventType.BID, "camp-001", 0, timestamp=base_time),
            Event.create(EventType.BID, "camp-001", 0, timestamp=base_time + timedelta(minutes=1)),
            Event.create(EventType.WIN, "camp-001", 2.0, timestamp=base_time + timedelta(minutes=2)),
            Event.create(EventType.IMPRESSION, "camp-001", 0, timestamp=base_time + timedelta(minutes=3)),
            Event.create(EventType.SPEND, "camp-001", 2.0, timestamp=base_time + timedelta(minutes=4)),
            Event.create(EventType.CLICK, "camp-001", 0.5, timestamp=base_time + timedelta(minutes=5)),
            Event.create(EventType.SPEND, "camp-001", 0.5, timestamp=base_time + timedelta(minutes=6)),
        ]
        
        db.record_events(events)
        
        state = db.get_campaign_state("camp-001")
        
        assert state.bids == 2
        assert state.wins == 1
        assert state.impressions == 1
        assert state.clicks == 1
        assert state.total_spend == 2.5  # 2.0 + 0.5
    
    def test_get_campaign_state_at_time(self, db):
        """Should return state as of specific timestamp."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        events = [
            Event.create(EventType.SPEND, "camp-001", 10.0, timestamp=base_time),
            Event.create(EventType.SPEND, "camp-001", 20.0, timestamp=base_time + timedelta(hours=1)),
            Event.create(EventType.SPEND, "camp-001", 30.0, timestamp=base_time + timedelta(hours=2)),
        ]
        
        db.record_events(events)
        
        # State at 1.5 hours should include first two spends
        state = db.get_campaign_state(
            "camp-001", 
            at_time=base_time + timedelta(hours=1, minutes=30)
        )
        
        assert state.total_spend == 30.0  # 10 + 20
    
    def test_get_campaign_state_tracks_last_event(self, db):
        """Should track timestamp of last event."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        last_time = base_time + timedelta(hours=5)
        
        events = [
            Event.create(EventType.BID, "camp-001", 1.0, timestamp=base_time),
            Event.create(EventType.WIN, "camp-001", 1.0, timestamp=base_time + timedelta(hours=2)),
            Event.create(EventType.IMPRESSION, "camp-001", 0, timestamp=last_time),
        ]
        
        db.record_events(events)
        
        state = db.get_campaign_state("camp-001")
        
        assert state.last_event == last_time
    
    def test_get_campaign_events_basic(self, db):
        """Should retrieve events for a campaign."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.BID, "camp-001", 2.0),
            Event.create(EventType.BID, "camp-002", 3.0),  # Different campaign
        ]
        
        db.record_events(events)
        
        camp_events = db.get_campaign_events("camp-001")
        
        assert len(camp_events) == 2
        assert all(e.campaign_id == "camp-001" for e in camp_events)
    
    def test_get_campaign_events_filter_by_type(self, db):
        """Should filter events by type."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.WIN, "camp-001", 1.0),
            Event.create(EventType.IMPRESSION, "camp-001", 0),
        ]
        
        db.record_events(events)
        
        bids = db.get_campaign_events("camp-001", event_type=EventType.BID)
        
        assert len(bids) == 1
        assert bids[0].event_type == EventType.BID
    
    def test_get_campaign_events_time_range(self, db):
        """Should filter events by time range."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        events = [
            Event.create(EventType.BID, "camp-001", 1.0, timestamp=base_time),
            Event.create(EventType.BID, "camp-001", 2.0, timestamp=base_time + timedelta(hours=1)),
            Event.create(EventType.BID, "camp-001", 3.0, timestamp=base_time + timedelta(hours=2)),
            Event.create(EventType.BID, "camp-001", 4.0, timestamp=base_time + timedelta(hours=3)),
        ]
        
        db.record_events(events)
        
        filtered = db.get_campaign_events(
            "camp-001",
            start_time=base_time + timedelta(minutes=30),
            end_time=base_time + timedelta(hours=2, minutes=30),
        )
        
        assert len(filtered) == 2
        assert filtered[0].amount == 2.0
        assert filtered[1].amount == 3.0
    
    def test_get_campaign_events_with_limit(self, db):
        """Should respect limit parameter."""
        events = [
            Event.create(EventType.BID, "camp-001", float(i))
            for i in range(10)
        ]
        
        db.record_events(events)
        
        limited = db.get_campaign_events("camp-001", limit=3)
        
        assert len(limited) == 3
    
    def test_get_campaign_events_ordered_by_timestamp(self, db):
        """Should return events ordered by timestamp ascending."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        # Insert out of order
        events = [
            Event.create(EventType.BID, "camp-001", 3.0, timestamp=base_time + timedelta(hours=2)),
            Event.create(EventType.BID, "camp-001", 1.0, timestamp=base_time),
            Event.create(EventType.BID, "camp-001", 2.0, timestamp=base_time + timedelta(hours=1)),
        ]
        
        db.record_events(events)
        
        result = db.get_campaign_events("camp-001")
        
        assert [e.amount for e in result] == [1.0, 2.0, 3.0]
    
    def test_get_all_campaigns(self, db):
        """Should return all unique campaign IDs."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.BID, "camp-002", 2.0),
            Event.create(EventType.BID, "camp-001", 3.0),  # Duplicate campaign
            Event.create(EventType.BID, "camp-003", 4.0),
        ]
        
        db.record_events(events)
        
        campaigns = db.get_all_campaigns()
        
        assert campaigns == ["camp-001", "camp-002", "camp-003"]
    
    def test_count_events_all(self, db):
        """Should count all events."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.WIN, "camp-001", 2.0),
            Event.create(EventType.BID, "camp-002", 3.0),
        ]
        
        db.record_events(events)
        
        assert db.count_events() == 3
    
    def test_count_events_by_campaign(self, db):
        """Should count events filtered by campaign."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.WIN, "camp-001", 2.0),
            Event.create(EventType.BID, "camp-002", 3.0),
        ]
        
        db.record_events(events)
        
        assert db.count_events(campaign_id="camp-001") == 2
        assert db.count_events(campaign_id="camp-002") == 1
    
    def test_count_events_by_type(self, db):
        """Should count events filtered by type."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.BID, "camp-001", 2.0),
            Event.create(EventType.WIN, "camp-001", 3.0),
        ]
        
        db.record_events(events)
        
        assert db.count_events(event_type=EventType.BID) == 2
        assert db.count_events(event_type=EventType.WIN) == 1
    
    def test_count_events_combined_filters(self, db):
        """Should count events with multiple filters."""
        events = [
            Event.create(EventType.BID, "camp-001", 1.0),
            Event.create(EventType.BID, "camp-001", 2.0),
            Event.create(EventType.WIN, "camp-001", 3.0),
            Event.create(EventType.BID, "camp-002", 4.0),
        ]
        
        db.record_events(events)
        
        assert db.count_events(campaign_id="camp-001", event_type=EventType.BID) == 2
    
    def test_context_manager(self, db_file):
        """Should work as context manager."""
        with GroundTruthDB(db_file) as db:
            event = Event.create(EventType.BID, "camp-001", 1.0)
            db.record_event(event)
        
        # Reopen to verify persistence
        with GroundTruthDB(db_file) as db:
            assert db.count_events() == 1
    
    def test_persistence(self, db_file):
        """Data should persist across connections."""
        # First connection: write data
        db1 = GroundTruthDB(db_file)
        events = [
            Event.create(EventType.SPEND, "camp-001", 100.0),
            Event.create(EventType.IMPRESSION, "camp-001", 0),
        ]
        db1.record_events(events)
        db1.close()
        
        # Second connection: verify data
        db2 = GroundTruthDB(db_file)
        state = db2.get_campaign_state("camp-001")
        db2.close()
        
        assert state.total_spend == 100.0
        assert state.impressions == 1


class TestGroundTruthDBIntegration:
    """Integration tests simulating realistic usage patterns."""
    
    @pytest.fixture
    def db(self):
        """Create an in-memory database."""
        db = GroundTruthDB(":memory:")
        yield db
        db.close()
    
    def test_full_campaign_lifecycle(self, db):
        """Simulate a complete campaign from start to finish."""
        campaign_id = "camp-lifecycle-001"
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        
        # Day 1: Campaign starts
        day1_events = []
        for i in range(100):
            bid_time = base_time + timedelta(minutes=i)
            day1_events.append(Event.create(
                EventType.BID, campaign_id, 2.0, timestamp=bid_time
            ))
            if i % 5 == 0:  # 20% win rate
                day1_events.append(Event.create(
                    EventType.WIN, campaign_id, 2.0, timestamp=bid_time + timedelta(seconds=1)
                ))
                day1_events.append(Event.create(
                    EventType.IMPRESSION, campaign_id, 0, timestamp=bid_time + timedelta(seconds=2)
                ))
                day1_events.append(Event.create(
                    EventType.SPEND, campaign_id, 2.0, timestamp=bid_time + timedelta(seconds=3)
                ))
        
        db.record_events(day1_events)
        
        # Verify Day 1 state
        day1_end = base_time + timedelta(hours=2)
        state = db.get_campaign_state(campaign_id, at_time=day1_end)
        
        assert state.bids == 100
        assert state.wins == 20
        assert state.impressions == 20
        assert state.total_spend == 40.0  # 20 wins * $2
    
    def test_multi_campaign_isolation(self, db):
        """Events from different campaigns should be isolated."""
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        
        # Campaign A: 50 bids, 10 wins
        for i in range(50):
            db.record_event(Event.create(EventType.BID, "camp-A", 1.0, timestamp=base_time + timedelta(minutes=i)))
            if i % 5 == 0:
                db.record_event(Event.create(EventType.WIN, "camp-A", 1.0, timestamp=base_time + timedelta(minutes=i, seconds=1)))
                db.record_event(Event.create(EventType.SPEND, "camp-A", 1.0, timestamp=base_time + timedelta(minutes=i, seconds=2)))
        
        # Campaign B: 30 bids, 15 wins (higher win rate)
        for i in range(30):
            db.record_event(Event.create(EventType.BID, "camp-B", 3.0, timestamp=base_time + timedelta(minutes=i)))
            if i % 2 == 0:
                db.record_event(Event.create(EventType.WIN, "camp-B", 3.0, timestamp=base_time + timedelta(minutes=i, seconds=1)))
                db.record_event(Event.create(EventType.SPEND, "camp-B", 3.0, timestamp=base_time + timedelta(minutes=i, seconds=2)))
        
        state_a = db.get_campaign_state("camp-A")
        state_b = db.get_campaign_state("camp-B")
        
        assert state_a.bids == 50
        assert state_a.wins == 10
        assert state_a.total_spend == 10.0
        
        assert state_b.bids == 30
        assert state_b.wins == 15
        assert state_b.total_spend == 45.0
    
    def test_point_in_time_queries(self, db):
        """Verify accurate point-in-time state retrieval."""
        campaign_id = "camp-time-001"
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        
        # Record spend events at different times
        spend_schedule = [
            (timedelta(hours=0), 10.0),
            (timedelta(hours=6), 20.0),
            (timedelta(hours=12), 30.0),
            (timedelta(hours=18), 40.0),
            (timedelta(hours=24), 50.0),
        ]
        
        for offset, amount in spend_schedule:
            db.record_event(Event.create(
                EventType.SPEND, campaign_id, amount, timestamp=base_time + offset
            ))
        
        # Query at various points in time
        assert db.get_campaign_state(campaign_id, base_time + timedelta(hours=3)).total_spend == 10.0
        assert db.get_campaign_state(campaign_id, base_time + timedelta(hours=9)).total_spend == 30.0
        assert db.get_campaign_state(campaign_id, base_time + timedelta(hours=15)).total_spend == 60.0
        assert db.get_campaign_state(campaign_id, base_time + timedelta(hours=21)).total_spend == 100.0
        assert db.get_campaign_state(campaign_id, base_time + timedelta(hours=30)).total_spend == 150.0
    
    def test_high_volume_performance(self, db):
        """Should handle high volume of events efficiently."""
        import time
        
        campaign_id = "camp-volume-001"
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        
        # Generate 10,000 events
        events = []
        for i in range(10000):
            events.append(Event.create(
                EventType.BID, campaign_id, 1.0, 
                timestamp=base_time + timedelta(seconds=i)
            ))
            if i % 10 == 0:
                events.append(Event.create(
                    EventType.WIN, campaign_id, 1.0,
                    timestamp=base_time + timedelta(seconds=i, milliseconds=1)
                ))
        
        # Measure insert time
        start = time.time()
        db.record_events(events)
        insert_time = time.time() - start
        
        # Measure query time
        start = time.time()
        state = db.get_campaign_state(campaign_id)
        query_time = time.time() - start
        
        # Verify correctness
        assert state.bids == 10000
        assert state.wins == 1000
        
        # Performance assertions (generous limits)
        assert insert_time < 5.0, f"Insert took {insert_time:.2f}s"
        assert query_time < 1.0, f"Query took {query_time:.2f}s"
