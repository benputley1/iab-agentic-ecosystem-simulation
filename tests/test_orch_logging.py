"""
Tests for the orchestration logging system.

Verifies event logging, narrative generation, and comparison analysis.
"""

import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from src.logging import (
    EventType,
    Scenario,
    SimulationEvent,
    EventIndex,
    NarrativeEngine,
    ComparisonAnalyzer,
    OrchestrationLogWriter,
    WriterConfig,
    create_bid_request_event,
    create_deal_event,
    create_context_rot_event,
    create_fee_extraction_event,
)
from src.orchestration import OrchestrationLogger, OrchLoggerConfig
from src.infrastructure.message_schemas import BidRequest, DealConfirmation


class TestEventModels:
    """Tests for event data models."""

    def test_simulation_event_creation(self):
        """Test creating a basic simulation event."""
        event = SimulationEvent(
            event_type=EventType.BID_REQUEST,
            scenario=Scenario.A,
            simulation_day=1,
            agent_id="buyer-001",
            agent_type="buyer",
        )

        assert event.event_id is not None
        assert event.event_type == EventType.BID_REQUEST
        assert event.scenario == Scenario.A
        assert event.simulation_day == 1
        assert event.agent_id == "buyer-001"

    def test_event_to_json(self):
        """Test JSON serialization of event."""
        event = SimulationEvent(
            event_type=EventType.DEAL_CREATED,
            scenario=Scenario.C,
            simulation_day=5,
            payload={"impressions": 100000, "cpm": 15.0},
        )

        json_str = event.to_json()
        assert "DEAL_CREATED" in json_str or "deal_created" in json_str
        assert "100000" in json_str

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "event_id": "test-123",
            "event_type": "bid_request",
            "scenario": "A",
            "timestamp": datetime.utcnow().isoformat(),
            "simulation_day": 1,
            "agent_id": "buyer-001",
            "payload": {"max_cpm": 20.0},
        }

        event = SimulationEvent.from_dict(data)
        assert event.event_id == "test-123"
        assert event.event_type == EventType.BID_REQUEST
        assert event.scenario == Scenario.A


class TestEventIndex:
    """Tests for event indexing."""

    def test_index_add_and_retrieve(self):
        """Test adding events and retrieving by index."""
        index = EventIndex()

        event1 = create_bid_request_event(
            scenario=Scenario.A,
            buyer_id="buyer-001",
            campaign_id="camp-001",
            request_id="req-001",
            impressions=100000,
            max_cpm=15.0,
            simulation_day=1,
        )

        event2 = create_deal_event(
            scenario=Scenario.A,
            deal_id="deal-001",
            request_id="req-001",
            buyer_id="buyer-001",
            seller_id="seller-001",
            campaign_id="camp-001",
            impressions=100000,
            cpm=12.0,
            total_cost=1200.0,
            exchange_fee=180.0,
            simulation_day=1,
        )

        index.add(event1)
        index.add(event2)

        assert len(index.events) == 2
        assert len(index.by_campaign["camp-001"]) == 2
        assert len(index.by_scenario[Scenario.A]) == 2

    def test_campaign_timeline(self):
        """Test getting campaign timeline."""
        index = EventIndex()

        for day in [1, 3, 5]:
            event = create_bid_request_event(
                scenario=Scenario.A,
                buyer_id="buyer-001",
                campaign_id="camp-test",
                request_id=f"req-{day}",
                impressions=100000,
                max_cpm=15.0,
                simulation_day=day,
            )
            index.add(event)

        timeline = index.get_campaign_timeline("camp-test")
        assert len(timeline) == 3


class TestEventFactories:
    """Tests for event factory functions."""

    def test_create_bid_request_event(self):
        """Test bid request event creation."""
        event = create_bid_request_event(
            scenario=Scenario.B,
            buyer_id="buyer-002",
            campaign_id="camp-002",
            request_id="req-002",
            impressions=500000,
            max_cpm=20.0,
            simulation_day=5,
        )

        assert event.event_type == EventType.BID_REQUEST
        assert event.scenario == Scenario.B
        assert event.agent_id == "buyer-002"
        assert event.payload["impressions_requested"] == 500000
        assert "500,000" in event.narrative_summary

    def test_create_deal_event(self):
        """Test deal event creation."""
        event = create_deal_event(
            scenario=Scenario.A,
            deal_id="deal-test",
            request_id="req-test",
            buyer_id="buyer-001",
            seller_id="seller-001",
            campaign_id="camp-test",
            impressions=1000000,
            cpm=15.0,
            total_cost=15000.0,
            exchange_fee=2250.0,
            simulation_day=10,
        )

        assert event.event_type == EventType.DEAL_CREATED
        assert event.deal_id == "deal-test"
        assert event.payload["fee_percentage"] == 15.0
        assert "exchange takes" in event.narrative_summary

    def test_create_context_rot_event(self):
        """Test context rot event creation."""
        event = create_context_rot_event(
            agent_id="buyer-003",
            keys_lost=50,
            survival_rate=0.8,
            simulation_day=15,
        )

        assert event.event_type == EventType.CONTEXT_DECAY
        assert event.scenario == Scenario.B
        assert event.payload["keys_lost"] == 50
        assert "80.0%" in event.narrative_summary

    def test_create_context_restart_event(self):
        """Test context restart event creation."""
        event = create_context_rot_event(
            agent_id="seller-001",
            keys_lost=100,
            survival_rate=0.0,
            simulation_day=20,
            is_restart=True,
        )

        assert event.event_type == EventType.CONTEXT_RESTART
        assert "RESTARTED" in event.narrative_summary
        assert event.narrative_detail is not None


class TestNarrativeEngine:
    """Tests for narrative generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.index = EventIndex()

        # Add events for a campaign
        for day in range(1, 6):
            # Bid request
            self.index.add(create_bid_request_event(
                scenario=Scenario.A,
                buyer_id="buyer-001",
                campaign_id="camp-narrative-test",
                request_id=f"req-{day}",
                impressions=200000,
                max_cpm=15.0,
                simulation_day=day,
            ))

            # Deal
            self.index.add(create_deal_event(
                scenario=Scenario.A,
                deal_id=f"deal-{day}",
                request_id=f"req-{day}",
                buyer_id="buyer-001",
                seller_id="seller-001",
                campaign_id="camp-narrative-test",
                impressions=180000,
                cpm=12.0,
                total_cost=2160.0,
                exchange_fee=324.0,
                simulation_day=day,
            ))

    def test_generate_campaign_narrative(self):
        """Test campaign narrative generation."""
        engine = NarrativeEngine(self.index)
        narrative = engine.generate_campaign_narrative("camp-narrative-test")

        assert narrative.campaign_id == "camp-narrative-test"
        assert narrative.deals_made == 5
        assert narrative.total_spend == 2160.0 * 5
        assert narrative.opening_summary != ""
        assert narrative.outcome_summary != ""

    def test_generate_scenario_narrative(self):
        """Test scenario narrative generation."""
        engine = NarrativeEngine(self.index)
        narrative = engine.generate_scenario_narrative(Scenario.A)

        assert narrative.scenario == Scenario.A
        assert narrative.total_deals == 5
        assert narrative.executive_summary != ""
        assert len(narrative.key_findings) > 0


class TestComparisonAnalyzer:
    """Tests for comparison analysis."""

    def setup_method(self):
        """Set up test fixtures with events for all scenarios."""
        self.index = EventIndex()

        # Scenario A events
        for i in range(3):
            self.index.add(create_deal_event(
                scenario=Scenario.A,
                deal_id=f"deal-a-{i}",
                request_id=f"req-a-{i}",
                buyer_id="buyer-001",
                seller_id="seller-001",
                campaign_id=f"camp-a-{i}",
                impressions=100000,
                cpm=15.0,
                total_cost=1500.0,
                exchange_fee=225.0,
                simulation_day=1,
            ))

        # Scenario B events (with context rot)
        for i in range(3):
            self.index.add(create_deal_event(
                scenario=Scenario.B,
                deal_id=f"deal-b-{i}",
                request_id=f"req-b-{i}",
                buyer_id="buyer-001",
                seller_id="seller-001",
                campaign_id=f"camp-b-{i}",
                impressions=90000,
                cpm=14.0,
                total_cost=1260.0,
                exchange_fee=0.0,
                simulation_day=1,
            ))

        # Add context rot for Scenario B
        self.index.add(create_context_rot_event(
            agent_id="buyer-001",
            keys_lost=25,
            survival_rate=0.75,
            simulation_day=1,
        ))

        # Scenario C events
        for i in range(3):
            self.index.add(create_deal_event(
                scenario=Scenario.C,
                deal_id=f"deal-c-{i}",
                request_id=f"req-c-{i}",
                buyer_id="buyer-001",
                seller_id="seller-001",
                campaign_id=f"camp-c-{i}",
                impressions=100000,
                cpm=15.0,
                total_cost=1500.0,
                exchange_fee=0.0,
                simulation_day=1,
            ))

    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        analyzer = ComparisonAnalyzer(self.index)
        report = analyzer.generate_comparison_report()

        assert report.scenario_a is not None
        assert report.scenario_b is not None
        assert report.scenario_c is not None

        # Check comparison metrics
        assert report.comparison.scenario_a_fees > 0
        assert report.comparison.scenario_b_fees == 0
        assert report.comparison.context_rot_events > 0

        # Check insights generated
        assert len(report.insights) > 0
        assert report.executive_summary != ""

    def test_fee_savings_calculation(self):
        """Test fee savings calculations."""
        analyzer = ComparisonAnalyzer(self.index)
        report = analyzer.generate_comparison_report()

        # Scenario A collected fees, C did not
        assert report.comparison.fee_savings_c_vs_a > 0


class TestOrchestrationLogger:
    """Tests for the main orchestration logger."""

    @pytest.mark.asyncio
    async def test_logger_lifecycle(self):
        """Test logger start and stop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchLoggerConfig(log_dir=Path(tmpdir))
            logger = OrchestrationLogger(config=config)

            await logger.start()
            assert logger._started

            outputs = await logger.stop()
            assert not logger._started

    @pytest.mark.asyncio
    async def test_log_events(self):
        """Test logging events through orchestration logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchLoggerConfig(log_dir=Path(tmpdir))

            async with OrchestrationLogger(config=config) as logger:
                # Create a bid request
                request = BidRequest(
                    buyer_id="buyer-001",
                    campaign_id="camp-001",
                    channel="display",
                    impressions_requested=100000,
                    max_cpm=15.0,
                )

                logger.log_bid_request(
                    scenario=Scenario.A,
                    request=request,
                    simulation_day=1,
                )

                assert logger.get_event_count() == 1

    @pytest.mark.asyncio
    async def test_log_and_generate_report(self):
        """Test full workflow: log events and generate report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchLoggerConfig(log_dir=Path(tmpdir))

            async with OrchestrationLogger(config=config) as logger:
                # Log deals for each scenario
                for scenario in [Scenario.A, Scenario.B, Scenario.C]:
                    deal = DealConfirmation(
                        deal_id=f"deal-{scenario.value}",
                        request_id=f"req-{scenario.value}",
                        buyer_id="buyer-001",
                        seller_id="seller-001",
                        impressions=100000,
                        cpm=15.0,
                        total_cost=1500.0,
                        exchange_fee=225.0 if scenario == Scenario.A else 0.0,
                        scenario=scenario.value,
                    )

                    logger.log_deal(
                        scenario=scenario,
                        deal=deal,
                        simulation_day=1,
                    )

                # Generate report
                report = logger.generate_report()

                assert report.comparison.scenario_a_deals == 1
                assert report.comparison.scenario_b_deals == 1
                assert report.comparison.scenario_c_deals == 1


class TestFileWriters:
    """Tests for file output writers."""

    def test_event_writer(self):
        """Test event writer output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WriterConfig(base_dir=Path(tmpdir))

            with OrchestrationLogWriter(config=config) as writer:
                event = create_deal_event(
                    scenario=Scenario.A,
                    deal_id="deal-write-test",
                    request_id="req-write-test",
                    buyer_id="buyer-001",
                    seller_id="seller-001",
                    campaign_id="camp-write-test",
                    impressions=100000,
                    cpm=15.0,
                    total_cost=1500.0,
                    exchange_fee=225.0,
                    simulation_day=1,
                )

                writer.log_event(event)
                assert writer.get_event_count() == 1

            # Check file was written
            event_files = list(config.events_dir.glob("*.jsonl"))
            assert len(event_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
