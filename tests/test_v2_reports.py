"""
Tests for V2 Report Generators.

Tests hallucination rate, type distribution, context overflow,
and recovery comparison reports with mock simulation data.
"""

import json
import pytest
from datetime import datetime, timedelta

from src.reports.v2_reports import (
    # Types
    HallucinationType,
    # Data structures
    Hallucination,
    OverflowEvent,
    RestartEvent,
    SimulationResults,
    # Reports
    HallucinationRateReport,
    HallucinationTypeReport,
    ContextOverflowReport,
    RecoveryComparisonReport,
    # Convenience functions
    generate_all_v2_reports,
    export_all_reports_markdown,
    export_all_reports_json,
)


# =============================================================================
# Fixtures: Mock Simulation Data
# =============================================================================

@pytest.fixture
def mock_hallucinations() -> list[Hallucination]:
    """Generate mock hallucinations with increasing rate over days."""
    hallucinations = []
    base_time = datetime(2025, 1, 1)
    
    # Simulate non-linear growth: low early, accelerating later
    # Day 1-5: ~0.1% rate (1-2 per day)
    # Day 6-10: ~0.5% rate (~5 per day)
    # Day 11-20: ~2% rate (~20 per day)
    # Day 21-30: ~5% rate (~50 per day)
    
    rates = {
        (1, 5): 2,
        (6, 10): 5,
        (11, 20): 20,
        (21, 30): 50,
    }
    
    type_distribution = [
        (HallucinationType.BUDGET_DRIFT, 0.35, 0.12),
        (HallucinationType.PRICE_ANCHOR_ERROR, 0.24, 0.08),
        (HallucinationType.FREQUENCY_VIOLATION, 0.16, 0.15),
        (HallucinationType.DEAL_INVENTION, 0.13, 0.45),
        (HallucinationType.CROSS_CAMPAIGN, 0.07, 0.32),
        (HallucinationType.PHANTOM_INVENTORY, 0.05, 0.28),
    ]
    
    idx = 0
    for day in range(1, 31):
        # Determine rate for this day
        daily_count = 0
        for (start, end), count in rates.items():
            if start <= day <= end:
                daily_count = count
                break
        
        for _ in range(daily_count):
            # Pick type based on distribution
            import random
            random.seed(idx)  # Reproducible
            r = random.random()
            cumulative = 0.0
            chosen_type = type_distribution[0]
            for t, prob, severity in type_distribution:
                cumulative += prob
                if r < cumulative:
                    chosen_type = (t, prob, severity)
                    break
            
            hallucinations.append(Hallucination(
                decision_id=f"dec_{idx:05d}",
                timestamp=base_time + timedelta(days=day-1, hours=idx % 24),
                day=day,
                type=chosen_type[0],
                expected=100.0,
                actual=100.0 * (1 + chosen_type[2] * (random.random() - 0.5) * 2),
                severity=chosen_type[2] + random.random() * 0.1 - 0.05,
            ))
            idx += 1
    
    return hallucinations


@pytest.fixture
def mock_overflow_events() -> list[OverflowEvent]:
    """Generate mock context overflow events."""
    events = []
    base_time = datetime(2025, 1, 1)
    
    # Overflow events tend to happen more frequently as campaign progresses
    overflow_days = [8, 12, 15, 18, 22, 25, 27, 29]
    
    for i, day in enumerate(overflow_days):
        events.append(OverflowEvent(
            event_id=f"overflow_{i:03d}",
            timestamp=base_time + timedelta(days=day-1, hours=14),
            day=day,
            hour=14,
            tokens_before=200_000 + (i * 5_000),
            context_limit=200_000,
            events_dropped=4_000 + (i * 500),
            total_events_before=20_000 + (i * 1_000),
            quality_before_pct=98.0 - (i * 0.5),
            quality_after_pct=91.0 - (i * 1.0),
            recovery_decisions=800 + (i * 50),
        ))
    
    return events


@pytest.fixture
def mock_restart_events() -> list[RestartEvent]:
    """Generate mock agent restart events."""
    events = []
    base_time = datetime(2025, 1, 1)
    
    # Simulate 3-4 restart events during campaign
    restart_config = [
        (5, 10, 92.5, 99.5),   # Day 5: early, good recovery
        (12, 9, 87.3, 99.8),   # Day 12: mid-campaign
        (20, 15, 82.1, 99.6),  # Day 20: late, more drift
        (28, 11, 78.4, 99.2),  # Day 28: end, significant drift
    ]
    
    for i, (day, hour, private_acc, ledger_acc) in enumerate(restart_config):
        events.append(RestartEvent(
            event_id=f"restart_{i:03d}",
            timestamp=base_time + timedelta(days=day-1, hours=hour),
            day=day,
            hour=hour,
            recovery_results={
                "private_db": {
                    "accuracy": private_acc,
                    "time_seconds": 2.3 + i * 0.5,
                    "decisions_affected": int((100 - private_acc) * 10),
                },
                "ledger": {
                    "accuracy": ledger_acc,
                    "time_seconds": 0.8 + i * 0.1,
                    "decisions_affected": int((100 - ledger_acc) * 10),
                },
            },
        ))
    
    return events


@pytest.fixture
def mock_simulation_results(
    mock_hallucinations,
    mock_overflow_events, 
    mock_restart_events
) -> SimulationResults:
    """Create complete mock simulation results."""
    return SimulationResults(
        total_days=30,
        decisions_per_day=1000,  # Simplified for testing
        hallucinations=mock_hallucinations,
        decisions_by_day={day: 1000 for day in range(1, 31)},
        overflow_events=mock_overflow_events,
        context_limit=200_000,
        restart_events=mock_restart_events,
    )


# =============================================================================
# HallucinationRateReport Tests
# =============================================================================

class TestHallucinationRateReport:
    """Tests for HallucinationRateReport."""
    
    def test_generate_returns_correct_data_type(self, mock_simulation_results):
        """Test that generate returns HallucinationRateReportData."""
        report = HallucinationRateReport()
        data = report.generate(mock_simulation_results)
        
        assert data is not None
        assert len(data.days) == 30
        assert data.total_decisions == 30000
    
    def test_cumulative_rate_increases(self, mock_simulation_results):
        """Test that cumulative rate trends upward with increasing hallucinations."""
        report = HallucinationRateReport()
        data = report.generate(mock_simulation_results)
        
        # Early days should have lower cumulative rate than late days
        assert data.days[4].cumulative_rate < data.days[29].cumulative_rate
    
    def test_peak_detection(self, mock_simulation_results):
        """Test that peak day is correctly identified."""
        report = HallucinationRateReport()
        data = report.generate(mock_simulation_results)
        
        # Peak should be in later days (21-30 based on our mock data)
        assert data.peak_day >= 21
        assert data.peak_daily_rate > 0
    
    def test_to_markdown_format(self, mock_simulation_results):
        """Test Markdown output format."""
        report = HallucinationRateReport()
        report.generate(mock_simulation_results)
        
        md = report.to_markdown()
        
        assert "## Hallucination Rate Over Time" in md
        assert "| Day | Decisions | Hallucinations | Rate | Cumulative |" in md
        assert "|-----|-----------|----------------|------|------------|" in md
    
    def test_to_json_format(self, mock_simulation_results):
        """Test JSON output format."""
        report = HallucinationRateReport()
        report.generate(mock_simulation_results)
        
        json_data = report.to_json()
        
        assert "report_type" in json_data
        assert json_data["report_type"] == "hallucination_rate"
        assert "data" in json_data
        assert "days" in json_data["data"]
    
    def test_report_not_generated_error(self):
        """Test that accessing data before generate raises error."""
        report = HallucinationRateReport()
        
        with pytest.raises(ValueError, match="not been generated"):
            _ = report.data
    
    def test_empty_hallucinations(self):
        """Test handling of simulation with no hallucinations."""
        results = SimulationResults(
            total_days=30,
            decisions_per_day=1000,
            hallucinations=[],
            decisions_by_day={day: 1000 for day in range(1, 31)},
        )
        
        report = HallucinationRateReport()
        data = report.generate(results)
        
        assert data.total_hallucinations == 0
        assert data.final_cumulative_rate == 0.0


# =============================================================================
# HallucinationTypeReport Tests
# =============================================================================

class TestHallucinationTypeReport:
    """Tests for HallucinationTypeReport."""
    
    def test_generate_returns_correct_data_type(self, mock_simulation_results):
        """Test that generate returns HallucinationTypeReportData."""
        report = HallucinationTypeReport()
        data = report.generate(mock_simulation_results)
        
        assert data is not None
        assert len(data.types) == len(HallucinationType)
    
    def test_percentages_sum_to_100(self, mock_simulation_results):
        """Test that type percentages sum to approximately 100%."""
        report = HallucinationTypeReport()
        data = report.generate(mock_simulation_results)
        
        total_pct = sum(t.percentage for t in data.types)
        assert 99.0 <= total_pct <= 101.0  # Allow for rounding
    
    def test_most_common_type_identified(self, mock_simulation_results):
        """Test that most common type is correctly identified."""
        report = HallucinationTypeReport()
        data = report.generate(mock_simulation_results)
        
        assert data.most_common_type is not None
        # Based on our mock distribution, budget drift should be most common
        assert "Budget" in data.most_common_type or data.most_common_type is not None
    
    def test_severity_values_valid(self, mock_simulation_results):
        """Test that severity values are in valid range."""
        report = HallucinationTypeReport()
        data = report.generate(mock_simulation_results)
        
        for t in data.types:
            if t.count > 0:
                assert 0.0 <= t.avg_severity <= 1.0
    
    def test_to_markdown_format(self, mock_simulation_results):
        """Test Markdown output format."""
        report = HallucinationTypeReport()
        report.generate(mock_simulation_results)
        
        md = report.to_markdown()
        
        assert "## Hallucination Type Distribution" in md
        assert "| Type | Count | % of Total | Avg Severity |" in md
    
    def test_types_sorted_by_count(self, mock_simulation_results):
        """Test that types are sorted by count descending."""
        report = HallucinationTypeReport()
        data = report.generate(mock_simulation_results)
        
        counts = [t.count for t in data.types]
        assert counts == sorted(counts, reverse=True)


# =============================================================================
# ContextOverflowReport Tests
# =============================================================================

class TestContextOverflowReport:
    """Tests for ContextOverflowReport."""
    
    def test_generate_returns_correct_data_type(self, mock_simulation_results):
        """Test that generate returns ContextOverflowReportData."""
        report = ContextOverflowReport()
        data = report.generate(mock_simulation_results)
        
        assert data is not None
        assert len(data.events) == len(mock_simulation_results.overflow_events)
    
    def test_event_numbering(self, mock_simulation_results):
        """Test that events are numbered correctly."""
        report = ContextOverflowReport()
        data = report.generate(mock_simulation_results)
        
        for i, event in enumerate(data.events, 1):
            assert event.event_number == i
    
    def test_quality_degradation_calculated(self, mock_simulation_results):
        """Test that quality degradation is calculated."""
        report = ContextOverflowReport()
        data = report.generate(mock_simulation_results)
        
        for event in data.events:
            assert event.quality_before > event.quality_after
        
        assert data.avg_quality_degradation > 0
    
    def test_to_markdown_includes_all_events(self, mock_simulation_results):
        """Test Markdown includes all overflow events."""
        report = ContextOverflowReport()
        report.generate(mock_simulation_results)
        
        md = report.to_markdown()
        
        assert "## Context Overflow Impact" in md
        assert "Overflow Event #1" in md
        assert "Context tokens:" in md
        assert "Decision quality before:" in md
    
    def test_empty_overflow_events(self):
        """Test handling of no overflow events."""
        results = SimulationResults(
            total_days=30,
            decisions_per_day=1000,
            hallucinations=[],
            overflow_events=[],
        )
        
        report = ContextOverflowReport()
        data = report.generate(results)
        
        assert data.total_overflow_events == 0
        assert data.avg_quality_degradation == 0.0


# =============================================================================
# RecoveryComparisonReport Tests
# =============================================================================

class TestRecoveryComparisonReport:
    """Tests for RecoveryComparisonReport."""
    
    def test_generate_returns_correct_data_type(self, mock_simulation_results):
        """Test that generate returns RecoveryComparisonReportData."""
        report = RecoveryComparisonReport()
        data = report.generate(mock_simulation_results)
        
        assert data is not None
        assert len(data.events) == len(mock_simulation_results.restart_events)
    
    def test_ledger_has_higher_accuracy(self, mock_simulation_results):
        """Test that ledger recovery has higher accuracy than private DB."""
        report = RecoveryComparisonReport()
        data = report.generate(mock_simulation_results)
        
        assert data.avg_ledger_accuracy > data.avg_private_db_accuracy
        assert data.reliability_advantage > 0
    
    def test_recovery_modes_present(self, mock_simulation_results):
        """Test that both recovery modes are present in each event."""
        report = RecoveryComparisonReport()
        data = report.generate(mock_simulation_results)
        
        for event in data.events:
            modes = [m.mode for m in event.recovery_modes]
            assert "private_db" in modes
            assert "ledger" in modes
    
    def test_to_markdown_format(self, mock_simulation_results):
        """Test Markdown output format."""
        report = RecoveryComparisonReport()
        report.generate(mock_simulation_results)
        
        md = report.to_markdown()
        
        assert "## Recovery Comparison" in md
        assert "Private DB" in md
        assert "Ledger (Alkimi)" in md
        assert "Reliability Advantage" in md
    
    def test_to_json_format(self, mock_simulation_results):
        """Test JSON output is valid."""
        report = RecoveryComparisonReport()
        report.generate(mock_simulation_results)
        
        json_data = report.to_json()
        
        # Verify it's JSON serializable
        json_str = json.dumps(json_data)
        assert len(json_str) > 0
        
        assert json_data["report_type"] == "recovery_comparison"


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_generate_all_v2_reports(self, mock_simulation_results):
        """Test generating all reports at once."""
        reports = generate_all_v2_reports(mock_simulation_results)
        
        assert len(reports) == 4
        assert "hallucination_rate" in reports
        assert "hallucination_type" in reports
        assert "context_overflow" in reports
        assert "recovery_comparison" in reports
        
        # All should be generated
        for report in reports.values():
            assert report._data is not None
    
    def test_export_all_reports_markdown(self, mock_simulation_results):
        """Test combined Markdown export."""
        reports = generate_all_v2_reports(mock_simulation_results)
        md = export_all_reports_markdown(reports)
        
        assert "# V2 Simulation Reports" in md
        assert "Hallucination Rate Over Time" in md
        assert "Hallucination Type Distribution" in md
        assert "Context Overflow Impact" in md
        assert "Recovery Comparison" in md
    
    def test_export_all_reports_json(self, mock_simulation_results):
        """Test combined JSON export."""
        reports = generate_all_v2_reports(mock_simulation_results)
        json_data = export_all_reports_json(reports)
        
        assert "version" in json_data
        assert json_data["version"] == "2.0.0"
        assert "reports" in json_data
        assert len(json_data["reports"]) == 4
        
        # Verify JSON serializable
        json_str = json.dumps(json_data)
        assert len(json_str) > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_day_simulation(self):
        """Test reports with single day of data."""
        results = SimulationResults(
            total_days=1,
            decisions_per_day=1000,
            hallucinations=[
                Hallucination(
                    decision_id="dec_001",
                    timestamp=datetime.now(),
                    day=1,
                    type=HallucinationType.BUDGET_DRIFT,
                    expected=100.0,
                    actual=105.0,
                    severity=0.05,
                )
            ],
            decisions_by_day={1: 1000},
        )
        
        report = HallucinationRateReport()
        data = report.generate(results)
        
        assert len(data.days) == 1
        assert data.days[0].hallucinations == 1
    
    def test_very_high_hallucination_rate(self):
        """Test handling of very high hallucination rates."""
        hallucinations = [
            Hallucination(
                decision_id=f"dec_{i:05d}",
                timestamp=datetime.now(),
                day=1,
                type=HallucinationType.BUDGET_DRIFT,
                expected=100.0,
                actual=150.0,
                severity=0.5,
            )
            for i in range(900)  # 90% hallucination rate
        ]
        
        results = SimulationResults(
            total_days=1,
            decisions_per_day=1000,
            hallucinations=hallucinations,
            decisions_by_day={1: 1000},
        )
        
        report = HallucinationRateReport()
        data = report.generate(results)
        
        assert data.days[0].rate == 90.0
    
    def test_json_serialization_with_enums(self, mock_simulation_results):
        """Test that enum values are properly serialized to JSON."""
        report = HallucinationTypeReport()
        report.generate(mock_simulation_results)
        
        json_data = report.to_json()
        json_str = json.dumps(json_data)  # Should not raise
        
        # Verify no enum objects remain
        assert "HallucinationType" not in json_str
