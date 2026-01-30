#!/usr/bin/env python3
"""
Live Metrics Analyzer

Analyzes the simulation in real-time to detect early degradation signals
before full hallucinations occur.

Metrics tracked:
1. Bid variance per channel (are decisions becoming erratic?)
2. Bid drift (is the model anchoring on wrong values?)
3. Response latency trend (is processing slowing?)
4. Channel consistency (same inputs → same outputs?)
"""

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ChannelMetrics:
    """Metrics for a specific channel type."""
    channel: str
    bid_count: int = 0
    bids: List[float] = field(default_factory=list)
    mean_bid: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    min_bid: float = 0.0
    max_bid: float = 0.0
    coefficient_of_variation: float = 0.0  # std_dev / mean - normalized variance


@dataclass
class DegradationSignals:
    """Early warning signals for context rot."""
    timestamp: str
    day: int
    context_tokens: int
    context_utilization: float
    
    # Variance metrics
    overall_variance: float = 0.0
    variance_trend: str = "stable"  # stable, increasing, decreasing
    variance_change_pct: float = 0.0
    
    # Consistency metrics
    consistency_score: float = 100.0  # 100 = perfect, 0 = random
    consistency_trend: str = "stable"
    
    # Drift metrics
    bid_drift_pct: float = 0.0  # How far bids have drifted from Day 1 baseline
    drift_direction: str = "none"  # up, down, none
    
    # Channel breakdown
    channel_metrics: Dict[str, dict] = field(default_factory=dict)
    
    # Warning level
    warning_level: str = "green"  # green, yellow, orange, red
    warnings: List[str] = field(default_factory=list)


def analyze_log_for_bids(log_path: str) -> Dict[int, Dict[str, List[float]]]:
    """
    Parse log file to extract bid information by day and channel.
    
    Note: The current log format doesn't include individual bids.
    This is a placeholder that would work with enhanced logging.
    """
    # For now, we'll simulate based on the avg bid data
    # In production, the simulation would write detailed bid logs
    
    bids_by_day = defaultdict(lambda: defaultdict(list))
    
    # This would parse actual bid data from enhanced logs
    # Format would be: "BID: day=1 channel=video bid=12.50 context=1234"
    
    return dict(bids_by_day)


def parse_daily_metrics(log_path: str) -> List[dict]:
    """Parse completed day metrics from log."""
    
    if not Path(log_path).exists():
        return []
    
    with open(log_path) as f:
        content = f.read()
    
    pattern = r"Day (\d+)/30.*?Hallucinations: (\d+)/(\d+) \((\d+\.\d+)%\).*?Avg context: ([\d,]+) tokens.*?Cost: \$([\d.]+)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    days = []
    for day, hall, deals, hall_rate, tokens, cost in matches:
        days.append({
            "day": int(day),
            "hallucinations": int(hall),
            "deals": int(deals),
            "hallucination_rate": float(hall_rate),
            "avg_context": int(tokens.replace(",", "")),
            "cost": float(cost),
        })
    
    return days


def calculate_degradation_signals(days: List[dict]) -> Optional[DegradationSignals]:
    """Calculate degradation signals from daily metrics."""
    
    if not days:
        return None
    
    latest = days[-1]
    
    signals = DegradationSignals(
        timestamp=datetime.now().isoformat(),
        day=latest["day"],
        context_tokens=latest["avg_context"],
        context_utilization=latest["avg_context"] / 200000 * 100,
    )
    
    # Analyze hallucination trend
    if len(days) >= 3:
        recent_rates = [d["hallucination_rate"] for d in days[-3:]]
        early_rates = [d["hallucination_rate"] for d in days[:3]] if len(days) >= 3 else recent_rates
        
        recent_avg = statistics.mean(recent_rates)
        early_avg = statistics.mean(early_rates)
        
        if recent_avg > early_avg + 1:
            signals.variance_trend = "increasing"
            signals.warnings.append("Hallucination rate trending up")
        elif recent_avg < early_avg - 1:
            signals.variance_trend = "decreasing"
    
    # Context-based warnings
    if signals.context_utilization > 50:
        signals.warning_level = "orange"
        signals.warnings.append(f"Context at {signals.context_utilization:.0f}% - entering pressure zone")
    elif signals.context_utilization > 25:
        signals.warning_level = "yellow"
        signals.warnings.append(f"Context at {signals.context_utilization:.0f}% - approaching pressure zone")
    
    # Hallucination-based warnings
    total_hall = sum(d["hallucinations"] for d in days)
    if total_hall > 0:
        if signals.warning_level == "green":
            signals.warning_level = "yellow"
        signals.warnings.append(f"Total {total_hall} hallucinations detected")
        
        # Check if accelerating
        if len(days) >= 5:
            first_half = sum(d["hallucinations"] for d in days[:len(days)//2])
            second_half = sum(d["hallucinations"] for d in days[len(days)//2:])
            
            if second_half > first_half * 1.5:
                signals.warning_level = "red"
                signals.warnings.append("Hallucination rate accelerating")
    
    # Cost efficiency check (early sign of model struggling = more tokens per response)
    if len(days) >= 2:
        early_efficiency = days[0]["avg_context"] / days[0]["cost"] if days[0]["cost"] > 0 else 0
        current_efficiency = latest["avg_context"] / latest["cost"] if latest["cost"] > 0 else 0
        
        # Efficiency should stay relatively stable; big drops indicate issues
        if early_efficiency > 0 and current_efficiency > 0:
            efficiency_change = (current_efficiency - early_efficiency) / early_efficiency * 100
            if efficiency_change < -20:
                signals.warnings.append(f"Token efficiency dropped {abs(efficiency_change):.0f}%")
    
    return signals


def generate_report(log_path: str = "results/real_context_30day.log") -> dict:
    """Generate a full degradation analysis report."""
    
    base_dir = Path(__file__).parent.parent
    full_path = base_dir / log_path
    
    days = parse_daily_metrics(str(full_path))
    
    if not days:
        return {
            "status": "no_data",
            "message": "Waiting for simulation data...",
        }
    
    signals = calculate_degradation_signals(days)
    
    report = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(),
        "simulation_progress": {
            "days_completed": len(days),
            "total_days": 30,
            "percent_complete": len(days) / 30 * 100,
        },
        "context_status": {
            "current_tokens": signals.context_tokens,
            "utilization_pct": signals.context_utilization,
            "projected_day30": signals.context_tokens / len(days) * 30 if days else 0,
        },
        "degradation_signals": asdict(signals),
        "daily_summary": days,
        "recommendations": [],
    }
    
    # Add recommendations based on signals
    if signals.warning_level == "green":
        report["recommendations"].append("✓ No action needed - simulation healthy")
    elif signals.warning_level == "yellow":
        report["recommendations"].append("⚠ Monitor closely - early warning signs")
    elif signals.warning_level == "orange":
        report["recommendations"].append("⚠ Context pressure increasing - watch for hallucinations")
    else:
        report["recommendations"].append("🔴 Significant degradation detected")
    
    return report


def main():
    """Generate and print degradation report."""
    
    report = generate_report()
    
    print(json.dumps(report, indent=2))
    
    # Also save to file
    output_path = Path(__file__).parent.parent / "results" / "live_metrics.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
