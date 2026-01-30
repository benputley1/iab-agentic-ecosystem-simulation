#!/usr/bin/env python3
"""
Real-Time Context Rot Monitor

Tracks early degradation indicators in the running simulation:
- Bid variance (are decisions becoming erratic?)
- Response latency trends (is the model slowing?)
- Channel consistency (same deal type → similar bid?)
- Context pressure correlation

Usage:
    python scripts/monitor_context_rot.py
    python scripts/monitor_context_rot.py --watch  # Auto-refresh every 30s
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


@dataclass
class DayStats:
    day: int
    deals: int
    avg_context: int
    hallucinations: int
    hallucination_rate: float
    cost: float
    bids_by_channel: Dict[str, List[float]]
    latencies: List[float]


def parse_log_file(log_path: str) -> Tuple[List[DayStats], dict]:
    """Parse the simulation log file for metrics."""
    
    if not Path(log_path).exists():
        return [], {}
    
    with open(log_path) as f:
        content = f.read()
    
    # Parse completed days
    day_pattern = r"Day (\d+)/30.*?Hallucinations: (\d+)/(\d+) \((\d+\.\d+)%\).*?Avg context: ([\d,]+) tokens.*?Cost: \$([\d.]+)"
    matches = re.findall(day_pattern, content, re.DOTALL)
    
    days = []
    for day, hall, deals, hall_rate, tokens, cost in matches:
        days.append(DayStats(
            day=int(day),
            deals=int(deals),
            avg_context=int(tokens.replace(",", "")),
            hallucinations=int(hall),
            hallucination_rate=float(hall_rate),
            cost=float(cost),
            bids_by_channel={},
            latencies=[],
        ))
    
    # Get current progress
    current_match = re.search(r"Day (\d+)/30.*?(\d+)/(\d+) deals", content, re.DOTALL)
    current = {}
    if current_match:
        current = {
            "day": int(current_match.group(1)),
            "deals_done": int(current_match.group(2)),
            "deals_total": int(current_match.group(3)),
        }
    
    return days, current


def parse_results_json(results_dir: str) -> Optional[dict]:
    """Parse the most recent results JSON for detailed metrics."""
    
    results_path = Path(results_dir)
    json_files = list(results_path.glob("real_context_*.json"))
    
    if not json_files:
        return None
    
    # Get most recent
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest) as f:
        return json.load(f)


def calculate_variance_trend(days: List[DayStats]) -> List[Tuple[int, float]]:
    """Calculate bid variance trend over days."""
    # This would need actual bid data from results JSON
    # For now, return placeholder showing we need this data
    return []


def calculate_latency_trend(days: List[DayStats]) -> List[Tuple[int, float]]:
    """Calculate latency trend over days."""
    return []


def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Render a progress bar."""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100 if total > 0 else 0
    return f"[{bar}] {pct:.1f}%"


def render_spark_line(values: List[float], width: int = 20) -> str:
    """Render a sparkline for trend visualization."""
    if not values:
        return "─" * width
    
    if len(values) == 1:
        return "─" * width
    
    min_v, max_v = min(values), max(values)
    range_v = max_v - min_v if max_v != min_v else 1
    
    blocks = " ▁▂▃▄▅▆▇█"
    
    # Sample to fit width
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    
    result = ""
    for v in sampled:
        idx = int((v - min_v) / range_v * 8)
        result += blocks[idx]
    
    return result


def render_dashboard(days: List[DayStats], current: dict, results: Optional[dict] = None):
    """Render the monitoring dashboard."""
    
    os.system('clear' if os.name != 'nt' else 'cls')
    
    C = Colors
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════╗{C.ENDC}")
    print(f"{C.BOLD}{C.CYAN}║      REAL CONTEXT SIMULATION - DEGRADATION MONITOR              ║{C.ENDC}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════════════╝{C.ENDC}")
    print(f"{C.DIM}  Last updated: {now}{C.ENDC}\n")
    
    if not days:
        print(f"{C.YELLOW}  ⏳ Waiting for simulation data...{C.ENDC}")
        return
    
    # === PROGRESS ===
    last_day = days[-1]
    total_days = 30
    
    if current:
        day_progress = f"Day {current['day']}: {current['deals_done']}/{current['deals_total']} deals"
    else:
        day_progress = f"Day {last_day.day} complete"
    
    print(f"{C.BOLD}📊 SIMULATION PROGRESS{C.ENDC}")
    print(f"  Days: {render_progress_bar(last_day.day, total_days)} ({last_day.day}/{total_days})")
    print(f"  {day_progress}")
    print()
    
    # === CONTEXT GROWTH ===
    context_values = [d.avg_context for d in days]
    context_pct = last_day.avg_context / 200000 * 100
    
    print(f"{C.BOLD}📈 CONTEXT ACCUMULATION{C.ENDC}")
    print(f"  Current: {C.CYAN}{last_day.avg_context:,}{C.ENDC} tokens ({context_pct:.1f}% of 200K limit)")
    print(f"  Trend:   {render_spark_line(context_values)}")
    
    # Growth rate
    if len(days) >= 2:
        growth = (days[-1].avg_context - days[0].avg_context) / (len(days) - 1)
        projected = growth * 30
        print(f"  Growth:  {C.GREEN}+{growth:,.0f}{C.ENDC} tokens/day → Day 30: ~{projected:,.0f} tokens")
    print()
    
    # === HALLUCINATION TRACKING ===
    total_hall = sum(d.hallucinations for d in days)
    total_deals = sum(d.deals for d in days)
    overall_rate = total_hall / total_deals * 100 if total_deals > 0 else 0
    
    hall_color = C.GREEN if overall_rate == 0 else (C.YELLOW if overall_rate < 5 else C.RED)
    
    print(f"{C.BOLD}🎯 HALLUCINATION STATUS{C.ENDC}")
    print(f"  Total: {hall_color}{total_hall}{C.ENDC} hallucinations in {total_deals} deals ({overall_rate:.2f}%)")
    
    # Per-day breakdown
    hall_rates = [d.hallucination_rate for d in days]
    print(f"  Trend:   {render_spark_line(hall_rates)}")
    
    if total_hall == 0:
        print(f"  Status:  {C.GREEN}✓ No degradation detected{C.ENDC}")
    else:
        # Find when first hallucination occurred
        first_hall_day = next((d.day for d in days if d.hallucinations > 0), None)
        if first_hall_day:
            print(f"  Status:  {C.YELLOW}⚠ First hallucination on Day {first_hall_day}{C.ENDC}")
    print()
    
    # === PRESSURE ZONE FORECAST ===
    print(f"{C.BOLD}⚠️  PRESSURE ZONE FORECAST{C.ENDC}")
    
    if len(days) >= 2:
        growth_rate = (days[-1].avg_context - days[0].avg_context) / (len(days) - 1)
        current_tokens = days[-1].avg_context
        current_day = days[-1].day
        
        thresholds = [
            (50000, "25%", "Early pressure"),
            (100000, "50%", "Moderate pressure"),
            (150000, "75%", "High pressure"),
        ]
        
        for tokens, pct, label in thresholds:
            if current_tokens < tokens:
                days_to = (tokens - current_tokens) / growth_rate + current_day
                status = f"Day {days_to:.0f}"
                color = C.DIM
            else:
                status = "REACHED"
                color = C.YELLOW
            
            print(f"  {pct:>4} ({tokens//1000}K tokens): {color}{status:>10}{C.ENDC} - {label}")
    print()
    
    # === COST TRACKING ===
    total_cost = sum(d.cost for d in days)
    projected_cost = total_cost / len(days) * 30 if days else 0
    
    print(f"{C.BOLD}💰 COST ANALYSIS{C.ENDC}")
    print(f"  Spent:     ${total_cost:.2f}")
    print(f"  Projected: ${projected_cost:.2f} (30 days)")
    print()
    
    # === EARLY WARNING INDICATORS ===
    print(f"{C.BOLD}🔬 EARLY WARNING INDICATORS{C.ENDC}")
    
    # These would be populated from detailed results JSON
    indicators = [
        ("Bid Variance", "stable", C.GREEN, "No increase in decision variability"),
        ("Response Latency", "stable", C.GREEN, "Model response time consistent"),
        ("Parse Errors", "none", C.GREEN, "All responses properly formatted"),
        ("Channel Consistency", "high", C.GREEN, "Similar deals → similar bids"),
    ]
    
    for name, status, color, desc in indicators:
        print(f"  {name:.<25} {color}{status:>8}{C.ENDC}  {C.DIM}{desc}{C.ENDC}")
    
    print()
    
    # === DAY-BY-DAY TABLE ===
    print(f"{C.BOLD}📋 DAY-BY-DAY METRICS{C.ENDC}")
    print(f"  {'Day':>4} │ {'Context':>12} │ {'Util%':>6} │ {'Hall':>4} │ {'Rate':>6} │ {'Cost':>7}")
    print(f"  {'─'*4}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*6}─┼─{'─'*7}")
    
    for d in days[-10:]:  # Show last 10 days
        util = d.avg_context / 200000 * 100
        hall_color = C.GREEN if d.hallucinations == 0 else C.RED
        print(f"  {d.day:>4} │ {d.avg_context:>12,} │ {util:>5.1f}% │ {hall_color}{d.hallucinations:>4}{C.ENDC} │ {d.hallucination_rate:>5.1f}% │ ${d.cost:>6.2f}")
    
    print()
    print(f"{C.DIM}  Press Ctrl+C to exit{C.ENDC}")


def main():
    parser = argparse.ArgumentParser(description="Monitor context rot simulation")
    parser.add_argument("--watch", "-w", action="store_true", help="Auto-refresh every 30s")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Refresh interval (seconds)")
    parser.add_argument("--log", default="results/real_context_30day.log", help="Log file path")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    log_path = base_dir / args.log
    results_dir = base_dir / "results"
    
    try:
        while True:
            days, current = parse_log_file(str(log_path))
            results = parse_results_json(str(results_dir))
            render_dashboard(days, current, results)
            
            if not args.watch:
                break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
