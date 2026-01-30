#!/usr/bin/env python3
"""
Context Pressure Monitoring Dashboard.

Analyzes results from pressure simulation runs and provides:
- Pressure ratio vs. recall accuracy visualization
- Price drift over time analysis
- Memory overflow event tracking
- Per-campaign metrics breakdown
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_results(path: str) -> Dict:
    """Load simulation results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print(f"\n{'='*width}")
    print(f" {title}")
    print(f"{'='*width}")


def print_section(title: str, width: int = 70) -> None:
    """Print a section header."""
    print(f"\n--- {title} {'-'*(width-len(title)-5)}")


def format_pct(value: float) -> str:
    """Format a value as percentage."""
    return f"{value*100:.1f}%" if value <= 1.0 else f"{value:.1f}%"


def analyze_pressure_vs_accuracy(results: Dict) -> None:
    """Analyze and display pressure level vs recall accuracy."""
    print_section("Pressure Level vs Recall Accuracy")
    
    pressure_stats = results.get("pressure_level_stats", {})
    
    if not pressure_stats:
        print("  No pressure stats available.")
        return
    
    # Header
    print(f"\n  {'Level':<12} {'Checks':>8} {'Accuracy':>10} {'Avg Drift':>10} {'Max Drift':>10} {'Expected':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for level in ["low", "moderate", "high", "critical", "overflow"]:
        if level in pressure_stats:
            stats = pressure_stats[level]
            checks = stats.get("recall_checks", 0)
            if checks > 0:
                accuracy = format_pct(stats.get("recall_accuracy", 0))
                avg_drift = format_pct(stats.get("avg_drift", 0))
                max_drift = format_pct(stats.get("max_drift", 0))
                expected = format_pct(stats.get("expected_accuracy", 0))
                
                # Indicator
                actual_acc = stats.get("recall_accuracy", 0)
                expected_acc = stats.get("expected_accuracy", 0)
                indicator = "✓" if actual_acc >= expected_acc * 0.9 else "✗"
                
                print(f"  {level:<12} {checks:>8} {accuracy:>10} {avg_drift:>10} {max_drift:>10} {expected:>10} {indicator}")


def analyze_drift_timeline(results: Dict) -> None:
    """Analyze price drift progression across campaigns."""
    print_section("Price Drift Timeline")
    
    campaigns = results.get("campaigns", [])
    if not campaigns:
        print("  No campaign data available.")
        return
    
    print(f"\n  Campaign                        Batches  Accuracy  Avg Drift  Max Drift  Overflow")
    print(f"  {'-'*32} {'-'*7} {'-'*9} {'-'*10} {'-'*10} {'-'*8}")
    
    for campaign in campaigns[:20]:  # Show first 20
        cid = campaign.get("campaign_id", "?")[:30]
        batches = campaign.get("batches_processed", 0)
        accuracy = format_pct(campaign.get("recall_accuracy", 0))
        avg_drift = format_pct(campaign.get("avg_drift", 0))
        max_drift = format_pct(campaign.get("max_drift", 0))
        overflow = campaign.get("memory_overflow_events", 0)
        
        print(f"  {cid:<32} {batches:>7} {accuracy:>9} {avg_drift:>10} {max_drift:>10} {overflow:>8}")
    
    if len(campaigns) > 20:
        print(f"\n  ... and {len(campaigns) - 20} more campaigns")


def analyze_memory_overflow(results: Dict) -> None:
    """Analyze memory overflow events."""
    print_section("Memory Overflow Analysis")
    
    campaigns = results.get("campaigns", [])
    if not campaigns:
        print("  No campaign data available.")
        return
    
    total_overflow = sum(c.get("memory_overflow_events", 0) for c in campaigns)
    total_batches = sum(c.get("batches_processed", 0) for c in campaigns)
    overflow_rate = total_overflow / total_batches if total_batches > 0 else 0
    
    print(f"\n  Total overflow events: {total_overflow}")
    print(f"  Total batches: {total_batches}")
    print(f"  Overflow rate: {format_pct(overflow_rate)}")
    
    # Find worst offenders
    overflows = [(c["campaign_id"], c.get("memory_overflow_events", 0)) for c in campaigns]
    overflows.sort(key=lambda x: x[1], reverse=True)
    
    if overflows and overflows[0][1] > 0:
        print(f"\n  Campaigns with most overflows:")
        for cid, count in overflows[:5]:
            if count > 0:
                print(f"    - {cid}: {count} events")


def analyze_reconciliation(results: Dict) -> None:
    """Analyze reconciliation failures."""
    print_section("Reconciliation Analysis")
    
    campaigns = results.get("campaigns", [])
    if not campaigns:
        print("  No campaign data available.")
        return
    
    total_failures = sum(c.get("reconciliation_failures", 0) for c in campaigns)
    total_drift_incidents = sum(c.get("price_drift_incidents", 0) for c in campaigns)
    
    print(f"\n  Total reconciliation failures: {total_failures}")
    print(f"  Total price drift incidents: {total_drift_incidents}")
    
    # Correlation with pressure
    high_pressure_failures = 0
    low_pressure_failures = 0
    
    for campaign in campaigns:
        pressure = campaign.get("pressure_ratio", 0)
        failures = campaign.get("reconciliation_failures", 0)
        
        if pressure > 0.5:
            high_pressure_failures += failures
        else:
            low_pressure_failures += failures
    
    print(f"\n  Failures at high pressure (>50%): {high_pressure_failures}")
    print(f"  Failures at low pressure (≤50%): {low_pressure_failures}")


def analyze_costs(results: Dict) -> None:
    """Analyze simulation costs."""
    print_section("Cost Analysis")
    
    cost = results.get("cost", {})
    config = results.get("config", {})
    
    api_cost = cost.get("total_api_cost_usd", 0)
    campaign_spend = cost.get("total_campaign_spend", 0)
    
    total_campaigns = config.get("num_buyers", 0) * config.get("num_campaigns_per_buyer", 0)
    
    print(f"\n  Total API cost: ${api_cost:.4f}")
    print(f"  Total campaign spend (simulated): ${campaign_spend:,.2f}")
    
    if total_campaigns > 0:
        print(f"  API cost per campaign: ${api_cost/total_campaigns:.6f}")
        print(f"  Campaign spend per campaign: ${campaign_spend/total_campaigns:,.2f}")


def print_summary(results: Dict) -> None:
    """Print executive summary."""
    print_header("CONTEXT PRESSURE SIMULATION REPORT")
    
    # Basic info
    sim_id = results.get("simulation_id", "unknown")
    status = results.get("status", "unknown")
    start = results.get("start_time", "?")
    end = results.get("end_time", "?")
    
    print(f"\n  Simulation ID: {sim_id}")
    print(f"  Status: {status}")
    print(f"  Time: {start} → {end}")
    
    # Config
    config = results.get("config", {})
    print(f"\n  Configuration:")
    print(f"    Buyers: {config.get('num_buyers', '?')}")
    print(f"    Campaigns/buyer: {config.get('num_campaigns_per_buyer', '?')}")
    print(f"    Impressions/campaign: {config.get('impressions_per_campaign', 0):,}")
    print(f"    Batch size: {config.get('batch_size', 0):,}")
    
    # Aggregate metrics
    metrics = results.get("aggregate_metrics", {})
    print(f"\n  Results:")
    print(f"    Total impressions: {metrics.get('total_impressions', 0):,}")
    print(f"    Total recall checks: {metrics.get('total_recall_checks', 0)}")
    print(f"    Overall recall accuracy: {format_pct(metrics.get('overall_recall_accuracy', 0))}")
    print(f"    Overall avg drift: {format_pct(metrics.get('overall_avg_drift', 0))}")
    print(f"    Price drift incidents: {metrics.get('total_price_drift_incidents', 0)}")


def generate_ascii_chart(results: Dict) -> None:
    """Generate ASCII chart of pressure vs accuracy."""
    print_section("Pressure vs Accuracy Chart")
    
    pressure_stats = results.get("pressure_level_stats", {})
    if not pressure_stats:
        print("  No data for chart.")
        return
    
    levels = ["low", "moderate", "high", "critical", "overflow"]
    max_width = 40
    
    print()
    for level in levels:
        if level in pressure_stats and pressure_stats[level].get("recall_checks", 0) > 0:
            accuracy = pressure_stats[level].get("recall_accuracy", 0)
            bar_len = int(accuracy * max_width)
            bar = "█" * bar_len + "░" * (max_width - bar_len)
            print(f"  {level:>10} │{bar}│ {accuracy*100:.1f}%")
    
    print(f"             └{'─'*max_width}┘")
    print(f"              0%{' '*(max_width//2-3)}50%{' '*(max_width//2-3)}100%")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and analyze context pressure simulation results"
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        help="Path to results JSON file (default: latest in results/)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only the summary"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis as JSON"
    )
    
    args = parser.parse_args()
    
    # Find results file
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_dir = Path(__file__).parent.parent / "results"
        pressure_files = list(results_dir.glob("pressure_simulation_*.json"))
        if not pressure_files:
            print("No pressure simulation results found in results/")
            print("Run: python scripts/run_pressure_simulation.py")
            return 1
        results_path = max(pressure_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {results_path.name}")
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return 1
    
    # Load and analyze
    results = load_results(str(results_path))
    
    if args.json:
        # Output structured analysis
        analysis = {
            "file": str(results_path),
            "summary": results.get("aggregate_metrics", {}),
            "pressure_stats": results.get("pressure_level_stats", {}),
            "cost": results.get("cost", {}),
        }
        print(json.dumps(analysis, indent=2))
        return 0
    
    # Print human-readable report
    print_summary(results)
    
    if not args.summary_only:
        analyze_pressure_vs_accuracy(results)
        generate_ascii_chart(results)
        analyze_drift_timeline(results)
        analyze_memory_overflow(results)
        analyze_reconciliation(results)
        analyze_costs(results)
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
