#!/usr/bin/env python3
"""
Multi-Agent Simulation Monitor

Tracks multi-agent isolation metrics:
- Per-agent context accumulation
- Per-agent hallucination rates
- Reconciliation failure rates
- Divergence severity over time

Usage:
    python scripts/monitor_multi_agent.py
    python scripts/monitor_multi_agent.py --watch  # Auto-refresh every 30s
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


def parse_results_json(results_dir: str) -> Optional[dict]:
    """Parse the most recent multi-agent results JSON."""
    
    results_path = Path(results_dir)
    json_files = list(results_path.glob("multi_agent_isolated_*.json"))
    
    if not json_files:
        return None
    
    # Get most recent
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest) as f:
        return json.load(f)


def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Render a progress bar."""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100 if total > 0 else 0
    return f"[{bar}] {pct:.1f}%"


def render_spark_line(values: List[float], width: int = 20) -> str:
    """Render a sparkline for trend visualization."""
    if not values or len(values) == 0:
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
        result += blocks[min(idx, 7)]
    
    return result


def render_dashboard(results: Optional[dict]):
    """Render the multi-agent monitoring dashboard."""
    
    os.system('clear' if os.name != 'nt' else 'cls')
    
    C = Colors
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{C.BOLD}{C.CYAN}╔═══════════════════════════════════════════════════════════════════════╗{C.ENDC}")
    print(f"{C.BOLD}{C.CYAN}║     MULTI-AGENT ISOLATED SIMULATION - RECONCILIATION MONITOR         ║{C.ENDC}")
    print(f"{C.BOLD}{C.CYAN}╚═══════════════════════════════════════════════════════════════════════╝{C.ENDC}")
    print(f"{C.DIM}  Last updated: {now}{C.ENDC}\n")
    
    if not results:
        print(f"{C.YELLOW}  ⏳ Waiting for simulation data...{C.ENDC}")
        print(f"{C.DIM}     Run: python scripts/run_multi_agent_isolated.py --days 3{C.ENDC}")
        return
    
    # === PROGRESS ===
    days_done = results.get('days_simulated', 0)
    status = results.get('status', 'unknown')
    
    print(f"{C.BOLD}📊 SIMULATION STATUS{C.ENDC}")
    print(f"  Status: {C.GREEN if status == 'completed' else C.YELLOW}{status.upper()}{C.ENDC}")
    print(f"  Days simulated: {days_done}")
    print()
    
    # === PER-AGENT METRICS ===
    metrics = results.get('multi_agent_metrics', {})
    agent_states = metrics.get('agent_states', {})
    
    if agent_states:
        print(f"{C.BOLD}🤖 PER-AGENT METRICS{C.ENDC}")
        
        # Separate buyers and sellers
        buyers = {k: v for k, v in agent_states.items() if v['agent_type'] == 'buyer'}
        sellers = {k: v for k, v in agent_states.items() if v['agent_type'] == 'seller'}
        
        # Buyers
        print(f"\n  {C.CYAN}BUYERS:{C.ENDC}")
        print(f"  {'Agent':12} │ {'Decisions':>10} │ {'Halluc':>7} │ {'Rate':>6} │ {'Avg Context':>12} │ {'Cost':>7}")
        print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*12}─┼─{'─'*7}")
        
        for agent_id in sorted(buyers.keys()):
            state = buyers[agent_id]
            hall_rate = state.get('hallucination_rate', 0) * 100
            hall_color = C.GREEN if hall_rate == 0 else (C.YELLOW if hall_rate < 5 else C.RED)
            
            print(f"  {agent_id:12} │ {state['total_decisions']:>10} │ "
                  f"{hall_color}{state['hallucinations']:>7}{C.ENDC} │ "
                  f"{hall_rate:>5.1f}% │ {state['avg_context_size']:>12,} │ "
                  f"${state['total_cost']:>6.2f}")
        
        # Sellers
        print(f"\n  {C.CYAN}SELLERS:{C.ENDC}")
        print(f"  {'Agent':12} │ {'Decisions':>10} │ {'Halluc':>7} │ {'Rate':>6} │ {'Avg Context':>12} │ {'Cost':>7}")
        print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*12}─┼─{'─'*7}")
        
        for agent_id in sorted(sellers.keys()):
            state = sellers[agent_id]
            hall_rate = state.get('hallucination_rate', 0) * 100
            hall_color = C.GREEN if hall_rate == 0 else (C.YELLOW if hall_rate < 5 else C.RED)
            
            print(f"  {agent_id:12} │ {state['total_decisions']:>10} │ "
                  f"{hall_color}{state['hallucinations']:>7}{C.ENDC} │ "
                  f"{hall_rate:>5.1f}% │ {state['avg_context_size']:>12,} │ "
                  f"${state['total_cost']:>6.2f}")
        
        print()
    
    # === RECONCILIATION METRICS ===
    total_recon = metrics.get('total_reconciliations', 0)
    recon_failures = metrics.get('reconciliation_failures', 0)
    recon_failure_rate = metrics.get('reconciliation_failure_rate', 0) * 100
    
    recon_color = C.GREEN if recon_failure_rate == 0 else (C.YELLOW if recon_failure_rate < 5 else C.RED)
    
    print(f"{C.BOLD}🔄 RECONCILIATION STATUS{C.ENDC}")
    print(f"  Total reconciliations: {total_recon:,}")
    print(f"  Failures: {recon_color}{recon_failures:,}{C.ENDC} ({recon_failure_rate:.2f}%)")
    
    # First divergence
    first_div_day = results.get('first_divergence_day')
    first_div_ctx = results.get('first_divergence_context_size')
    
    if first_div_day:
        print(f"  First divergence: {C.YELLOW}Day {first_div_day}{C.ENDC} (context: ~{first_div_ctx:,} tokens)")
    else:
        print(f"  First divergence: {C.GREEN}None detected{C.ENDC}")
    
    # Divergence severity distribution
    severity_counts = metrics.get('divergence_severity_counts', {})
    if severity_counts:
        print(f"\n  Divergence Severity:")
        for severity in ['none', 'minor', 'moderate', 'severe']:
            count = severity_counts.get(severity, 0)
            if severity == 'none':
                color = C.GREEN
            elif severity == 'minor':
                color = C.CYAN
            elif severity == 'moderate':
                color = C.YELLOW
            else:
                color = C.RED
            print(f"    {severity.capitalize():10}: {color}{count:>6}{C.ENDC}")
    
    print()
    
    # === RECONCILIATION TREND ===
    daily_metrics = results.get('daily_metrics', [])
    
    if daily_metrics:
        print(f"{C.BOLD}📈 RECONCILIATION FAILURE TREND{C.ENDC}")
        
        failure_rates = [
            m.get('reconciliation_failure_rate', 0) * 100 
            for m in daily_metrics
        ]
        
        print(f"  Trend: {render_spark_line(failure_rates)}")
        
        # Show if trend is increasing
        if len(failure_rates) >= 3:
            early_avg = sum(failure_rates[:len(failure_rates)//3]) / (len(failure_rates)//3)
            late_avg = sum(failure_rates[-len(failure_rates)//3:]) / (len(failure_rates)//3)
            
            if late_avg > early_avg * 1.5:
                print(f"  {C.RED}⚠ Divergence accelerating!{C.ENDC} Early: {early_avg:.1f}% → Late: {late_avg:.1f}%")
            elif late_avg > early_avg:
                print(f"  {C.YELLOW}⚠ Divergence increasing{C.ENDC} Early: {early_avg:.1f}% → Late: {late_avg:.1f}%")
            else:
                print(f"  {C.GREEN}✓ Stable{C.ENDC} Early: {early_avg:.1f}% → Late: {late_avg:.1f}%")
        
        print()
    
    # === CONTEXT GROWTH ===
    if daily_metrics:
        print(f"{C.BOLD}📊 CONTEXT ACCUMULATION{C.ENDC}")
        
        # Get buyer and seller context trends
        buyer_contexts = [m.get('avg_buyer_context', 0) for m in daily_metrics]
        seller_contexts = [m.get('avg_seller_context', 0) for m in daily_metrics]
        
        latest = daily_metrics[-1]
        
        print(f"  Buyers:  {latest.get('avg_buyer_context', 0):>8,} tokens avg | Trend: {render_spark_line(buyer_contexts, 15)}")
        print(f"  Sellers: {latest.get('avg_seller_context', 0):>8,} tokens avg | Trend: {render_spark_line(seller_contexts, 15)}")
        
        print()
    
    # === DAY-BY-DAY TABLE ===
    if daily_metrics:
        print(f"{C.BOLD}📋 DAY-BY-DAY METRICS{C.ENDC}")
        print(f"  {'Day':>4} │ {'Deals':>6} │ {'Recon Fails':>12} │ {'Rate':>6} │ {'Buyer Ctx':>10} │ {'Seller Ctx':>11}")
        print(f"  {'─'*4}─┼─{'─'*6}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*10}─┼─{'─'*11}")
        
        for dm in daily_metrics[-10:]:  # Last 10 days
            day = dm.get('day', 0)
            deals = dm.get('deals_processed', 0)
            fails = dm.get('reconciliation_failures', 0)
            rate = dm.get('reconciliation_failure_rate', 0) * 100
            buyer_ctx = dm.get('avg_buyer_context', 0)
            seller_ctx = dm.get('avg_seller_context', 0)
            
            rate_color = C.GREEN if rate == 0 else (C.YELLOW if rate < 5 else C.RED)
            
            print(f"  {day:>4} │ {deals:>6} │ "
                  f"{rate_color}{fails:>12}{C.ENDC} │ "
                  f"{rate:>5.1f}% │ {buyer_ctx:>10,} │ {seller_ctx:>11,}")
        
        print()
    
    # === KEY INSIGHT ===
    print(f"{C.BOLD}💡 KEY INSIGHT{C.ENDC}")
    
    if recon_failure_rate == 0:
        print(f"  {C.GREEN}✓ Perfect agreement{C.ENDC} - all agents synchronized")
    elif recon_failure_rate < 1:
        print(f"  {C.YELLOW}⚠ Minor divergence detected{C.ENDC} - early signs of memory drift")
    elif recon_failure_rate < 5:
        print(f"  {C.YELLOW}⚠ Moderate divergence{C.ENDC} - agents' memories starting to differ")
    else:
        print(f"  {C.RED}⚠ SIGNIFICANT DIVERGENCE{C.ENDC} - buyer/seller records misaligned")
    
    print(f"\n  This demonstrates the A2A problem: WITHOUT a shared ledger,")
    print(f"  independent agents accumulate conflicting memories → disputes.")
    print(f"  WITH Alkimi's blockchain, both parties reference the same truth.")
    
    print()
    print(f"{C.DIM}  Press Ctrl+C to exit{C.ENDC}")


def main():
    parser = argparse.ArgumentParser(description="Monitor multi-agent simulation")
    parser.add_argument("--watch", "-w", action="store_true", help="Auto-refresh every 30s")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Refresh interval (seconds)")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / args.results_dir
    
    try:
        while True:
            results = parse_results_json(str(results_dir))
            render_dashboard(results)
            
            if not args.watch:
                break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
