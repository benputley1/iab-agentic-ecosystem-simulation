#!/usr/bin/env python3
"""
Real LLM 30-Day Simulation Runner.

Runs the IAB agentic ecosystem simulation with:
- Real Anthropic Claude API calls (Haiku for cost efficiency)
- Realistic campaign briefs from campaign_briefs.json
- Context window pressure simulation
- Hallucination detection and metrics
- Full cost tracking

Usage:
    python scripts/run_real_llm_simulation.py --days 30
    python scripts/run_real_llm_simulation.py --days 5 --dry-run  # Cost estimate only
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from llm.pricing_agent import PricingAgent, PricingDecision, HallucinationMetrics
from data.campaign_loader import CampaignBriefLoader


@dataclass
class DayMetrics:
    """Metrics for a single simulation day."""
    day: int
    deals_processed: int = 0
    hallucinations: int = 0
    hallucination_rate: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    api_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    context_health_pct: float = 100.0
    avg_bid_cpm: float = 0.0
    bid_variance: float = 0.0
    hallucination_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Complete simulation results."""
    simulation_id: str
    start_time: str
    end_time: Optional[str] = None
    days_simulated: int = 0
    status: str = "running"
    
    # Aggregate metrics
    total_deals: int = 0
    total_hallucinations: int = 0
    total_hallucination_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_api_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Per-day breakdown
    daily_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context rot analysis
    context_rot_curve: List[Dict[str, float]] = field(default_factory=list)
    hallucination_growth_curve: List[Dict[str, float]] = field(default_factory=list)
    
    # Final analysis
    peak_hallucination_rate: float = 0.0
    peak_hallucination_day: int = 0
    context_health_day_30: float = 0.0
    correlation_context_hallucination: float = 0.0


class RealLLMSimulation:
    """
    Run a 30-day simulation with real LLM calls.
    
    This demonstrates context rot in pure A2A scenarios by:
    1. Loading realistic campaign briefs
    2. Making real Claude API calls for pricing decisions
    3. Simulating context window pressure (limiting history access)
    4. Tracking hallucinations as context degrades
    """
    
    # Context decay: each day, agents lose access to older history
    CONTEXT_DECAY_RATE = 0.02  # 2% decay per day
    
    # Sonnet pricing (per IAB specs)
    INPUT_COST_PER_1M = 3.00
    OUTPUT_COST_PER_1M = 15.00
    
    def __init__(
        self,
        days: int = 30,
        deals_per_day: int = 500,
        model: str = "claude-sonnet-4-20250514",
        dry_run: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the simulation.
        
        Args:
            days: Number of simulation days (default: 30)
            deals_per_day: Bid requests per day (default: 500 for cost efficiency)
            model: Claude model to use (default: Haiku)
            dry_run: If True, estimate costs without making API calls
            verbose: Print progress updates
        """
        self.days = days
        self.deals_per_day = deals_per_day
        self.model = model
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Initialize components
        self.loader = CampaignBriefLoader(
            Path(__file__).parent.parent / "data" / "campaign_briefs.json"
        )
        self.loader.load()
        
        # Create pricing agent (one per buyer, simulating context accumulation)
        if not dry_run:
            self.agent = PricingAgent(
                model=model,
                max_context_history=1000,  # Will be limited by context_limit
            )
        else:
            self.agent = None
        
        # Results tracking
        self.result = SimulationResult(
            simulation_id=f"sim-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            start_time=datetime.utcnow().isoformat(),
        )
    
    def estimate_cost(self) -> Dict[str, Any]:
        """Estimate API costs without making calls."""
        # Rough estimates based on prompt/response sizes
        avg_input_tokens = 350  # Prompt with context
        avg_output_tokens = 15   # Just a number
        
        total_calls = self.days * self.deals_per_day
        total_input = total_calls * avg_input_tokens
        total_output = total_calls * avg_output_tokens
        
        input_cost = total_input * self.INPUT_COST_PER_1M / 1_000_000
        output_cost = total_output * self.OUTPUT_COST_PER_1M / 1_000_000
        total_cost = input_cost + output_cost
        
        return {
            "total_api_calls": total_calls,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(total_cost, 2),
            "cost_breakdown": {
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
            },
            "model": self.model,
            "note": "Actual costs may vary ±20% based on prompt complexity",
        }
    
    def calculate_context_limit(self, day: int) -> int:
        """
        Calculate how much context history the agent can access on a given day.
        
        Simulates context rot: as days progress, agents can access less history
        due to context window pressure from accumulated conversation.
        
        Args:
            day: Simulation day (1-30)
        
        Returns:
            Number of historical transactions the agent can reference
        """
        # Start with full context access, degrade over time
        # Day 1: 100% access (100 transactions)
        # Day 30: ~55% access with 2% daily decay
        
        base_context = 100
        retention = (1 - self.CONTEXT_DECAY_RATE) ** day
        return max(int(base_context * retention), 5)  # Minimum 5 for basic reasoning
    
    def calculate_context_health(self, day: int) -> float:
        """Calculate context health percentage for a day."""
        return (1 - self.CONTEXT_DECAY_RATE) ** day * 100
    
    def run_day(self, day: int) -> DayMetrics:
        """
        Run simulation for a single day.
        
        Args:
            day: Day number (1-based)
        
        Returns:
            DayMetrics for the day
        """
        metrics = DayMetrics(day=day)
        context_limit = self.calculate_context_limit(day)
        context_health = self.calculate_context_health(day)
        metrics.context_health_pct = context_health
        
        if self.verbose:
            print(f"\n--- Day {day}/{self.days} | Context: {context_limit} txns ({context_health:.1f}% health) ---", flush=True)
        
        # Generate deal stream for this day
        deals = self.loader.generate_deal_stream(
            day=day,
            requests_per_day=self.deals_per_day,
            buyer_id="buyer-001",
        )
        
        bids = []
        latencies = []
        
        for i, deal in enumerate(deals):
            if self.dry_run:
                # Simulate without API calls
                metrics.deals_processed += 1
                continue
            
            # Make real pricing decision with limited context
            decision = self.agent.make_pricing_decision(
                deal_opportunity=deal,
                context_limit=context_limit,
                include_ground_truth=False,  # Agents don't see floor/max (realistic)
            )
            
            metrics.deals_processed += 1
            metrics.input_tokens += decision.input_tokens
            metrics.output_tokens += decision.output_tokens
            latencies.append(decision.latency_ms)
            bids.append(decision.bid_cpm)
            
            if decision.is_hallucination:
                metrics.hallucinations += 1
                reason = decision.hallucination_reason or "unknown"
                metrics.hallucination_types[reason] = metrics.hallucination_types.get(reason, 0) + 1
            
            # Progress indicator
            if self.verbose and (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(deals)} deals...", flush=True)
        
        # Calculate day metrics
        if metrics.deals_processed > 0:
            metrics.hallucination_rate = metrics.hallucinations / metrics.deals_processed
            
            if not self.dry_run:
                metrics.api_cost_usd = (
                    (metrics.input_tokens * self.INPUT_COST_PER_1M / 1_000_000) +
                    (metrics.output_tokens * self.OUTPUT_COST_PER_1M / 1_000_000)
                )
                metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
                metrics.avg_bid_cpm = sum(bids) / len(bids) if bids else 0
                
                if len(bids) > 1:
                    mean = metrics.avg_bid_cpm
                    metrics.bid_variance = sum((b - mean) ** 2 for b in bids) / len(bids)
        
        if self.verbose and not self.dry_run:
            print(f"  Hallucinations: {metrics.hallucinations}/{metrics.deals_processed} ({metrics.hallucination_rate*100:.1f}%)")
            print(f"  API Cost: ${metrics.api_cost_usd:.4f}")
            print(f"  Avg Bid: ${metrics.avg_bid_cpm:.2f} CPM")
        
        return metrics
    
    def run(self) -> SimulationResult:
        """
        Run the full simulation.
        
        Returns:
            SimulationResult with all metrics and analysis
        """
        print(f"\n{'='*60}")
        print(f"IAB Agentic Ecosystem Simulation - Real LLM Mode")
        print(f"{'='*60}")
        print(f"Days: {self.days}")
        print(f"Deals/day: {self.deals_per_day}")
        print(f"Model: {self.model}")
        print(f"Dry run: {self.dry_run}")
        
        # Cost estimate
        estimate = self.estimate_cost()
        print(f"\nCost estimate: ${estimate['estimated_cost_usd']:.2f}")
        print(f"Total API calls: {estimate['total_api_calls']:,}")
        
        if self.dry_run:
            print("\n[DRY RUN MODE - No API calls will be made]")
            self.result.status = "dry_run"
            self.result.end_time = datetime.utcnow().isoformat()
            return self.result
        
        print(f"\nStarting simulation...")
        start = time.time()
        
        try:
            for day in range(1, self.days + 1):
                day_metrics = self.run_day(day)
                self.result.daily_metrics.append(asdict(day_metrics))
                
                # Update aggregates
                self.result.total_deals += day_metrics.deals_processed
                self.result.total_hallucinations += day_metrics.hallucinations
                self.result.total_input_tokens += day_metrics.input_tokens
                self.result.total_output_tokens += day_metrics.output_tokens
                self.result.total_api_cost_usd += day_metrics.api_cost_usd
                
                # Track curves
                self.result.context_rot_curve.append({
                    "day": day,
                    "context_health_pct": day_metrics.context_health_pct,
                })
                self.result.hallucination_growth_curve.append({
                    "day": day,
                    "hallucination_rate": day_metrics.hallucination_rate,
                    "hallucinations": day_metrics.hallucinations,
                })
                
                # Track peaks
                if day_metrics.hallucination_rate > self.result.peak_hallucination_rate:
                    self.result.peak_hallucination_rate = day_metrics.hallucination_rate
                    self.result.peak_hallucination_day = day
                
                self.result.days_simulated = day
            
            # Final calculations
            if self.result.total_deals > 0:
                self.result.total_hallucination_rate = (
                    self.result.total_hallucinations / self.result.total_deals
                )
            
            if self.result.daily_metrics:
                total_latency = sum(d.get("avg_latency_ms", 0) for d in self.result.daily_metrics)
                self.result.avg_latency_ms = total_latency / len(self.result.daily_metrics)
                self.result.context_health_day_30 = self.calculate_context_health(30)
            
            self.result.status = "completed"
            
        except KeyboardInterrupt:
            print("\n\n[Interrupted by user]")
            self.result.status = "interrupted"
        
        except Exception as e:
            print(f"\n\n[Error: {e}]")
            self.result.status = f"error: {e}"
        
        finally:
            elapsed = time.time() - start
            self.result.end_time = datetime.utcnow().isoformat()
        
        # Print summary
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Days simulated: {self.result.days_simulated}")
        print(f"Total deals: {self.result.total_deals:,}")
        print(f"Total hallucinations: {self.result.total_hallucinations}")
        print(f"Overall hallucination rate: {self.result.total_hallucination_rate*100:.2f}%")
        print(f"Peak hallucination rate: {self.result.peak_hallucination_rate*100:.2f}% (day {self.result.peak_hallucination_day})")
        print(f"Context health day 30: {self.result.context_health_day_30:.1f}%")
        print(f"\nAPI Usage:")
        print(f"  Input tokens: {self.result.total_input_tokens:,}")
        print(f"  Output tokens: {self.result.total_output_tokens:,}")
        print(f"  Total cost: ${self.result.total_api_cost_usd:.2f}")
        print(f"  Avg latency: {self.result.avg_latency_ms:.0f}ms")
        print(f"\nElapsed time: {elapsed:.1f}s")
        
        return self.result
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save results to JSON file.
        
        Args:
            output_path: Path to save results (default: results/real_llm_TIMESTAMP.json)
        
        Returns:
            Path where results were saved
        """
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/real_llm_{timestamp}.json"
        
        output_file = Path(__file__).parent.parent / output_path
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(asdict(self.result), f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Run IAB Real LLM Simulation")
    parser.add_argument("--days", type=int, default=30, help="Simulation days (default: 30)")
    parser.add_argument("--deals-per-day", type=int, default=500, help="Deals per day (default: 500)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Claude model")
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs without API calls")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    sim = RealLLMSimulation(
        days=args.days,
        deals_per_day=args.deals_per_day,
        model=args.model,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    
    result = sim.run()
    
    if not args.dry_run:
        sim.save_results(args.output)
    
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
