#!/usr/bin/env python3
"""
Real Context Pressure Simulation.

Runs the IAB agentic ecosystem simulation with ACTUAL context accumulation:
- Full conversation history passed to each LLM call
- No artificial context limiting
- Measures real token growth and its correlation with hallucinations
- Authentic context window pressure as history accumulates

This is the "real life" version - we let context grow naturally and measure
when/how the LLM starts making worse decisions.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import anthropic
from data.campaign_loader import CampaignBriefLoader


@dataclass
class Decision:
    """A single pricing decision with full context."""
    deal_id: int
    day: int
    channel: str
    campaign_id: str
    impressions: int
    cpm_floor: float
    cpm_max: float
    bid_cpm: float
    raw_response: str
    is_hallucination: bool
    hallucination_reason: Optional[str]
    input_tokens: int
    output_tokens: int
    cumulative_tokens: int  # Total tokens used so far in simulation
    latency_ms: float
    timestamp: str


@dataclass 
class DayMetrics:
    """Metrics for a single simulation day."""
    day: int
    deals_processed: int = 0
    hallucinations: int = 0
    hallucination_rate: float = 0.0
    day_input_tokens: int = 0
    day_output_tokens: int = 0
    cumulative_input_tokens: int = 0
    avg_context_size: int = 0  # Avg input tokens per call (shows context growth)
    api_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    avg_bid_cpm: float = 0.0
    hallucination_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Complete simulation results."""
    simulation_id: str
    start_time: str
    end_time: Optional[str] = None
    mode: str = "real_context"
    days_simulated: int = 0
    status: str = "running"
    
    # Aggregate metrics
    total_deals: int = 0
    total_hallucinations: int = 0
    total_hallucination_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_api_cost_usd: float = 0.0
    
    # Context growth tracking
    final_context_size: int = 0  # Input tokens on last call
    context_growth_curve: List[Dict[str, Any]] = field(default_factory=list)
    hallucination_by_context_size: List[Dict[str, Any]] = field(default_factory=list)
    
    # Per-day breakdown
    daily_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Key findings
    first_hallucination_at_tokens: Optional[int] = None
    hallucination_spike_threshold: Optional[int] = None  # Token count where rate > 5%


class RealContextSimulation:
    """
    Simulation with REAL context accumulation.
    
    Unlike the simulated version, this:
    - Passes FULL transaction history to each LLM call
    - Tracks actual token count growth
    - Measures hallucinations as function of real context size
    - Shows authentic context window pressure
    """
    
    # Sonnet pricing per 1M tokens
    INPUT_COST_PER_1M = 3.00
    OUTPUT_COST_PER_1M = 15.00
    
    def __init__(
        self,
        days: int = 30,
        deals_per_day: int = 100,
        model: str = "claude-sonnet-4-20250514",
        max_history_tokens: int = 150000,  # Stop adding history if we hit this
        verbose: bool = True,
    ):
        self.days = days
        self.deals_per_day = deals_per_day
        self.model = model
        self.max_history_tokens = max_history_tokens
        self.verbose = verbose
        
        # API client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Campaign loader
        self.loader = CampaignBriefLoader(
            Path(__file__).parent.parent / "data" / "campaign_briefs.json"
        )
        self.loader.load()
        
        # Conversation history - grows throughout simulation
        self.conversation_history: List[Dict[str, Any]] = []
        self.history_token_estimate: int = 0
        
        # All decisions for analysis
        self.all_decisions: List[Decision] = []
        
        # Cumulative token tracking
        self.cumulative_input_tokens: int = 0
        self.cumulative_output_tokens: int = 0
        
        # Results
        self.result = SimulationResult(
            simulation_id=f"real-ctx-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            start_time=datetime.now(timezone.utc).isoformat(),
        )
    
    def _build_prompt_with_history(self, deal: Dict[str, Any]) -> str:
        """Build prompt including full conversation history."""
        
        # System context
        prompt = """You are an AI ad buyer agent in a programmatic advertising ecosystem.

Your role: Make optimal CPM bid decisions based on deal parameters and your transaction history.

"""
        
        # Add transaction history if we have any
        if self.conversation_history:
            prompt += f"=== YOUR TRANSACTION HISTORY ({len(self.conversation_history)} deals) ===\n"
            
            # Include ALL history (this is the key difference from simulated version)
            for i, h in enumerate(self.conversation_history):
                prompt += f"{i+1}. Day {h['day']} | {h['channel']} | {h['campaign']}: "
                prompt += f"bid ${h['bid']:.2f} CPM for {h['impressions']:,} imps "
                prompt += f"(floor: ${h['floor']:.2f}, max: ${h['max']:.2f})\n"
            
            prompt += "\n"
        
        # Current deal
        prompt += f"""=== CURRENT DEAL OPPORTUNITY ===
Channel: {deal['channel']}
Campaign: {deal['campaign_name']} ({deal['campaign_id']})
Advertiser: {deal['advertiser']}
Category: {deal['category']}
Objective: {deal['objective']}
Impressions: {deal['impressions']:,}
Seller: {deal['seller_name']} (quality: {deal['quality_score']:.0%})

Market guidance:
- {deal['channel'].upper()} typical range: ${deal['cpm_floor']:.2f} - ${deal['cpm_max']:.2f} CPM
- Consider quality score and campaign objectives
- Balance competitive bidding with cost efficiency

What CPM should you bid? Respond with ONLY a number (e.g., "12.50"). No other text."""

        return prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4
    
    def _parse_bid(self, response: str) -> tuple[float, Optional[str]]:
        """Parse bid from response."""
        cleaned = response.lower().replace("$", "").replace("cpm", "").strip()
        
        try:
            return float(cleaned), None
        except ValueError:
            pass
        
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            try:
                return float(match.group(1)), None
            except ValueError:
                pass
        
        return 0.0, f"parse_failed: {response[:50]}"
    
    def _detect_hallucination(
        self,
        bid: float,
        deal: Dict[str, Any],
        parse_error: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        """Detect if bid is a hallucination."""
        floor = deal['cpm_floor']
        max_cpm = deal['cpm_max']
        
        if parse_error:
            return True, "parse_error"
        if bid < 0:
            return True, "negative_bid"
        if bid > 100:
            return True, "absurd_high"
        if bid > max_cpm * 2:
            return True, "above_2x_max"
        if bid < floor * 0.5:
            return True, "below_half_floor"
        
        return False, None
    
    def make_decision(self, deal: Dict[str, Any], day: int, deal_id: int) -> Decision:
        """Make a pricing decision with full context."""
        
        prompt = self._build_prompt_with_history(deal)
        prompt_tokens_est = self._estimate_tokens(prompt)
        
        start = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            
            latency = (time.time() - start) * 1000
            raw = response.content[0].text.strip()
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            bid, parse_error = self._parse_bid(raw)
            is_hall, hall_reason = self._detect_hallucination(bid, deal, parse_error)
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            raw = f"API_ERROR: {e}"
            input_tokens = prompt_tokens_est
            output_tokens = 0
            bid = 0.0
            is_hall = True
            hall_reason = "api_error"
        
        # Update cumulative tokens
        self.cumulative_input_tokens += input_tokens
        self.cumulative_output_tokens += output_tokens
        
        decision = Decision(
            deal_id=deal_id,
            day=day,
            channel=deal['channel'],
            campaign_id=deal['campaign_id'],
            impressions=deal['impressions'],
            cpm_floor=deal['cpm_floor'],
            cpm_max=deal['cpm_max'],
            bid_cpm=bid,
            raw_response=raw,
            is_hallucination=is_hall,
            hallucination_reason=hall_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cumulative_tokens=self.cumulative_input_tokens,
            latency_ms=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        # Add to history (this grows the context for future calls)
        self.conversation_history.append({
            'day': day,
            'channel': deal['channel'],
            'campaign': deal['campaign_id'],
            'impressions': deal['impressions'],
            'floor': deal['cpm_floor'],
            'max': deal['cpm_max'],
            'bid': bid,
        })
        self.history_token_estimate += 50  # Rough estimate per history entry
        
        return decision
    
    def run_day(self, day: int) -> DayMetrics:
        """Run simulation for one day."""
        
        metrics = DayMetrics(day=day)
        
        if self.verbose:
            ctx_size = self.history_token_estimate
            print(f"\n--- Day {day}/{self.days} | History: {len(self.conversation_history)} txns (~{ctx_size:,} tokens) ---", flush=True)
        
        # Generate deals
        deals = self.loader.generate_deal_stream(
            day=day,
            requests_per_day=self.deals_per_day,
        )
        
        day_decisions = []
        latencies = []
        bids = []
        input_tokens_list = []
        
        for i, deal in enumerate(deals):
            decision = self.make_decision(deal, day, len(self.all_decisions) + 1)
            self.all_decisions.append(decision)
            day_decisions.append(decision)
            
            metrics.deals_processed += 1
            metrics.day_input_tokens += decision.input_tokens
            metrics.day_output_tokens += decision.output_tokens
            latencies.append(decision.latency_ms)
            bids.append(decision.bid_cpm)
            input_tokens_list.append(decision.input_tokens)
            
            if decision.is_hallucination:
                metrics.hallucinations += 1
                reason = decision.hallucination_reason or "unknown"
                metrics.hallucination_types[reason] = metrics.hallucination_types.get(reason, 0) + 1
                
                # Track first hallucination
                if self.result.first_hallucination_at_tokens is None:
                    self.result.first_hallucination_at_tokens = decision.cumulative_tokens
            
            # Progress
            if self.verbose and (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(deals)} deals | Context: {decision.input_tokens:,} tokens", flush=True)
        
        # Calculate metrics
        if metrics.deals_processed > 0:
            metrics.hallucination_rate = metrics.hallucinations / metrics.deals_processed
            metrics.cumulative_input_tokens = self.cumulative_input_tokens
            metrics.avg_context_size = sum(input_tokens_list) // len(input_tokens_list)
            metrics.avg_latency_ms = sum(latencies) / len(latencies)
            metrics.avg_bid_cpm = sum(bids) / len(bids) if bids else 0
            metrics.api_cost_usd = (
                (metrics.day_input_tokens * self.INPUT_COST_PER_1M / 1_000_000) +
                (metrics.day_output_tokens * self.OUTPUT_COST_PER_1M / 1_000_000)
            )
        
        # Track context growth and hallucination correlation
        self.result.context_growth_curve.append({
            "day": day,
            "history_entries": len(self.conversation_history),
            "avg_input_tokens": metrics.avg_context_size,
            "cumulative_tokens": self.cumulative_input_tokens,
        })
        
        self.result.hallucination_by_context_size.append({
            "day": day,
            "avg_context_tokens": metrics.avg_context_size,
            "hallucination_rate": metrics.hallucination_rate,
            "hallucinations": metrics.hallucinations,
            "deals": metrics.deals_processed,
        })
        
        if self.verbose:
            print(f"  Hallucinations: {metrics.hallucinations}/{metrics.deals_processed} ({metrics.hallucination_rate*100:.1f}%)", flush=True)
            print(f"  Avg context: {metrics.avg_context_size:,} tokens | Cost: ${metrics.api_cost_usd:.4f}", flush=True)
        
        return metrics
    
    def run(self) -> SimulationResult:
        """Run the full simulation."""
        
        print(f"\n{'='*60}")
        print("IAB Agentic Simulation - REAL CONTEXT MODE")
        print(f"{'='*60}")
        print(f"Days: {self.days}")
        print(f"Deals/day: {self.deals_per_day}")
        print(f"Model: {self.model}")
        print(f"Max history tokens: {self.max_history_tokens:,}")
        print(f"\nThis simulation uses REAL context accumulation.")
        print("History grows naturally - no artificial limiting.")
        print(f"{'='*60}\n", flush=True)
        
        start = time.time()
        
        try:
            for day in range(1, self.days + 1):
                day_metrics = self.run_day(day)
                self.result.daily_metrics.append(asdict(day_metrics))
                
                # Update aggregates
                self.result.total_deals += day_metrics.deals_processed
                self.result.total_hallucinations += day_metrics.hallucinations
                self.result.total_input_tokens += day_metrics.day_input_tokens
                self.result.total_output_tokens += day_metrics.day_output_tokens
                self.result.total_api_cost_usd += day_metrics.api_cost_usd
                self.result.days_simulated = day
                
                # Check for hallucination spike threshold
                if (self.result.hallucination_spike_threshold is None and 
                    day_metrics.hallucination_rate > 0.05):
                    self.result.hallucination_spike_threshold = day_metrics.avg_context_size
            
            self.result.status = "completed"
            
        except KeyboardInterrupt:
            print("\n[Interrupted]", flush=True)
            self.result.status = "interrupted"
        except Exception as e:
            print(f"\n[Error: {e}]", flush=True)
            self.result.status = f"error: {e}"
        
        # Final calculations
        elapsed = time.time() - start
        self.result.end_time = datetime.now(timezone.utc).isoformat()
        
        if self.result.total_deals > 0:
            self.result.total_hallucination_rate = (
                self.result.total_hallucinations / self.result.total_deals
            )
        
        if self.all_decisions:
            self.result.final_context_size = self.all_decisions[-1].input_tokens
        
        # Summary
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Days: {self.result.days_simulated}")
        print(f"Total deals: {self.result.total_deals:,}")
        print(f"Total hallucinations: {self.result.total_hallucinations}")
        print(f"Overall rate: {self.result.total_hallucination_rate*100:.2f}%")
        print(f"\nContext Growth:")
        print(f"  Final context size: {self.result.final_context_size:,} tokens")
        print(f"  First hallucination at: {self.result.first_hallucination_at_tokens or 'N/A'} cumulative tokens")
        print(f"  Spike threshold (>5%): {self.result.hallucination_spike_threshold or 'N/A'} avg tokens")
        print(f"\nCost: ${self.result.total_api_cost_usd:.2f}")
        print(f"Time: {elapsed:.1f}s", flush=True)
        
        return self.result
    
    def save_results(self, path: Optional[str] = None) -> str:
        """Save results to JSON."""
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"results/real_context_{ts}.json"
        
        output = Path(__file__).parent.parent / path
        output.parent.mkdir(exist_ok=True)
        
        with open(output, "w") as f:
            json.dump(asdict(self.result), f, indent=2)
        
        print(f"\nResults: {output}", flush=True)
        return str(output)


def main():
    parser = argparse.ArgumentParser(description="Real Context IAB Simulation")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--deals-per-day", type=int, default=100)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    sim = RealContextSimulation(
        days=args.days,
        deals_per_day=args.deals_per_day,
        model=args.model,
    )
    
    result = sim.run()
    sim.save_results(args.output)
    
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
