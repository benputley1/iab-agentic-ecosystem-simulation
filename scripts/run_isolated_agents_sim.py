#!/usr/bin/env python3
"""
Multi-Agent Isolation Simulation.

This simulation demonstrates the IAB A2A problem by:
1. Creating SEPARATE agent instances for each buyer and seller
2. Each agent maintains its OWN isolated context and memory
3. After each deal, BOTH buyer and seller record it from their perspective
4. Reconciliation checks detect when memories diverge
5. Proves that WITHOUT a shared ledger, disputes are inevitable

Key insight: As context pressure builds, agents start making errors.
Their isolated memories diverge. Without a source of truth, there's
no way to resolve who is "right" - this is the problem Alkimi solves.
"""

import argparse
import json
import os
import sys
import time
import random
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
from models.agent_state import (
    AgentState,
    ReconciliationResult,
    MultiAgentMetrics,
    Discrepancy,
    reconcile_deal,
)


@dataclass
class IsolatedAgent:
    """
    An isolated agent with its own context, memory, and API client.
    
    Each agent is completely independent - no shared state.
    """
    agent_id: str
    agent_type: str  # "buyer" or "seller"
    model: str = "claude-sonnet-4-20250514"
    
    # Isolated context - grows independently
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Agent's "database" of deals
    deal_records: Dict[str, dict] = field(default_factory=dict)
    
    # Metrics
    total_decisions: int = 0
    hallucinations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    latencies: List[float] = field(default_factory=list)
    
    # API client (initialized separately)
    client: Any = None
    
    def __post_init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def estimate_context_tokens(self) -> int:
        """Estimate current context size in tokens."""
        # Rough estimate: 50 tokens per history entry
        return len(self.context_history) * 50
    
    def get_state(self) -> AgentState:
        """Get current state snapshot."""
        return AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            context_history=self.context_history.copy(),
            context_tokens=self.estimate_context_tokens(),
            deal_records={k: v for k, v in self.deal_records.items()},
            total_decisions=self.total_decisions,
            hallucinations=self.hallucinations,
            hallucination_rate=self.hallucinations / max(self.total_decisions, 1),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_cost_usd=self.total_cost_usd,
            avg_latency_ms=sum(self.latencies) / max(len(self.latencies), 1),
        )


@dataclass
class DealResult:
    """Result of a single deal between buyer and seller."""
    deal_id: str
    day: int
    buyer_id: str
    seller_id: str
    channel: str
    impressions: int
    cpm_floor: float
    cpm_max: float
    buyer_bid: float
    seller_accepted_cpm: float
    buyer_hallucinated: bool
    seller_hallucinated: bool
    buyer_input_tokens: int
    seller_input_tokens: int
    reconciliation: Optional[ReconciliationResult] = None


class IsolatedAgentSimulation:
    """
    Multi-agent simulation with complete isolation.
    
    Demonstrates:
    - Independent context accumulation per agent
    - Memory divergence over time
    - Reconciliation failures as evidence of the A2A problem
    """
    
    # Sonnet pricing
    INPUT_COST_PER_1M = 3.00
    OUTPUT_COST_PER_1M = 15.00
    
    def __init__(
        self,
        num_buyers: int = 3,
        num_sellers: int = 3,
        days: int = 30,
        deals_per_day: int = 50,  # Lower than before - each deal needs 2 API calls
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
    ):
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.days = days
        self.deals_per_day = deals_per_day
        self.model = model
        self.verbose = verbose
        
        # Create ISOLATED agents
        self.buyers: Dict[str, IsolatedAgent] = {}
        self.sellers: Dict[str, IsolatedAgent] = {}
        
        for i in range(1, num_buyers + 1):
            agent_id = f"buyer_{i}"
            self.buyers[agent_id] = IsolatedAgent(
                agent_id=agent_id,
                agent_type="buyer",
                model=model,
            )
        
        for i in range(1, num_sellers + 1):
            agent_id = f"seller_{i}"
            self.sellers[agent_id] = IsolatedAgent(
                agent_id=agent_id,
                agent_type="seller",
                model=model,
            )
        
        # Campaign loader
        self.loader = CampaignBriefLoader(
            Path(__file__).parent.parent / "data" / "campaign_briefs.json"
        )
        self.loader.load()
        
        # Track all deals and reconciliations
        self.all_deals: List[DealResult] = []
        self.reconciliation_results: List[ReconciliationResult] = []
        
        # Results
        self.start_time = datetime.now(timezone.utc)
    
    def _build_buyer_prompt(self, agent: IsolatedAgent, deal: dict) -> str:
        """Build prompt for buyer to make a bid."""
        prompt = f"""You are {agent.agent_id.upper()}, an AI ad buyer agent.

"""
        # Add this buyer's history (isolated context)
        if agent.context_history:
            prompt += f"=== YOUR TRANSACTION HISTORY ({len(agent.context_history)} deals) ===\n"
            for i, h in enumerate(agent.context_history[-50:], 1):  # Last 50
                prompt += f"{i}. Day {h['day']} | {h['channel']}: bid ${h['bid']:.2f} CPM\n"
            prompt += "\n"
        
        prompt += f"""=== DEAL OPPORTUNITY ===
Channel: {deal['channel']}
Campaign: {deal['campaign_name']}
Impressions: {deal['impressions']:,}
Seller: {deal['seller_name']}
Market range: ${deal['cpm_floor']:.2f} - ${deal['cpm_max']:.2f} CPM

What CPM should you bid? Respond with ONLY a number."""
        
        return prompt
    
    def _build_seller_prompt(self, agent: IsolatedAgent, deal: dict, buyer_bid: float) -> str:
        """Build prompt for seller to evaluate a bid."""
        prompt = f"""You are {agent.agent_id.upper()}, an AI ad seller agent.

"""
        # Add this seller's history (isolated context)
        if agent.context_history:
            prompt += f"=== YOUR TRANSACTION HISTORY ({len(agent.context_history)} deals) ===\n"
            for i, h in enumerate(agent.context_history[-50:], 1):
                prompt += f"{i}. Day {h['day']} | {h['channel']}: accepted ${h['accepted']:.2f} CPM\n"
            prompt += "\n"
        
        prompt += f"""=== INCOMING BID ===
Channel: {deal['channel']}
Campaign: {deal['campaign_name']}
Impressions: {deal['impressions']:,}
Buyer bid: ${buyer_bid:.2f} CPM
Your floor: ${deal['cpm_floor']:.2f} CPM
Market max: ${deal['cpm_max']:.2f} CPM

What CPM do you accept for this deal? Respond with ONLY a number.
(You can accept the bid as-is, counter-offer, or accept a different amount based on your strategy)"""
        
        return prompt
    
    def _parse_cpm(self, response: str) -> tuple[float, bool, Optional[str]]:
        """Parse CPM from response. Returns (cpm, is_hallucination, reason)."""
        cleaned = response.lower().replace("$", "").replace("cpm", "").strip()
        
        try:
            cpm = float(cleaned)
            if cpm < 0:
                return 0.0, True, "negative_value"
            if cpm > 100:
                return 0.0, True, "absurd_high"
            return cpm, False, None
        except ValueError:
            pass
        
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            try:
                cpm = float(match.group(1))
                if cpm < 0:
                    return 0.0, True, "negative_value"
                if cpm > 100:
                    return 0.0, True, "absurd_high"
                return cpm, False, None
            except ValueError:
                pass
        
        return 0.0, True, "parse_error"
    
    def _make_api_call(self, agent: IsolatedAgent, prompt: str) -> tuple[str, int, int, float]:
        """Make API call and return (response, input_tokens, output_tokens, latency_ms)."""
        if not agent.client:
            return "API_ERROR: No client", 0, 0, 0.0
        
        start = time.time()
        try:
            response = agent.client.messages.create(
                model=agent.model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            latency = (time.time() - start) * 1000
            return (
                response.content[0].text.strip(),
                response.usage.input_tokens,
                response.usage.output_tokens,
                latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return f"API_ERROR: {e}", 0, 0, latency
    
    def execute_deal(self, deal: dict, day: int, deal_num: int) -> DealResult:
        """
        Execute a single deal between a buyer and seller.
        
        Both agents make independent decisions, then we reconcile.
        """
        deal_id = f"d{day:02d}_{deal_num:04d}"
        
        # Randomly select buyer and seller
        buyer_id = random.choice(list(self.buyers.keys()))
        seller_id = random.choice(list(self.sellers.keys()))
        buyer = self.buyers[buyer_id]
        seller = self.sellers[seller_id]
        
        # BUYER MAKES BID (using buyer's isolated context)
        buyer_prompt = self._build_buyer_prompt(buyer, deal)
        buyer_response, buyer_in, buyer_out, buyer_latency = self._make_api_call(buyer, buyer_prompt)
        buyer_bid, buyer_hall, buyer_reason = self._parse_cpm(buyer_response)
        
        # Update buyer metrics
        buyer.total_decisions += 1
        buyer.total_input_tokens += buyer_in
        buyer.total_output_tokens += buyer_out
        buyer.latencies.append(buyer_latency)
        if buyer_hall:
            buyer.hallucinations += 1
            # Use fallback bid on hallucination
            buyer_bid = (deal['cpm_floor'] + deal['cpm_max']) / 2
        
        # SELLER EVALUATES (using seller's isolated context)
        seller_prompt = self._build_seller_prompt(seller, deal, buyer_bid)
        seller_response, seller_in, seller_out, seller_latency = self._make_api_call(seller, seller_prompt)
        seller_cpm, seller_hall, seller_reason = self._parse_cpm(seller_response)
        
        # Update seller metrics
        seller.total_decisions += 1
        seller.total_input_tokens += seller_in
        seller.total_output_tokens += seller_out
        seller.latencies.append(seller_latency)
        if seller_hall:
            seller.hallucinations += 1
            # Use buyer's bid on hallucination
            seller_cpm = buyer_bid
        
        # RECORD DEAL FROM EACH PERSPECTIVE (this is where divergence happens!)
        # Buyer records what THEY think happened
        buyer_record = {
            "deal_id": deal_id,
            "day": day,
            "channel": deal['channel'],
            "counterparty_id": seller_id,
            "agreed_cpm": buyer_bid,  # Buyer thinks their bid was accepted
            "impressions": deal['impressions'],
            "my_bid": buyer_bid,
            "their_bid": None,  # Buyer doesn't know seller's internal valuation
            "raw_response": buyer_response,
            "was_hallucination": buyer_hall,
            "context_tokens": buyer.estimate_context_tokens(),
            "recorded_by": buyer_id,
        }
        buyer.deal_records[deal_id] = buyer_record
        
        # Add to buyer's context history (shapes future decisions)
        buyer.context_history.append({
            "day": day,
            "channel": deal['channel'],
            "impressions": deal['impressions'],
            "bid": buyer_bid,
            "outcome": "accepted",
        })
        
        # Seller records what THEY think happened
        seller_record = {
            "deal_id": deal_id,
            "day": day,
            "channel": deal['channel'],
            "counterparty_id": buyer_id,
            "agreed_cpm": seller_cpm,  # Seller records THEIR accepted price
            "impressions": deal['impressions'],
            "my_bid": seller_cpm,
            "their_bid": buyer_bid,
            "raw_response": seller_response,
            "was_hallucination": seller_hall,
            "context_tokens": seller.estimate_context_tokens(),
            "recorded_by": seller_id,
        }
        seller.deal_records[deal_id] = seller_record
        
        # Add to seller's context history
        seller.context_history.append({
            "day": day,
            "channel": deal['channel'],
            "impressions": deal['impressions'],
            "accepted": seller_cpm,
            "buyer_offer": buyer_bid,
        })
        
        # RECONCILIATION CHECK - do buyer and seller agree?
        recon = reconcile_deal(
            deal_id=deal_id,
            buyer_record=buyer_record,
            seller_record=seller_record,
            day=day,
            cpm_tolerance=0.01,  # 1% tolerance
        )
        self.reconciliation_results.append(recon)
        
        return DealResult(
            deal_id=deal_id,
            day=day,
            buyer_id=buyer_id,
            seller_id=seller_id,
            channel=deal['channel'],
            impressions=deal['impressions'],
            cpm_floor=deal['cpm_floor'],
            cpm_max=deal['cpm_max'],
            buyer_bid=buyer_bid,
            seller_accepted_cpm=seller_cpm,
            buyer_hallucinated=buyer_hall,
            seller_hallucinated=seller_hall,
            buyer_input_tokens=buyer_in,
            seller_input_tokens=seller_in,
            reconciliation=recon,
        )
    
    def run_day(self, day: int) -> dict:
        """Run one day of the simulation."""
        if self.verbose:
            print(f"\n--- Day {day}/{self.days} ---", flush=True)
        
        # Generate deals for this day
        deals = self.loader.generate_deal_stream(
            day=day,
            requests_per_day=self.deals_per_day,
        )
        
        day_results = []
        day_failures = 0
        
        for i, deal in enumerate(deals):
            result = self.execute_deal(deal, day, i + 1)
            self.all_deals.append(result)
            day_results.append(result)
            
            if result.reconciliation and not result.reconciliation.matched:
                day_failures += 1
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(deals)} deals | Failures: {day_failures}", flush=True)
        
        # Calculate day metrics
        day_hall_buyer = sum(1 for r in day_results if r.buyer_hallucinated)
        day_hall_seller = sum(1 for r in day_results if r.seller_hallucinated)
        failure_rate = day_failures / len(day_results) if day_results else 0
        
        if self.verbose:
            print(f"  Deals: {len(day_results)} | Reconciliation failures: {day_failures} ({failure_rate*100:.1f}%)")
            print(f"  Buyer hallucinations: {day_hall_buyer} | Seller: {day_hall_seller}")
            
            # Show per-agent context sizes
            buyer_tokens = {b.agent_id: b.estimate_context_tokens() for b in self.buyers.values()}
            seller_tokens = {s.agent_id: s.estimate_context_tokens() for s in self.sellers.values()}
            print(f"  Buyer contexts: {buyer_tokens}")
            print(f"  Seller contexts: {seller_tokens}", flush=True)
        
        return {
            "day": day,
            "deals": len(day_results),
            "failures": day_failures,
            "failure_rate": failure_rate,
            "buyer_hallucinations": day_hall_buyer,
            "seller_hallucinations": day_hall_seller,
        }
    
    def get_metrics(self) -> MultiAgentMetrics:
        """Calculate aggregate metrics."""
        metrics = MultiAgentMetrics(
            num_buyers=len(self.buyers),
            num_sellers=len(self.sellers),
            total_deals=len(self.all_deals),
            reconciliation_attempts=len(self.reconciliation_results),
        )
        
        # Per-agent states
        for agent_id, agent in self.buyers.items():
            metrics.buyer_states[agent_id] = agent.get_state().to_dict()
        for agent_id, agent in self.sellers.items():
            metrics.seller_states[agent_id] = agent.get_state().to_dict()
        
        # Reconciliation stats
        successes = [r for r in self.reconciliation_results if r.matched]
        failures = [r for r in self.reconciliation_results if not r.matched]
        
        metrics.reconciliation_successes = len(successes)
        metrics.reconciliation_failures = len(failures)
        metrics.failure_rate = len(failures) / max(len(self.reconciliation_results), 1)
        
        # Discrepancy breakdown
        for r in failures:
            for d in r.discrepancies:
                if d.field_name == "agreed_cpm":
                    metrics.cpm_discrepancies += 1
                elif d.field_name == "impressions":
                    metrics.impression_discrepancies += 1
                elif d.field_name == "channel":
                    metrics.channel_discrepancies += 1
                
                if d.severity == "minor":
                    metrics.minor_failures += 1
                elif d.severity == "moderate":
                    metrics.moderate_failures += 1
                elif d.severity == "severe":
                    metrics.severe_failures += 1
        
        # Context correlation
        if failures:
            metrics.avg_buyer_tokens_on_failure = sum(r.buyer_context_tokens for r in failures) / len(failures)
            metrics.avg_seller_tokens_on_failure = sum(r.seller_context_tokens for r in failures) / len(failures)
            metrics.failures_with_hallucination = sum(1 for r in failures if r.buyer_hallucinated or r.seller_hallucinated)
            metrics.failures_without_hallucination = len(failures) - metrics.failures_with_hallucination
        
        if successes:
            metrics.avg_buyer_tokens_on_success = sum(r.buyer_context_tokens for r in successes) / len(successes)
            metrics.avg_seller_tokens_on_success = sum(r.seller_context_tokens for r in successes) / len(successes)
        
        return metrics
    
    def run(self) -> dict:
        """Run the full simulation."""
        print(f"\n{'='*70}")
        print("MULTI-AGENT ISOLATION SIMULATION")
        print(f"{'='*70}")
        print(f"Buyers: {self.num_buyers} | Sellers: {self.num_sellers}")
        print(f"Days: {self.days} | Deals/day: {self.deals_per_day}")
        print(f"Model: {self.model}")
        print(f"\nEach agent has ISOLATED context - no shared memory.")
        print("Reconciliation checks detect when buyer/seller records diverge.")
        print(f"{'='*70}\n", flush=True)
        
        day_metrics = []
        
        try:
            for day in range(1, self.days + 1):
                dm = self.run_day(day)
                day_metrics.append(dm)
                
        except KeyboardInterrupt:
            print("\n[Interrupted]", flush=True)
        
        # Final metrics
        metrics = self.get_metrics()
        
        # Summary
        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"\n📊 OVERVIEW:")
        print(f"  Total deals: {metrics.total_deals}")
        print(f"  Reconciliation failures: {metrics.reconciliation_failures} ({metrics.failure_rate*100:.1f}%)")
        
        print(f"\n🔍 FAILURE BREAKDOWN:")
        print(f"  CPM discrepancies: {metrics.cpm_discrepancies}")
        print(f"  Impression discrepancies: {metrics.impression_discrepancies}")
        print(f"  Minor: {metrics.minor_failures} | Moderate: {metrics.moderate_failures} | Severe: {metrics.severe_failures}")
        
        print(f"\n🤖 PER-AGENT STATS:")
        for agent_id, state in metrics.buyer_states.items():
            print(f"  {agent_id}: {state['context_tokens']} tokens, {state['hallucinations']} hallucinations")
        for agent_id, state in metrics.seller_states.items():
            print(f"  {agent_id}: {state['context_tokens']} tokens, {state['hallucinations']} hallucinations")
        
        print(f"\n📈 CONTEXT CORRELATION:")
        print(f"  Avg buyer tokens on failure: {metrics.avg_buyer_tokens_on_failure:,.0f}")
        print(f"  Avg buyer tokens on success: {metrics.avg_buyer_tokens_on_success:,.0f}")
        print(f"  Failures with hallucination: {metrics.failures_with_hallucination}")
        print(f"  Failures without hallucination: {metrics.failures_without_hallucination}")
        
        return {
            "simulation_id": f"isolated-{self.start_time.strftime('%Y%m%d-%H%M%S')}",
            "config": {
                "num_buyers": self.num_buyers,
                "num_sellers": self.num_sellers,
                "days": self.days,
                "deals_per_day": self.deals_per_day,
                "model": self.model,
            },
            "day_metrics": day_metrics,
            "aggregate_metrics": metrics.to_dict(),
            "reconciliation_failures": [r.to_dict() for r in self.reconciliation_results if not r.matched],
        }
    
    def save_results(self, path: Optional[str] = None) -> str:
        """Save results to JSON."""
        results = self.run()
        
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"results/isolated_agents_{ts}.json"
        
        output = Path(__file__).parent.parent / path
        output.parent.mkdir(exist_ok=True)
        
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved: {output}")
        return str(output)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Isolation Simulation")
    parser.add_argument("--buyers", type=int, default=3, help="Number of buyer agents")
    parser.add_argument("--sellers", type=int, default=3, help="Number of seller agents")
    parser.add_argument("--days", type=int, default=3, help="Days to simulate")
    parser.add_argument("--deals-per-day", type=int, default=20, help="Deals per day")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    sim = IsolatedAgentSimulation(
        num_buyers=args.buyers,
        num_sellers=args.sellers,
        days=args.days,
        deals_per_day=args.deals_per_day,
        model=args.model,
    )
    
    sim.save_results(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
