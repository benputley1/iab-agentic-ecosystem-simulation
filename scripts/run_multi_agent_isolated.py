#!/usr/bin/env python3
"""
Multi-Agent Isolated Context Simulation.

The CORRECT architecture for IAB A2A simulation:
- Separate PricingAgent instances for each buyer/seller
- Each agent has isolated context history and memory
- Deals recorded from BOTH perspectives (buyer + seller)
- Reconciliation checks detect divergence

This proves the core A2A problem: WITHOUT a shared ledger,
buyer and seller memories diverge over time, leading to disputes.
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
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from llm.pricing_agent import PricingAgent, PricingDecision
from models.agent_state import AgentState, ReconciliationResult, MultiAgentMetrics, reconcile_deal
from data.campaign_loader import CampaignBriefLoader


@dataclass
class DayMetrics:
    """Metrics for a single simulation day."""
    day: int
    deals_processed: int = 0
    
    # Per-agent metrics
    buyer_hallucinations: Dict[str, int] = field(default_factory=dict)
    seller_hallucinations: Dict[str, int] = field(default_factory=dict)
    
    # Reconciliation
    reconciliation_failures: int = 0
    reconciliation_failure_rate: float = 0.0
    
    # Costs
    total_cost_usd: float = 0.0
    
    # Context pressure
    avg_buyer_context: int = 0
    avg_seller_context: int = 0
    avg_context_overall: int = 0


@dataclass
class SimulationResult:
    """Complete multi-agent simulation results."""
    simulation_id: str
    start_time: str
    end_time: Optional[str] = None
    mode: str = "multi_agent_isolated"
    days_simulated: int = 0
    status: str = "running"
    
    # Multi-agent metrics
    multi_agent_metrics: Optional[MultiAgentMetrics] = None
    
    # Daily breakdown
    daily_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Key findings
    first_divergence_day: Optional[int] = None
    first_divergence_context_size: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = asdict(self)
        if self.multi_agent_metrics:
            result['multi_agent_metrics'] = self.multi_agent_metrics.to_dict()
        return result


class MultiAgentSimulation:
    """
    Simulation with ISOLATED agent instances.
    
    Creates separate buyers and sellers with independent:
    - Context histories
    - Deal records (their "database")
    - API calls
    - Memory/hallucination patterns
    
    Enables reconciliation checks to detect when memories diverge.
    """
    
    def __init__(
        self,
        days: int = 30,
        deals_per_day: int = 100,
        num_buyers: int = 3,
        num_sellers: int = 3,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
    ):
        self.days = days
        self.deals_per_day = deals_per_day
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.model = model
        self.verbose = verbose
        
        # API key check
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        # Campaign loader
        self.loader = CampaignBriefLoader(
            Path(__file__).parent.parent / "data" / "campaign_briefs.json"
        )
        self.loader.load()
        
        # Create ISOLATED agent instances
        self.buyer_agents: Dict[str, PricingAgent] = {}
        self.seller_agents: Dict[str, PricingAgent] = {}
        
        # Track agent states separately
        self.agent_states: Dict[str, AgentState] = {}
        
        for i in range(1, num_buyers + 1):
            agent_id = f"buyer_{i}"
            self.buyer_agents[agent_id] = PricingAgent(
                agent_id=agent_id,
                agent_type="buyer",
                model=model,
            )
            self.agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                agent_type="buyer",
            )
        
        for i in range(1, num_sellers + 1):
            agent_id = f"seller_{i}"
            self.seller_agents[agent_id] = PricingAgent(
                agent_id=agent_id,
                agent_type="seller",
                model=model,
            )
            self.agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                agent_type="seller",
            )
        
        # Multi-agent metrics
        self.metrics = MultiAgentMetrics(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
        )
        
        # Track reconciliation results
        self.all_reconciliations: List[ReconciliationResult] = []
        
        # Results
        self.result = SimulationResult(
            simulation_id=f"multi-agent-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            start_time=datetime.now(timezone.utc).isoformat(),
            multi_agent_metrics=self.metrics,
        )
    
    def update_agent_state(self, agent_id: str, decision: PricingDecision):
        """Update agent state with a new decision."""
        state = self.agent_states[agent_id]
        state.total_decisions += 1
        state.total_input_tokens += decision.input_tokens
        state.total_output_tokens += decision.output_tokens
        
        if decision.is_hallucination:
            state.hallucinations += 1
        
        state.hallucination_rate = state.hallucinations / state.total_decisions
        
        # Calculate cost (Sonnet pricing)
        INPUT_COST_PER_1M = 3.00
        OUTPUT_COST_PER_1M = 15.00
        state.total_cost_usd += (
            (decision.input_tokens * INPUT_COST_PER_1M / 1_000_000) +
            (decision.output_tokens * OUTPUT_COST_PER_1M / 1_000_000)
        )
        
        # Update context tracking
        state.context_tokens = decision.input_tokens
    
    def simulate_deal(
        self,
        deal: Dict[str, Any],
        day: int,
        deal_id: int,
    ) -> ReconciliationResult:
        """
        Simulate a deal using separate buyer and seller agents.
        
        Flow:
        1. Random buyer makes bid (their context)
        2. Random seller evaluates (their context - may differ!)
        3. Both record the deal from their perspective
        4. Reconciliation check detects divergence
        """
        
        # Select random buyer and seller
        buyer_id = random.choice(list(self.buyer_agents.keys()))
        seller_id = random.choice(list(self.seller_agents.keys()))
        
        buyer_agent = self.buyer_agents[buyer_id]
        seller_agent = self.seller_agents[seller_id]
        
        # === BUYER'S DECISION (buyer's isolated context) ===
        buyer_decision = buyer_agent.make_pricing_decision(
            deal_opportunity=deal,
            context_limit=None,  # Use full context
            include_ground_truth=False,
        )
        
        # Update buyer's state
        self.update_agent_state(buyer_id, buyer_decision)
        
        # === SELLER'S DECISION (seller's isolated context) ===
        # Seller evaluates the buyer's bid
        seller_deal = deal.copy()
        seller_deal['buyer_bid'] = buyer_decision.bid_cpm
        
        seller_decision = seller_agent.make_pricing_decision(
            deal_opportunity=seller_deal,
            context_limit=None,
            include_ground_truth=False,
        )
        
        # Update seller's state
        self.update_agent_state(seller_id, seller_decision)
        
        # === DEAL EXECUTION ===
        # Accept if seller's price <= buyer's bid
        deal_accepted = seller_decision.bid_cpm <= buyer_decision.bid_cpm
        
        if deal_accepted:
            # Agreement! But what did they agree to?
            # This is where divergence can creep in...
            
            # Buyer records what THEY think happened
            buyer_record = {
                'deal_id': f"deal_{deal_id}",
                'day': day,
                'channel': deal['channel'],
                'agreed_cpm': buyer_decision.bid_cpm,  # Buyer's view
                'impressions': deal['impressions'],
                'recorded_by': buyer_id,
                'counterparty_id': seller_id,
                'was_hallucination': buyer_decision.is_hallucination,
                'context_tokens': self.agent_states[buyer_id].context_tokens,
                'timestamp': buyer_decision.timestamp,
            }
            buyer_agent.record_deal(buyer_record)
            
            # Seller records what THEY think happened
            # In a perfect world, this matches buyer_record exactly
            # With context pressure, divergence appears...
            seller_record = {
                'deal_id': f"deal_{deal_id}",
                'day': day,
                'channel': deal['channel'],
                'agreed_cpm': seller_decision.bid_cpm,  # Seller's view (may differ!)
                'impressions': deal['impressions'],  # May also differ if hallucinating
                'recorded_by': seller_id,
                'counterparty_id': buyer_id,
                'was_hallucination': seller_decision.is_hallucination,
                'context_tokens': self.agent_states[seller_id].context_tokens,
                'timestamp': seller_decision.timestamp,
            }
            seller_agent.record_deal(seller_record)
            
            # === RECONCILIATION ===
            reconciliation = reconcile_deal(
                deal_id=f"deal_{deal_id}",
                buyer_record=buyer_record,
                seller_record=seller_record,
                day=day,
            )
            
            # Track reconciliation
            self.all_reconciliations.append(reconciliation)
            self.metrics.total_deals += 1
            self.metrics.reconciliation_attempts += 1
            
            if reconciliation.matched:
                self.metrics.reconciliation_successes += 1
            else:
                self.metrics.reconciliation_failures += 1
                
                # Track first divergence
                if self.result.first_divergence_day is None:
                    self.result.first_divergence_day = day
                    self.result.first_divergence_context_size = (
                        reconciliation.buyer_context_tokens + reconciliation.seller_context_tokens
                    ) // 2
                
                # Update severity counts
                severity = reconciliation.severity_level
                if severity == "minor":
                    self.metrics.minor_failures += 1
                elif severity == "moderate":
                    self.metrics.moderate_failures += 1
                elif severity == "severe":
                    self.metrics.severe_failures += 1
                
                # Track hallucination involvement
                if reconciliation.buyer_hallucinated or reconciliation.seller_hallucinated:
                    self.metrics.failures_with_hallucination += 1
                else:
                    self.metrics.failures_without_hallucination += 1
            
            # Update failure rate
            if self.metrics.reconciliation_attempts > 0:
                self.metrics.failure_rate = (
                    self.metrics.reconciliation_failures / self.metrics.reconciliation_attempts
                )
            
            return reconciliation
        
        else:
            # Deal rejected - no reconciliation needed
            return ReconciliationResult(
                deal_id=f"deal_{deal_id}",
                day=day,
                matched=True,  # No deal, so no divergence
                buyer_id=buyer_id,
                seller_id=seller_id,
                buyer_cpm=buyer_decision.bid_cpm,
                seller_cpm=seller_decision.bid_cpm,
                buyer_impressions=deal['impressions'],
                seller_impressions=deal['impressions'],
            )
    
    def run_day(self, day: int) -> DayMetrics:
        """Run simulation for one day."""
        
        metrics = DayMetrics(day=day)
        
        if self.verbose:
            print(f"\n--- Day {day}/{self.days} ---", flush=True)
        
        # Generate deals
        deals = self.loader.generate_deal_stream(
            day=day,
            requests_per_day=self.deals_per_day,
        )
        
        for i, deal in enumerate(deals):
            reconciliation = self.simulate_deal(
                deal=deal,
                day=day,
                deal_id=metrics.deals_processed + 1,
            )
            
            metrics.deals_processed += 1
            
            if not reconciliation.matched:
                metrics.reconciliation_failures += 1
            
            # Progress
            if self.verbose and (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(deals)} deals processed", flush=True)
        
        # Calculate metrics
        if metrics.deals_processed > 0:
            metrics.reconciliation_failure_rate = (
                metrics.reconciliation_failures / metrics.deals_processed
            )
        
        # Calculate per-agent hallucinations
        for agent_id, state in self.agent_states.items():
            if state.agent_type == "buyer":
                metrics.buyer_hallucinations[agent_id] = state.hallucinations
            else:
                metrics.seller_hallucinations[agent_id] = state.hallucinations
        
        # Calculate average context sizes
        buyer_contexts = [
            state.context_tokens 
            for state in self.agent_states.values() 
            if state.agent_type == "buyer"
        ]
        seller_contexts = [
            state.context_tokens 
            for state in self.agent_states.values() 
            if state.agent_type == "seller"
        ]
        
        if buyer_contexts:
            metrics.avg_buyer_context = sum(buyer_contexts) // len(buyer_contexts)
        if seller_contexts:
            metrics.avg_seller_context = sum(seller_contexts) // len(seller_contexts)
        
        all_contexts = buyer_contexts + seller_contexts
        if all_contexts:
            metrics.avg_context_overall = sum(all_contexts) // len(all_contexts)
        
        # Total cost
        metrics.total_cost_usd = sum(
            state.total_cost_usd for state in self.agent_states.values()
        )
        
        if self.verbose:
            print(f"  Reconciliation failures: {metrics.reconciliation_failures}/{metrics.deals_processed} ({metrics.reconciliation_failure_rate*100:.1f}%)", flush=True)
            print(f"  Avg context: buyers={metrics.avg_buyer_context:,}, sellers={metrics.avg_seller_context:,}", flush=True)
            print(f"  Cost: ${metrics.total_cost_usd:.4f}", flush=True)
        
        return metrics
    
    def run(self) -> SimulationResult:
        """Run the full multi-agent simulation."""
        
        print(f"\n{'='*70}")
        print("IAB A2A Simulation - MULTI-AGENT ISOLATED ARCHITECTURE")
        print(f"{'='*70}")
        print(f"Days: {self.days}")
        print(f"Deals/day: {self.deals_per_day}")
        print(f"Buyers: {self.num_buyers} (isolated agents)")
        print(f"Sellers: {self.num_sellers} (isolated agents)")
        print(f"Model: {self.model}")
        print(f"\nEach agent has SEPARATE context and memory.")
        print("Reconciliation checks detect when memories diverge.")
        print(f"{'='*70}\n", flush=True)
        
        start = time.time()
        
        try:
            for day in range(1, self.days + 1):
                day_metrics = self.run_day(day)
                self.result.daily_metrics.append(asdict(day_metrics))
                self.result.days_simulated = day
            
            self.result.status = "completed"
            
        except KeyboardInterrupt:
            print("\n[Interrupted]", flush=True)
            self.result.status = "interrupted"
        except Exception as e:
            print(f"\n[Error: {e}]", flush=True)
            self.result.status = f"error: {e}"
            import traceback
            traceback.print_exc()
        
        # Final summary
        elapsed = time.time() - start
        self.result.end_time = datetime.now(timezone.utc).isoformat()
        
        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Days: {self.result.days_simulated}")
        # Update metrics with final agent states
        for agent_id, state in self.agent_states.items():
            if state.agent_type == "buyer":
                self.metrics.buyer_states[agent_id] = state.to_dict()
            else:
                self.metrics.seller_states[agent_id] = state.to_dict()
        
        print(f"\nPer-Agent Metrics:")
        for agent_id, state in sorted(self.agent_states.items()):
            print(f"  {agent_id:12} ({state.agent_type:6}): "
                  f"{state.total_decisions:4} decisions, "
                  f"{state.hallucinations:3} hallucinations ({state.hallucination_rate*100:.1f}%), "
                  f"{state.context_tokens:6,} context tokens")
        
        print(f"\nReconciliation:")
        print(f"  Total: {self.metrics.reconciliation_attempts}")
        print(f"  Successes: {self.metrics.reconciliation_successes}")
        print(f"  Failures: {self.metrics.reconciliation_failures} ({self.metrics.failure_rate*100:.1f}%)")
        print(f"  First divergence: Day {self.result.first_divergence_day or 'N/A'} "
              f"(~{self.result.first_divergence_context_size or 0:,} tokens)")
        
        print(f"\nDivergence Severity:")
        print(f"  Minor:    {self.metrics.minor_failures}")
        print(f"  Moderate: {self.metrics.moderate_failures}")
        print(f"  Severe:   {self.metrics.severe_failures}")
        
        total_cost = sum(state.total_cost_usd for state in self.agent_states.values())
        print(f"\nTotal Cost: ${total_cost:.2f}")
        print(f"Time: {elapsed:.1f}s", flush=True)
        
        return self.result
    
    def save_results(self, path: Optional[str] = None) -> str:
        """Save results to JSON."""
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"results/multi_agent_isolated_{ts}.json"
        
        output = Path(__file__).parent.parent / path
        output.parent.mkdir(exist_ok=True)
        
        with open(output, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2)
        
        print(f"\nResults: {output}", flush=True)
        return str(output)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Isolated IAB Simulation")
    parser.add_argument("--days", type=int, default=5, help="Days to simulate")
    parser.add_argument("--deals-per-day", type=int, default=100)
    parser.add_argument("--buyers", type=int, default=3)
    parser.add_argument("--sellers", type=int, default=3)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    sim = MultiAgentSimulation(
        days=args.days,
        deals_per_day=args.deals_per_day,
        num_buyers=args.buyers,
        num_sellers=args.sellers,
        model=args.model,
    )
    
    result = sim.run()
    sim.save_results(args.output)
    
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
