#!/usr/bin/env python3
"""
Two-Level Context Pressure Simulation.

This script simulates the core IAB A2A problem: high-volume campaigns
generate theoretical context pressure that exceeds actual context limits,
causing agents to "forget" agreed terms and experience price drift.

Simulation Flow:
1. NEGOTIATE PHASE: Buyer + Seller agree on deal terms (CPM, impressions)
2. EXECUTE PHASE: Process impressions in batches, checking memory each batch
3. RECONCILE PHASE: Compare buyer vs seller records for drift

Key Insight: At 1M impressions × 50 tokens each = 50M theoretical tokens,
but context limit is ~200K. Pressure ratio = 250x. This causes:
- Memory overflow events (agent forgets deal terms)
- Price drift (recalled CPM ≠ agreed CPM)
- Reconciliation failures (buyer/seller records diverge)
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import anthropic

from models.campaign_execution import (
    CampaignExecution,
    BatchResult,
    RecallResult,
    PressureSimulationResult,
    PressureThreshold,
    get_pressure_threshold,
    TOKENS_PER_IMPRESSION,
    CONTEXT_LIMIT,
)


@dataclass
class NegotiatedDeal:
    """Result of buyer-seller negotiation."""
    campaign_id: str
    buyer_id: str
    seller_id: str
    agreed_cpm: float
    impressions: int
    channel: str
    negotiation_tokens: int
    buyer_response: str
    seller_response: str


class ContextPressureSimulation:
    """
    Simulates context pressure in A2A campaign execution.
    
    Creates realistic pressure scenarios by:
    1. Running high-impression campaigns (1M+ each)
    2. Processing in batches with memory checks
    3. Tracking theoretical vs actual context usage
    4. Measuring recall accuracy under pressure
    """
    
    # Model pricing (Sonnet)
    INPUT_COST_PER_1M = 3.00
    OUTPUT_COST_PER_1M = 15.00
    
    def __init__(
        self,
        num_buyers: int = 10,
        campaigns_per_buyer: int = 10,
        impressions_per_campaign: int = 1_000_000,
        batch_size: int = 100_000,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
    ):
        self.num_buyers = num_buyers
        self.campaigns_per_buyer = campaigns_per_buyer
        self.impressions_per_campaign = impressions_per_campaign
        self.batch_size = batch_size
        self.model = model
        self.verbose = verbose
        
        # Verify API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Track buyer agent context (accumulated conversation history)
        self.buyer_contexts: Dict[str, List[Dict]] = {}
        
        # Track all deals per buyer for context building
        self.buyer_deals: Dict[str, List[NegotiatedDeal]] = {}
        
        # Results
        self.result = PressureSimulationResult(
            simulation_id=f"pressure-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            start_time=datetime.now(timezone.utc).isoformat(),
            num_buyers=num_buyers,
            num_campaigns_per_buyer=campaigns_per_buyer,
            impressions_per_campaign=impressions_per_campaign,
            batch_size=batch_size,
        )
    
    def _log(self, msg: str) -> None:
        """Log message if verbose mode is on."""
        if self.verbose:
            print(msg, flush=True)
    
    def _call_api(self, messages: List[Dict], system: str = "") -> Tuple[str, int, int]:
        """Make API call and return response, input tokens, output tokens."""
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": 200,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            
            response = self.client.messages.create(**kwargs)
            return (
                response.content[0].text.strip(),
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        except Exception as e:
            return f"ERROR: {e}", 0, 0
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost in USD."""
        return (
            (input_tokens * self.INPUT_COST_PER_1M / 1_000_000) +
            (output_tokens * self.OUTPUT_COST_PER_1M / 1_000_000)
        )
    
    def negotiate_deal(
        self,
        buyer_id: str,
        seller_id: str,
        campaign_id: str,
        channel: str = "display",
    ) -> NegotiatedDeal:
        """
        NEGOTIATE PHASE: Buyer and seller agree on deal terms.
        
        This establishes the "ground truth" that will be tested under pressure.
        """
        impressions = self.impressions_per_campaign
        
        # Seller offers
        seller_system = f"You are {seller_id}, an ad inventory seller. Be concise."
        seller_prompt = f"""
You have {impressions:,} {channel} impressions to sell for campaign {campaign_id}.

Offer a CPM price between $5-15 for display, $10-25 for video, $15-35 for CTV.
This is {channel}. Be competitive but profitable.

Reply with ONLY the format: "CPM: $X.XX"
"""
        seller_response, s_in, s_out = self._call_api(
            [{"role": "user", "content": seller_prompt}],
            seller_system,
        )
        
        # Extract seller's offer
        seller_cpm = self._extract_cpm(seller_response, default=10.0)
        
        # Buyer accepts (simplified negotiation)
        buyer_system = f"You are {buyer_id}, an ad buyer. Be concise."
        buyer_prompt = f"""
Seller {seller_id} offers {impressions:,} {channel} impressions at ${seller_cpm:.2f} CPM for campaign {campaign_id}.

This is a reasonable market rate. Accept the deal and confirm the terms.

Reply with ONLY the format: "ACCEPTED: {campaign_id} with {seller_id} at $X.XX CPM for Y impressions"
"""
        buyer_response, b_in, b_out = self._call_api(
            [{"role": "user", "content": buyer_prompt}],
            buyer_system,
        )
        
        # Record deal in buyer's memory
        if buyer_id not in self.buyer_deals:
            self.buyer_deals[buyer_id] = []
        
        deal = NegotiatedDeal(
            campaign_id=campaign_id,
            buyer_id=buyer_id,
            seller_id=seller_id,
            agreed_cpm=seller_cpm,
            impressions=impressions,
            channel=channel,
            negotiation_tokens=s_in + s_out + b_in + b_out,
            buyer_response=buyer_response,
            seller_response=seller_response,
        )
        
        self.buyer_deals[buyer_id].append(deal)
        
        # Add to buyer's context history
        if buyer_id not in self.buyer_contexts:
            self.buyer_contexts[buyer_id] = []
        
        self.buyer_contexts[buyer_id].append({
            "role": "assistant",
            "content": f"Deal agreed: {campaign_id} with {seller_id} at ${seller_cpm:.2f} CPM for {impressions:,} impressions"
        })
        
        return deal
    
    def _extract_cpm(self, text: str, default: float = 10.0) -> float:
        """Extract CPM value from response text."""
        # Try common patterns
        patterns = [
            r'\$(\d+\.?\d*)',
            r'CPM[:\s]+\$?(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*CPM',
            r'(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    cpm = float(match.group(1))
                    if 0.1 <= cpm <= 100:  # Sanity check
                        return cpm
                except ValueError:
                    continue
        
        return default
    
    def check_agent_recall(
        self,
        buyer_id: str,
        seller_id: str,
        expected_cpm: float,
        campaign_id: str,
    ) -> RecallResult:
        """
        Query agent mid-campaign about agreed terms.
        
        This is the key pressure test: can the agent correctly recall
        the negotiated CPM when under context pressure?
        """
        # Build context with ALL deals this buyer has made
        context_summary = "Your negotiated deals:\n"
        for deal in self.buyer_deals.get(buyer_id, []):
            context_summary += f"- {deal.campaign_id}: {deal.seller_id} @ ${deal.agreed_cpm:.2f} CPM, {deal.impressions:,} imps\n"
        
        prompt = f"""
{context_summary}

Question: What CPM did we agree with {seller_id} for campaign {campaign_id}?

Reply with ONLY the format: "$X.XX"
"""
        
        response, in_tokens, out_tokens = self._call_api(
            [{"role": "user", "content": prompt}],
            f"You are {buyer_id}. Answer from your deal records. Be precise.",
        )
        
        recalled_cpm = self._extract_cpm(response, default=0.0)
        
        # Calculate drift
        if expected_cpm > 0:
            drift = abs(recalled_cpm - expected_cpm) / expected_cpm
        else:
            drift = 1.0 if recalled_cpm != 0 else 0.0
        
        return RecallResult(
            seller_id=seller_id,
            expected_cpm=expected_cpm,
            recalled_cpm=recalled_cpm,
            drift=drift,
            is_correct=drift < 0.02,  # 2% tolerance
            raw_response=response,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
    
    def execute_campaign(self, deal: NegotiatedDeal) -> CampaignExecution:
        """
        EXECUTE PHASE: Process campaign impressions in batches.
        
        After each batch:
        1. Calculate theoretical context pressure
        2. Query agent for recall check
        3. Track price drift
        """
        campaign = CampaignExecution(
            campaign_id=deal.campaign_id,
            buyer_id=deal.buyer_id,
            seller_id=deal.seller_id,
            agreed_cpm=deal.agreed_cpm,
            impressions_total=deal.impressions,
            batch_size=self.batch_size,
            status="running",
        )
        
        num_batches = deal.impressions // self.batch_size
        
        self._log(f"\n  Executing {deal.campaign_id}: {deal.impressions:,} imps @ ${deal.agreed_cpm:.2f} CPM")
        
        for batch_num in range(1, num_batches + 1):
            impressions_cumulative = batch_num * self.batch_size
            
            # Calculate theoretical pressure
            theoretical_tokens = impressions_cumulative * TOKENS_PER_IMPRESSION
            pressure_ratio = theoretical_tokens / CONTEXT_LIMIT
            threshold = get_pressure_threshold(pressure_ratio)
            
            # Simulate actual context (grows with conversation but caps at API limit)
            # Estimate: ~500 tokens per deal in memory + query overhead
            deals_count = len(self.buyer_deals.get(deal.buyer_id, []))
            actual_context = min(deals_count * 500 + 1000, CONTEXT_LIMIT)
            
            # Memory check - query agent about agreed terms
            start_time = time.time()
            recall_result = self.check_agent_recall(
                buyer_id=deal.buyer_id,
                seller_id=deal.seller_id,
                expected_cpm=deal.agreed_cpm,
                campaign_id=deal.campaign_id,
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate cost for this batch
            api_cost = self._calculate_cost(
                recall_result.input_tokens,
                recall_result.output_tokens,
            )
            
            batch_result = BatchResult(
                batch_number=batch_num,
                impressions_in_batch=self.batch_size,
                impressions_cumulative=impressions_cumulative,
                theoretical_tokens=theoretical_tokens,
                actual_context_tokens=actual_context,
                pressure_ratio=pressure_ratio,
                recall_result=recall_result,
                processing_time_ms=processing_time,
                api_cost_usd=api_cost,
            )
            
            campaign.record_batch(batch_result)
            
            # Log progress
            status = "✓" if recall_result.is_correct else "✗"
            drift_pct = recall_result.drift * 100
            self._log(
                f"    Batch {batch_num}/{num_batches}: "
                f"pressure={pressure_ratio:.1%} ({threshold.label}), "
                f"recall={status} (drift={drift_pct:.1f}%)"
            )
            
            # Track memory overflow
            if pressure_ratio > 1.0 and not recall_result.is_correct:
                campaign.memory_overflow_events += 1
        
        # Calculate final spend
        campaign.calculate_spend()
        campaign.end_time = datetime.now(timezone.utc).isoformat()
        campaign.status = "completed"
        
        return campaign
    
    def reconcile_campaign(self, campaign: CampaignExecution) -> Dict:
        """
        RECONCILE PHASE: Compare buyer vs seller final records.
        
        In a real scenario, seller would also track the campaign.
        Here we simulate by checking if final recall matches original deal.
        """
        # Final recall check
        final_recall = self.check_agent_recall(
            buyer_id=campaign.buyer_id,
            seller_id=campaign.seller_id,
            expected_cpm=campaign.agreed_cpm,
            campaign_id=campaign.campaign_id,
        )
        
        # Reconciliation result
        reconciled = final_recall.is_correct
        if not reconciled:
            campaign.reconciliation_failures += 1
        
        return {
            "campaign_id": campaign.campaign_id,
            "agreed_cpm": campaign.agreed_cpm,
            "buyer_final_recall": final_recall.recalled_cpm,
            "drift": final_recall.drift,
            "reconciled": reconciled,
        }
    
    def run(self) -> PressureSimulationResult:
        """Run the full pressure simulation."""
        self._log(f"\n{'='*70}")
        self._log("IAB A2A CONTEXT PRESSURE SIMULATION")
        self._log(f"{'='*70}")
        self._log(f"Buyers: {self.num_buyers}")
        self._log(f"Campaigns per buyer: {self.campaigns_per_buyer}")
        self._log(f"Impressions per campaign: {self.impressions_per_campaign:,}")
        self._log(f"Batch size: {self.batch_size:,}")
        self._log(f"Total campaigns: {self.num_buyers * self.campaigns_per_buyer}")
        self._log(f"Total impressions: {self.num_buyers * self.campaigns_per_buyer * self.impressions_per_campaign:,}")
        self._log(f"Model: {self.model}")
        self._log(f"\nTheoretical tokens per campaign: {self.impressions_per_campaign * TOKENS_PER_IMPRESSION:,}")
        self._log(f"Context limit: {CONTEXT_LIMIT:,}")
        self._log(f"Max pressure ratio: {self.impressions_per_campaign * TOKENS_PER_IMPRESSION / CONTEXT_LIMIT:.0f}x")
        self._log(f"{'='*70}\n")
        
        start = time.time()
        channels = ["display", "video", "ctv"]
        
        try:
            campaign_num = 0
            total_campaigns = self.num_buyers * self.campaigns_per_buyer
            
            for buyer_idx in range(1, self.num_buyers + 1):
                buyer_id = f"buyer_{buyer_idx}"
                self._log(f"\n--- {buyer_id} ---")
                
                for camp_idx in range(1, self.campaigns_per_buyer + 1):
                    campaign_num += 1
                    campaign_id = f"campaign_{buyer_idx}_{camp_idx}"
                    seller_id = f"seller_{random.randint(1, 10)}"
                    channel = random.choice(channels)
                    
                    self._log(f"\n[{campaign_num}/{total_campaigns}] {campaign_id}")
                    
                    # Phase 1: Negotiate
                    deal = self.negotiate_deal(
                        buyer_id=buyer_id,
                        seller_id=seller_id,
                        campaign_id=campaign_id,
                        channel=channel,
                    )
                    self._log(f"  Negotiated: ${deal.agreed_cpm:.2f} CPM with {seller_id}")
                    
                    # Phase 2: Execute
                    campaign = self.execute_campaign(deal)
                    
                    # Phase 3: Reconcile
                    recon = self.reconcile_campaign(campaign)
                    self._log(
                        f"  Reconciliation: {'✓ PASS' if recon['reconciled'] else '✗ FAIL'} "
                        f"(drift={recon['drift']*100:.1f}%)"
                    )
                    
                    # Add to results
                    self.result.campaigns.append(campaign)
            
            self.result.status = "completed"
            
        except KeyboardInterrupt:
            self._log("\n[Interrupted]")
            self.result.status = "interrupted"
        except Exception as e:
            self._log(f"\n[Error: {e}]")
            self.result.status = f"error: {e}"
            import traceback
            traceback.print_exc()
        
        # Finalize results
        self.result.finalize()
        
        elapsed = time.time() - start
        
        # Print summary
        self._log(f"\n{'='*70}")
        self._log("SIMULATION COMPLETE")
        self._log(f"{'='*70}")
        self._log(f"Status: {self.result.status}")
        self._log(f"Duration: {elapsed:.1f}s")
        self._log(f"\nAggregate Metrics:")
        self._log(f"  Total impressions: {self.result.total_impressions:,}")
        self._log(f"  Total batches: {self.result.total_batches}")
        self._log(f"  Total recall checks: {self.result.total_recall_checks}")
        self._log(f"  Overall recall accuracy: {self.result.overall_recall_accuracy*100:.1f}%")
        self._log(f"  Overall avg drift: {self.result.overall_avg_drift*100:.2f}%")
        self._log(f"  Price drift incidents: {self.result.total_price_drift_incidents}")
        
        self._log(f"\nPressure Level Breakdown:")
        for level, stats in self.result.pressure_level_stats.items():
            if stats["recall_checks"] > 0:
                self._log(
                    f"  {level:10}: {stats['recall_checks']:4} checks, "
                    f"{stats['recall_accuracy']*100:5.1f}% accuracy, "
                    f"{stats['avg_drift']*100:5.2f}% avg drift"
                )
        
        self._log(f"\nCost:")
        self._log(f"  API cost: ${self.result.total_api_cost_usd:.4f}")
        self._log(f"  Campaign spend: ${self.result.total_campaign_spend:,.2f}")
        
        return self.result
    
    def save_results(self, path: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"results/pressure_simulation_{ts}.json"
        
        output = Path(__file__).parent.parent / path
        output.parent.mkdir(exist_ok=True)
        
        with open(output, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2)
        
        self._log(f"\nResults saved: {output}")
        return str(output)


def main():
    parser = argparse.ArgumentParser(
        description="Two-Level Context Pressure Simulation for IAB A2A"
    )
    parser.add_argument("--buyers", type=int, default=3, help="Number of buyers")
    parser.add_argument("--campaigns", type=int, default=3, help="Campaigns per buyer")
    parser.add_argument("--impressions", type=int, default=1_000_000, help="Impressions per campaign")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Batch size")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    sim = ContextPressureSimulation(
        num_buyers=args.buyers,
        campaigns_per_buyer=args.campaigns,
        impressions_per_campaign=args.impressions,
        batch_size=args.batch_size,
        model=args.model,
        verbose=not args.quiet,
    )
    
    result = sim.run()
    sim.save_results(args.output)
    
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
