#!/usr/bin/env python3
"""
Run N-Day Comparison with REAL LLM API Calls

This script runs all three scenarios with actual Claude API calls for pricing decisions,
replacing the simulated hallucination approach with real LLM behavior.

Key differences from run_5day_comparison.py:
- Scenario B actually calls Claude for each pricing decision
- Hallucinations are detected from real Claude responses (not random())
- Context rot is simulated by limiting what history Claude can see
- Full API cost and token tracking

Usage:
    # Run 5-day comparison with real LLM
    python scripts/run_real_llm_comparison.py --days 5

    # Run quick test (3 days, fewer deals)
    python scripts/run_real_llm_comparison.py --days 3 --test

    # Full 30-day simulation
    python scripts/run_real_llm_comparison.py --days 30
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Import our LLM pricing agent
from src.llm.pricing_agent import PricingAgent, HallucinationMetrics


@dataclass
class DealOpportunity:
    """A deal opportunity to be processed across scenarios."""
    day: int
    buyer_id: str
    seller_id: str
    campaign_id: str
    channel: str
    impressions: int
    cpm: float  # Market rate
    cpm_floor: float  # Minimum acceptable
    cpm_max: float  # Maximum reasonable


def generate_deal_opportunities(
    days: int, 
    num_buyers: int, 
    num_sellers: int, 
    seed: int = 42,
    test_mode: bool = False,
) -> List[DealOpportunity]:
    """
    Generate deterministic deal opportunities for all scenarios.
    """
    import random
    rng = random.Random(seed)
    
    opportunities = []
    
    # Channel-specific CPM ranges
    channel_cpms = {
        "display": (5.0, 15.0),
        "video": (10.0, 25.0),
        "ctv": (15.0, 35.0),
    }
    channels = list(channel_cpms.keys())
    
    for day in range(1, days + 1):
        for i in range(num_buyers):
            buyer_id = f"buyer-{i+1:03d}"
            for j in range(num_sellers):
                seller_id = f"seller-{j+1:03d}"
                
                # In test mode, reduce deal probability
                deal_prob = 0.15 if test_mode else 0.30
                
                if rng.random() < deal_prob:
                    channel = rng.choice(channels)
                    floor, ceiling = channel_cpms[channel]
                    
                    # Market rate is somewhere in the middle
                    cpm = rng.uniform(floor * 1.1, ceiling * 0.9)
                    
                    opportunities.append(DealOpportunity(
                        day=day,
                        buyer_id=buyer_id,
                        seller_id=seller_id,
                        campaign_id=f"camp-{buyer_id}-day{day}-{j+1:03d}",
                        channel=channel,
                        impressions=rng.randint(50000, 500000),
                        cpm=cpm,
                        cpm_floor=floor,
                        cpm_max=ceiling,
                    ))
    
    return opportunities


async def run_scenario_a(opportunities: List[DealOpportunity]) -> Dict[str, Any]:
    """
    Scenario A: Rent-seeking exchange (15% fees).
    
    No LLM calls - exchange sets prices algorithmically.
    """
    logger.info("\n" + "="*60)
    logger.info("SCENARIO A: Rent-Seeking Exchange (15% fee)")
    logger.info("="*60)
    
    results = {
        "scenario": "A",
        "name": "Rent-Seeking Exchange (15%)",
        "fee_pct": 15.0,
        "deals": 0,
        "deal_opportunities": len(opportunities),
        "total_spend": 0.0,
        "total_fees": 0.0,
        "total_impressions": 0,
        "context_rot_events": 0,
        "hallucinations": 0,
        "hallucination_cost": 0.0,
        "failed_deals": 0,
        "daily_results": [],
        "api_cost_usd": 0.0,  # No API calls
    }
    
    by_day = {}
    for opp in opportunities:
        by_day.setdefault(opp.day, []).append(opp)
    
    days = max(opp.day for opp in opportunities) if opportunities else 0
    
    for day in range(1, days + 1):
        day_opps = by_day.get(day, [])
        day_spend = 0.0
        day_fees = 0.0
        day_impressions = 0
        day_deals = 0
        
        for opp in day_opps:
            # Use market rate, exchange takes 15%
            gross_cost = (opp.impressions / 1000) * opp.cpm
            exchange_fee = gross_cost * 0.15
            
            day_deals += 1
            day_spend += gross_cost
            day_fees += exchange_fee
            day_impressions += opp.impressions
        
        results["deals"] += day_deals
        results["total_spend"] += day_spend
        results["total_fees"] += day_fees
        results["total_impressions"] += day_impressions
        
        results["daily_results"].append({
            "day": day,
            "deals": day_deals,
            "spend": day_spend,
            "fees": day_fees,
            "impressions": day_impressions,
        })
        
        logger.info(f"scenario_a.day_{day}", deals=day_deals, spend=f"${day_spend:.2f}", fees=f"${day_fees:.2f}")
    
    if results["total_impressions"] > 0:
        results["avg_cpm"] = (results["total_spend"] / results["total_impressions"]) * 1000
    
    return results


async def run_scenario_b_real_llm(opportunities: List[DealOpportunity]) -> Dict[str, Any]:
    """
    Scenario B: Direct A2A with REAL LLM pricing decisions.
    
    This is where we actually call Claude for each pricing decision.
    Context rot is simulated by limiting the context window.
    """
    logger.info("\n" + "="*60)
    logger.info("SCENARIO B: Direct A2A with Real LLM (context rot)")
    logger.info("="*60)
    
    # Initialize the real pricing agent
    agent = PricingAgent()
    
    results = {
        "scenario": "B",
        "name": "Direct A2A (Real LLM, context rot)",
        "fee_pct": 0.0,
        "deals": 0,
        "deal_opportunities": len(opportunities),
        "total_spend": 0.0,
        "total_fees": 0.0,
        "total_impressions": 0,
        "context_rot_events": 0,
        "hallucinations": 0,
        "hallucination_cost": 0.0,
        "failed_deals": 0,
        "daily_results": [],
        "api_cost_usd": 0.0,
        "llm_decisions": [],
    }
    
    by_day = {}
    for opp in opportunities:
        by_day.setdefault(opp.day, []).append(opp)
    
    days = max(opp.day for opp in opportunities) if opportunities else 0
    
    # Context rot simulation: context window shrinks over time
    # Day 1: full context (50), Day 30: severely limited (5)
    base_context_limit = 50
    context_decay_rate = 0.05  # 5% per day
    
    for day in range(1, days + 1):
        day_opps = by_day.get(day, [])
        day_spend = 0.0
        day_impressions = 0
        day_deals = 0
        day_hallucinations = 0
        day_hallucination_cost = 0.0
        day_failed = 0
        
        # Calculate context limit for this day (simulates context rot)
        context_integrity = max(0.1, 1.0 - (day - 1) * context_decay_rate)
        context_limit = int(base_context_limit * context_integrity)
        
        if day > 1 and context_limit < base_context_limit:
            results["context_rot_events"] += 1
        
        for opp in day_opps:
            # Build deal dict for the LLM
            deal_dict = {
                "channel": opp.channel,
                "impressions": opp.impressions,
                "cpm_floor": opp.cpm_floor,
                "cpm_max": opp.cpm_max,
                "seller_id": opp.seller_id,
                "buyer_id": opp.buyer_id,
            }
            
            # Make REAL LLM pricing decision
            decision = agent.make_pricing_decision(
                deal_opportunity=deal_dict,
                context_limit=context_limit,
                include_ground_truth=False,  # Don't give Claude the answer
            )
            
            # Log the decision
            results["llm_decisions"].append({
                "day": day,
                "deal": opp.campaign_id,
                "channel": opp.channel,
                "market_cpm": opp.cpm,
                "bid_cpm": decision.bid_cpm,
                "raw_response": decision.raw_response,
                "is_hallucination": decision.is_hallucination,
                "hallucination_reason": decision.hallucination_reason,
                "context_limit": context_limit,
                "tokens": decision.input_tokens + decision.output_tokens,
            })
            
            if decision.is_hallucination:
                day_hallucinations += 1
                
                # Hallucination cost calculation:
                # If bid was too high, we overpaid
                # If bid was invalid/too low, deal failed
                if decision.bid_cpm > opp.cpm_max * 1.5:
                    # Overpaid
                    correct_cost = (opp.impressions / 1000) * opp.cpm
                    inflated_cost = (opp.impressions / 1000) * decision.bid_cpm
                    day_hallucination_cost += (inflated_cost - correct_cost)
                    
                    day_deals += 1
                    day_spend += inflated_cost
                    day_impressions += opp.impressions
                else:
                    # Deal failed (bid too low or invalid)
                    day_failed += 1
            else:
                # Normal deal with LLM-determined price
                # Use the LLM's bid (clamped to reasonable range)
                effective_cpm = max(opp.cpm_floor, min(decision.bid_cpm, opp.cpm_max))
                total_cost = (opp.impressions / 1000) * effective_cpm
                
                day_deals += 1
                day_spend += total_cost
                day_impressions += opp.impressions
        
        results["deals"] += day_deals
        results["total_spend"] += day_spend
        results["total_impressions"] += day_impressions
        results["hallucinations"] += day_hallucinations
        results["hallucination_cost"] += day_hallucination_cost
        results["failed_deals"] += day_failed
        
        results["daily_results"].append({
            "day": day,
            "deals": day_deals,
            "spend": day_spend,
            "impressions": day_impressions,
            "context_limit": context_limit,
            "context_integrity": context_integrity,
            "hallucinations": day_hallucinations,
            "hallucination_cost": day_hallucination_cost,
            "failed_deals": day_failed,
        })
        
        logger.info(
            f"scenario_b.day_{day}",
            deals=day_deals,
            spend=f"${day_spend:.2f}",
            context=f"{context_limit}/{base_context_limit}",
            hallucinations=day_hallucinations,
            failed=day_failed,
        )
    
    # Get API cost summary
    cost_summary = agent.get_cost_summary()
    results["api_cost_usd"] = cost_summary["total_api_cost_usd"]
    results["total_tokens"] = cost_summary["total_input_tokens"] + cost_summary["total_output_tokens"]
    results["avg_latency_ms"] = cost_summary["avg_latency_ms"]
    
    if results["total_impressions"] > 0:
        results["avg_cpm"] = (results["total_spend"] / results["total_impressions"]) * 1000
    
    return results


async def run_scenario_c_real_llm(opportunities: List[DealOpportunity]) -> Dict[str, Any]:
    """
    Scenario C: Alkimi ledger-backed (5% fee, FULL context from ledger).
    
    Uses real LLM but with full context (no rot) since ledger provides ground truth.
    """
    logger.info("\n" + "="*60)
    logger.info("SCENARIO C: Alkimi Ledger-Backed (5% fee, full context)")
    logger.info("="*60)
    
    ALKIMI_FEE_PCT = 0.05
    BLOCKCHAIN_COST_PER_DEAL = 0.001
    
    # Initialize the pricing agent with full context
    agent = PricingAgent()
    
    results = {
        "scenario": "C",
        "name": "Alkimi Ledger-Backed (5% fee)",
        "fee_pct": ALKIMI_FEE_PCT * 100,
        "deals": 0,
        "deal_opportunities": len(opportunities),
        "total_spend": 0.0,
        "total_fees": 0.0,
        "blockchain_costs": 0.0,
        "total_alkimi_costs": 0.0,
        "total_impressions": 0,
        "context_rot_events": 0,
        "hallucinations": 0,
        "hallucination_cost": 0.0,
        "failed_deals": 0,
        "daily_results": [],
        "api_cost_usd": 0.0,
        "llm_decisions": [],
    }
    
    by_day = {}
    for opp in opportunities:
        by_day.setdefault(opp.day, []).append(opp)
    
    days = max(opp.day for opp in opportunities) if opportunities else 0
    
    for day in range(1, days + 1):
        day_opps = by_day.get(day, [])
        day_deals = 0
        day_spend = 0.0
        day_fees = 0.0
        day_impressions = 0
        day_blockchain_cost = 0.0
        day_hallucinations = 0
        
        for opp in day_opps:
            # Build deal dict with GROUND TRUTH from ledger
            deal_dict = {
                "channel": opp.channel,
                "impressions": opp.impressions,
                "cpm_floor": opp.cpm_floor,
                "cpm_max": opp.cpm_max,
                "seller_id": opp.seller_id,
                "buyer_id": opp.buyer_id,
            }
            
            # Make LLM decision with FULL CONTEXT (ledger provides history)
            # Also include ground truth bounds to reduce hallucinations
            decision = agent.make_pricing_decision(
                deal_opportunity=deal_dict,
                context_limit=None,  # Full context - no rot!
                include_ground_truth=True,  # Ledger provides bounds
            )
            
            results["llm_decisions"].append({
                "day": day,
                "deal": opp.campaign_id,
                "channel": opp.channel,
                "market_cpm": opp.cpm,
                "bid_cpm": decision.bid_cpm,
                "raw_response": decision.raw_response,
                "is_hallucination": decision.is_hallucination,
                "hallucination_reason": decision.hallucination_reason,
                "context_limit": "unlimited",
                "tokens": decision.input_tokens + decision.output_tokens,
            })
            
            if decision.is_hallucination:
                day_hallucinations += 1
                results["hallucinations"] += 1
                # Even with hallucination, ledger can correct via validation
                # Use market rate instead
                effective_cpm = opp.cpm
            else:
                # Use LLM's bid (clamped to bounds)
                effective_cpm = max(opp.cpm_floor, min(decision.bid_cpm, opp.cpm_max))
            
            # Calculate with Alkimi 5% fee
            gross_cost = (opp.impressions / 1000) * effective_cpm
            alkimi_fee = gross_cost * ALKIMI_FEE_PCT
            blockchain_cost = BLOCKCHAIN_COST_PER_DEAL
            
            day_deals += 1
            day_spend += gross_cost
            day_fees += alkimi_fee
            day_impressions += opp.impressions
            day_blockchain_cost += blockchain_cost
        
        results["deals"] += day_deals
        results["total_spend"] += day_spend
        results["total_fees"] += day_fees
        results["total_impressions"] += day_impressions
        results["blockchain_costs"] += day_blockchain_cost
        
        results["daily_results"].append({
            "day": day,
            "deals": day_deals,
            "spend": day_spend,
            "alkimi_fee": day_fees,
            "impressions": day_impressions,
            "blockchain_cost": day_blockchain_cost,
            "hallucinations": day_hallucinations,
        })
        
        logger.info(
            f"scenario_c.day_{day}",
            deals=day_deals,
            spend=f"${day_spend:.2f}",
            alkimi_fee=f"${day_fees:.2f}",
            hallucinations=day_hallucinations,
        )
    
    results["total_alkimi_costs"] = results["total_fees"] + results["blockchain_costs"]
    
    # Get API cost summary
    cost_summary = agent.get_cost_summary()
    results["api_cost_usd"] = cost_summary["total_api_cost_usd"]
    results["total_tokens"] = cost_summary["total_input_tokens"] + cost_summary["total_output_tokens"]
    results["avg_latency_ms"] = cost_summary["avg_latency_ms"]
    
    if results["total_impressions"] > 0:
        results["avg_cpm"] = (results["total_spend"] / results["total_impressions"]) * 1000
    
    return results


async def run_all_scenarios_real_llm(
    days: int,
    num_buyers: int = 3,
    num_sellers: int = 3,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Run all scenarios with real LLM API calls."""
    
    start_time = datetime.utcnow()
    
    # Generate deal opportunities
    opportunities = generate_deal_opportunities(
        days, num_buyers, num_sellers, seed=42, test_mode=test_mode
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generated {len(opportunities)} deal opportunities for {days} days")
    logger.info(f"Buyers: {num_buyers}, Sellers: {num_sellers}")
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"{'='*60}\n")
    
    results = {
        "start_time": start_time.isoformat(),
        "days": days,
        "real_llm": True,
        "test_mode": test_mode,
        "num_buyers": num_buyers,
        "num_sellers": num_sellers,
        "total_opportunities": len(opportunities),
        "scenarios": {},
    }
    
    # Run Scenario A (no LLM)
    try:
        results["scenarios"]["A"] = await run_scenario_a(opportunities)
    except Exception as e:
        logger.error(f"scenario_a.failed: {e}")
        import traceback
        results["scenarios"]["A"] = {"error": str(e), "traceback": traceback.format_exc()}
    
    # Run Scenario B with REAL LLM
    try:
        results["scenarios"]["B"] = await run_scenario_b_real_llm(opportunities)
    except Exception as e:
        logger.error(f"scenario_b.failed: {e}")
        import traceback
        results["scenarios"]["B"] = {"error": str(e), "traceback": traceback.format_exc()}
    
    # Run Scenario C with REAL LLM (full context)
    try:
        results["scenarios"]["C"] = await run_scenario_c_real_llm(opportunities)
    except Exception as e:
        logger.error(f"scenario_c.failed: {e}")
        import traceback
        results["scenarios"]["C"] = {"error": str(e), "traceback": traceback.format_exc()}
    
    results["end_time"] = datetime.utcnow().isoformat()
    
    # Calculate total API costs
    total_api_cost = sum(
        results["scenarios"].get(s, {}).get("api_cost_usd", 0)
        for s in ["A", "B", "C"]
    )
    results["total_api_cost_usd"] = total_api_cost
    
    return results


def print_comparison_table(results: Dict[str, Any]):
    """Print a comprehensive comparison table."""
    
    days = results.get('days', 5)
    print(f"\n{'='*90}")
    print(f"{days}-DAY SIMULATION COMPARISON (REAL LLM)")
    print(f"{'='*90}")
    print(f"Mode: Real LLM API Calls")
    print(f"Days: {days}")
    print(f"Deal Opportunities: {results.get('total_opportunities', 'N/A')}")
    print(f"Total API Cost: ${results.get('total_api_cost_usd', 0):.4f}")
    print()
    
    a = results.get("scenarios", {}).get("A", {})
    b = results.get("scenarios", {}).get("B", {})
    c = results.get("scenarios", {}).get("C", {})
    
    # Build comparison rows
    rows = [
        ("Name", a.get("name", "N/A"), b.get("name", "N/A"), c.get("name", "N/A")),
        ("Fee Structure", "15% exchange", "0% (+ hallucination)", f"{c.get('fee_pct', 5):.0f}% Alkimi"),
        ("Successful Deals", str(a.get("deals", 0)), str(b.get("deals", 0)), str(c.get("deals", 0))),
        ("Failed Deals", "0", str(b.get("failed_deals", 0)), str(c.get("failed_deals", 0))),
        ("Total Spend", f"${a.get('total_spend', 0):,.2f}", f"${b.get('total_spend', 0):,.2f}", f"${c.get('total_spend', 0):,.2f}"),
        ("Platform Fees", f"${a.get('total_fees', 0):,.2f}", "$0.00", f"${c.get('total_fees', 0):,.2f}"),
        ("Hallucination Cost", "N/A", f"${b.get('hallucination_cost', 0):,.2f}", "$0.00"),
        ("Blockchain Cost", "N/A", "N/A", f"${c.get('blockchain_costs', 0):.4f}"),
        ("Total Overhead", f"${a.get('total_fees', 0):,.2f}", f"${b.get('hallucination_cost', 0):,.2f}", f"${c.get('total_alkimi_costs', 0):,.2f}"),
        ("Hallucinations", "N/A", str(b.get("hallucinations", 0)), str(c.get("hallucinations", 0))),
        ("Context Rot Events", "N/A", str(b.get("context_rot_events", 0)), "0"),
        ("API Cost", "$0.00", f"${b.get('api_cost_usd', 0):.4f}", f"${c.get('api_cost_usd', 0):.4f}"),
    ]
    
    # Print table
    col_widths = [max(len(str(r[i])) for r in rows) for i in range(4)]
    col_widths = [max(cw, len(h)) for cw, h in zip(col_widths, ["Metric", "Scenario A", "Scenario B", "Scenario C"])]
    
    def print_row(row):
        print("| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")
    
    print("-" * (sum(col_widths) + 13))
    print_row(["Metric", "Scenario A", "Scenario B", "Scenario C"])
    print("-" * (sum(col_widths) + 13))
    for row in rows:
        print_row(row)
    print("-" * (sum(col_widths) + 13))
    
    # Print key findings
    print("\n" + "="*90)
    print("KEY FINDINGS (REAL LLM)")
    print("="*90)
    
    b_hall = b.get("hallucinations", 0)
    c_hall = c.get("hallucinations", 0)
    b_deals = b.get("deals", 0) + b.get("failed_deals", 0)
    c_deals = c.get("deals", 0) + c.get("failed_deals", 0)
    
    b_rate = (b_hall / b_deals * 100) if b_deals > 0 else 0
    c_rate = (c_hall / c_deals * 100) if c_deals > 0 else 0
    
    print(f"\n1. REAL Hallucination Rates (from Claude API):")
    print(f"   - Scenario B (context rot): {b_hall}/{b_deals} = {b_rate:.1f}%")
    print(f"   - Scenario C (full context): {c_hall}/{c_deals} = {c_rate:.1f}%")
    
    if c_rate < b_rate:
        reduction = ((b_rate - c_rate) / b_rate * 100) if b_rate > 0 else 100
        print(f"   → {reduction:.0f}% reduction with ledger-backed context!")
    
    print(f"\n2. Context Rot Impact:")
    print(f"   - Scenario B: {b.get('context_rot_events', 0)} context degradation events")
    print(f"   - Scenario B failed deals: {b.get('failed_deals', 0)}")
    print(f"   - Scenario C: Zero context rot (ledger provides ground truth)")
    
    print(f"\n3. Cost Comparison:")
    a_overhead = a.get("total_fees", 0)
    b_overhead = b.get("hallucination_cost", 0) + b.get("api_cost_usd", 0)
    c_overhead = c.get("total_alkimi_costs", 0) + c.get("api_cost_usd", 0)
    print(f"   - Scenario A: ${a_overhead:,.2f} (15% fees)")
    print(f"   - Scenario B: ${b_overhead:,.2f} (hallucinations + API)")
    print(f"   - Scenario C: ${c_overhead:,.2f} (5% fees + blockchain + API)")
    
    if c_overhead < a_overhead:
        savings = a_overhead - c_overhead
        savings_pct = (savings / a_overhead * 100) if a_overhead > 0 else 0
        print(f"   → Alkimi saves ${savings:,.2f} ({savings_pct:.0f}%) vs traditional!")
    
    print(f"\n4. API Costs Incurred:")
    print(f"   - Scenario B: ${b.get('api_cost_usd', 0):.4f}")
    print(f"   - Scenario C: ${c.get('api_cost_usd', 0):.4f}")
    print(f"   - TOTAL: ${results.get('total_api_cost_usd', 0):.4f}")
    
    # Sample LLM decisions
    print("\n" + "="*90)
    print("SAMPLE LLM DECISIONS (from Scenario B)")
    print("="*90)
    
    b_decisions = b.get("llm_decisions", [])[:5]
    for d in b_decisions:
        hall_indicator = "⚠️ HALLUCINATION" if d.get("is_hallucination") else "✓"
        print(f"\n  {d.get('channel', '?').upper()} deal:")
        print(f"    Response: {d.get('raw_response', 'N/A')[:60]}")
        print(f"    Bid: ${d.get('bid_cpm', 0):.2f} (market: ${d.get('market_cpm', 0):.2f}) {hall_indicator}")
        if d.get("is_hallucination"):
            print(f"    Reason: {d.get('hallucination_reason', 'unknown')}")
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Run N-day comparison with REAL LLM API calls")
    parser.add_argument("--days", type=int, default=5, help="Number of simulation days")
    parser.add_argument("--test", action="store_true", help="Test mode (fewer deals)")
    parser.add_argument("--output", type=str, help="Output file for results JSON")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set! Cannot run real LLM simulation.")
        return 1
    
    logger.info(f"Running {args.days}-day simulation with REAL LLM API calls...")
    
    results = asyncio.run(run_all_scenarios_real_llm(
        days=args.days,
        test_mode=args.test,
    ))
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    output_file = args.output or f"results_real_llm_{args.days}day.json"
    with open(output_file, "w") as f:
        # Remove llm_decisions for cleaner JSON (can be verbose)
        clean_results = results.copy()
        for s in ["B", "C"]:
            if s in clean_results.get("scenarios", {}):
                clean_results["scenarios"][s] = {
                    k: v for k, v in clean_results["scenarios"][s].items() 
                    if k != "llm_decisions"
                }
        json.dump(clean_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
