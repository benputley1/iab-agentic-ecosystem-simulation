#!/usr/bin/env python3
"""
Run IAB Simulation with Real LLM Integration

This script runs a short simulation using the IAB buyer and seller agents
with real LLM calls to the IAB OpenDirect server.

Usage:
    # Mock mode (no API costs)
    python scripts/run_iab_simulation.py --mock --days 5

    # Real LLM mode (requires ANTHROPIC_API_KEY)
    python scripts/run_iab_simulation.py --days 5
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


async def run_simulation(
    days: int = 5,
    buyers: int = 3,
    sellers: int = 3,
    mock_llm: bool = False,
    use_a2a: bool = False,
) -> dict:
    """
    Run a simulation with IAB-integrated agents.
    
    Args:
        days: Number of simulation days
        buyers: Number of buyer agents
        sellers: Number of seller agents
        mock_llm: If True, use mock LLM (no API costs)
        use_a2a: If True, use A2A natural language protocol
        
    Returns:
        Simulation results dictionary
    """
    from src.agents.seller.iab_adapter import IABSellerAdapter
    from src.agents.buyer.iab_wrapper import IABBuyerWrapper, Campaign
    from src.infrastructure.message_schemas import DealType
    
    results = {
        "start_time": datetime.utcnow().isoformat(),
        "config": {
            "days": days,
            "buyers": buyers,
            "sellers": sellers,
            "mock_llm": mock_llm,
            "use_a2a": use_a2a,
        },
        "days_completed": 0,
        "total_deals": 0,
        "total_impressions": 0,
        "total_spend": 0.0,
        "total_llm_calls": 0,
        "total_llm_cost": 0.0,
        "deals_by_day": [],
        "errors": [],
    }
    
    logger.info(
        "simulation.starting",
        days=days,
        buyers=buyers,
        sellers=sellers,
        mock_llm=mock_llm,
        use_a2a=use_a2a,
    )
    
    # Initialize sellers (these don't need Redis for basic pricing)
    seller_adapters = []
    for i in range(sellers):
        adapter = IABSellerAdapter(
            seller_id=f"seller-{i+1:03d}",
            scenario="B",
            mock_llm=mock_llm,
            inventory_seed=42 + i,
        )
        adapter._init_iab_components()
        adapter._products = adapter._inventory.generate_catalog(num_products=3)
        seller_adapters.append(adapter)
    
    # Initialize buyers with campaigns
    buyer_wrappers = []
    for i in range(buyers):
        wrapper = IABBuyerWrapper(
            buyer_id=f"buyer-{i+1:03d}",
            scenario="B",
            mock_llm=mock_llm,
        )
        
        # Add campaigns
        for j in range(2):  # 2 campaigns per buyer
            campaign = Campaign(
                campaign_id=f"camp-{i+1:03d}-{j+1:02d}",
                name=f"Campaign {i+1}-{j+1}",
                budget=50000.0,
                target_impressions=500000,
                target_cpm=15.0 + (i * 2),  # Vary CPM by buyer
                channel="display" if j == 0 else "video",
            )
            wrapper.add_campaign(campaign)
        
        if not mock_llm:
            try:
                await wrapper.connect()
            except Exception as e:
                logger.warning(f"Could not connect buyer {wrapper.buyer_id}: {e}")
        
        buyer_wrappers.append(wrapper)
    
    try:
        # Run simulation day by day
        for day in range(1, days + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Simulation Day {day}/{days}")
            logger.info(f"{'='*60}")
            
            day_deals = 0
            day_impressions = 0
            day_spend = 0.0
            
            # Each buyer tries to make deals
            for buyer in buyer_wrappers:
                active_campaigns = buyer.get_active_campaigns()
                
                if not active_campaigns:
                    continue
                
                for campaign in active_campaigns:
                    # Find matching inventory
                    for seller in seller_adapters:
                        # Find products matching campaign channel
                        matching_products = [
                            p for p in seller.products.values()
                            if p.inventory_type == campaign.channel
                        ]
                        
                        if not matching_products:
                            continue
                        
                        product = matching_products[0]
                        
                        # Create bid request
                        from src.infrastructure.message_schemas import BidRequest
                        
                        request = BidRequest(
                            request_id=f"req-{day}-{buyer.buyer_id}-{campaign.campaign_id}",
                            buyer_id=buyer.buyer_id,
                            campaign_id=campaign.campaign_id,
                            channel=campaign.channel,
                            impressions_requested=min(100000, campaign.remaining_impressions),
                            max_cpm=campaign.target_cpm * 1.2,
                            targeting=campaign.targeting,
                        )
                        
                        # Seller evaluates request
                        evaluation = await seller.evaluate_request(request, product)
                        
                        if evaluation["accept"]:
                            # Record deal
                            deal_impressions = min(evaluation["offer_impressions"], 100000)
                            if deal_impressions == 0:
                                deal_impressions = min(100000, campaign.remaining_impressions)
                            
                            deal_cost = (deal_impressions / 1000) * evaluation["offer_cpm"]
                            
                            campaign.impressions_delivered += deal_impressions
                            campaign.spend += deal_cost
                            
                            day_deals += 1
                            day_impressions += deal_impressions
                            day_spend += deal_cost
                            
                            logger.info(
                                "deal.completed",
                                buyer=buyer.buyer_id,
                                seller=seller.seller_id,
                                campaign=campaign.campaign_id,
                                impressions=deal_impressions,
                                cpm=evaluation["offer_cpm"],
                                cost=f"${deal_cost:.2f}",
                            )
                        
                        # Only one deal per campaign per day
                        break
            
            # Track LLM usage
            day_llm_calls = sum(s.llm_stats["calls"] for s in seller_adapters)
            day_llm_calls += sum(b.llm_stats["calls"] for b in buyer_wrappers)
            day_llm_cost = sum(s.llm_stats["cost"] for s in seller_adapters)
            day_llm_cost += sum(b.llm_stats["cost"] for b in buyer_wrappers)
            
            results["days_completed"] = day
            results["total_deals"] += day_deals
            results["total_impressions"] += day_impressions
            results["total_spend"] += day_spend
            results["total_llm_calls"] = day_llm_calls
            results["total_llm_cost"] = day_llm_cost
            
            results["deals_by_day"].append({
                "day": day,
                "deals": day_deals,
                "impressions": day_impressions,
                "spend": day_spend,
            })
            
            logger.info(
                "day.completed",
                day=day,
                deals=day_deals,
                impressions=f"{day_impressions:,}",
                spend=f"${day_spend:.2f}",
            )
        
    finally:
        # Disconnect buyers
        for buyer in buyer_wrappers:
            if not mock_llm:
                try:
                    await buyer.disconnect()
                except Exception:
                    pass
    
    results["end_time"] = datetime.utcnow().isoformat()
    
    logger.info(f"\n{'='*60}")
    logger.info("SIMULATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Days: {results['days_completed']}")
    logger.info(f"Total deals: {results['total_deals']}")
    logger.info(f"Total impressions: {results['total_impressions']:,}")
    logger.info(f"Total spend: ${results['total_spend']:.2f}")
    logger.info(f"LLM calls: {results['total_llm_calls']}")
    logger.info(f"LLM cost: ${results['total_llm_cost']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run IAB simulation")
    parser.add_argument("--days", type=int, default=5, help="Number of simulation days")
    parser.add_argument("--buyers", type=int, default=3, help="Number of buyer agents")
    parser.add_argument("--sellers", type=int, default=3, help="Number of seller agents")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API costs)")
    parser.add_argument("--a2a", action="store_true", help="Use A2A natural language protocol")
    parser.add_argument("--output", type=str, help="Output file for results JSON")
    
    args = parser.parse_args()
    
    # Check for API key if not mock mode
    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set - running in mock mode")
        args.mock = True
    
    results = asyncio.run(run_simulation(
        days=args.days,
        buyers=args.buyers,
        sellers=args.sellers,
        mock_llm=args.mock,
        use_a2a=args.a2a,
    ))
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
