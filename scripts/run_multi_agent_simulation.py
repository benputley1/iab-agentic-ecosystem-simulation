#!/usr/bin/env python3
"""
Multi-Agent Hierarchy Simulation Runner.

Runs comparative simulations across Scenarios A, B, and C with
full multi-agent hierarchies.

Usage:
    python scripts/run_multi_agent_simulation.py \\
        --days 30 \\
        --campaigns 5 \\
        --scenario a,b,c \\
        --output results.json

Examples:
    # Run all scenarios for 30 days with 5 campaigns
    python scripts/run_multi_agent_simulation.py --days 30 --campaigns 5
    
    # Run only scenario A for quick test
    python scripts/run_multi_agent_simulation.py --days 5 --campaigns 2 --scenario a
    
    # Run B vs C comparison
    python scripts/run_multi_agent_simulation.py --days 15 --scenario b,c --output bc_comparison.json
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scenarios.multi_agent_scenario_a import MultiAgentScenarioA, MultiAgentScenarioAConfig
from scenarios.multi_agent_scenario_b import MultiAgentScenarioB, MultiAgentScenarioBConfig
from scenarios.multi_agent_scenario_c import MultiAgentScenarioC, MultiAgentScenarioCConfig
from agents.buyer.models import (
    Campaign,
    CampaignObjectives,
    AudienceSpec,
    Channel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("multi_agent_runner")


def random_channel_mix() -> dict[str, float]:
    """Generate random channel mix that sums to 1.0."""
    channels = [Channel.DISPLAY.value, Channel.VIDEO.value, Channel.CTV.value, 
                Channel.MOBILE_APP.value, Channel.NATIVE.value]
    
    # Random selection of 2-4 channels
    n_channels = random.randint(2, 4)
    selected = random.sample(channels, n_channels)
    
    # Random weights
    weights = [random.random() for _ in selected]
    total = sum(weights)
    
    return {ch: w / total for ch, w in zip(selected, weights)}


def generate_audience_spec() -> AudienceSpec:
    """Generate a realistic audience specification."""
    segments = random.sample([
        "auto_intenders", "luxury_shoppers", "tech_enthusiasts",
        "sports_fans", "travel_planners", "health_conscious",
        "young_professionals", "parents", "gamers", "foodies",
    ], random.randint(2, 4))
    
    demographics = {
        "age_min": random.choice([18, 25, 35, 45]),
        "age_max": random.choice([45, 55, 65, 100]),
        "gender": random.choice(["all", "male", "female"]),
        "income": random.choice(["all", "high", "medium", "low"]),
    }
    
    geo_targets = random.sample([
        "US", "UK", "CA", "DE", "FR", "JP", "AU", "BR",
    ], random.randint(1, 3))
    
    device_types = random.sample([
        "desktop", "mobile", "tablet", "ctv",
    ], random.randint(2, 4))
    
    return AudienceSpec(
        segments=segments,
        demographics=demographics,
        geo_targets=geo_targets,
        device_types=device_types,
    )


def generate_test_campaigns(n: int, seed: Optional[int] = None) -> list[Campaign]:
    """Generate n test campaigns with realistic parameters.
    
    Args:
        n: Number of campaigns to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of Campaign objects
    """
    if seed is not None:
        random.seed(seed)
    
    advertisers = [
        "TechCorp", "AutoMaker", "RetailGiant", "FinanceHub", "TravelCo",
        "FoodBrand", "SportsFan", "HealthFirst", "EntertainCo", "EduLearn",
    ]
    
    campaigns = []
    start_date = date.today()
    
    for i in range(n):
        campaign_id = f"camp-{i:03d}"
        
        # Random budget between $50k and $500k
        total_budget = random.uniform(50000, 500000)
        
        # Random objectives
        objectives = CampaignObjectives(
            reach_target=random.randint(100000, 1000000),
            frequency_cap=random.randint(2, 5),
            cpm_target=random.uniform(10, 35),
            channel_mix=random_channel_mix(),
            viewability_target=random.uniform(0.6, 0.85),
            brand_safety_level=random.choice(["standard", "high", "very_high"]),
        )
        
        # Campaign duration: 14-45 days
        duration_days = random.randint(14, 45)
        end_date = start_date + timedelta(days=duration_days)
        
        campaign = Campaign(
            campaign_id=campaign_id,
            name=f"Test Campaign {i}",
            advertiser=advertisers[i % len(advertisers)],
            total_budget=total_budget,
            start_date=start_date,
            end_date=end_date,
            objectives=objectives,
            audience=generate_audience_spec(),
            priority=random.randint(1, 3),
        )
        
        campaigns.append(campaign)
        logger.debug(f"Generated campaign {campaign_id}: ${total_budget:,.2f}")
    
    return campaigns


def create_scenario(
    scenario_code: str,
    num_buyers: int = 3,
    num_sellers: int = 3,
    mock_llm: bool = True,
    seed: Optional[int] = None,
):
    """Create a scenario instance.
    
    Args:
        scenario_code: Scenario code (a, b, or c)
        num_buyers: Number of buyer systems
        num_sellers: Number of seller systems
        mock_llm: Use mock LLM
        seed: Random seed
        
    Returns:
        Scenario instance
    """
    code = scenario_code.lower()
    
    if code == "a":
        return MultiAgentScenarioA(
            config=MultiAgentScenarioAConfig(),
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            mock_llm=mock_llm,
            seed=seed,
        )
    elif code == "b":
        return MultiAgentScenarioB(
            config=MultiAgentScenarioBConfig(),
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            mock_llm=mock_llm,
            seed=seed,
        )
    elif code == "c":
        return MultiAgentScenarioC(
            config=MultiAgentScenarioCConfig(),
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            mock_llm=mock_llm,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario_code}")


async def run_scenario(
    scenario,
    campaigns: list[Campaign],
    days: int,
) -> dict[str, Any]:
    """Run a single scenario.
    
    Args:
        scenario: Scenario instance
        campaigns: Campaigns to run
        days: Number of simulation days
        
    Returns:
        Results dictionary
    """
    scenario_name = scenario.scenario_name
    logger.info(f"Starting scenario: {scenario_name}")
    
    try:
        # Setup
        await scenario.setup()
        
        # Add campaigns
        scenario.add_campaigns(campaigns)
        
        # Run simulation
        all_deals = []
        daily_results = []
        
        for day in range(1, days + 1):
            logger.info(f"[{scenario_name}] Day {day}/{days}")
            
            day_deals = await scenario.run_day(day)
            all_deals.extend(day_deals)
            
            daily_results.append({
                "day": day,
                "deals": len(day_deals),
                "impressions": sum(d.impressions for d in day_deals),
                "spend": sum(d.total_cost for d in day_deals),
                "revenue": sum(d.seller_revenue for d in day_deals),
                "fees": sum(d.exchange_fee for d in day_deals),
            })
        
        # Collect results
        metrics = scenario.metrics.to_dict()
        hierarchy_metrics = scenario.get_hierarchy_metrics()
        
        results = {
            "scenario": scenario.scenario_id,
            "scenario_name": scenario_name,
            "days_run": days,
            "campaigns": len(campaigns),
            "metrics": metrics,
            "hierarchy_metrics": hierarchy_metrics,
            "daily_results": daily_results,
            "summary": {
                "total_deals": metrics.get("total_deals", 0),
                "total_impressions": metrics.get("total_impressions", 0),
                "total_spend": metrics.get("total_buyer_spend", 0),
                "total_revenue": metrics.get("total_seller_revenue", 0),
                "total_fees": metrics.get("total_exchange_fees", 0),
                "average_cpm": metrics.get("average_cpm", 0),
                "fee_rate": metrics.get("intermediary_take_rate", 0),
                "context_rot_events": metrics.get("context_rot_events", 0),
                "recovery_rate": metrics.get("recovery_success_rate", 0),
            },
        }
        
        return results
        
    finally:
        await scenario.teardown()


async def run_comparison(
    scenario_codes: list[str],
    campaigns: list[Campaign],
    days: int,
    num_buyers: int = 3,
    num_sellers: int = 3,
    mock_llm: bool = True,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run comparison across multiple scenarios.
    
    Args:
        scenario_codes: List of scenario codes to run
        campaigns: Campaigns to use for all scenarios
        days: Number of simulation days
        num_buyers: Number of buyer systems
        num_sellers: Number of seller systems
        mock_llm: Use mock LLM
        seed: Random seed
        
    Returns:
        Comparison results
    """
    logger.info(f"Running comparison: scenarios={scenario_codes}, days={days}, campaigns={len(campaigns)}")
    
    results = {}
    
    for code in scenario_codes:
        # Create fresh copy of campaigns for each scenario
        campaign_copies = [
            Campaign(
                campaign_id=c.campaign_id,
                name=c.name,
                advertiser=c.advertiser,
                total_budget=c.total_budget,
                start_date=c.start_date,
                end_date=c.end_date,
                objectives=c.objectives,
                audience=c.audience,
                priority=c.priority,
            )
            for c in campaigns
        ]
        
        scenario = create_scenario(
            code,
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            mock_llm=mock_llm,
            seed=seed,
        )
        
        scenario_results = await run_scenario(scenario, campaign_copies, days)
        results[code.upper()] = scenario_results
    
    # Generate comparison summary
    comparison = generate_comparison_summary(results)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "scenarios": scenario_codes,
            "days": days,
            "campaigns": len(campaigns),
            "num_buyers": num_buyers,
            "num_sellers": num_sellers,
            "mock_llm": mock_llm,
            "seed": seed,
        },
        "scenario_results": results,
        "comparison": comparison,
    }


def generate_comparison_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Generate comparison summary across scenarios.
    
    Args:
        results: Results from each scenario
        
    Returns:
        Comparison summary
    """
    comparison = {
        "metrics": {},
        "rankings": {},
        "insights": [],
    }
    
    # Extract key metrics for comparison
    metrics_to_compare = [
        "total_deals", "total_impressions", "total_spend", 
        "total_revenue", "total_fees", "average_cpm", 
        "fee_rate", "context_rot_events", "recovery_rate",
    ]
    
    for metric in metrics_to_compare:
        comparison["metrics"][metric] = {}
        for scenario_code, scenario_results in results.items():
            value = scenario_results.get("summary", {}).get(metric, 0)
            comparison["metrics"][metric][scenario_code] = value
    
    # Generate rankings
    for metric in ["total_impressions", "total_revenue", "recovery_rate"]:
        values = comparison["metrics"].get(metric, {})
        if values:
            ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric] = [s[0] for s in ranked]
    
    # Lower is better for fees
    for metric in ["fee_rate", "context_rot_events"]:
        values = comparison["metrics"].get(metric, {})
        if values:
            ranked = sorted(values.items(), key=lambda x: x[1])  # Ascending
            comparison["rankings"][metric] = [s[0] for s in ranked]
    
    # Generate insights
    if "A" in results and "B" in results:
        a_fees = results["A"]["summary"].get("fee_rate", 0)
        b_fees = results["B"]["summary"].get("fee_rate", 0)
        comparison["insights"].append(
            f"Scenario A has {a_fees:.1f}% fees vs Scenario B's {b_fees:.1f}% (direct A2A)"
        )
    
    if "A" in results and "C" in results:
        a_fees = results["A"]["summary"].get("fee_rate", 0)
        c_fees = results["C"]["summary"].get("fee_rate", 0)
        fee_savings = a_fees - c_fees
        if fee_savings > 0:
            comparison["insights"].append(
                f"Scenario C saves {fee_savings:.1f}% in fees vs traditional exchange"
            )
    
    if "B" in results and "C" in results:
        b_rot = results["B"]["summary"].get("context_rot_events", 0)
        c_rot = results["C"]["summary"].get("context_rot_events", 0)
        comparison["insights"].append(
            f"Scenario B: {b_rot} context rot events vs C: {c_rot} (ledger recovery)"
        )
    
    return comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Hierarchy Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --days 30 --campaigns 5
  %(prog)s --days 5 --scenario a --mock
  %(prog)s --days 15 --scenario b,c --output comparison.json
        """,
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of simulation days (default: 30)",
    )
    
    parser.add_argument(
        "--campaigns",
        type=int,
        default=5,
        help="Number of campaigns to generate (default: 5)",
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default="a,b,c",
        help="Comma-separated scenarios to run: a, b, c (default: a,b,c)",
    )
    
    parser.add_argument(
        "--buyers",
        type=int,
        default=3,
        help="Number of buyer systems (default: 3)",
    )
    
    parser.add_argument(
        "--sellers",
        type=int,
        default=3,
        help="Number of seller systems (default: 3)",
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock LLM (default: True)",
    )
    
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM (requires API keys)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="multi_agent_results.json",
        help="Output file path (default: multi_agent_results.json)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse scenarios
    scenario_codes = [s.strip() for s in args.scenario.split(",")]
    for code in scenario_codes:
        if code.lower() not in ["a", "b", "c"]:
            logger.error(f"Invalid scenario: {code}")
            sys.exit(1)
    
    # Use real LLM if specified
    mock_llm = not args.real_llm
    
    logger.info("=" * 60)
    logger.info("Multi-Agent Hierarchy Simulation")
    logger.info("=" * 60)
    logger.info(f"Scenarios: {scenario_codes}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Campaigns: {args.campaigns}")
    logger.info(f"Buyers: {args.buyers}")
    logger.info(f"Sellers: {args.sellers}")
    logger.info(f"Mock LLM: {mock_llm}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)
    
    # Generate campaigns
    campaigns = generate_test_campaigns(args.campaigns, seed=args.seed)
    logger.info(f"Generated {len(campaigns)} test campaigns")
    
    # Run comparison
    results = await run_comparison(
        scenario_codes=scenario_codes,
        campaigns=campaigns,
        days=args.days,
        num_buyers=args.buyers,
        num_sellers=args.sellers,
        mock_llm=mock_llm,
        seed=args.seed,
    )
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison = results.get("comparison", {})
    
    # Print metrics table
    print("\nKey Metrics by Scenario:")
    print("-" * 60)
    
    metrics = comparison.get("metrics", {})
    scenarios = list(results.get("scenario_results", {}).keys())
    
    print(f"{'Metric':<25} " + " ".join(f"{s:>12}" for s in scenarios))
    print("-" * 60)
    
    for metric, values in metrics.items():
        row = f"{metric:<25} "
        for s in scenarios:
            v = values.get(s, 0)
            if isinstance(v, float):
                row += f"{v:>12.2f}"
            else:
                row += f"{v:>12,}"
        print(row)
    
    # Print insights
    insights = comparison.get("insights", [])
    if insights:
        print("\nKey Insights:")
        print("-" * 60)
        for insight in insights:
            print(f"• {insight}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
