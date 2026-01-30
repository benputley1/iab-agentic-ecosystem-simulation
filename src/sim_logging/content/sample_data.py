"""
Sample data generator for demonstrating content generation.

Generates realistic simulation events to demonstrate the content
generation pipeline without running the full simulation.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

from ..events import (
    EventIndex,
    SimulationEvent,
    EventType,
    Scenario,
    create_bid_request_event,
    create_deal_event,
    create_context_rot_event,
    create_hallucination_event,
    create_state_recovery_event,
    create_fee_extraction_event,
    create_blockchain_cost_event,
    create_day_summary_event,
)


# Realistic parameters based on IMPLEMENTATION_PLAN.md
NUM_BUYERS = 5
NUM_SELLERS = 5
CAMPAIGNS_PER_BUYER = 10
SIMULATION_DAYS = 30
EXCHANGE_FEE_RATE = 0.15  # 15%

# Context rot parameters (Scenario B)
BASE_DECAY_RATE = 0.1  # 10% daily chance of decay
RESTART_PROBABILITY = 0.05  # 5% chance of full restart
HALLUCINATION_PROBABILITY = 0.3  # 30% chance after context loss

# Blockchain costs (Scenario C)
SUI_GAS_PER_TX = 0.001  # $0.001 per transaction
WALRUS_PER_KB = 0.0001  # $0.0001 per KB stored


def generate_sample_simulation(
    days: int = SIMULATION_DAYS,
    buyers: int = NUM_BUYERS,
    sellers: int = NUM_SELLERS,
    campaigns_per_buyer: int = CAMPAIGNS_PER_BUYER,
    seed: Optional[int] = 42,
) -> EventIndex:
    """
    Generate a complete sample simulation for all three scenarios.

    Creates realistic event data that demonstrates the key differences
    between scenarios A, B, and C.

    Args:
        days: Number of simulation days
        buyers: Number of buyer agents
        sellers: Number of seller agents
        campaigns_per_buyer: Campaigns per buyer
        seed: Random seed for reproducibility

    Returns:
        EventIndex populated with sample events
    """
    if seed is not None:
        random.seed(seed)

    index = EventIndex()
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    # Generate campaign definitions
    campaigns = []
    for buyer_idx in range(buyers):
        buyer_id = f"buyer-{buyer_idx + 1:03d}"
        for camp_idx in range(campaigns_per_buyer):
            campaign = {
                "campaign_id": f"camp-{buyer_idx + 1:03d}-{camp_idx + 1:03d}",
                "buyer_id": buyer_id,
                "correlation_id": f"corr-{camp_idx + 1:03d}",  # Same campaign tracked across scenarios
                "target_impressions": random.randint(500_000, 5_000_000),
                "max_cpm": random.uniform(2.0, 8.0),
                "start_day": random.randint(0, days // 2),
            }
            campaigns.append(campaign)

    # Generate events for each scenario
    for scenario in [Scenario.A, Scenario.B, Scenario.C]:
        _generate_scenario_events(
            index=index,
            scenario=scenario,
            campaigns=campaigns,
            sellers=sellers,
            days=days,
            start_time=start_time,
        )

    return index


def _generate_scenario_events(
    index: EventIndex,
    scenario: Scenario,
    campaigns: list[dict],
    sellers: int,
    days: int,
    start_time: datetime,
) -> None:
    """Generate events for a single scenario."""

    # Track agent state for context rot
    agent_context_health = {f"buyer-{i + 1:03d}": 1.0 for i in range(5)}
    agent_context_health.update({f"seller-{i + 1:03d}": 1.0 for i in range(sellers)})

    # Track campaign progress
    campaign_progress = {c["campaign_id"]: 0 for c in campaigns}

    # Scenario start event
    index.add(SimulationEvent(
        event_type=EventType.SCENARIO_STARTED,
        scenario=scenario,
        timestamp=start_time,
        simulation_day=0,
        narrative_summary=f"Scenario {scenario.value} simulation started",
    ))

    for day in range(days):
        day_time = start_time + timedelta(days=day)
        day_deals = 0
        day_spend = 0.0
        day_fees = 0.0
        day_context_losses = 0
        day_hallucinations = 0
        day_blockchain_costs = 0.0

        # Simulate context rot for Scenario B
        if scenario == Scenario.B:
            for agent_id, health in list(agent_context_health.items()):
                if random.random() < BASE_DECAY_RATE * (day / days + 0.5):
                    # Context decay
                    keys_lost = random.randint(5, 50)
                    is_restart = random.random() < RESTART_PROBABILITY

                    if is_restart:
                        keys_lost = random.randint(100, 500)
                        agent_context_health[agent_id] = 0.1
                    else:
                        agent_context_health[agent_id] = max(0.1, health - 0.1)

                    event = create_context_rot_event(
                        agent_id=agent_id,
                        keys_lost=keys_lost,
                        survival_rate=agent_context_health[agent_id],
                        simulation_day=day,
                        is_restart=is_restart,
                    )
                    event.timestamp = day_time + timedelta(hours=random.randint(0, 23))
                    index.add(event)
                    day_context_losses += 1

                    # Hallucination after context loss
                    if random.random() < HALLUCINATION_PROBABILITY:
                        claim_types = [
                            ("deal_history", f"5 deals with seller-{random.randint(1, sellers):03d}", "0 deals"),
                            ("price_floor", f"${random.uniform(1, 3):.2f}", f"${random.uniform(3, 6):.2f}"),
                            ("inventory_level", f"{random.randint(1000000, 5000000):,}", "0"),
                        ]
                        claim_type, claimed, actual = random.choice(claim_types)

                        event = create_hallucination_event(
                            scenario=scenario,
                            agent_id=agent_id,
                            agent_type="buyer" if "buyer" in agent_id else "seller",
                            claim_type=claim_type,
                            claimed_value=claimed,
                            actual_value=actual,
                            simulation_day=day,
                            impact_description="Led to suboptimal bidding decision",
                        )
                        event.timestamp = day_time + timedelta(hours=random.randint(0, 23))
                        index.add(event)
                        day_hallucinations += 1

        # State recovery for Scenario C (periodically demonstrate)
        if scenario == Scenario.C and day % 7 == 0 and day > 0:
            for agent_id in list(agent_context_health.keys())[:2]:
                records = random.randint(50, 200)
                event = create_state_recovery_event(
                    agent_id=agent_id,
                    agent_type="buyer" if "buyer" in agent_id else "seller",
                    simulation_day=day,
                    records_recovered=records,
                    recovery_accuracy=1.0,  # 100% for ledger
                    source="alkimi_ledger",
                )
                event.timestamp = day_time + timedelta(hours=6)
                index.add(event)

        # Generate deals for active campaigns
        for campaign in campaigns:
            if campaign["start_day"] > day:
                continue
            if campaign_progress[campaign["campaign_id"]] >= campaign["target_impressions"]:
                continue

            # Probability of deal today
            if random.random() > 0.7:
                continue

            # Select seller
            seller_id = f"seller-{random.randint(1, sellers):03d}"

            # Generate impressions (affected by context rot in Scenario B)
            base_impressions = random.randint(50_000, 500_000)

            # Context rot degrades deal efficiency
            if scenario == Scenario.B:
                buyer_health = agent_context_health.get(campaign["buyer_id"], 0.5)
                seller_health = agent_context_health.get(seller_id, 0.5)
                efficiency = (buyer_health + seller_health) / 2
                impressions = int(base_impressions * efficiency)
            else:
                impressions = base_impressions

            impressions = min(impressions, campaign["target_impressions"] - campaign_progress[campaign["campaign_id"]])
            if impressions <= 0:
                continue

            campaign_progress[campaign["campaign_id"]] += impressions

            # Calculate costs
            cpm = random.uniform(2.0, campaign["max_cpm"])
            total_cost = (impressions / 1000) * cpm

            # Exchange fee (Scenario A only)
            exchange_fee = total_cost * EXCHANGE_FEE_RATE if scenario == Scenario.A else 0.0

            # Create deal event
            deal_id = str(uuid.uuid4())[:8]
            request_id = str(uuid.uuid4())[:8]

            deal_event = create_deal_event(
                scenario=scenario,
                deal_id=deal_id,
                request_id=request_id,
                buyer_id=campaign["buyer_id"],
                seller_id=seller_id,
                campaign_id=campaign["campaign_id"],
                impressions=impressions,
                cpm=cpm,
                total_cost=total_cost,
                exchange_fee=exchange_fee,
                simulation_day=day,
                ledger_entry_id=f"ledger-{deal_id}" if scenario == Scenario.C else None,
                correlation_id=campaign["correlation_id"],
            )
            deal_event.timestamp = day_time + timedelta(hours=random.randint(8, 20))
            index.add(deal_event)

            day_deals += 1
            day_spend += total_cost
            day_fees += exchange_fee

            # Fee extraction event for Scenario A
            if scenario == Scenario.A:
                fee_event = create_fee_extraction_event(
                    deal_id=deal_id,
                    buyer_id=campaign["buyer_id"],
                    seller_id=seller_id,
                    gross_amount=total_cost,
                    fee_amount=exchange_fee,
                    fee_percentage=EXCHANGE_FEE_RATE * 100,
                    simulation_day=day,
                )
                fee_event.timestamp = deal_event.timestamp + timedelta(seconds=1)
                index.add(fee_event)

            # Blockchain cost event for Scenario C
            if scenario == Scenario.C:
                payload_bytes = random.randint(200, 500)
                blockchain_cost = SUI_GAS_PER_TX + (payload_bytes / 1024) * WALRUS_PER_KB

                cost_event = create_blockchain_cost_event(
                    transaction_id=f"tx-{deal_id}",
                    transaction_type="deal_record",
                    payload_bytes=payload_bytes,
                    sui_gas=SUI_GAS_PER_TX,
                    walrus_cost=(payload_bytes / 1024) * WALRUS_PER_KB,
                    total_usd=blockchain_cost,
                    simulation_day=day,
                )
                cost_event.timestamp = deal_event.timestamp + timedelta(seconds=2)
                index.add(cost_event)
                day_blockchain_costs += blockchain_cost

        # Day summary event
        summary_event = create_day_summary_event(
            scenario=scenario,
            simulation_day=day,
            deals_count=day_deals,
            total_spend=day_spend,
            total_fees=day_fees,
            context_losses=day_context_losses,
            hallucinations=day_hallucinations,
            blockchain_costs=day_blockchain_costs,
        )
        summary_event.timestamp = day_time + timedelta(hours=23, minutes=59)
        index.add(summary_event)

    # Scenario complete event
    index.add(SimulationEvent(
        event_type=EventType.SCENARIO_COMPLETED,
        scenario=scenario,
        timestamp=start_time + timedelta(days=days),
        simulation_day=days,
        narrative_summary=f"Scenario {scenario.value} simulation completed after {days} days",
    ))
