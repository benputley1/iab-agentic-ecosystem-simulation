"""
RTB Simulation Scenario Engines.

Implements three simulation scenarios for comparing programmatic advertising approaches:

Scenario A: Current State (Rent-Seeking Exchanges)
- Exchange agent intermediates all transactions
- Extracts 10-20% fees on each deal
- Runs second-price auctions
- Centralized state in exchange database

Scenario B: IAB Pure A2A (Direct Buyer-Seller)
- Direct agent-to-agent communication per IAB spec
- No exchange intermediary (passive infrastructure only)
- In-memory state with context rot simulation
- Hallucination injection for testing data integrity

Scenario C: Alkimi Ledger-Backed (Coming Soon)
- Same A2A flow as Scenario B
- Beads = Walrus blob proxy (immutable records)
- Internal ledger = Sui object proxy
"""

from .base import (
    BaseScenario,
    ScenarioConfig,
    ScenarioMetrics,
)
from .scenario_a import ScenarioA, run_scenario_a
from .scenario_b import ScenarioB

__all__ = [
    # Base classes
    "BaseScenario",
    "ScenarioConfig",
    "ScenarioMetrics",
    # Scenario A
    "ScenarioA",
    "run_scenario_a",
    # Scenario B
    "ScenarioB",
]
