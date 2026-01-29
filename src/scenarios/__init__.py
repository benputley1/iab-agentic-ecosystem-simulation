"""
Scenario engines for RTB simulation.

Three scenarios comparing programmatic advertising approaches:
- Scenario A: Current state with rent-seeking exchange (10-20% fee extraction)
- Scenario B: IAB Pure A2A (direct buyer<->seller, context rot simulation)
- Scenario C: Alkimi ledger-backed (Beads = immutable records, Sui/Walrus proxy)
"""

from .base import BaseScenario, ScenarioConfig, ScenarioMetrics
from .scenario_a import ScenarioA, run_scenario_a

__all__ = [
    "BaseScenario",
    "ScenarioConfig",
    "ScenarioMetrics",
    "ScenarioA",
    "run_scenario_a",
]
