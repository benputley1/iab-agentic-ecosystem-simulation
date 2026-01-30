"""
RTB Simulation Scenario Engines.

Implements scenarios for comparing programmatic advertising approaches:

Scenario A: Current State (Rent-Seeking Exchanges)
- Exchange agent intermediates all transactions
- Extracts 10-20% fees on each deal
- Runs second-price auctions
- Centralized state in exchange database
- Exchange serves as arbiter for disputes

Scenario B: IAB Pure A2A (Direct Buyer-Seller)
- Direct agent-to-agent communication per IAB spec
- No exchange intermediary (passive infrastructure only)
- **Private databases** - each agent maintains own state
- **No reconciliation mechanism** - disputes unresolvable
- Cross-agent state divergence over time

Scenario C: Alkimi Ledger-Backed
- Same A2A flow as Scenario B (no exchange fees)
- Beads = Walrus blob proxy (immutable records)
- Internal ledger = Sui object proxy
- **Shared source of truth** for reconciliation
- 100% dispute resolution via ledger arbitration

Key Research Focus (v2.0):
- Cross-agent reconciliation, not single-agent context rot
- Multi-agent state divergence with private databases
- Unresolvable disputes when no shared source of truth
"""

from .base import (
    BaseScenario,
    ScenarioConfig,
    ScenarioMetrics,
)
from .scenario_a import ScenarioA, run_scenario_a
from .scenario_b import ScenarioB
from .scenario_c import ScenarioC
from .reconciliation import (
    ReconciliationSimulator,
    ReconciliationEngine,
    DiscrepancyInjector,
    DiscrepancyConfig,
    ReconciliationMetrics,
    ReconciliationResult,
    CampaignRecord,
    DiscrepancySource,
    ResolutionOutcome,
)

# Multi-Agent Hierarchy Scenarios
from .multi_agent_scenario_a import (
    MultiAgentScenarioA,
    MultiAgentScenarioAConfig,
    DealCycleResult,
)
from .multi_agent_scenario_b import (
    MultiAgentScenarioB,
    MultiAgentScenarioBConfig,
    DirectDealResult,
)
from .multi_agent_scenario_c import (
    MultiAgentScenarioC,
    MultiAgentScenarioCConfig,
    LedgerClient,
    LedgerState,
    LedgerDealResult,
)

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
    # Scenario C
    "ScenarioC",
    # Reconciliation (new focus)
    "ReconciliationSimulator",
    "ReconciliationEngine",
    "DiscrepancyInjector",
    "DiscrepancyConfig",
    "ReconciliationMetrics",
    "ReconciliationResult",
    "CampaignRecord",
    "DiscrepancySource",
    "ResolutionOutcome",
    # Multi-Agent Hierarchy Scenarios
    "MultiAgentScenarioA",
    "MultiAgentScenarioAConfig",
    "DealCycleResult",
    "MultiAgentScenarioB",
    "MultiAgentScenarioBConfig",
    "DirectDealResult",
    "MultiAgentScenarioC",
    "MultiAgentScenarioCConfig",
    "LedgerClient",
    "LedgerState",
    "LedgerDealResult",
]
