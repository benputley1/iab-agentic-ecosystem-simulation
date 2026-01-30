"""
Multi-Campaign State Management.

Provides state management for concurrent campaigns with support for:
- Portfolio-level state tracking
- Cross-campaign coordination
- Volatile (in-memory) state for Scenario B
- Ledger-backed state for Scenario C
- Agent state synchronization

Key Research Focus:
- Context rot across multiple concurrent campaigns
- State divergence between agents with private databases
- Reconciliation via shared ledger vs volatile state
"""

from .campaign_portfolio import (
    CampaignPortfolio,
    CampaignState,
    CampaignMetrics,
    PortfolioView,
    StateUpdate,
    Conflict,
    Deal,
)
from .cross_campaign import (
    CrossCampaignState,
    Commit,
    PacingState,
    ContentionResult,
)
from .volatile import (
    VolatileStateManager,
)
from .ledger_backed import (
    LedgerBackedStateManager,
    StateSnapshot,
    VerificationResult,
)
from .sync import (
    StateSync,
    SyncResult,
    Divergence,
)

__all__ = [
    # Campaign Portfolio
    "CampaignPortfolio",
    "CampaignState",
    "CampaignMetrics",
    "PortfolioView",
    "StateUpdate",
    "Conflict",
    "Deal",
    # Cross-Campaign
    "CrossCampaignState",
    "Commit",
    "PacingState",
    "ContentionResult",
    # Volatile State (Scenario B)
    "VolatileStateManager",
    # Ledger-Backed State (Scenario C)
    "LedgerBackedStateManager",
    "StateSnapshot",
    "VerificationResult",
    # State Sync
    "StateSync",
    "SyncResult",
    "Divergence",
]
