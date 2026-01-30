"""
Context Recovery Mechanisms.

Provides recovery capabilities for lost context based on scenario:
- Scenario A: Exchange can recover ~60% via logs
- Scenario B: No recovery (0%)
- Scenario C: Full recovery via ledger (100%)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

if TYPE_CHECKING:
    from src.agents.base import AgentContext
    from .window import ContextEntry

logger = structlog.get_logger(__name__)


class RecoverySource(str, Enum):
    """Sources for context recovery."""
    LEDGER = "ledger"           # Full state from blockchain ledger
    EXCHANGE_LOGS = "exchange"  # Partial from exchange transaction logs
    PEER_AGENTS = "peers"       # Reconstructed from peer agent state
    NONE = "none"               # No recovery possible


class RecoveryResult(BaseModel):
    """Result of a context recovery attempt."""
    recovery_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    scenario: str  # "A", "B", or "C"
    source: RecoverySource
    
    # Recovery stats
    success: bool = False
    entries_requested: int = 0
    entries_recovered: int = 0
    tokens_requested: int = 0
    tokens_recovered: int = 0
    
    # Recovered items
    recovered_keys: list[str] = Field(default_factory=list)
    failed_keys: list[str] = Field(default_factory=list)
    
    # Quality metrics
    recovery_rate: float = 0.0  # entries_recovered / entries_requested
    fidelity: float = 1.0       # How accurate recovered data is (0-1)
    
    # Timing
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    
    # Error info
    error: str | None = None


class ContextRecovery:
    """
    Context recovery mechanisms.
    
    Provides different recovery capabilities based on scenario
    and available infrastructure.
    """
    
    def __init__(
        self,
        ledger_client: Any | None = None,
        exchange_client: Any | None = None,
        exchange_recovery_rate: float = 0.60,  # 60% recoverable from logs
        exchange_fidelity: float = 0.95,       # 95% accurate
        peer_recovery_rate: float = 0.30,      # 30% from peers
        peer_fidelity: float = 0.80            # 80% accurate
    ):
        """
        Initialize recovery system.
        
        Args:
            ledger_client: Client for ledger operations (Scenario C)
            exchange_client: Client for exchange log access (Scenario A)
            exchange_recovery_rate: Percentage recoverable from exchange
            exchange_fidelity: Accuracy of exchange recovery
            peer_recovery_rate: Percentage recoverable from peer agents
            peer_fidelity: Accuracy of peer recovery
        """
        self.ledger_client = ledger_client
        self.exchange_client = exchange_client
        self.exchange_recovery_rate = exchange_recovery_rate
        self.exchange_fidelity = exchange_fidelity
        self.peer_recovery_rate = peer_recovery_rate
        self.peer_fidelity = peer_fidelity
        
        self.log = structlog.get_logger(__name__)
        
        # Recovery history
        self._recovery_history: list[RecoveryResult] = []
    
    async def attempt_recovery(
        self,
        agent_id: str,
        lost_entries: list["ContextEntry"],
        scenario: str
    ) -> RecoveryResult:
        """
        Attempt to recover lost context.
        
        Routes to appropriate recovery mechanism based on scenario.
        
        Args:
            agent_id: Agent that lost context
            lost_entries: List of lost context entries
            scenario: "A", "B", or "C"
            
        Returns:
            RecoveryResult with recovery details
        """
        start_time = time.time()
        
        result = RecoveryResult(
            agent_id=agent_id,
            scenario=scenario,
            source=RecoverySource.NONE,
            entries_requested=len(lost_entries),
            tokens_requested=sum(e.tokens for e in lost_entries)
        )
        
        try:
            if scenario == "C":
                result = await self._recover_from_ledger(
                    agent_id, lost_entries, result
                )
            elif scenario == "A":
                result = await self._recover_from_exchange(
                    agent_id, lost_entries, result
                )
            else:
                # Scenario B - no recovery mechanism
                result.source = RecoverySource.NONE
                result.success = False
                result.recovery_rate = 0.0
                result.failed_keys = [e.entry_id for e in lost_entries]
                
                self.log.info(
                    "no_recovery_available",
                    agent_id=agent_id,
                    scenario=scenario,
                    entries_lost=len(lost_entries)
                )
                
        except Exception as e:
            result.error = str(e)
            self.log.error(
                "recovery_failed",
                agent_id=agent_id,
                scenario=scenario,
                error=str(e)
            )
        
        result.duration_ms = (time.time() - start_time) * 1000
        self._recovery_history.append(result)
        
        return result
    
    async def _recover_from_ledger(
        self,
        agent_id: str,
        lost_entries: list["ContextEntry"],
        result: RecoveryResult
    ) -> RecoveryResult:
        """
        Full recovery from ledger (Scenario C).
        
        Ledger contains complete state snapshots, enabling
        100% recovery with 100% fidelity.
        """
        result.source = RecoverySource.LEDGER
        
        # In Scenario C, everything is recoverable from ledger
        if self.ledger_client:
            # Real ledger recovery would go here
            recovered = await self.ledger_client.get_agent_state(agent_id)
            result.entries_recovered = len(lost_entries)
            result.tokens_recovered = sum(e.tokens for e in lost_entries)
        else:
            # Simulated ledger recovery
            result.entries_recovered = len(lost_entries)
            result.tokens_recovered = sum(e.tokens for e in lost_entries)
        
        result.success = True
        result.recovery_rate = 1.0
        result.fidelity = 1.0
        result.recovered_keys = [e.entry_id for e in lost_entries]
        
        self.log.info(
            "ledger_recovery_complete",
            agent_id=agent_id,
            entries_recovered=result.entries_recovered,
            tokens_recovered=result.tokens_recovered
        )
        
        return result
    
    async def _recover_from_exchange(
        self,
        agent_id: str,
        lost_entries: list["ContextEntry"],
        result: RecoveryResult
    ) -> RecoveryResult:
        """
        Partial recovery from exchange logs (Scenario A).
        
        Exchange logs contain transaction records that can be
        used to reconstruct ~60% of context with ~95% fidelity.
        """
        import random
        
        result.source = RecoverySource.EXCHANGE_LOGS
        
        recovered_count = 0
        recovered_tokens = 0
        
        for entry in lost_entries:
            # Probability of recovery depends on entry type
            # Transaction-related entries more likely to be in logs
            if entry.entry_type in ["transaction", "deal", "bid"]:
                recovery_prob = self.exchange_recovery_rate * 1.2
            else:
                recovery_prob = self.exchange_recovery_rate
            
            recovery_prob = min(1.0, recovery_prob)
            
            if random.random() < recovery_prob:
                recovered_count += 1
                recovered_tokens += entry.tokens
                result.recovered_keys.append(entry.entry_id)
            else:
                result.failed_keys.append(entry.entry_id)
        
        result.entries_recovered = recovered_count
        result.tokens_recovered = recovered_tokens
        result.success = recovered_count > 0
        result.recovery_rate = (
            recovered_count / len(lost_entries) if lost_entries else 0
        )
        result.fidelity = self.exchange_fidelity
        
        self.log.info(
            "exchange_recovery_complete",
            agent_id=agent_id,
            entries_recovered=recovered_count,
            recovery_rate=result.recovery_rate,
            fidelity=result.fidelity
        )
        
        return result
    
    async def recover_from_ledger(
        self,
        agent_id: str
    ) -> dict[str, Any]:
        """
        Full recovery from ledger (Scenario C).
        
        Retrieves complete agent context from the ledger.
        
        Args:
            agent_id: Agent to recover
            
        Returns:
            Full agent context as dict
        """
        self.log.info(
            "full_ledger_recovery_requested",
            agent_id=agent_id
        )
        
        if self.ledger_client:
            # Real ledger query
            return await self.ledger_client.get_agent_state(agent_id)
        
        # Simulated response
        return {
            "agent_id": agent_id,
            "recovered": True,
            "source": "ledger",
            "context": {},
            "timestamp": time.time()
        }
    
    async def recover_from_exchange_logs(
        self,
        agent_id: str,
        time_range_hours: float = 24.0
    ) -> dict[str, Any]:
        """
        Partial recovery from exchange logs (Scenario A).
        
        Reconstructs context from exchange transaction logs.
        Only ~60% of context can be recovered this way.
        
        Args:
            agent_id: Agent to recover
            time_range_hours: How far back to look in logs
            
        Returns:
            Partial agent context as dict
        """
        self.log.info(
            "exchange_log_recovery_requested",
            agent_id=agent_id,
            time_range_hours=time_range_hours
        )
        
        if self.exchange_client:
            # Real exchange log query
            logs = await self.exchange_client.get_agent_transactions(
                agent_id=agent_id,
                hours=time_range_hours
            )
            # Process logs to reconstruct context
            return self._reconstruct_from_logs(agent_id, logs)
        
        # Simulated response
        return {
            "agent_id": agent_id,
            "recovered": True,
            "partial": True,
            "source": "exchange_logs",
            "recovery_rate": self.exchange_recovery_rate,
            "fidelity": self.exchange_fidelity,
            "context": {},
            "timestamp": time.time()
        }
    
    def _reconstruct_from_logs(
        self,
        agent_id: str,
        logs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Reconstruct context from transaction logs.
        
        This is a simplified reconstruction - real implementation
        would parse log entries to rebuild state.
        """
        context: dict[str, Any] = {
            "agent_id": agent_id,
            "recovered": True,
            "partial": True,
            "source": "exchange_logs",
            "items": {}
        }
        
        for log_entry in logs:
            # Extract relevant context from each log
            if "context_key" in log_entry:
                context["items"][log_entry["context_key"]] = log_entry.get("value")
        
        context["recovery_rate"] = self.exchange_recovery_rate
        context["fidelity"] = self.exchange_fidelity
        
        return context
    
    async def recover_from_peers(
        self,
        agent_id: str,
        peer_agents: list[str]
    ) -> RecoveryResult:
        """
        Attempt recovery by querying peer agents.
        
        Peers may have copies of shared context that can
        help reconstruct lost information.
        
        Args:
            agent_id: Agent that lost context
            peer_agents: List of peer agent IDs to query
            
        Returns:
            RecoveryResult with peer recovery details
        """
        result = RecoveryResult(
            agent_id=agent_id,
            scenario="peer_recovery",
            source=RecoverySource.PEER_AGENTS
        )
        
        # Simulated peer recovery
        # In real implementation, would query each peer's shared context
        result.success = len(peer_agents) > 0
        result.recovery_rate = self.peer_recovery_rate if peer_agents else 0.0
        result.fidelity = self.peer_fidelity
        
        self.log.info(
            "peer_recovery_attempted",
            agent_id=agent_id,
            peer_count=len(peer_agents),
            recovery_rate=result.recovery_rate
        )
        
        self._recovery_history.append(result)
        return result
    
    def get_recovery_history(self) -> list[RecoveryResult]:
        """Get history of recovery attempts."""
        return self._recovery_history.copy()
    
    def get_success_rate(self) -> float:
        """Calculate overall recovery success rate."""
        if not self._recovery_history:
            return 0.0
        successes = sum(1 for r in self._recovery_history if r.success)
        return successes / len(self._recovery_history)
    
    def get_average_recovery_rate(self) -> float:
        """Calculate average context recovery rate."""
        if not self._recovery_history:
            return 0.0
        return sum(r.recovery_rate for r in self._recovery_history) / len(self._recovery_history)
    
    def clear_history(self) -> None:
        """Clear recovery history."""
        self._recovery_history.clear()
