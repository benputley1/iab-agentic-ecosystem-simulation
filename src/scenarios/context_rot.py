"""
Shared Context Rot Simulation for RTB Scenarios.

This module provides context rot simulation that can be used across
Scenario A and B, with configurable parameters and recovery mechanisms.

Context rot models the reality that AI agents:
1. Have limited context windows
2. Lose state on restarts
3. May hallucinate based on stale/corrupted embeddings
4. Cannot perfectly recall transaction history

Key insight: BOTH exchange-mediated (A) and direct A2A (B) scenarios
involve AI agents that suffer context rot. The difference is:
- Scenario A: Exchange provides partial verification/correction
- Scenario B: No verification layer, errors compound
- Scenario C: Ledger provides perfect recovery (zero rot)
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import structlog

logger = structlog.get_logger()


class RecoverySource(str, Enum):
    """Sources for context recovery."""
    NONE = "none"              # No recovery attempted/available
    EXCHANGE = "exchange"      # Exchange transaction records
    LEDGER = "ledger"          # Blockchain ledger (Scenario C)
    PEER = "peer"              # Peer-to-peer state sharing
    CHECKPOINT = "checkpoint"  # Local checkpoint files


@dataclass
class ContextRotConfig:
    """Configuration for context rot simulation."""

    # Base decay rate per simulation day
    # 2% default = ~55% context remains after 30 days
    decay_rate: float = 0.02

    # Probability of full context wipe (agent restart/crash)
    restart_probability: float = 0.005

    # Maximum memory items an agent can retain
    max_memory_items: int = 100

    # Days before memory starts decaying (grace period)
    grace_period_days: int = 3

    # Recovery configuration (varies by scenario)
    recovery_source: RecoverySource = RecoverySource.NONE
    recovery_accuracy: float = 0.0  # 0-1, how much can be recovered


@dataclass
class AgentMemory:
    """
    In-memory state for an agent subject to context rot.
    
    This memory is volatile and subject to decay/loss.
    Used in Scenario A and B to track what agents "remember".
    """

    agent_id: str
    agent_type: str  # "buyer" or "seller"

    # Transaction history (limited, decaying)
    deal_history: dict[str, dict] = field(default_factory=dict)

    # Negotiation context (ephemeral)
    pending_requests: dict[str, dict] = field(default_factory=dict)
    pending_responses: dict[str, dict] = field(default_factory=dict)

    # Partner relationship memory
    partner_reputation: dict[str, float] = field(default_factory=dict)
    partner_history: dict[str, list[str]] = field(default_factory=dict)

    # Context rot tracking
    rot_events: int = 0
    keys_lost_total: int = 0
    last_rot_day: int = 0
    
    # Recovery tracking
    recovery_events: int = 0
    keys_recovered_total: int = 0
    last_recovery_day: int = 0

    def memory_size(self) -> int:
        """Calculate total memory items."""
        return (
            len(self.deal_history)
            + len(self.pending_requests)
            + len(self.pending_responses)
            + len(self.partner_reputation)
        )

    def to_dict(self) -> dict:
        """Serialize memory state."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "deal_count": len(self.deal_history),
            "pending_requests": len(self.pending_requests),
            "pending_responses": len(self.pending_responses),
            "partner_count": len(self.partner_reputation),
            "rot_events": self.rot_events,
            "keys_lost_total": self.keys_lost_total,
            "recovery_events": self.recovery_events,
            "keys_recovered_total": self.keys_recovered_total,
        }


@dataclass
class ContextRotEvent:
    """Record of a context rot event."""
    agent_id: str
    agent_type: str
    simulation_day: int
    is_restart: bool  # True = full wipe, False = gradual decay
    keys_lost: int
    keys_lost_names: list[str]
    recovery_attempted: bool
    recovery_successful: bool
    keys_recovered: int
    recovery_source: RecoverySource
    recovery_accuracy: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ContextRotSimulator:
    """
    Simulates context loss over time for AI agents.

    Models the degradation that occurs when:
    - Agents have limited context windows
    - No external state persistence exists
    - Memory compaction or restarts occur
    
    Used in BOTH Scenario A and B with different recovery configurations.
    """

    def __init__(
        self,
        config: Optional[ContextRotConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or ContextRotConfig()
        self._random = random.Random(seed)
        self._events: list[ContextRotEvent] = []

    def apply_daily_decay(
        self,
        memory: AgentMemory,
        simulation_day: int,
    ) -> ContextRotEvent:
        """
        Apply daily context decay to agent memory.

        Probability of losing each memory item increases with:
        - Number of days passed
        - Total memory size

        Args:
            memory: Agent's memory to decay
            simulation_day: Current simulation day

        Returns:
            ContextRotEvent describing what happened
        """
        # Skip during grace period
        if simulation_day <= self.config.grace_period_days:
            return ContextRotEvent(
                agent_id=memory.agent_id,
                agent_type=memory.agent_type,
                simulation_day=simulation_day,
                is_restart=False,
                keys_lost=0,
                keys_lost_names=[],
                recovery_attempted=False,
                recovery_successful=False,
                keys_recovered=0,
                recovery_source=RecoverySource.NONE,
                recovery_accuracy=1.0,
            )

        # Calculate survival rate for this day
        # By day 30: ~55% of original context remains (0.98^30 ≈ 0.545)
        days_active = simulation_day - self.config.grace_period_days
        survival_rate = (1 - self.config.decay_rate) ** days_active

        lost_keys = []

        # Decay deal history
        for deal_id in list(memory.deal_history.keys()):
            if self._random.random() > survival_rate:
                del memory.deal_history[deal_id]
                lost_keys.append(f"deal:{deal_id[:8]}")

        # Decay partner reputation (older relationships decay faster)
        for partner_id in list(memory.partner_reputation.keys()):
            if self._random.random() > survival_rate * 0.9:
                del memory.partner_reputation[partner_id]
                lost_keys.append(f"reputation:{partner_id[:8]}")

        # Clear stale pending items (aggressive decay)
        for req_id in list(memory.pending_requests.keys()):
            if self._random.random() > survival_rate * 0.8:
                del memory.pending_requests[req_id]
                lost_keys.append(f"pending:{req_id[:8]}")

        # Attempt recovery if configured
        keys_recovered = 0
        recovery_attempted = len(lost_keys) > 0 and self.config.recovery_source != RecoverySource.NONE
        recovery_successful = False
        
        if recovery_attempted:
            # Recovery catches some percentage of lost keys
            recoverable = int(len(lost_keys) * self.config.recovery_accuracy)
            keys_recovered = recoverable
            recovery_successful = recoverable > 0
            
            # Note: We don't actually restore the keys here because the
            # recovery is partial - agent has to re-fetch from exchange/ledger
            # This models the overhead of recovery operations

        # Update tracking
        if lost_keys:
            memory.rot_events += 1
            memory.keys_lost_total += len(lost_keys) - keys_recovered
            memory.last_rot_day = simulation_day
            
            if keys_recovered > 0:
                memory.recovery_events += 1
                memory.keys_recovered_total += keys_recovered
                memory.last_recovery_day = simulation_day

            logger.warning(
                "context_rot.decay",
                agent_id=memory.agent_id,
                day=simulation_day,
                keys_lost=len(lost_keys),
                keys_recovered=keys_recovered,
                net_loss=len(lost_keys) - keys_recovered,
                survival_rate=f"{survival_rate:.2%}",
                recovery_source=self.config.recovery_source.value,
            )

        event = ContextRotEvent(
            agent_id=memory.agent_id,
            agent_type=memory.agent_type,
            simulation_day=simulation_day,
            is_restart=False,
            keys_lost=len(lost_keys),
            keys_lost_names=lost_keys,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful,
            keys_recovered=keys_recovered,
            recovery_source=self.config.recovery_source,
            recovery_accuracy=self.config.recovery_accuracy,
        )
        self._events.append(event)
        
        return event

    def check_restart(
        self,
        memory: AgentMemory,
        simulation_day: int,
    ) -> Optional[ContextRotEvent]:
        """
        Check if agent experiences a restart (full context wipe).

        Simulates scenarios where:
        - Agent process crashes and restarts
        - Context window is exceeded and truncated
        - Session expires

        Args:
            memory: Agent's memory
            simulation_day: Current simulation day

        Returns:
            ContextRotEvent if restart occurred, None otherwise
        """
        # Restart probability increases slightly over time
        adjusted_prob = self.config.restart_probability * (1 + simulation_day * 0.01)

        if self._random.random() >= adjusted_prob:
            return None

        # Full context wipe
        keys_lost = memory.memory_size()
        lost_keys_names = [
            f"deal:{k[:8]}" for k in memory.deal_history.keys()
        ] + [
            f"reputation:{k[:8]}" for k in memory.partner_reputation.keys()
        ] + [
            f"pending:{k[:8]}" for k in memory.pending_requests.keys()
        ]
        
        memory.deal_history.clear()
        memory.pending_requests.clear()
        memory.pending_responses.clear()
        memory.partner_reputation.clear()
        memory.partner_history.clear()

        # Attempt recovery
        keys_recovered = 0
        recovery_attempted = self.config.recovery_source != RecoverySource.NONE
        recovery_successful = False
        
        if recovery_attempted:
            # Recovery can restore significant portion after restart
            # But not everything - some context is truly ephemeral
            recoverable = int(keys_lost * self.config.recovery_accuracy * 0.8)
            keys_recovered = recoverable
            recovery_successful = recoverable > 0

        memory.rot_events += 1
        memory.keys_lost_total += keys_lost - keys_recovered
        
        if keys_recovered > 0:
            memory.recovery_events += 1
            memory.keys_recovered_total += keys_recovered

        logger.error(
            "context_rot.restart",
            agent_id=memory.agent_id,
            day=simulation_day,
            keys_lost=keys_lost,
            keys_recovered=keys_recovered,
            net_loss=keys_lost - keys_recovered,
            recovery_source=self.config.recovery_source.value,
        )

        event = ContextRotEvent(
            agent_id=memory.agent_id,
            agent_type=memory.agent_type,
            simulation_day=simulation_day,
            is_restart=True,
            keys_lost=keys_lost,
            keys_lost_names=lost_keys_names,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful,
            keys_recovered=keys_recovered,
            recovery_source=self.config.recovery_source,
            recovery_accuracy=self.config.recovery_accuracy,
        )
        self._events.append(event)
        
        return event

    def get_events(self) -> list[ContextRotEvent]:
        """Get all recorded context rot events."""
        return list(self._events)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        total_lost = sum(e.keys_lost for e in self._events)
        total_recovered = sum(e.keys_recovered for e in self._events)
        restarts = sum(1 for e in self._events if e.is_restart)
        decays = sum(1 for e in self._events if not e.is_restart and e.keys_lost > 0)
        
        return {
            "total_events": len(self._events),
            "restart_events": restarts,
            "decay_events": decays,
            "total_keys_lost": total_lost,
            "total_keys_recovered": total_recovered,
            "net_keys_lost": total_lost - total_recovered,
            "recovery_rate": total_recovered / total_lost if total_lost > 0 else 1.0,
            "config": {
                "decay_rate": self.config.decay_rate,
                "restart_probability": self.config.restart_probability,
                "recovery_source": self.config.recovery_source.value,
                "recovery_accuracy": self.config.recovery_accuracy,
            },
        }


# Preset configurations for each scenario
SCENARIO_A_ROT_CONFIG = ContextRotConfig(
    decay_rate=0.02,
    restart_probability=0.005,
    grace_period_days=3,
    recovery_source=RecoverySource.EXCHANGE,
    recovery_accuracy=0.60,  # Exchange catches ~60% of errors via transaction logs
)

SCENARIO_B_ROT_CONFIG = ContextRotConfig(
    decay_rate=0.02,
    restart_probability=0.005,
    grace_period_days=3,
    recovery_source=RecoverySource.NONE,
    recovery_accuracy=0.0,  # No recovery mechanism - errors compound
)

SCENARIO_C_ROT_CONFIG = ContextRotConfig(
    decay_rate=0.0,  # No decay - ledger provides perfect persistence
    restart_probability=0.0,  # Restarts don't matter - ledger recovers all
    grace_period_days=0,
    recovery_source=RecoverySource.LEDGER,
    recovery_accuracy=1.0,  # Perfect recovery from immutable ledger
)
