"""
State Synchronization Between Agents.

Manages state synchronization between buyer and seller agents:

In Scenario B: Agents try to sync via messages (unreliable)
- Message loss causes state divergence
- No authoritative source for conflict resolution
- Drift accumulates over time

In Scenario C: Agents sync via ledger (reliable)
- Ledger provides authoritative state
- Divergence is detected and corrected
- All agents converge to same state
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Literal
from enum import Enum
import structlog

logger = structlog.get_logger()


class SyncStatus(str, Enum):
    """Status of synchronization attempt."""
    SUCCESS = "success"  # States synchronized
    PARTIAL = "partial"  # Some entries synced
    FAILED = "failed"  # Sync failed
    DIVERGED = "diverged"  # Unresolvable divergence
    TIMEOUT = "timeout"  # Sync timed out


class DivergenceType(str, Enum):
    """Type of state divergence."""
    MISSING_A = "missing_a"  # Agent A missing entry
    MISSING_B = "missing_b"  # Agent B missing entry
    VALUE_MISMATCH = "value_mismatch"  # Different values
    VERSION_CONFLICT = "version_conflict"  # Conflicting versions
    TIMING_CONFLICT = "timing_conflict"  # Timing-based conflict


@dataclass
class Divergence:
    """
    A detected divergence between two agents' state.
    
    In Scenario B, divergences may be unresolvable.
    In Scenario C, divergences are resolved via ledger.
    """
    key: str
    divergence_type: DivergenceType
    
    value_a: Any = None
    value_b: Any = None
    
    # Resolution
    resolved: bool = False
    resolution_source: Optional[str] = None  # "a", "b", "ledger", "merge"
    resolved_value: Any = None
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0  # 0-1, how confident we are in resolution
    
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "type": self.divergence_type.value,
            "value_a": str(self.value_a)[:100] if self.value_a else None,
            "value_b": str(self.value_b)[:100] if self.value_b else None,
            "resolved": self.resolved,
            "resolution_source": self.resolution_source,
            "confidence": self.confidence,
        }


@dataclass
class SyncResult:
    """
    Result of attempting to synchronize state between agents.
    """
    status: SyncStatus
    
    agent_a_id: str
    agent_b_id: str
    
    # Sync statistics
    keys_compared: int = 0
    keys_synced: int = 0
    keys_diverged: int = 0
    keys_unresolvable: int = 0
    
    # Details
    divergences: list[Divergence] = field(default_factory=list)
    sync_method: str = ""  # "message", "ledger", "direct"
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    
    # Retry info
    attempts: int = 1
    retries_needed: int = 0
    
    @property
    def is_success(self) -> bool:
        return self.status == SyncStatus.SUCCESS
    
    @property
    def sync_rate(self) -> float:
        if self.keys_compared == 0:
            return 1.0
        return self.keys_synced / self.keys_compared
    
    @property
    def unresolvable_rate(self) -> float:
        if self.keys_compared == 0:
            return 0.0
        return self.keys_unresolvable / self.keys_compared
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "is_success": self.is_success,
            "agents": {
                "a": self.agent_a_id,
                "b": self.agent_b_id,
            },
            "keys": {
                "compared": self.keys_compared,
                "synced": self.keys_synced,
                "diverged": self.keys_diverged,
                "unresolvable": self.keys_unresolvable,
            },
            "rates": {
                "sync_rate": round(self.sync_rate * 100, 1),
                "unresolvable_rate": round(self.unresolvable_rate * 100, 1),
            },
            "sync_method": self.sync_method,
            "duration_ms": self.duration_ms,
            "attempts": self.attempts,
            "divergences": [d.to_dict() for d in self.divergences[:10]],
        }


class MessageChannel:
    """
    Simulated message channel for Scenario B sync.
    
    Messages can be lost, delayed, or arrive out of order.
    This demonstrates the unreliability of message-based sync.
    """
    
    def __init__(
        self,
        loss_rate: float = 0.05,
        delay_ms_range: tuple[int, int] = (10, 500),
        seed: Optional[int] = None,
    ):
        import random
        self.loss_rate = loss_rate
        self.delay_ms_range = delay_ms_range
        self._random = random.Random(seed)
        
        # Statistics
        self.messages_sent = 0
        self.messages_lost = 0
        self.messages_delivered = 0
    
    async def send(self, message: dict) -> tuple[bool, Optional[dict]]:
        """
        Attempt to send message. Returns (delivered, response).
        
        May lose message or return late/corrupted response.
        """
        self.messages_sent += 1
        
        # Simulate message loss
        if self._random.random() < self.loss_rate:
            self.messages_lost += 1
            logger.debug("message_channel.message_lost")
            return False, None
        
        # Simulate delay
        delay_ms = self._random.randint(*self.delay_ms_range)
        await asyncio.sleep(delay_ms / 1000.0)
        
        self.messages_delivered += 1
        
        # Return success with echoed message as response
        return True, {"ack": True, "original": message}
    
    def statistics(self) -> dict:
        return {
            "sent": self.messages_sent,
            "lost": self.messages_lost,
            "delivered": self.messages_delivered,
            "loss_rate": self.loss_rate,
            "actual_loss_rate": self.messages_lost / max(1, self.messages_sent),
        }


class StateSync:
    """
    Synchronize state between agents.
    
    In Scenario B: Agents try to sync via messages (unreliable)
        - Message loss causes missed updates
        - No way to verify sync completed
        - Divergence accumulates
    
    In Scenario C: Agents sync via ledger (reliable)
        - Ledger provides single source of truth
        - Both agents sync to same ledger state
        - Divergence is detected and corrected
    """
    
    def __init__(
        self,
        message_loss_rate: float = 0.05,
        max_retries: int = 3,
        seed: Optional[int] = None,
    ):
        self.message_loss_rate = message_loss_rate
        self.max_retries = max_retries
        self._channel = MessageChannel(loss_rate=message_loss_rate, seed=seed)
        
        import random
        self._random = random.Random(seed)
        
        # Statistics
        self.syncs_attempted = 0
        self.syncs_succeeded = 0
        self.syncs_failed = 0
        self.total_divergences = 0
        self.total_unresolvable = 0
    
    async def sync_buyer_seller(
        self,
        buyer_state: dict,
        seller_state: dict,
        use_ledger: bool = False,
        ledger_state: Optional[dict] = None,
    ) -> SyncResult:
        """
        Attempt to sync buyer/seller state.
        
        Args:
            buyer_state: Buyer's current state dict
            seller_state: Seller's current state dict
            use_ledger: If True, use ledger as source of truth
            ledger_state: Authoritative ledger state (if use_ledger)
        
        Returns:
            SyncResult with outcome and details
        """
        self.syncs_attempted += 1
        start_time = datetime.now()
        
        result = SyncResult(
            status=SyncStatus.FAILED,
            agent_a_id="buyer",
            agent_b_id="seller",
            sync_method="ledger" if use_ledger else "message",
            started_at=start_time,
        )
        
        # Detect divergences
        divergences = await self.detect_divergence(buyer_state, seller_state)
        result.divergences = divergences
        result.keys_diverged = len(divergences)
        result.keys_compared = len(set(buyer_state.keys()) | set(seller_state.keys()))
        
        if use_ledger and ledger_state is not None:
            # Scenario C: Use ledger to resolve all divergences
            for div in divergences:
                if div.key in ledger_state:
                    div.resolved = True
                    div.resolution_source = "ledger"
                    div.resolved_value = ledger_state[div.key]
                    div.confidence = 1.0
                    result.keys_synced += 1
                else:
                    # Key not in ledger - unresolvable (shouldn't happen in C)
                    div.resolved = False
                    result.keys_unresolvable += 1
            
            result.status = SyncStatus.SUCCESS if result.keys_unresolvable == 0 else SyncStatus.PARTIAL
            
        else:
            # Scenario B: Try to sync via messages (unreliable)
            for div in divergences:
                resolved = await self._try_message_sync(div)
                if resolved:
                    result.keys_synced += 1
                else:
                    result.keys_unresolvable += 1
            
            # Determine status
            if result.keys_unresolvable == 0:
                result.status = SyncStatus.SUCCESS
            elif result.keys_synced > 0:
                result.status = SyncStatus.PARTIAL
            else:
                result.status = SyncStatus.DIVERGED
        
        result.completed_at = datetime.now()
        result.duration_ms = int((result.completed_at - start_time).total_seconds() * 1000)
        
        # Update statistics
        if result.is_success:
            self.syncs_succeeded += 1
        else:
            self.syncs_failed += 1
        self.total_divergences += len(divergences)
        self.total_unresolvable += result.keys_unresolvable
        
        logger.info(
            "state_sync.complete",
            status=result.status.value,
            keys_compared=result.keys_compared,
            keys_synced=result.keys_synced,
            keys_unresolvable=result.keys_unresolvable,
            method=result.sync_method,
        )
        
        return result
    
    async def _try_message_sync(self, divergence: Divergence) -> bool:
        """
        Try to resolve divergence via message exchange.
        
        This is unreliable - messages can be lost.
        Returns True if resolved, False otherwise.
        """
        for attempt in range(self.max_retries):
            # Try to get acknowledgment from both sides
            delivered, response = await self._channel.send({
                "type": "sync_request",
                "key": divergence.key,
                "value": divergence.value_a,
            })
            
            if not delivered:
                continue
            
            # Even if delivered, there's a chance of disagreement
            # In Scenario B, neither side has authority
            if self._random.random() < 0.7:  # 70% chance one side accepts
                divergence.resolved = True
                divergence.resolution_source = self._random.choice(["a", "b"])
                divergence.resolved_value = divergence.value_a if divergence.resolution_source == "a" else divergence.value_b
                divergence.confidence = 0.6  # Low confidence without ledger
                return True
        
        # Failed to resolve after retries
        divergence.resolved = False
        return False
    
    async def detect_divergence(
        self,
        agent_a: dict,
        agent_b: dict,
    ) -> list[Divergence]:
        """
        Detect state divergence between agents.
        
        Compares two state dictionaries and returns list of divergences.
        """
        divergences = []
        
        keys_a = set(agent_a.keys())
        keys_b = set(agent_b.keys())
        
        # Keys only in A
        for key in keys_a - keys_b:
            divergences.append(Divergence(
                key=key,
                divergence_type=DivergenceType.MISSING_B,
                value_a=agent_a[key],
                value_b=None,
            ))
        
        # Keys only in B
        for key in keys_b - keys_a:
            divergences.append(Divergence(
                key=key,
                divergence_type=DivergenceType.MISSING_A,
                value_a=None,
                value_b=agent_b[key],
            ))
        
        # Keys in both - check for value mismatches
        for key in keys_a & keys_b:
            val_a = agent_a[key]
            val_b = agent_b[key]
            
            if val_a != val_b:
                # Check if it's a version/timing conflict
                div_type = DivergenceType.VALUE_MISMATCH
                
                # If values are dicts with version/timestamp, check for conflicts
                if isinstance(val_a, dict) and isinstance(val_b, dict):
                    if "version" in val_a and "version" in val_b:
                        if val_a["version"] != val_b["version"]:
                            div_type = DivergenceType.VERSION_CONFLICT
                    if "timestamp" in val_a and "timestamp" in val_b:
                        if val_a["timestamp"] != val_b["timestamp"]:
                            div_type = DivergenceType.TIMING_CONFLICT
                
                divergences.append(Divergence(
                    key=key,
                    divergence_type=div_type,
                    value_a=val_a,
                    value_b=val_b,
                ))
        
        return divergences
    
    async def simulate_drift(
        self,
        initial_state: dict,
        num_operations: int = 100,
        drift_rate: float = 0.1,
    ) -> tuple[dict, dict, list[str]]:
        """
        Simulate state drift between two agents.
        
        Starting from the same initial state, simulate operations
        where some updates are missed by one agent.
        
        Returns: (agent_a_state, agent_b_state, drifted_keys)
        """
        state_a = initial_state.copy()
        state_b = initial_state.copy()
        drifted_keys = []
        
        for i in range(num_operations):
            key = f"key_{self._random.randint(0, 20)}"
            value = f"value_{i}_{self._random.randint(0, 1000)}"
            
            # Both agents receive the update
            state_a[key] = value
            
            # But B might miss it due to message loss
            if self._random.random() > drift_rate:
                state_b[key] = value
            else:
                drifted_keys.append(key)
        
        return state_a, state_b, list(set(drifted_keys))
    
    def statistics(self) -> dict:
        """Get sync statistics."""
        return {
            "syncs_attempted": self.syncs_attempted,
            "syncs_succeeded": self.syncs_succeeded,
            "syncs_failed": self.syncs_failed,
            "success_rate": self.syncs_succeeded / max(1, self.syncs_attempted),
            "total_divergences": self.total_divergences,
            "total_unresolvable": self.total_unresolvable,
            "unresolvable_rate": self.total_unresolvable / max(1, self.total_divergences),
            "message_channel": self._channel.statistics(),
        }
