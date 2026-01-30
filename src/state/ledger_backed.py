"""
Ledger-Backed State Manager for Scenario C.

All state persisted to Sui blockchain, providing:
- Guaranteed persistence
- Full recovery after restart
- Verification against source of truth
- Cross-agent state consistency

This demonstrates the advantage of shared ledger:
- Agents can always recover their state
- State divergence is detectable and correctable
- Disputes are resolvable via ledger arbitration
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol
from enum import Enum
import structlog

logger = structlog.get_logger()


class VerificationStatus(str, Enum):
    """Result of state verification against ledger."""
    VERIFIED = "verified"  # Local matches ledger
    DIVERGED = "diverged"  # Local differs from ledger
    MISSING_LOCAL = "missing_local"  # Ledger has state we don't
    MISSING_LEDGER = "missing_ledger"  # We have state ledger doesn't
    ERROR = "error"  # Verification failed


@dataclass
class StateSnapshot:
    """
    Complete snapshot of state for recovery/verification.
    
    This is what gets persisted to the ledger and used
    for full state recovery after agent restart.
    """
    agent_id: str
    timestamp: datetime
    
    # State data
    entries: dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    checksum: str = ""
    entry_count: int = 0
    total_bytes: int = 0
    
    # Ledger reference
    ledger_tx: Optional[str] = None
    ledger_object_id: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of state."""
        content = json.dumps(self.entries, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "entry_count": self.entry_count,
            "total_bytes": self.total_bytes,
            "checksum": self.checksum,
            "ledger_tx": self.ledger_tx,
            "ledger_object_id": self.ledger_object_id,
            "entries": self.entries,
        }


@dataclass
class VerificationResult:
    """Result of verifying local state against ledger."""
    status: VerificationStatus
    
    local_checksum: str
    ledger_checksum: str
    
    # Differences found
    missing_local: list[str] = field(default_factory=list)  # Keys in ledger but not local
    missing_ledger: list[str] = field(default_factory=list)  # Keys in local but not ledger
    value_differences: list[dict] = field(default_factory=list)  # Keys with different values
    
    # Counts
    local_entries: int = 0
    ledger_entries: int = 0
    matching_entries: int = 0
    
    verified_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED
    
    @property
    def divergence_count(self) -> int:
        return len(self.missing_local) + len(self.missing_ledger) + len(self.value_differences)
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "is_verified": self.is_verified,
            "local_checksum": self.local_checksum,
            "ledger_checksum": self.ledger_checksum,
            "local_entries": self.local_entries,
            "ledger_entries": self.ledger_entries,
            "matching_entries": self.matching_entries,
            "divergence_count": self.divergence_count,
            "missing_local": self.missing_local,
            "missing_ledger": self.missing_ledger,
            "value_differences": self.value_differences[:10],  # Limit for readability
        }


class LedgerClient(Protocol):
    """Protocol for ledger interaction."""
    
    async def read_state(self, agent_id: str) -> Optional[StateSnapshot]:
        """Read state from ledger."""
        ...
    
    async def write_state(self, snapshot: StateSnapshot) -> str:
        """Write state to ledger, returns transaction ID."""
        ...
    
    async def read_key(self, agent_id: str, key: str) -> Optional[Any]:
        """Read single key from ledger."""
        ...
    
    async def write_key(self, agent_id: str, key: str, value: Any) -> str:
        """Write single key to ledger, returns transaction ID."""
        ...


class MockLedgerClient:
    """
    Mock ledger client for testing.
    
    In production, this would be replaced with actual Sui client.
    """
    
    def __init__(self):
        self._state: dict[str, StateSnapshot] = {}
        self._keys: dict[str, dict[str, Any]] = {}  # agent_id -> {key: value}
        self._tx_counter = 0
    
    async def read_state(self, agent_id: str) -> Optional[StateSnapshot]:
        return self._state.get(agent_id)
    
    async def write_state(self, snapshot: StateSnapshot) -> str:
        self._tx_counter += 1
        tx_id = f"tx_{self._tx_counter:06d}"
        snapshot.ledger_tx = tx_id
        snapshot.ledger_object_id = f"obj_{snapshot.agent_id}_{self._tx_counter}"
        self._state[snapshot.agent_id] = snapshot
        # Also update keys
        if snapshot.agent_id not in self._keys:
            self._keys[snapshot.agent_id] = {}
        self._keys[snapshot.agent_id] = snapshot.entries.copy()
        return tx_id
    
    async def read_key(self, agent_id: str, key: str) -> Optional[Any]:
        if agent_id in self._keys:
            return self._keys[agent_id].get(key)
        return None
    
    async def write_key(self, agent_id: str, key: str, value: Any) -> str:
        self._tx_counter += 1
        tx_id = f"tx_{self._tx_counter:06d}"
        if agent_id not in self._keys:
            self._keys[agent_id] = {}
        self._keys[agent_id][key] = value
        return tx_id


class LedgerBackedStateManager:
    """
    Ledger-backed state for Scenario C.
    
    All state persisted to Sui blockchain, providing:
    - Guaranteed persistence (survives restarts)
    - Full recovery from ledger at any time
    - Verification that local state matches ledger
    - Cross-agent consistency via shared source of truth
    
    Operations:
    - get: Read from local cache, verify against ledger optionally
    - set: Write to local AND ledger
    - recover: Full state recovery from ledger
    - verify: Check local matches ledger
    """
    
    def __init__(
        self,
        agent_id: str,
        ledger_client: Optional[LedgerClient] = None,
        auto_persist: bool = True,
        verify_on_read: bool = False,
    ):
        self.agent_id = agent_id
        self.ledger_client = ledger_client or MockLedgerClient()
        self.auto_persist = auto_persist
        self.verify_on_read = verify_on_read
        
        # Local cache
        self._local_state: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        # Statistics
        self._reads: int = 0
        self._writes: int = 0
        self._ledger_reads: int = 0
        self._ledger_writes: int = 0
        self._verifications: int = 0
        self._recoveries: int = 0
        self._divergences_detected: int = 0
        
        self._created_at = datetime.now()
        self._last_verification: Optional[datetime] = None
        self._last_recovery: Optional[datetime] = None
        
        logger.info(
            "ledger_backed_state.initialized",
            agent_id=agent_id,
            auto_persist=auto_persist,
        )
    
    async def get(self, key: str, verify: Optional[bool] = None) -> Optional[Any]:
        """
        Get state from ledger (always available).
        
        Reads from local cache first, optionally verifies against ledger.
        Unlike volatile state, this value can always be recovered.
        """
        async with self._lock:
            self._reads += 1
            
            # Check local cache first
            local_value = self._local_state.get(key)
            
            # Verify against ledger if requested
            should_verify = verify if verify is not None else self.verify_on_read
            
            if should_verify:
                self._ledger_reads += 1
                ledger_value = await self.ledger_client.read_key(self.agent_id, key)
                
                if local_value != ledger_value:
                    self._divergences_detected += 1
                    logger.warning(
                        "ledger_backed_state.divergence",
                        key=key,
                        local=local_value,
                        ledger=ledger_value,
                    )
                    # Ledger is source of truth
                    local_value = ledger_value
                    self._local_state[key] = ledger_value
            
            return local_value
    
    async def set(self, key: str, value: Any) -> Optional[str]:
        """
        Set state to ledger (persisted).
        
        Writes to both local cache and ledger. Returns ledger transaction ID.
        """
        async with self._lock:
            self._writes += 1
            
            # Update local
            self._local_state[key] = value
            
            # Persist to ledger if enabled
            tx_id = None
            if self.auto_persist:
                self._ledger_writes += 1
                tx_id = await self.ledger_client.write_key(self.agent_id, key, value)
                
                logger.debug(
                    "ledger_backed_state.persisted",
                    key=key,
                    tx_id=tx_id,
                )
            
            return tx_id
    
    async def delete(self, key: str) -> bool:
        """Delete a key from state."""
        async with self._lock:
            if key in self._local_state:
                del self._local_state[key]
                # Also write null to ledger to mark deletion
                if self.auto_persist:
                    await self.ledger_client.write_key(self.agent_id, key, None)
                return True
            return False
    
    async def recover_from_ledger(self) -> StateSnapshot:
        """
        Full state recovery from ledger.
        
        This is the key advantage over volatile state:
        After restart, state can be completely recovered.
        """
        async with self._lock:
            self._recoveries += 1
            self._ledger_reads += 1
            
            # Read full state from ledger
            ledger_snapshot = await self.ledger_client.read_state(self.agent_id)
            
            if ledger_snapshot:
                # Replace local state with ledger state
                self._local_state = ledger_snapshot.entries.copy()
                self._last_recovery = datetime.now()
                
                logger.info(
                    "ledger_backed_state.recovered",
                    agent_id=self.agent_id,
                    entries_recovered=len(self._local_state),
                    ledger_tx=ledger_snapshot.ledger_tx,
                )
                
                return ledger_snapshot
            else:
                # No ledger state - start fresh
                self._local_state = {}
                self._last_recovery = datetime.now()
                
                logger.info(
                    "ledger_backed_state.no_ledger_state",
                    agent_id=self.agent_id,
                )
                
                return StateSnapshot(
                    agent_id=self.agent_id,
                    timestamp=datetime.now(),
                    entries={},
                    entry_count=0,
                )
    
    async def persist_snapshot(self) -> StateSnapshot:
        """Persist complete state snapshot to ledger."""
        async with self._lock:
            self._ledger_writes += 1
            
            # Create snapshot
            snapshot = StateSnapshot(
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                entries=self._local_state.copy(),
                entry_count=len(self._local_state),
                total_bytes=len(json.dumps(self._local_state)),
            )
            snapshot.checksum = snapshot.calculate_checksum()
            
            # Persist to ledger
            tx_id = await self.ledger_client.write_state(snapshot)
            snapshot.ledger_tx = tx_id
            
            logger.info(
                "ledger_backed_state.snapshot_persisted",
                agent_id=self.agent_id,
                entries=snapshot.entry_count,
                checksum=snapshot.checksum[:16] + "...",
                tx_id=tx_id,
            )
            
            return snapshot
    
    async def verify_state(self, local_snapshot: Optional[StateSnapshot] = None) -> VerificationResult:
        """
        Verify local state matches ledger.
        
        This detects any divergence between local cache and ledger,
        enabling correction before disputes arise.
        """
        async with self._lock:
            self._verifications += 1
            self._ledger_reads += 1
            self._last_verification = datetime.now()
            
            # Get local snapshot
            if local_snapshot is None:
                local_snapshot = StateSnapshot(
                    agent_id=self.agent_id,
                    timestamp=datetime.now(),
                    entries=self._local_state.copy(),
                    entry_count=len(self._local_state),
                )
                local_snapshot.checksum = local_snapshot.calculate_checksum()
            
            # Get ledger snapshot
            ledger_snapshot = await self.ledger_client.read_state(self.agent_id)
            
            if ledger_snapshot is None:
                return VerificationResult(
                    status=VerificationStatus.MISSING_LEDGER,
                    local_checksum=local_snapshot.checksum,
                    ledger_checksum="",
                    missing_ledger=list(self._local_state.keys()),
                    local_entries=len(self._local_state),
                    ledger_entries=0,
                )
            
            # Compare checksums first (fast path)
            local_checksum = local_snapshot.checksum or local_snapshot.calculate_checksum()
            ledger_checksum = ledger_snapshot.checksum or ledger_snapshot.calculate_checksum()
            
            if local_checksum == ledger_checksum:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    local_checksum=local_checksum,
                    ledger_checksum=ledger_checksum,
                    local_entries=len(self._local_state),
                    ledger_entries=len(ledger_snapshot.entries),
                    matching_entries=len(self._local_state),
                )
            
            # Checksums differ - find differences
            local_keys = set(self._local_state.keys())
            ledger_keys = set(ledger_snapshot.entries.keys())
            
            missing_local = list(ledger_keys - local_keys)
            missing_ledger = list(local_keys - ledger_keys)
            
            value_differences = []
            for key in local_keys & ledger_keys:
                if self._local_state[key] != ledger_snapshot.entries[key]:
                    value_differences.append({
                        "key": key,
                        "local": self._local_state[key],
                        "ledger": ledger_snapshot.entries[key],
                    })
            
            self._divergences_detected += 1
            
            matching = len(local_keys & ledger_keys) - len(value_differences)
            
            result = VerificationResult(
                status=VerificationStatus.DIVERGED,
                local_checksum=local_checksum,
                ledger_checksum=ledger_checksum,
                missing_local=missing_local,
                missing_ledger=missing_ledger,
                value_differences=value_differences,
                local_entries=len(self._local_state),
                ledger_entries=len(ledger_snapshot.entries),
                matching_entries=matching,
            )
            
            logger.warning(
                "ledger_backed_state.divergence_detected",
                agent_id=self.agent_id,
                missing_local=len(missing_local),
                missing_ledger=len(missing_ledger),
                value_diffs=len(value_differences),
            )
            
            return result
    
    async def sync_from_ledger(self) -> int:
        """
        Sync local state from ledger, fixing any divergence.
        
        Returns number of entries updated.
        """
        verification = await self.verify_state()
        
        if verification.is_verified:
            return 0
        
        # Recover full state from ledger
        await self.recover_from_ledger()
        
        return verification.divergence_count
    
    async def get_statistics(self) -> dict:
        """Get statistics about state operations."""
        async with self._lock:
            return {
                "agent_id": self.agent_id,
                "created_at": self._created_at.isoformat(),
                "current_entries": len(self._local_state),
                "operations": {
                    "reads": self._reads,
                    "writes": self._writes,
                    "ledger_reads": self._ledger_reads,
                    "ledger_writes": self._ledger_writes,
                    "verifications": self._verifications,
                    "recoveries": self._recoveries,
                },
                "integrity": {
                    "divergences_detected": self._divergences_detected,
                    "last_verification": self._last_verification.isoformat() if self._last_verification else None,
                    "last_recovery": self._last_recovery.isoformat() if self._last_recovery else None,
                },
                "config": {
                    "auto_persist": self.auto_persist,
                    "verify_on_read": self.verify_on_read,
                },
            }
    
    async def compare_with(self, other: "LedgerBackedStateManager") -> dict:
        """
        Compare state with another agent's ledger-backed state.
        
        Unlike volatile state, both agents can verify against ledger
        to determine who has correct state.
        """
        # Both verify against their own ledger state
        my_verification = await self.verify_state()
        their_verification = await other.verify_state()
        
        async with self._lock:
            my_keys = set(self._local_state.keys())
        
        other_keys = set(await other.keys())
        
        return {
            "my_agent": self.agent_id,
            "their_agent": other.agent_id,
            "my_verification": my_verification.to_dict(),
            "their_verification": their_verification.to_dict(),
            "shared_keys": len(my_keys & other_keys),
            "note": "In Scenario C, divergence can be resolved by both agents syncing from ledger",
            "resolution": "BOTH agents can recover correct state from their respective ledger snapshots",
        }
    
    async def keys(self) -> list[str]:
        """Get all keys."""
        return list(self._local_state.keys())
    
    def to_dict(self) -> dict:
        """Serialize current state."""
        return {
            "agent_id": self.agent_id,
            "entries": self._local_state.copy(),
            "statistics": {
                "entries": len(self._local_state),
                "writes": self._writes,
                "reads": self._reads,
                "ledger_writes": self._ledger_writes,
            },
        }
