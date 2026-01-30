"""
Volatile State Manager for Scenario B.

In-memory state with no persistence or recovery.
This demonstrates the problem with private agent databases:
- State is lost on restart
- No recovery mechanism
- No way to reconcile after failure

This is the "fragmented state" scenario that leads to
unresolvable disputes when agents have no shared source of truth.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import structlog

logger = structlog.get_logger()


@dataclass
class VolatileEntry:
    """A single entry in volatile state."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
        }


class VolatileStateManager:
    """
    Volatile (in-memory) state for Scenario B.
    
    No persistence, no recovery. This is what happens when
    agents maintain private databases without a shared ledger.
    
    Problems demonstrated:
    1. State lost on restart
    2. No way to verify state correctness
    3. No recovery from corruption
    4. Divergence between agents undetectable
    
    This manager tracks access patterns to show what would
    be lost in case of failure.
    """
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self._state: dict[str, VolatileEntry] = {}
        self._lock = asyncio.Lock()
        
        # Statistics for demonstrating loss
        self._total_writes: int = 0
        self._total_reads: int = 0
        self._total_value_bytes: int = 0
        self._restart_count: int = 0
        self._keys_lost_total: int = 0
        self._value_lost_total: int = 0
        
        self._created_at = datetime.now()
        
        logger.info(
            "volatile_state.initialized",
            agent_id=agent_id,
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get state value (may be lost after restart).
        
        In Scenario B, there is no guarantee this value
        is correct or matches other agents' state.
        """
        async with self._lock:
            self._total_reads += 1
            
            entry = self._state.get(key)
            if entry is None:
                logger.debug(
                    "volatile_state.get_miss",
                    key=key,
                    agent_id=self.agent_id,
                )
                return None
            
            entry.access_count += 1
            
            logger.debug(
                "volatile_state.get_hit",
                key=key,
                agent_id=self.agent_id,
                access_count=entry.access_count,
            )
            
            return entry.value
    
    async def set(self, key: str, value: Any) -> None:
        """
        Set state value (not persisted).
        
        This value exists ONLY in memory. On restart, it is gone.
        There is no backup, no journal, no way to recover.
        """
        async with self._lock:
            self._total_writes += 1
            
            # Track value size for loss statistics
            try:
                value_bytes = len(json.dumps(value))
            except (TypeError, ValueError):
                value_bytes = 0
            
            if key in self._state:
                # Update existing
                entry = self._state[key]
                old_bytes = len(json.dumps(entry.value)) if entry.value else 0
                self._total_value_bytes -= old_bytes
                
                entry.value = value
                entry.updated_at = datetime.now()
            else:
                # New entry
                entry = VolatileEntry(key=key, value=value)
                self._state[key] = entry
            
            self._total_value_bytes += value_bytes
            
            logger.debug(
                "volatile_state.set",
                key=key,
                agent_id=self.agent_id,
                value_bytes=value_bytes,
            )
    
    async def delete(self, key: str) -> bool:
        """Delete a key from state."""
        async with self._lock:
            if key in self._state:
                entry = self._state.pop(key)
                try:
                    value_bytes = len(json.dumps(entry.value))
                    self._total_value_bytes -= value_bytes
                except (TypeError, ValueError):
                    pass
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._state
    
    async def keys(self) -> list[str]:
        """Get all keys (for inspection)."""
        return list(self._state.keys())
    
    async def simulate_restart(self) -> dict:
        """
        Simulate agent restart, losing all state.
        
        Returns statistics about what was lost.
        This demonstrates the fragility of Scenario B.
        """
        async with self._lock:
            # Record what will be lost
            keys_lost = len(self._state)
            entries_lost = [e.to_dict() for e in self._state.values()]
            value_bytes_lost = self._total_value_bytes
            
            # Most accessed keys (highest impact loss)
            sorted_entries = sorted(
                self._state.values(),
                key=lambda e: e.access_count,
                reverse=True,
            )
            high_impact_keys = [
                {"key": e.key, "access_count": e.access_count}
                for e in sorted_entries[:5]
            ]
            
            # Clear all state
            self._state.clear()
            
            # Update statistics
            self._restart_count += 1
            self._keys_lost_total += keys_lost
            self._value_lost_total += value_bytes_lost
            self._total_value_bytes = 0
            
            loss_report = {
                "restart_number": self._restart_count,
                "keys_lost": keys_lost,
                "value_bytes_lost": value_bytes_lost,
                "high_impact_keys": high_impact_keys,
                "entries_lost_preview": entries_lost[:10],  # First 10
                "cumulative_keys_lost": self._keys_lost_total,
                "cumulative_bytes_lost": self._value_lost_total,
            }
            
            logger.warning(
                "volatile_state.restart_simulated",
                agent_id=self.agent_id,
                keys_lost=keys_lost,
                bytes_lost=value_bytes_lost,
                restart_count=self._restart_count,
            )
            
            return loss_report
    
    async def simulate_partial_corruption(self, corruption_rate: float = 0.1) -> dict:
        """
        Simulate partial data corruption.
        
        Randomly corrupts a percentage of entries.
        This represents data loss that goes undetected without ledger verification.
        """
        import random
        
        async with self._lock:
            keys = list(self._state.keys())
            num_to_corrupt = int(len(keys) * corruption_rate)
            
            corrupted_keys = random.sample(keys, min(num_to_corrupt, len(keys)))
            
            for key in corrupted_keys:
                entry = self._state[key]
                # Corrupt by zeroing or truncating values
                if isinstance(entry.value, (int, float)):
                    entry.value = 0
                elif isinstance(entry.value, str):
                    entry.value = entry.value[:len(entry.value)//2] + "[CORRUPTED]"
                elif isinstance(entry.value, dict):
                    entry.value = {"CORRUPTED": True, "original_keys": list(entry.value.keys())}
                elif isinstance(entry.value, list):
                    entry.value = entry.value[:len(entry.value)//2]
            
            corruption_report = {
                "total_keys": len(keys),
                "keys_corrupted": len(corrupted_keys),
                "corruption_rate": corruption_rate,
                "corrupted_keys": corrupted_keys,
                "note": "In Scenario B, this corruption is UNDETECTABLE without a source of truth",
            }
            
            logger.warning(
                "volatile_state.corruption_simulated",
                agent_id=self.agent_id,
                keys_corrupted=len(corrupted_keys),
            )
            
            return corruption_report
    
    async def get_statistics(self) -> dict:
        """Get statistics about state usage and potential loss."""
        async with self._lock:
            return {
                "agent_id": self.agent_id,
                "created_at": self._created_at.isoformat(),
                "current_keys": len(self._state),
                "current_value_bytes": self._total_value_bytes,
                "total_reads": self._total_reads,
                "total_writes": self._total_writes,
                "restart_count": self._restart_count,
                "cumulative_keys_lost": self._keys_lost_total,
                "cumulative_bytes_lost": self._value_lost_total,
                "at_risk": {
                    "keys": len(self._state),
                    "bytes": self._total_value_bytes,
                    "note": "All current state would be lost on restart",
                },
            }
    
    async def compare_with(self, other: "VolatileStateManager") -> dict:
        """
        Compare state with another agent's volatile state.
        
        This shows how states can diverge in Scenario B.
        Without a ledger, there's no way to know who is "right".
        """
        async with self._lock:
            my_keys = set(self._state.keys())
            other_keys = set(await other.keys())
            
            only_mine = my_keys - other_keys
            only_theirs = other_keys - my_keys
            shared = my_keys & other_keys
            
            # Check for value differences on shared keys
            value_differences = []
            for key in shared:
                my_value = self._state[key].value
                their_value = await other.get(key)
                
                if my_value != their_value:
                    value_differences.append({
                        "key": key,
                        "my_value": my_value,
                        "their_value": their_value,
                    })
            
            return {
                "my_agent": self.agent_id,
                "their_agent": other.agent_id,
                "total_keys_mine": len(my_keys),
                "total_keys_theirs": len(other_keys),
                "shared_keys": len(shared),
                "only_in_mine": list(only_mine),
                "only_in_theirs": list(only_theirs),
                "value_differences": value_differences,
                "divergence_detected": len(only_mine) > 0 or len(only_theirs) > 0 or len(value_differences) > 0,
                "note": "In Scenario B, there is NO way to determine which agent has the correct state",
            }
    
    def to_dict(self) -> dict:
        """Serialize current state (for debugging)."""
        return {
            "agent_id": self.agent_id,
            "entries": {
                key: entry.to_dict()
                for key, entry in self._state.items()
            },
            "statistics": {
                "total_keys": len(self._state),
                "total_writes": self._total_writes,
                "total_reads": self._total_reads,
                "restart_count": self._restart_count,
            },
        }
