"""
State Management for Multi-Agent Hierarchy.

Handles agent state persistence, snapshots for recovery,
and different state backends for Scenario B (volatile) and C (ledger-backed).
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class StateVersion(BaseModel):
    """Version tracking for state changes."""
    version: int = 0
    timestamp: float = Field(default_factory=time.time)
    agent_id: str = ""
    change_description: str = ""
    
    def increment(self, agent_id: str, description: str = "") -> "StateVersion":
        """Create new version."""
        return StateVersion(
            version=self.version + 1,
            timestamp=time.time(),
            agent_id=agent_id,
            change_description=description
        )


class AgentState(BaseModel):
    """
    Base class for agent state.
    
    Tracks the mutable state that an agent maintains across interactions.
    Each agent type will extend this with domain-specific fields.
    """
    state_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    agent_type: str = ""
    
    # Version tracking
    version: StateVersion = Field(default_factory=StateVersion)
    
    # Timestamps
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    # Generic state storage
    data: dict[str, Any] = Field(default_factory=dict)
    
    # Status tracking
    is_active: bool = True
    error_count: int = 0
    last_error: str | None = None
    
    def update(self, **kwargs) -> None:
        """Update state fields and version."""
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key] = value
            elif hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = time.time()
        self.version = self.version.increment(self.agent_id, f"Updated: {list(kwargs.keys())}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get from generic data dict."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set in generic data dict."""
        self.data[key] = value
        self.updated_at = time.time()
    
    def record_error(self, error: str) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.last_error = error
        self.updated_at = time.time()
    
    def to_snapshot(self) -> "StateSnapshot":
        """Create a snapshot for recovery."""
        return StateSnapshot(
            state_id=self.state_id,
            agent_id=self.agent_id,
            version=self.version,
            snapshot_time=time.time(),
            state_data=self.model_dump()
        )


class StateSnapshot(BaseModel):
    """
    Immutable snapshot of agent state for recovery.
    
    Used for checkpointing and rollback capabilities.
    """
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    state_id: str = ""
    agent_id: str = ""
    version: StateVersion = Field(default_factory=StateVersion)
    snapshot_time: float = Field(default_factory=time.time)
    state_data: dict[str, Any] = Field(default_factory=dict)
    
    # Recovery metadata
    recovery_point: bool = False  # Marked as safe recovery point
    description: str = ""
    
    def restore(self, state_class: type[AgentState] = AgentState) -> AgentState:
        """Restore state from snapshot."""
        return state_class.model_validate(self.state_data)


class StateBackend(ABC):
    """Abstract backend for state persistence."""
    
    @abstractmethod
    async def save(self, state: AgentState) -> bool:
        """Persist state."""
        ...
    
    @abstractmethod
    async def load(self, state_id: str) -> AgentState | None:
        """Load state by ID."""
        ...
    
    @abstractmethod
    async def save_snapshot(self, snapshot: StateSnapshot) -> bool:
        """Save a snapshot."""
        ...
    
    @abstractmethod
    async def load_snapshot(self, snapshot_id: str) -> StateSnapshot | None:
        """Load a snapshot."""
        ...
    
    @abstractmethod
    async def list_snapshots(self, state_id: str) -> list[StateSnapshot]:
        """List all snapshots for a state."""
        ...


class VolatileStateBackend(StateBackend):
    """
    In-memory state backend for Scenario B (no ledger).
    
    State is lost on restart, simulating pure LLM context management.
    """
    
    def __init__(self):
        self._states: dict[str, AgentState] = {}
        self._snapshots: dict[str, StateSnapshot] = {}
        self._state_snapshots: dict[str, list[str]] = {}  # state_id -> snapshot_ids
        self.log = structlog.get_logger(__name__)
    
    async def save(self, state: AgentState) -> bool:
        """Save state to memory."""
        self._states[state.state_id] = state
        self.log.debug("state_saved_volatile", state_id=state.state_id)
        return True
    
    async def load(self, state_id: str) -> AgentState | None:
        """Load state from memory."""
        return self._states.get(state_id)
    
    async def save_snapshot(self, snapshot: StateSnapshot) -> bool:
        """Save snapshot to memory."""
        self._snapshots[snapshot.snapshot_id] = snapshot
        
        # Track snapshot for its state
        if snapshot.state_id not in self._state_snapshots:
            self._state_snapshots[snapshot.state_id] = []
        self._state_snapshots[snapshot.state_id].append(snapshot.snapshot_id)
        
        self.log.debug(
            "snapshot_saved_volatile",
            snapshot_id=snapshot.snapshot_id,
            state_id=snapshot.state_id
        )
        return True
    
    async def load_snapshot(self, snapshot_id: str) -> StateSnapshot | None:
        """Load snapshot from memory."""
        return self._snapshots.get(snapshot_id)
    
    async def list_snapshots(self, state_id: str) -> list[StateSnapshot]:
        """List all snapshots for a state."""
        snapshot_ids = self._state_snapshots.get(state_id, [])
        return [
            self._snapshots[sid]
            for sid in snapshot_ids
            if sid in self._snapshots
        ]
    
    def clear(self) -> None:
        """Clear all state (simulates restart/loss)."""
        self._states.clear()
        self._snapshots.clear()
        self._state_snapshots.clear()
        self.log.warning("volatile_state_cleared")


class FileStateBackend(StateBackend):
    """
    File-based state backend for development/testing.
    
    Persists state to JSON files.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.states_path = base_path / "states"
        self.snapshots_path = base_path / "snapshots"
        
        # Create directories
        self.states_path.mkdir(parents=True, exist_ok=True)
        self.snapshots_path.mkdir(parents=True, exist_ok=True)
        
        self.log = structlog.get_logger(__name__)
    
    async def save(self, state: AgentState) -> bool:
        """Save state to file."""
        try:
            file_path = self.states_path / f"{state.state_id}.json"
            file_path.write_text(state.model_dump_json(indent=2))
            self.log.debug("state_saved_file", path=str(file_path))
            return True
        except Exception as e:
            self.log.error("state_save_failed", error=str(e))
            return False
    
    async def load(self, state_id: str) -> AgentState | None:
        """Load state from file."""
        try:
            file_path = self.states_path / f"{state_id}.json"
            if not file_path.exists():
                return None
            data = json.loads(file_path.read_text())
            return AgentState.model_validate(data)
        except Exception as e:
            self.log.error("state_load_failed", state_id=state_id, error=str(e))
            return None
    
    async def save_snapshot(self, snapshot: StateSnapshot) -> bool:
        """Save snapshot to file."""
        try:
            file_path = self.snapshots_path / f"{snapshot.snapshot_id}.json"
            file_path.write_text(snapshot.model_dump_json(indent=2))
            return True
        except Exception as e:
            self.log.error("snapshot_save_failed", error=str(e))
            return False
    
    async def load_snapshot(self, snapshot_id: str) -> StateSnapshot | None:
        """Load snapshot from file."""
        try:
            file_path = self.snapshots_path / f"{snapshot_id}.json"
            if not file_path.exists():
                return None
            data = json.loads(file_path.read_text())
            return StateSnapshot.model_validate(data)
        except Exception as e:
            self.log.error("snapshot_load_failed", snapshot_id=snapshot_id, error=str(e))
            return None
    
    async def list_snapshots(self, state_id: str) -> list[StateSnapshot]:
        """List all snapshots for a state."""
        snapshots = []
        for file_path in self.snapshots_path.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                snapshot = StateSnapshot.model_validate(data)
                if snapshot.state_id == state_id:
                    snapshots.append(snapshot)
            except Exception:
                continue
        return sorted(snapshots, key=lambda s: s.snapshot_time)


class LedgerStateBackend(StateBackend):
    """
    Ledger-backed state for Scenario C (Alkimi Ledger on Sui).
    
    State is persisted to blockchain, providing immutable audit trail
    and cross-agent verification capability.
    
    NOTE: This is a stub implementation. Full integration with Sui
    blockchain will be implemented in rs-0107 (Protocol Handlers).
    """
    
    def __init__(
        self,
        sui_endpoint: str = "https://fullnode.testnet.sui.io:443",
        package_id: str | None = None
    ):
        self.sui_endpoint = sui_endpoint
        self.package_id = package_id
        self.log = structlog.get_logger(__name__)
        
        # Fallback to file storage until Sui integration is complete
        self._fallback = FileStateBackend(Path("/tmp/ledger_state"))
        
        self.log.info(
            "ledger_backend_initialized",
            endpoint=sui_endpoint,
            package_id=package_id,
            note="Using file fallback until Sui integration complete"
        )
    
    async def save(self, state: AgentState) -> bool:
        """Save state to ledger (or fallback)."""
        # TODO: Implement Sui object creation/update
        # For now, use file fallback
        self.log.debug(
            "ledger_save_fallback",
            state_id=state.state_id,
            note="Would commit to Sui blockchain"
        )
        return await self._fallback.save(state)
    
    async def load(self, state_id: str) -> AgentState | None:
        """Load state from ledger (or fallback)."""
        # TODO: Implement Sui object fetch
        return await self._fallback.load(state_id)
    
    async def save_snapshot(self, snapshot: StateSnapshot) -> bool:
        """Save snapshot to ledger (or fallback)."""
        # TODO: Implement Sui event emission for snapshot
        self.log.debug(
            "ledger_snapshot_fallback",
            snapshot_id=snapshot.snapshot_id,
            note="Would emit Sui event"
        )
        return await self._fallback.save_snapshot(snapshot)
    
    async def load_snapshot(self, snapshot_id: str) -> StateSnapshot | None:
        """Load snapshot from ledger (or fallback)."""
        return await self._fallback.load_snapshot(snapshot_id)
    
    async def list_snapshots(self, state_id: str) -> list[StateSnapshot]:
        """List snapshots from ledger (or fallback)."""
        return await self._fallback.list_snapshots(state_id)
    
    async def verify_state(self, state: AgentState) -> bool:
        """
        Verify state against ledger.
        
        Key advantage of Scenario C - any agent can verify
        that state hasn't been corrupted.
        """
        # TODO: Implement Sui state verification
        self.log.debug(
            "ledger_verify_stub",
            state_id=state.state_id,
            note="Would verify against Sui blockchain"
        )
        return True


class StateManager:
    """
    High-level state management for agents.
    
    Handles state lifecycle, snapshots, and recovery.
    """
    
    def __init__(
        self,
        backend: StateBackend,
        snapshot_interval: int = 10,  # Snapshot every N updates
        max_snapshots: int = 100
    ):
        self.backend = backend
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        
        self._update_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self.log = structlog.get_logger(__name__)
    
    async def initialize_state(
        self,
        agent_id: str,
        agent_type: str,
        initial_data: dict[str, Any] | None = None
    ) -> AgentState:
        """Create and persist initial state for an agent."""
        state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            data=initial_data or {}
        )
        
        await self.backend.save(state)
        self._update_counts[state.state_id] = 0
        
        self.log.info(
            "state_initialized",
            state_id=state.state_id,
            agent_id=agent_id,
            agent_type=agent_type
        )
        
        return state
    
    async def update_state(
        self,
        state: AgentState,
        **updates
    ) -> AgentState:
        """Update state with automatic snapshotting."""
        async with self._lock:
            state.update(**updates)
            await self.backend.save(state)
            
            # Track updates for snapshot interval
            count = self._update_counts.get(state.state_id, 0) + 1
            self._update_counts[state.state_id] = count
            
            # Create snapshot if interval reached
            if count % self.snapshot_interval == 0:
                await self.create_snapshot(state, description=f"Auto-snapshot at update {count}")
            
            return state
    
    async def create_snapshot(
        self,
        state: AgentState,
        description: str = "",
        recovery_point: bool = False
    ) -> StateSnapshot:
        """Create a named snapshot."""
        snapshot = state.to_snapshot()
        snapshot.description = description
        snapshot.recovery_point = recovery_point
        
        await self.backend.save_snapshot(snapshot)
        
        # Prune old snapshots if needed
        snapshots = await self.backend.list_snapshots(state.state_id)
        if len(snapshots) > self.max_snapshots:
            # Keep recovery points and recent snapshots
            to_remove = [
                s for s in snapshots[:-self.max_snapshots]
                if not s.recovery_point
            ]
            # Note: Actual removal would require backend delete method
            self.log.debug(
                "snapshots_would_prune",
                count=len(to_remove)
            )
        
        self.log.info(
            "snapshot_created",
            snapshot_id=snapshot.snapshot_id,
            state_id=state.state_id,
            recovery_point=recovery_point
        )
        
        return snapshot
    
    async def recover_from_snapshot(
        self,
        snapshot_id: str,
        state_class: type[AgentState] = AgentState
    ) -> AgentState | None:
        """Recover state from a snapshot."""
        snapshot = await self.backend.load_snapshot(snapshot_id)
        if not snapshot:
            self.log.error("snapshot_not_found", snapshot_id=snapshot_id)
            return None
        
        state = snapshot.restore(state_class)
        await self.backend.save(state)
        
        self.log.info(
            "state_recovered",
            snapshot_id=snapshot_id,
            state_id=state.state_id
        )
        
        return state
    
    async def get_latest_recovery_point(
        self,
        state_id: str
    ) -> StateSnapshot | None:
        """Get the most recent recovery point snapshot."""
        snapshots = await self.backend.list_snapshots(state_id)
        recovery_points = [s for s in snapshots if s.recovery_point]
        
        if not recovery_points:
            return None
        
        return max(recovery_points, key=lambda s: s.snapshot_time)


# Convenience type aliases
VolatileState = VolatileStateBackend  # For Scenario B
LedgerBackedState = LedgerStateBackend  # For Scenario C
