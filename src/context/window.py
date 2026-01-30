"""
Context Window Management.

Manages context window size and token limits with support for
tracking integrity and degradation over time.

Different models have different limits:
- Opus: 200K tokens
- Sonnet: 200K tokens

But effective context degrades over conversation length.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class WindowState(str, Enum):
    """State of the context window."""
    HEALTHY = "healthy"      # < 50% utilized
    WARNING = "warning"      # 50-80% utilized
    CRITICAL = "critical"    # > 80% utilized
    OVERFLOW = "overflow"    # Truncation required


class ContextEntry(BaseModel):
    """
    Individual entry in the context window.
    
    Tracks content, tokens, and metadata for each piece
    of context in an agent's window.
    """
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    tokens: int
    timestamp: float = Field(default_factory=time.time)
    importance: float = 0.5  # 0-1, higher = keep longer
    source: str = ""  # which agent added this
    
    # Metadata
    entry_type: str = "general"  # general, task, result, system
    parent_entry_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    
    # Decay tracking
    original_tokens: int | None = None
    decay_factor: float = 1.0  # 1.0 = no decay, 0.0 = fully decayed
    
    def apply_decay(self, rate: float) -> None:
        """
        Apply decay to this entry.
        
        Decay reduces effective importance over time.
        """
        if self.original_tokens is None:
            self.original_tokens = self.tokens
        
        self.decay_factor = max(0.0, self.decay_factor - rate)
        self.importance = self.importance * self.decay_factor
    
    @property
    def effective_importance(self) -> float:
        """Importance adjusted for decay."""
        return self.importance * self.decay_factor
    
    @property
    def age_seconds(self) -> float:
        """Age of entry in seconds."""
        return time.time() - self.timestamp


class ContextWindowManager(BaseModel):
    """
    Manages context window size and token limits.
    
    Tracks entries, handles truncation, and calculates
    context integrity metrics.
    """
    window_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    
    # Token limits
    max_tokens: int = 200000  # Model-dependent
    reserved_tokens: int = 4000  # Reserve for response
    warning_threshold: float = 0.5
    critical_threshold: float = 0.8
    
    # Current state
    current_tokens: int = 0
    history: list[ContextEntry] = Field(default_factory=list)
    
    # Integrity tracking
    total_tokens_added: int = 0
    total_tokens_truncated: int = 0
    truncation_count: int = 0
    
    # Timestamps
    created_at: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)
    
    @property
    def available_tokens(self) -> int:
        """Tokens available for new content."""
        return self.max_tokens - self.current_tokens - self.reserved_tokens
    
    @property
    def utilization(self) -> float:
        """Current utilization ratio (0.0-1.0)."""
        return self.current_tokens / self.max_tokens
    
    @property
    def state(self) -> WindowState:
        """Current window state based on utilization."""
        if self.utilization >= 1.0:
            return WindowState.OVERFLOW
        elif self.utilization >= self.critical_threshold:
            return WindowState.CRITICAL
        elif self.utilization >= self.warning_threshold:
            return WindowState.WARNING
        return WindowState.HEALTHY
    
    def add_entry(self, entry: ContextEntry) -> int:
        """
        Add entry to the window.
        
        Returns tokens used. May trigger truncation if needed.
        """
        # Check if we need to truncate first
        if entry.tokens > self.available_tokens:
            needed = entry.tokens - self.available_tokens
            self.truncate_oldest(needed)
        
        self.history.append(entry)
        self.current_tokens += entry.tokens
        self.total_tokens_added += entry.tokens
        self.last_updated = time.time()
        
        logger.debug(
            "context_entry_added",
            window_id=self.window_id,
            entry_id=entry.entry_id,
            tokens=entry.tokens,
            current_tokens=self.current_tokens,
            utilization=self.utilization
        )
        
        return entry.tokens
    
    def add_content(
        self,
        content: str,
        source: str = "",
        importance: float = 0.5,
        entry_type: str = "general",
        tags: list[str] | None = None
    ) -> ContextEntry:
        """
        Add content string to the window.
        
        Convenience method that creates a ContextEntry.
        """
        # Estimate tokens (rough: 4 chars per token)
        tokens = len(content) // 4 + 1
        
        entry = ContextEntry(
            content=content,
            tokens=tokens,
            source=source,
            importance=importance,
            entry_type=entry_type,
            tags=tags or []
        )
        
        self.add_entry(entry)
        return entry
    
    def can_fit(self, tokens: int) -> bool:
        """Check if tokens can fit without truncation."""
        return tokens <= self.available_tokens
    
    def truncate_oldest(self, tokens_needed: int) -> list[ContextEntry]:
        """
        Truncate oldest/least important entries to make room.
        
        Uses importance-weighted age to prioritize what to remove.
        Returns the truncated entries.
        """
        if tokens_needed <= 0:
            return []
        
        truncated: list[ContextEntry] = []
        tokens_freed = 0
        
        # Sort by effective importance (low importance + old = remove first)
        # Score = importance * decay / age_factor
        def removal_priority(entry: ContextEntry) -> float:
            age_factor = max(1.0, entry.age_seconds / 3600)  # Hours
            return entry.effective_importance / age_factor
        
        # Sort entries (lowest priority first)
        sorted_entries = sorted(self.history, key=removal_priority)
        
        for entry in sorted_entries:
            if tokens_freed >= tokens_needed:
                break
            
            truncated.append(entry)
            tokens_freed += entry.tokens
            self.history.remove(entry)
            self.current_tokens -= entry.tokens
        
        self.total_tokens_truncated += tokens_freed
        self.truncation_count += 1
        self.last_updated = time.time()
        
        logger.info(
            "context_truncated",
            window_id=self.window_id,
            entries_removed=len(truncated),
            tokens_freed=tokens_freed,
            tokens_needed=tokens_needed
        )
        
        return truncated
    
    def get_context_integrity(self) -> float:
        """
        Calculate context integrity (% of original retained).
        
        1.0 = all context retained
        0.0 = all context lost
        """
        if self.total_tokens_added == 0:
            return 1.0
        
        retained = self.total_tokens_added - self.total_tokens_truncated
        return max(0.0, retained / self.total_tokens_added)
    
    def get_entries_by_source(self, source: str) -> list[ContextEntry]:
        """Get all entries from a specific source."""
        return [e for e in self.history if e.source == source]
    
    def get_entries_by_type(self, entry_type: str) -> list[ContextEntry]:
        """Get all entries of a specific type."""
        return [e for e in self.history if e.entry_type == entry_type]
    
    def get_entries_by_importance(
        self,
        min_importance: float = 0.0
    ) -> list[ContextEntry]:
        """Get entries above importance threshold."""
        return [
            e for e in self.history 
            if e.effective_importance >= min_importance
        ]
    
    def get_recent_entries(self, count: int = 10) -> list[ContextEntry]:
        """Get most recent entries."""
        return sorted(
            self.history, 
            key=lambda e: e.timestamp, 
            reverse=True
        )[:count]
    
    def apply_decay(self, rate: float = 0.05) -> int:
        """
        Apply decay to all entries.
        
        Returns number of entries affected.
        """
        affected = 0
        for entry in self.history:
            if entry.decay_factor > 0:
                entry.apply_decay(rate)
                affected += 1
        
        logger.debug(
            "decay_applied",
            window_id=self.window_id,
            rate=rate,
            entries_affected=affected
        )
        
        return affected
    
    def prune_decayed(self, min_importance: float = 0.1) -> list[ContextEntry]:
        """
        Remove entries that have decayed below threshold.
        
        Returns pruned entries.
        """
        pruned: list[ContextEntry] = []
        
        for entry in list(self.history):
            if entry.effective_importance < min_importance:
                pruned.append(entry)
                self.history.remove(entry)
                self.current_tokens -= entry.tokens
                self.total_tokens_truncated += entry.tokens
        
        if pruned:
            self.truncation_count += 1
            self.last_updated = time.time()
            
            logger.info(
                "decayed_entries_pruned",
                window_id=self.window_id,
                entries_pruned=len(pruned),
                min_importance=min_importance
            )
        
        return pruned
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of window state."""
        return {
            "window_id": self.window_id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "utilization": self.utilization,
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
            "entry_count": len(self.history),
            "integrity": self.get_context_integrity(),
            "truncation_count": self.truncation_count,
            "total_tokens_added": self.total_tokens_added,
            "total_tokens_truncated": self.total_tokens_truncated
        }
    
    def clear(self) -> None:
        """Clear all entries from the window."""
        cleared_tokens = self.current_tokens
        self.history.clear()
        self.current_tokens = 0
        self.last_updated = time.time()
        
        logger.info(
            "window_cleared",
            window_id=self.window_id,
            tokens_cleared=cleared_tokens
        )
