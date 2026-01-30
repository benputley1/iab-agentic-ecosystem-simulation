"""
V2 CLI Commands - Context Window Hallucination Testing

Extends the CLI with new flags for:
- Token pressure tracking
- Decision chain tracking
- Restart simulation
- Full V2 mode (all features)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class VolumeProfile(str, Enum):
    """Volume profile presets for simulation."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"
    
    @property
    def daily_requests(self) -> int:
        """Get daily bid request volume for profile."""
        return {
            VolumeProfile.SMALL: 10_000,
            VolumeProfile.MEDIUM: 100_000,
            VolumeProfile.LARGE: 1_000_000,
            VolumeProfile.ENTERPRISE: 10_000_000,
        }[self]
    
    @property
    def bid_rate(self) -> float:
        """Get bid rate for profile."""
        return {
            VolumeProfile.SMALL: 0.3,
            VolumeProfile.MEDIUM: 0.2,
            VolumeProfile.LARGE: 0.1,
            VolumeProfile.ENTERPRISE: 0.05,
        }[self]


@dataclass
class V2Config:
    """Configuration for V2 simulation modes."""
    
    # Context pressure settings
    context_pressure: bool = False
    volume_profile: VolumeProfile = VolumeProfile.MEDIUM
    context_limit: int = 200_000
    compression_loss: float = 0.20
    
    # Decision chain settings
    decision_chain: bool = False
    lookback_window: int = 100
    
    # Restart test settings
    restart_test: bool = False
    crash_probability: float = 0.01
    recovery_modes: List[str] = field(default_factory=lambda: ["private_db", "ledger"])
    
    # Full V2 mode
    full_v2: bool = False
    
    def __post_init__(self):
        """Enable all features if full_v2 is set."""
        if self.full_v2:
            self.context_pressure = True
            self.decision_chain = True
            self.restart_test = True
    
    @property
    def is_v2_enabled(self) -> bool:
        """Check if any V2 feature is enabled."""
        return any([
            self.context_pressure,
            self.decision_chain,
            self.restart_test,
            self.full_v2,
        ])
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "context_pressure": {
                "enabled": self.context_pressure,
                "volume_profile": self.volume_profile.value if isinstance(self.volume_profile, VolumeProfile) else self.volume_profile,
                "context_limit": self.context_limit,
                "compression_loss": self.compression_loss,
            },
            "decision_chain": {
                "enabled": self.decision_chain,
                "lookback_window": self.lookback_window,
            },
            "restart_test": {
                "enabled": self.restart_test,
                "crash_probability": self.crash_probability,
                "recovery_modes": self.recovery_modes,
            },
            "full_v2": self.full_v2,
        }
    
    @classmethod
    def from_cli_args(
        cls,
        context_pressure: bool = False,
        volume_profile: Optional[str] = None,
        context_limit: int = 200_000,
        compression_loss: float = 0.20,
        decision_chain: bool = False,
        lookback_window: int = 100,
        restart_test: bool = False,
        crash_probability: float = 0.01,
        recovery_modes: Optional[str] = None,
        full_v2: bool = False,
    ) -> "V2Config":
        """Create V2Config from CLI arguments."""
        # Parse volume profile
        vp = VolumeProfile.MEDIUM
        if volume_profile:
            vp = VolumeProfile(volume_profile.lower())
        
        # Parse recovery modes
        modes = ["private_db", "ledger"]
        if recovery_modes:
            modes = [m.strip() for m in recovery_modes.split(",")]
        
        return cls(
            context_pressure=context_pressure,
            volume_profile=vp,
            context_limit=context_limit,
            compression_loss=compression_loss,
            decision_chain=decision_chain,
            lookback_window=lookback_window,
            restart_test=restart_test,
            crash_probability=crash_probability,
            recovery_modes=modes,
            full_v2=full_v2,
        )


def validate_v2_config(config: V2Config) -> List[str]:
    """
    Validate V2 configuration, returning list of warnings/errors.
    
    Returns empty list if all valid.
    """
    issues = []
    
    # Validate context_limit
    if config.context_limit < 1000:
        issues.append(f"context_limit too low: {config.context_limit}. Minimum: 1000")
    if config.context_limit > 2_000_000:
        issues.append(f"context_limit very high: {config.context_limit}. May cause memory issues.")
    
    # Validate compression_loss
    if not 0.0 <= config.compression_loss <= 1.0:
        issues.append(f"compression_loss must be 0.0-1.0, got: {config.compression_loss}")
    
    # Validate lookback_window
    if config.lookback_window < 1:
        issues.append(f"lookback_window must be >= 1, got: {config.lookback_window}")
    if config.lookback_window > 10000:
        issues.append(f"lookback_window very high: {config.lookback_window}. May be slow.")
    
    # Validate crash_probability
    if not 0.0 <= config.crash_probability <= 1.0:
        issues.append(f"crash_probability must be 0.0-1.0, got: {config.crash_probability}")
    
    # Validate recovery_modes
    valid_modes = {"private_db", "ledger", "context", "checkpoint"}
    for mode in config.recovery_modes:
        if mode not in valid_modes:
            issues.append(f"Unknown recovery mode: {mode}. Valid: {valid_modes}")
    
    return issues


def get_v2_feature_summary(config: V2Config) -> str:
    """Get a human-readable summary of enabled V2 features."""
    if not config.is_v2_enabled:
        return "V2 features: None enabled (V1 mode)"
    
    features = []
    
    if config.context_pressure:
        features.append(
            f"Context Pressure (profile={config.volume_profile.value}, "
            f"limit={config.context_limit:,}, loss={config.compression_loss:.0%})"
        )
    
    if config.decision_chain:
        features.append(f"Decision Chain (lookback={config.lookback_window})")
    
    if config.restart_test:
        features.append(
            f"Restart Test (crash_prob={config.crash_probability:.2%}, "
            f"modes={','.join(config.recovery_modes)})"
        )
    
    if config.full_v2:
        return "Full V2 Mode: " + " | ".join(features)
    
    return "V2 features: " + " | ".join(features)
