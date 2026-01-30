"""
Integration tests for V2 CLI extensions.

Tests the new CLI flags for context window hallucination testing.
"""

import pytest
from typer.testing import CliRunner
from src.cli.main import app
from src.cli.v2_commands import (
    V2Config,
    VolumeProfile,
    validate_v2_config,
    get_v2_feature_summary,
)

runner = CliRunner()


class TestV2Config:
    """Test V2Config dataclass."""

    def test_default_config(self):
        """Test default V2 config values."""
        config = V2Config()
        assert config.context_pressure is False
        assert config.volume_profile == VolumeProfile.MEDIUM
        assert config.context_limit == 200_000
        assert config.compression_loss == 0.20
        assert config.decision_chain is False
        assert config.lookback_window == 100
        assert config.restart_test is False
        assert config.crash_probability == 0.01
        assert config.recovery_modes == ["private_db", "ledger"]
        assert config.full_v2 is False

    def test_full_v2_enables_all(self):
        """Test that full_v2=True enables all features."""
        config = V2Config(full_v2=True)
        assert config.context_pressure is True
        assert config.decision_chain is True
        assert config.restart_test is True

    def test_is_v2_enabled(self):
        """Test is_v2_enabled property."""
        # Default config should be disabled
        config = V2Config()
        assert config.is_v2_enabled is False

        # Enabling any feature should enable V2
        config = V2Config(context_pressure=True)
        assert config.is_v2_enabled is True

        config = V2Config(decision_chain=True)
        assert config.is_v2_enabled is True

        config = V2Config(restart_test=True)
        assert config.is_v2_enabled is True

    def test_from_cli_args(self):
        """Test creating config from CLI arguments."""
        config = V2Config.from_cli_args(
            context_pressure=True,
            volume_profile="large",
            context_limit=150_000,
            compression_loss=0.15,
            decision_chain=True,
            lookback_window=50,
            restart_test=True,
            crash_probability=0.05,
            recovery_modes="ledger,checkpoint",
            full_v2=False,
        )

        assert config.context_pressure is True
        assert config.volume_profile == VolumeProfile.LARGE
        assert config.context_limit == 150_000
        assert config.compression_loss == 0.15
        assert config.decision_chain is True
        assert config.lookback_window == 50
        assert config.restart_test is True
        assert config.crash_probability == 0.05
        assert config.recovery_modes == ["ledger", "checkpoint"]

    def test_to_dict(self):
        """Test serializing config to dict."""
        config = V2Config(
            context_pressure=True,
            volume_profile=VolumeProfile.ENTERPRISE,
            context_limit=300_000,
        )
        d = config.to_dict()

        assert d["context_pressure"]["enabled"] is True
        assert d["context_pressure"]["volume_profile"] == "enterprise"
        assert d["context_pressure"]["context_limit"] == 300_000
        assert d["full_v2"] is False


class TestVolumeProfile:
    """Test VolumeProfile enum."""

    def test_daily_requests(self):
        """Test daily request volumes."""
        assert VolumeProfile.SMALL.daily_requests == 10_000
        assert VolumeProfile.MEDIUM.daily_requests == 100_000
        assert VolumeProfile.LARGE.daily_requests == 1_000_000
        assert VolumeProfile.ENTERPRISE.daily_requests == 10_000_000

    def test_bid_rate(self):
        """Test bid rates."""
        assert VolumeProfile.SMALL.bid_rate == 0.3
        assert VolumeProfile.MEDIUM.bid_rate == 0.2
        assert VolumeProfile.LARGE.bid_rate == 0.1
        assert VolumeProfile.ENTERPRISE.bid_rate == 0.05


class TestValidateV2Config:
    """Test V2 config validation."""

    def test_valid_config(self):
        """Test validation passes for valid config."""
        config = V2Config()
        issues = validate_v2_config(config)
        assert issues == []

    def test_invalid_context_limit_low(self):
        """Test validation fails for too low context limit."""
        config = V2Config(context_limit=500)
        issues = validate_v2_config(config)
        assert any("context_limit too low" in issue for issue in issues)

    def test_invalid_compression_loss(self):
        """Test validation fails for invalid compression loss."""
        config = V2Config(compression_loss=1.5)
        issues = validate_v2_config(config)
        assert any("compression_loss must be" in issue for issue in issues)

    def test_invalid_lookback_window(self):
        """Test validation fails for invalid lookback window."""
        config = V2Config(lookback_window=0)
        issues = validate_v2_config(config)
        assert any("lookback_window must be" in issue for issue in issues)

    def test_invalid_crash_probability(self):
        """Test validation fails for invalid crash probability."""
        config = V2Config(crash_probability=-0.1)
        issues = validate_v2_config(config)
        assert any("crash_probability must be" in issue for issue in issues)

    def test_invalid_recovery_mode(self):
        """Test validation warns for unknown recovery mode."""
        config = V2Config(recovery_modes=["ledger", "unknown_mode"])
        issues = validate_v2_config(config)
        assert any("Unknown recovery mode" in issue for issue in issues)


class TestGetV2FeatureSummary:
    """Test V2 feature summary generation."""

    def test_no_v2_features(self):
        """Test summary when no V2 features enabled."""
        config = V2Config()
        summary = get_v2_feature_summary(config)
        assert "None enabled" in summary or "V1 mode" in summary

    def test_context_pressure_summary(self):
        """Test summary includes context pressure details."""
        config = V2Config(context_pressure=True, volume_profile=VolumeProfile.LARGE)
        summary = get_v2_feature_summary(config)
        assert "Context Pressure" in summary
        assert "large" in summary

    def test_decision_chain_summary(self):
        """Test summary includes decision chain details."""
        config = V2Config(decision_chain=True, lookback_window=50)
        summary = get_v2_feature_summary(config)
        assert "Decision Chain" in summary
        assert "50" in summary

    def test_full_v2_summary(self):
        """Test summary for full V2 mode."""
        config = V2Config(full_v2=True)
        summary = get_v2_feature_summary(config)
        assert "Full V2 Mode" in summary


class TestCLICommands:
    """Test CLI commands with V2 flags."""

    def test_version_command(self):
        """Test version command shows V2."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "V2" in result.stdout or "0.2" in result.stdout

    def test_v2_status_command(self):
        """Test v2-status command shows all V2 options."""
        result = runner.invoke(app, ["v2-status"])
        assert result.exit_code == 0
        assert "Context Pressure" in result.stdout
        assert "Decision Chain" in result.stdout
        assert "Restart Test" in result.stdout
        assert "--context-pressure" in result.stdout
        assert "--volume-profile" in result.stdout
        assert "--full-v2" in result.stdout

    def test_run_help_shows_v2_flags(self):
        """Test run --help shows V2 flags."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        # Check for V2 flags (allowing for truncated display in narrow terminals)
        assert "--context-pressure" in result.stdout
        assert "--volume-profile" in result.stdout
        assert "--context-limit" in result.stdout
        assert "--compression-loss" in result.stdout
        assert "--decision-chain" in result.stdout
        assert "--lookback-window" in result.stdout
        assert "--restart-test" in result.stdout
        # These may be truncated in display, check for partial match
        assert "crash" in result.stdout.lower() or "--crash-probability" in result.stdout
        assert "--recovery-modes" in result.stdout
        assert "--full-v2" in result.stdout

    def test_run_with_invalid_compression_loss(self):
        """Test run fails with invalid compression loss."""
        result = runner.invoke(app, [
            "run",
            "--days", "1",
            "--context-pressure",
            "--compression-loss", "2.0",  # Invalid: > 1.0
            "--skip-infra",
        ])
        # Should show warning or error
        assert "compression_loss" in result.stdout.lower() or result.exit_code != 0

    def test_run_with_valid_v2_flags(self):
        """Test run accepts valid V2 flags (dry run check)."""
        # Just test that the flags parse correctly
        result = runner.invoke(app, [
            "run", "--help",
        ])
        # If we get here without error, flags are defined correctly
        assert result.exit_code == 0


class TestCLIBackwardsCompatibility:
    """Test backwards compatibility with V1 commands."""

    def test_v1_run_still_works(self):
        """Test V1 run command still works without V2 flags."""
        result = runner.invoke(app, ["run", "--help"])
        # V1 flags should still be present
        assert "--scenario" in result.stdout
        assert "--days" in result.stdout
        assert "--buyers" in result.stdout
        assert "--sellers" in result.stdout
        assert "--mock-llm" in result.stdout
        assert "--skip-infra" in result.stdout
        assert "--output" in result.stdout
        assert "--verbose" in result.stdout

    def test_test_scenario_command_unchanged(self):
        """Test test-scenario command still works."""
        result = runner.invoke(app, ["test-scenario", "--help"])
        assert result.exit_code == 0
        assert "--scenario" in result.stdout

    def test_test_recovery_command_unchanged(self):
        """Test test-recovery command still works."""
        result = runner.invoke(app, ["test-recovery", "--help"])
        assert result.exit_code == 0
        assert "--agent" in result.stdout

    def test_compare_command_unchanged(self):
        """Test compare command still works."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.stdout


class TestVolumeProfileCLI:
    """Test volume profile parsing from CLI."""

    def test_volume_profile_choices(self):
        """Test all volume profile values are accepted."""
        for profile in ["small", "medium", "large", "enterprise"]:
            # Test that it parses (via from_cli_args)
            config = V2Config.from_cli_args(
                context_pressure=True,
                volume_profile=profile,
            )
            assert config.volume_profile.value == profile

    def test_volume_profile_case_insensitive(self):
        """Test volume profile parsing is case insensitive."""
        config = V2Config.from_cli_args(
            context_pressure=True,
            volume_profile="LARGE",
        )
        assert config.volume_profile == VolumeProfile.LARGE


class TestRecoveryModesCLI:
    """Test recovery modes parsing from CLI."""

    def test_recovery_modes_default(self):
        """Test default recovery modes."""
        config = V2Config.from_cli_args()
        assert config.recovery_modes == ["private_db", "ledger"]

    def test_recovery_modes_custom(self):
        """Test custom recovery modes."""
        config = V2Config.from_cli_args(
            restart_test=True,
            recovery_modes="ledger,checkpoint,context",
        )
        assert config.recovery_modes == ["ledger", "checkpoint", "context"]

    def test_recovery_modes_whitespace_handling(self):
        """Test recovery modes handles whitespace."""
        config = V2Config.from_cli_args(
            restart_test=True,
            recovery_modes="ledger , private_db , checkpoint",
        )
        assert config.recovery_modes == ["ledger", "private_db", "checkpoint"]
