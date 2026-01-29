"""Tests to verify the repository scaffold is correctly set up."""

import pytest
from pathlib import Path


class TestScaffold:
    """Verify repository structure is correct."""

    def test_pyproject_exists(self):
        """pyproject.toml should exist."""
        assert Path("pyproject.toml").exists()

    def test_docker_compose_exists(self):
        """docker-compose.yml should exist."""
        assert Path("docker/docker-compose.yml").exists()

    def test_sql_init_scripts_exist(self):
        """PostgreSQL init scripts should exist."""
        assert Path("docker/postgres/init.sql").exists()
        assert Path("docker/postgres/ground_truth.sql").exists()
        assert Path("docker/postgres/ledger.sql").exists()

    def test_source_directories_exist(self):
        """Source directories should exist."""
        expected_dirs = [
            "src",
            "src/infrastructure",
            "src/agents",
            "src/agents/buyer",
            "src/agents/seller",
            "src/agents/exchange",
            "src/agents/ucp",
            "src/scenarios",
            "src/orchestration",
            "src/metrics",
            "src/logging",
            "src/models",
            "src/config",
            "src/data",
        ]
        for dir_path in expected_dirs:
            assert Path(dir_path).is_dir(), f"Directory {dir_path} should exist"

    def test_test_directories_exist(self):
        """Test directories should exist."""
        assert Path("tests").is_dir()
        assert Path("tests/integration").is_dir()
        assert Path("tests/hallucination").is_dir()

    def test_env_example_exists(self):
        """Environment example file should exist."""
        assert Path(".env.example").exists()

    def test_grafana_config_exists(self):
        """Grafana configuration should exist."""
        assert Path("docker/grafana/provisioning/datasources/datasources.yml").exists()
        assert Path("docker/grafana/provisioning/dashboards/dashboards.yml").exists()


class TestPyprojectContent:
    """Verify pyproject.toml has required content."""

    def test_has_required_dependencies(self):
        """pyproject.toml should have required dependencies."""
        content = Path("pyproject.toml").read_text()
        required = ["pydantic", "sqlalchemy", "redis", "influxdb-client", "typer"]
        for dep in required:
            assert dep in content, f"Dependency {dep} should be in pyproject.toml"

    def test_has_pytest_config(self):
        """pyproject.toml should have pytest configuration."""
        content = Path("pyproject.toml").read_text()
        assert "pytest" in content
        assert "asyncio_mode" in content


class TestDockerComposeContent:
    """Verify docker-compose.yml has required services."""

    def test_has_required_services(self):
        """docker-compose.yml should define required services."""
        content = Path("docker/docker-compose.yml").read_text()
        required_services = ["postgres", "redis", "influxdb", "grafana"]
        for service in required_services:
            assert service in content, f"Service {service} should be in docker-compose.yml"

    def test_has_healthchecks(self):
        """Services should have health checks."""
        content = Path("docker/docker-compose.yml").read_text()
        assert "healthcheck" in content


class TestFixtures:
    """Test that fixtures work correctly."""

    def test_sample_campaign_fixture(self, sample_campaign):
        """Sample campaign fixture should have required fields."""
        assert "id" in sample_campaign
        assert "buyer_id" in sample_campaign
        assert "total_budget" in sample_campaign
        assert "target_cpm" in sample_campaign
        assert "scenario" in sample_campaign

    def test_sample_publisher_fixture(self, sample_publisher):
        """Sample publisher fixture should have required fields."""
        assert "id" in sample_publisher
        assert "name" in sample_publisher
        assert "floor_cpm" in sample_publisher
        assert "channels" in sample_publisher
