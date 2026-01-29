"""Configuration for buyer agents in RTB simulation."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class BuyerAgentSettings(BaseSettings):
    """Settings for buyer agents in RTB simulation.

    These settings configure how IAB buyer-agent CrewAI flows operate
    within the simulation environment.
    """

    # LLM Configuration
    anthropic_api_key: str = ""
    default_llm_model: str = "anthropic/claude-3-haiku-20240307"
    manager_llm_model: str = "anthropic/claude-3-haiku-20240307"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Mock LLM for testing (no API calls)
    mock_llm: bool = True

    # CrewAI Settings
    crew_memory_enabled: bool = False  # Disable for simulation speed
    crew_verbose: bool = False
    crew_max_iterations: int = 5  # Fewer iterations for simulation

    # Simulation Behavior
    scenario: str = "A"  # Default scenario
    hallucination_rate: float = 0.05  # 5% hallucination rate
    context_decay_rate: float = 0.02  # 2% per simulated day

    # Bidding Strategy
    default_bid_strategy: str = "target_cpm"  # target_cpm, maximize_reach, floor_plus
    max_cpm_multiplier: float = 1.2  # Max bid = target * multiplier
    min_cpm_floor: float = 5.0  # Never bid below this

    # Campaign Defaults
    default_channel: str = "display"
    default_deal_type: str = "OA"  # Open Auction

    # Redis Configuration (inherited from infrastructure)
    redis_url: str = "redis://localhost:6379"
    consumer_group_prefix: str = "buyer"

    model_config = {
        "env_prefix": "RTB_BUYER_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_buyer_settings() -> BuyerAgentSettings:
    """Get cached buyer settings instance."""
    return BuyerAgentSettings()


buyer_settings = get_buyer_settings()
