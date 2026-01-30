"""Tests for Buyer L2 Channel Specialists.

Tests cover:
- Each specialist's channel-specific logic
- L1 context receiving (via context passing)
- L3 delegation planning
- System prompts and capabilities

Author: Alkimi Exchange
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.agents.base.context import AgentContext, ContextPriority
from src.agents.base.specialist import (
    SpecialistAgent,
    DelegationRequest,
    DelegationResult,
    SpecialistAgentState,
)
from src.agents.base.functional import FunctionalAgent
from src.agents.buyer.l2_branding import BrandingSpecialist, create_branding_specialist
from src.agents.buyer.l2_mobile_app import MobileAppSpecialist, create_mobile_app_specialist
from src.agents.buyer.l2_ctv import CTVSpecialist, create_ctv_specialist
from src.agents.buyer.l2_performance import PerformanceSpecialist, create_performance_specialist
from src.agents.buyer.l2_dsp import DSPSpecialist, create_dsp_specialist


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Create a mock AsyncAnthropic client."""
    mock_client = MagicMock()
    
    # Create async mock for messages.create
    async def mock_create(**kwargs):
        response = MagicMock()
        response.content = [MagicMock(text=json.dumps({
            "considerations": ["Test consideration"],
            "approach": "Test approach",
            "risks": ["Test risk"],
            "required_capabilities": ["test_capability"]
        }))]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        return response
    
    mock_client.messages.create = mock_create
    return mock_client


@pytest.fixture
def sample_context():
    """Create a sample AgentContext."""
    ctx = AgentContext(
        agent_id="test-agent",
        task_description="Test campaign execution",
        task_constraints=["budget_limit", "brand_safety"]
    )
    ctx.add_item("campaign_id", "camp-001", ContextPriority.CRITICAL)
    ctx.add_item("budget", 100000, ContextPriority.HIGH)
    ctx.add_item("objective", "awareness", ContextPriority.HIGH)
    return ctx


@pytest.fixture
def mock_functional_agent():
    """Create a mock L3 functional agent."""
    agent = MagicMock(spec=FunctionalAgent)
    agent.agent_id = "research"
    agent.name = "ResearchAgent"
    
    async def mock_execute(context, task):
        return {
            "success": True,
            "results": ["result1", "result2"],
            "final_response": "Task completed"
        }
    
    async def mock_initialize():
        pass
    
    async def mock_cleanup():
        pass
    
    agent.execute = mock_execute
    agent.initialize = mock_initialize
    agent.cleanup = mock_cleanup
    return agent


# ============================================================================
# Branding Specialist Tests
# ============================================================================

class TestBrandingSpecialist:
    """Tests for BrandingSpecialist."""
    
    def test_channel_attribute(self, mock_anthropic_client):
        """BrandingSpecialist has correct channel."""
        specialist = BrandingSpecialist(anthropic_client=mock_anthropic_client)
        assert specialist.channel == "branding"
        assert specialist.name == "BrandingSpecialist"
    
    def test_capabilities(self, mock_anthropic_client):
        """BrandingSpecialist has branding capabilities."""
        specialist = BrandingSpecialist(anthropic_client=mock_anthropic_client)
        capabilities = specialist.get_channel_capabilities()
        
        assert "premium_display" in capabilities
        assert "brand_safety_verification" in capabilities
        assert "viewability_optimization" in capabilities
        assert "reach_frequency_planning" in capabilities
    
    def test_system_prompt_content(self, mock_anthropic_client):
        """BrandingSpecialist system prompt mentions brand safety."""
        specialist = BrandingSpecialist(anthropic_client=mock_anthropic_client)
        prompt = specialist.get_system_prompt()
        
        assert "brand safety" in prompt.lower()
        assert "viewability" in prompt.lower()
        assert "premium" in prompt.lower()
        assert "L2" in prompt  # Hierarchy level
    
    @pytest.mark.asyncio
    async def test_plan_execution_creates_delegations(
        self,
        mock_anthropic_client,
        sample_context,
        mock_functional_agent,
    ):
        """Test plan_execution creates appropriate delegations."""
        specialist = BrandingSpecialist(anthropic_client=mock_anthropic_client)
        specialist.register_functional_agent(mock_functional_agent)
        
        delegations = await specialist.plan_execution(
            sample_context,
            "Launch Q1 brand awareness campaign"
        )
        
        assert len(delegations) >= 1
        assert all(isinstance(d, DelegationRequest) for d in delegations)
    
    def test_factory_function(self, mock_anthropic_client):
        """Test create_branding_specialist factory."""
        specialist = create_branding_specialist(anthropic_client=mock_anthropic_client)
        assert isinstance(specialist, BrandingSpecialist)
        assert specialist.channel == "branding"


# ============================================================================
# Mobile App Specialist Tests
# ============================================================================

class TestMobileAppSpecialist:
    """Tests for MobileAppSpecialist."""
    
    def test_channel_attribute(self, mock_anthropic_client):
        """MobileAppSpecialist has correct channel."""
        specialist = MobileAppSpecialist(anthropic_client=mock_anthropic_client)
        assert specialist.channel == "mobile_app"
        assert specialist.name == "MobileAppSpecialist"
    
    def test_capabilities(self, mock_anthropic_client):
        """MobileAppSpecialist has mobile capabilities."""
        specialist = MobileAppSpecialist(anthropic_client=mock_anthropic_client)
        capabilities = specialist.get_channel_capabilities()
        
        assert "cpi_optimization" in capabilities
        assert "attribution_tracking" in capabilities
        assert "rewarded_video" in capabilities
        assert "fraud_prevention" in capabilities
    
    def test_system_prompt_content(self, mock_anthropic_client):
        """MobileAppSpecialist system prompt mentions CPI."""
        specialist = MobileAppSpecialist(anthropic_client=mock_anthropic_client)
        prompt = specialist.get_system_prompt()
        
        assert "cpi" in prompt.lower()
        assert "app" in prompt.lower()
        assert "attribution" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_plan_execution_creates_delegations(
        self,
        mock_anthropic_client,
        sample_context,
        mock_functional_agent,
    ):
        """Test plan_execution creates appropriate delegations."""
        specialist = MobileAppSpecialist(anthropic_client=mock_anthropic_client)
        specialist.register_functional_agent(mock_functional_agent)
        
        delegations = await specialist.plan_execution(
            sample_context,
            "Drive app installs for gaming app"
        )
        
        assert len(delegations) >= 1
        assert all(isinstance(d, DelegationRequest) for d in delegations)
    
    def test_factory_function(self, mock_anthropic_client):
        """Test create_mobile_app_specialist factory."""
        specialist = create_mobile_app_specialist(anthropic_client=mock_anthropic_client)
        assert isinstance(specialist, MobileAppSpecialist)


# ============================================================================
# CTV Specialist Tests
# ============================================================================

class TestCTVSpecialist:
    """Tests for CTVSpecialist."""
    
    def test_channel_attribute(self, mock_anthropic_client):
        """CTVSpecialist has correct channel."""
        specialist = CTVSpecialist(anthropic_client=mock_anthropic_client)
        assert specialist.channel == "ctv"
        assert specialist.name == "CTVSpecialist"
    
    def test_capabilities(self, mock_anthropic_client):
        """CTVSpecialist has CTV capabilities."""
        specialist = CTVSpecialist(anthropic_client=mock_anthropic_client)
        capabilities = specialist.get_channel_capabilities()
        
        assert "household_reach" in capabilities
        assert "streaming_platform_selection" in capabilities
        assert "programmatic_guaranteed" in capabilities
        assert "cross_device_attribution" in capabilities
    
    def test_system_prompt_content(self, mock_anthropic_client):
        """CTVSpecialist system prompt mentions streaming."""
        specialist = CTVSpecialist(anthropic_client=mock_anthropic_client)
        prompt = specialist.get_system_prompt()
        
        assert "streaming" in prompt.lower()
        assert "household" in prompt.lower()
        assert "ctv" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_plan_execution_creates_delegations(
        self,
        mock_anthropic_client,
        sample_context,
        mock_functional_agent,
    ):
        """Test plan_execution creates appropriate delegations."""
        specialist = CTVSpecialist(anthropic_client=mock_anthropic_client)
        specialist.register_functional_agent(mock_functional_agent)
        
        delegations = await specialist.plan_execution(
            sample_context,
            "Run CTV campaign on premium streaming"
        )
        
        assert len(delegations) >= 1
        assert all(isinstance(d, DelegationRequest) for d in delegations)
    
    def test_factory_function(self, mock_anthropic_client):
        """Test create_ctv_specialist factory."""
        specialist = create_ctv_specialist(anthropic_client=mock_anthropic_client)
        assert isinstance(specialist, CTVSpecialist)


# ============================================================================
# Performance Specialist Tests
# ============================================================================

class TestPerformanceSpecialist:
    """Tests for PerformanceSpecialist."""
    
    def test_channel_attribute(self, mock_anthropic_client):
        """PerformanceSpecialist has correct channel."""
        specialist = PerformanceSpecialist(anthropic_client=mock_anthropic_client)
        assert specialist.channel == "performance"
        assert specialist.name == "PerformanceSpecialist"
    
    def test_capabilities(self, mock_anthropic_client):
        """PerformanceSpecialist has performance capabilities."""
        specialist = PerformanceSpecialist(anthropic_client=mock_anthropic_client)
        capabilities = specialist.get_channel_capabilities()
        
        assert "roas_optimization" in capabilities
        assert "conversion_tracking" in capabilities
        assert "retargeting_strategies" in capabilities
        assert "attribution_modeling" in capabilities
    
    def test_system_prompt_content(self, mock_anthropic_client):
        """PerformanceSpecialist system prompt mentions ROAS."""
        specialist = PerformanceSpecialist(anthropic_client=mock_anthropic_client)
        prompt = specialist.get_system_prompt()
        
        assert "roas" in prompt.lower()
        assert "conversion" in prompt.lower()
        assert "attribution" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_plan_execution_creates_delegations(
        self,
        mock_anthropic_client,
        sample_context,
        mock_functional_agent,
    ):
        """Test plan_execution creates appropriate delegations."""
        specialist = PerformanceSpecialist(anthropic_client=mock_anthropic_client)
        specialist.register_functional_agent(mock_functional_agent)
        
        delegations = await specialist.plan_execution(
            sample_context,
            "Optimize for ROAS on e-commerce campaign"
        )
        
        assert len(delegations) >= 1
        assert all(isinstance(d, DelegationRequest) for d in delegations)
    
    def test_factory_function(self, mock_anthropic_client):
        """Test create_performance_specialist factory."""
        specialist = create_performance_specialist(anthropic_client=mock_anthropic_client)
        assert isinstance(specialist, PerformanceSpecialist)


# ============================================================================
# DSP Specialist Tests
# ============================================================================

class TestDSPSpecialist:
    """Tests for DSPSpecialist."""
    
    def test_channel_attribute(self, mock_anthropic_client):
        """DSPSpecialist has correct channel."""
        specialist = DSPSpecialist(anthropic_client=mock_anthropic_client)
        assert specialist.channel == "dsp"
        assert specialist.name == "DSPSpecialist"
    
    def test_capabilities(self, mock_anthropic_client):
        """DSPSpecialist has DSP capabilities."""
        specialist = DSPSpecialist(anthropic_client=mock_anthropic_client)
        capabilities = specialist.get_channel_capabilities()
        
        assert "rtb_optimization" in capabilities
        assert "supply_path_optimization" in capabilities
        assert "deal_prioritization" in capabilities
        assert "bid_shading" in capabilities
    
    def test_system_prompt_content(self, mock_anthropic_client):
        """DSPSpecialist system prompt mentions RTB."""
        specialist = DSPSpecialist(anthropic_client=mock_anthropic_client)
        prompt = specialist.get_system_prompt()
        
        assert "rtb" in prompt.lower()
        assert "bidding" in prompt.lower()
        assert "supply path" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_plan_execution_creates_delegations(
        self,
        mock_anthropic_client,
        sample_context,
        mock_functional_agent,
    ):
        """Test plan_execution creates appropriate delegations."""
        specialist = DSPSpecialist(anthropic_client=mock_anthropic_client)
        specialist.register_functional_agent(mock_functional_agent)
        
        delegations = await specialist.plan_execution(
            sample_context,
            "Optimize DSP bidding strategy"
        )
        
        assert len(delegations) >= 1
        assert all(isinstance(d, DelegationRequest) for d in delegations)
    
    def test_factory_function(self, mock_anthropic_client):
        """Test create_dsp_specialist factory."""
        specialist = create_dsp_specialist(anthropic_client=mock_anthropic_client)
        assert isinstance(specialist, DSPSpecialist)


# ============================================================================
# L3 Delegation Tests
# ============================================================================

class TestL3Delegation:
    """Test L3 agent delegation mechanics."""
    
    def test_register_functional_agent(self, mock_anthropic_client, mock_functional_agent):
        """Test registering L3 functional agents."""
        specialist = BrandingSpecialist(anthropic_client=mock_anthropic_client)
        
        specialist.register_functional_agent(mock_functional_agent)
        
        assert mock_functional_agent.agent_id in specialist._functional_agents
    
    @pytest.mark.asyncio
    async def test_delegation_context_additions(
        self,
        mock_anthropic_client,
        sample_context,
        mock_functional_agent,
    ):
        """Test that delegations include appropriate context additions."""
        specialist = BrandingSpecialist(anthropic_client=mock_anthropic_client)
        specialist.register_functional_agent(mock_functional_agent)
        
        delegations = await specialist.plan_execution(
            sample_context,
            "Test objective"
        )
        
        # Branding specialist should include brand safety context
        research_delegation = next(
            (d for d in delegations if "research" in d.functional_agent_id.lower()),
            None
        )
        
        if research_delegation:
            assert "channel" in research_delegation.context_additions
            assert research_delegation.context_additions["channel"] == "branding"


# ============================================================================
# All Specialists Instantiation Test
# ============================================================================

class TestAllSpecialistsInstantiation:
    """Ensure all specialists can be instantiated."""
    
    def test_all_specialists_instantiate(self, mock_anthropic_client):
        """All 5 specialists should instantiate correctly."""
        specialists = [
            BrandingSpecialist(anthropic_client=mock_anthropic_client),
            MobileAppSpecialist(anthropic_client=mock_anthropic_client),
            CTVSpecialist(anthropic_client=mock_anthropic_client),
            PerformanceSpecialist(anthropic_client=mock_anthropic_client),
            DSPSpecialist(anthropic_client=mock_anthropic_client),
        ]
        
        channels = [s.channel for s in specialists]
        
        assert "branding" in channels
        assert "mobile_app" in channels
        assert "ctv" in channels
        assert "performance" in channels
        assert "dsp" in channels
        
        # All should be SpecialistAgent subclasses
        for s in specialists:
            assert isinstance(s, SpecialistAgent)
            assert s.get_system_prompt() is not None
            assert len(s.get_channel_capabilities()) > 0
    
    def test_all_specialists_have_unique_channels(self, mock_anthropic_client):
        """Each specialist should have a unique channel."""
        specialists = [
            BrandingSpecialist(anthropic_client=mock_anthropic_client),
            MobileAppSpecialist(anthropic_client=mock_anthropic_client),
            CTVSpecialist(anthropic_client=mock_anthropic_client),
            PerformanceSpecialist(anthropic_client=mock_anthropic_client),
            DSPSpecialist(anthropic_client=mock_anthropic_client),
        ]
        
        channels = [s.channel for s in specialists]
        assert len(channels) == len(set(channels)), "Channels should be unique"


# ============================================================================
# Context Flow Tests
# ============================================================================

class TestContextFlow:
    """Test context passing from L1 through L2."""
    
    @pytest.mark.asyncio
    async def test_context_propagation(
        self,
        mock_anthropic_client,
        sample_context,
        mock_functional_agent,
    ):
        """Test that context flows through to delegation requests."""
        specialist = BrandingSpecialist(anthropic_client=mock_anthropic_client)
        specialist.register_functional_agent(mock_functional_agent)
        
        # Add L1-specific context
        sample_context.add_item("l1_budget_allocation", 50000, ContextPriority.HIGH)
        sample_context.add_item("l1_priority", "high", ContextPriority.MEDIUM)
        
        delegations = await specialist.plan_execution(
            sample_context,
            "Execute brand campaign"
        )
        
        # Delegations should be created based on context
        assert len(delegations) >= 1
        
        # Each delegation should have priority set
        for d in delegations:
            assert d.priority in [
                ContextPriority.CRITICAL,
                ContextPriority.HIGH,
                ContextPriority.MEDIUM,
                ContextPriority.LOW
            ]
