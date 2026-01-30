"""
Tests for IAB Simulation Protocol Handlers.

Tests A2A, UCP, AA protocols and inter-level communication.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta

from src.protocols.a2a import (
    A2AProtocol,
    A2AMessage,
    A2AResponse,
    Offer,
    NegotiationResult,
    NegotiationStatus,
    MessagePriority,
)
from src.protocols.ucp import (
    UCPProtocol,
    UCPEmbedding,
    AudienceSpec,
    Inventory,
    MatchResult,
    EmbeddingType,
)
from src.protocols.aa import (
    AAProtocol,
    AudienceSegment,
    CampaignBrief,
    ValidationResult,
    ReachEstimate,
    SegmentType,
    ValidationStatus,
)
from src.protocols.inter_level import (
    InterLevelProtocol,
    ContextSerializer,
    AgentContext,
    Task,
    Result,
    DelegationResult,
    AckResult,
    TaskPriority,
    TaskStatus,
    ResultStatus,
)


# =============================================================================
# A2A Protocol Tests
# =============================================================================

class TestA2AMessage:
    """Tests for A2AMessage dataclass."""
    
    def test_create_message(self):
        """Test message creation with defaults."""
        msg = A2AMessage(
            sender="agent_1",
            recipient="agent_2",
            content="Hello, can we negotiate?",
        )
        
        assert msg.sender == "agent_1"
        assert msg.recipient == "agent_2"
        assert msg.content == "Hello, can we negotiate?"
        assert msg.priority == MessagePriority.NORMAL
        assert msg.message_id is not None
        assert msg.timestamp is not None
    
    def test_message_serialization(self):
        """Test message to_dict and from_dict."""
        msg = A2AMessage(
            sender="agent_1",
            recipient="agent_2",
            content="Test message",
            context={"key": "value"},
            priority=MessagePriority.HIGH,
        )
        
        data = msg.to_dict()
        restored = A2AMessage.from_dict(data)
        
        assert restored.sender == msg.sender
        assert restored.recipient == msg.recipient
        assert restored.content == msg.content
        assert restored.context == msg.context
        assert restored.priority == msg.priority


class TestOffer:
    """Tests for negotiation Offer."""
    
    def test_create_offer(self):
        """Test offer creation."""
        offer = Offer.create(
            offerer="buyer_1",
            recipient="seller_1",
            terms={"cpm": 5.0, "impressions": 1000000},
            description="Initial offer for premium inventory",
        )
        
        assert offer.offerer == "buyer_1"
        assert offer.recipient == "seller_1"
        assert offer.terms["cpm"] == 5.0
        assert offer.offer_id is not None
        assert offer.valid_until is not None
    
    def test_offer_expiration(self):
        """Test offer expiration check."""
        # Expired offer
        expired = Offer(
            offer_id="test",
            offerer="a",
            recipient="b",
            terms={},
            description="",
            valid_until=datetime.utcnow() - timedelta(seconds=10),
        )
        assert expired.is_expired()
        
        # Valid offer
        valid = Offer.create(
            offerer="a",
            recipient="b",
            terms={},
            description="",
            valid_for_seconds=300,
        )
        assert not valid.is_expired()


class TestA2AProtocol:
    """Tests for A2AProtocol."""
    
    @pytest.fixture
    def protocol(self):
        return A2AProtocol(agent_id="test_agent")
    
    @pytest.mark.asyncio
    async def test_send_message(self, protocol):
        """Test sending a message."""
        response = await protocol.send_message(
            recipient="other_agent",
            message="Hello there!",
        )
        
        assert response.success
        assert response.original_message_id is not None
    
    @pytest.mark.asyncio
    async def test_deliver_and_receive_message(self, protocol):
        """Test delivering and receiving a message."""
        msg = A2AMessage(
            sender="other_agent",
            recipient="test_agent",
            content="Incoming message",
        )
        
        await protocol.deliver_message(msg)
        received = await protocol.receive_message(timeout=1.0)
        
        assert received.content == "Incoming message"
        assert received.sender == "other_agent"
    
    def test_conversation_tracking(self, protocol):
        """Test conversation retrieval."""
        # Initially empty
        assert protocol.get_conversation("conv_1") == []
        
        # After sending messages (they get tracked)
        asyncio.run(protocol.send_message(
            recipient="other",
            message="Test",
            conversation_id="conv_1",
        ))
        
        conv = protocol.get_conversation("conv_1")
        assert len(conv) == 1
        assert conv[0].content == "Test"


# =============================================================================
# UCP Protocol Tests
# =============================================================================

class TestAudienceSpec:
    """Tests for AudienceSpec."""
    
    def test_create_spec(self):
        """Test audience spec creation."""
        spec = AudienceSpec(
            target_attributes=["in_market_auto", "high_income"],
            preferred_attributes=["age_25_34"],
            geo_targets=["US", "UK"],
        )
        
        assert len(spec.target_attributes) == 2
        assert "in_market_auto" in spec.target_attributes
    
    def test_spec_serialization(self):
        """Test spec to_dict and from_dict."""
        spec = AudienceSpec(
            target_attributes=["sports_enthusiast"],
            min_reach=1000000,
        )
        
        data = spec.to_dict()
        restored = AudienceSpec.from_dict(data)
        
        assert restored.target_attributes == spec.target_attributes
        assert restored.min_reach == spec.min_reach


class TestUCPEmbedding:
    """Tests for UCPEmbedding."""
    
    def test_create_embedding(self):
        """Test embedding creation."""
        embedding = UCPEmbedding(
            embedding_type="user_intent",
            dimension=512,
            vector=[0.1] * 512,
        )
        
        assert embedding.dimension == 512
        assert len(embedding.vector) == 512
    
    def test_embedding_normalization(self):
        """Test embedding normalization."""
        embedding = UCPEmbedding(
            dimension=3,
            vector=[3.0, 4.0, 0.0],
        )
        
        normalized = embedding.normalize()
        
        # L2 norm should be 1
        import math
        magnitude = math.sqrt(sum(x * x for x in normalized.vector))
        assert abs(magnitude - 1.0) < 0.001
    
    def test_dimension_validation(self):
        """Test dimension mismatch raises error."""
        with pytest.raises(ValueError):
            UCPEmbedding(
                dimension=512,
                vector=[0.1] * 100,  # Wrong size
            )


class TestUCPProtocol:
    """Tests for UCPProtocol."""
    
    @pytest.fixture
    def protocol(self):
        return UCPProtocol(dimension=128)  # Smaller for tests
    
    @pytest.mark.asyncio
    async def test_create_query_embedding(self, protocol):
        """Test query embedding creation."""
        spec = AudienceSpec(
            target_attributes=["in_market_auto", "high_income"],
            geo_targets=["US"],
        )
        
        embedding = await protocol.create_query_embedding(spec)
        
        assert embedding.embedding_type == EmbeddingType.USER_INTENT.value
        assert len(embedding.vector) == 128
        assert embedding.source_hash is not None
    
    @pytest.mark.asyncio
    async def test_create_inventory_embedding(self, protocol):
        """Test inventory embedding creation."""
        inventory = Inventory(
            inventory_id="inv_1",
            name="Premium Auto Sites",
            available_attributes=["in_market_auto", "age_25_34"],
            estimated_reach=5000000,
        )
        
        embedding = await protocol.create_inventory_embedding(inventory)
        
        assert embedding.embedding_type == EmbeddingType.INVENTORY.value
        assert len(embedding.vector) == 128
    
    @pytest.mark.asyncio
    async def test_embedding_matching(self, protocol):
        """Test matching query to inventory."""
        # Create similar spec and inventory
        spec = AudienceSpec(
            target_attributes=["in_market_auto", "high_income"],
        )
        inventory = Inventory(
            inventory_id="inv_1",
            name="Auto Inventory",
            available_attributes=["in_market_auto", "high_income", "age_35_44"],
            estimated_reach=10000000,
        )
        
        query_emb = await protocol.create_query_embedding(spec)
        inv_emb = await protocol.create_inventory_embedding(inventory)
        
        result = await protocol.match_embeddings(
            query_emb, inv_emb,
            query_spec=spec,
            inventory_data=inventory,
        )
        
        assert isinstance(result, MatchResult)
        assert 0 <= result.similarity_score <= 1
        assert result.coverage_percentage == 1.0  # All target attrs available
        assert "in_market_auto" in result.matched_capabilities
        assert len(result.missing_capabilities) == 0


# =============================================================================
# AA Protocol Tests
# =============================================================================

class TestCampaignBrief:
    """Tests for CampaignBrief."""
    
    def test_create_brief(self):
        """Test brief creation."""
        brief = CampaignBrief.create(
            advertiser="Acme Auto",
            objective="brand_awareness",
            budget=100000,
            target_audience="luxury car buyers aged 35-54",
        )
        
        assert brief.brief_id is not None
        assert brief.budget == 100000
    
    def test_brief_serialization(self):
        """Test brief to_dict and from_dict."""
        brief = CampaignBrief.create(
            advertiser="Test",
            objective="conversions",
            budget=50000,
            target_audience="tech enthusiasts",
            geo_targets=["US", "UK"],
        )
        
        data = brief.to_dict()
        restored = CampaignBrief.from_dict(data)
        
        assert restored.advertiser == brief.advertiser
        assert restored.budget == brief.budget
        assert restored.geo_targets == brief.geo_targets


class TestAAProtocol:
    """Tests for AAProtocol."""
    
    @pytest.fixture
    def protocol(self):
        return AAProtocol()
    
    @pytest.mark.asyncio
    async def test_discover_segments(self, protocol):
        """Test segment discovery from brief."""
        brief = CampaignBrief.create(
            advertiser="Auto Brand",
            objective="awareness",
            budget=200000,
            target_audience="luxury auto buyers with high income",
        )
        
        segments = await protocol.discover_segments(brief)
        
        assert len(segments) > 0
        # Should find auto-related segment
        auto_segment = next(
            (s for s in segments if "auto" in s.name.lower()),
            None
        )
        assert auto_segment is not None
    
    @pytest.mark.asyncio
    async def test_validate_targeting(self, protocol):
        """Test targeting validation."""
        segments = [
            AudienceSegment(
                segment_id="seg_1",
                name="Test Segment",
                segment_type=SegmentType.THIRD_PARTY,
                attributes=["in_market_auto"],
                estimated_size=5000000,
            )
        ]
        
        result = await protocol.validate_targeting(
            segments,
            inventory_id="inv_1",
            min_reach=1000,
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_achievable
        assert result.status in [ValidationStatus.VALID, ValidationStatus.PARTIAL]
    
    @pytest.mark.asyncio
    async def test_estimate_reach(self, protocol):
        """Test reach estimation."""
        segments = [
            AudienceSegment(
                segment_id="seg_1",
                name="Test",
                segment_type=SegmentType.BEHAVIORAL,
                estimated_size=10000000,
                cpm_range=(2.0, 4.0),
            )
        ]
        
        estimate = await protocol.estimate_reach(
            segments,
            budget=50000,
            target_frequency=3.0,
        )
        
        assert isinstance(estimate, ReachEstimate)
        assert estimate.total_reach > 0
        assert estimate.impressions > 0
        assert estimate.effective_cpm > 0


# =============================================================================
# Inter-Level Protocol Tests
# =============================================================================

class TestAgentContext:
    """Tests for AgentContext."""
    
    def test_create_context(self):
        """Test context creation."""
        ctx = AgentContext.create(
            agent_id="agent_1",
            level=2,
            working_memory={"current_task": "planning"},
        )
        
        assert ctx.context_id is not None
        assert ctx.level == 2
        assert ctx.working_memory["current_task"] == "planning"
    
    def test_context_expiration(self):
        """Test context expiration."""
        # Non-expiring context
        ctx = AgentContext.create(agent_id="a", level=1)
        assert not ctx.is_expired()
        
        # Expired context
        expired = AgentContext.create(
            agent_id="a",
            level=1,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert expired.is_expired()


class TestContextSerializer:
    """Tests for ContextSerializer."""
    
    @pytest.fixture
    def serializer(self):
        return ContextSerializer(default_limit=500)
    
    def test_token_counting(self, serializer):
        """Test token count estimation."""
        ctx = AgentContext.create(
            agent_id="agent_1",
            level=1,
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )
        
        tokens = serializer.to_tokens(ctx)
        assert tokens > 0
    
    def test_serialization_roundtrip(self, serializer):
        """Test serialize/deserialize roundtrip."""
        ctx = AgentContext.create(
            agent_id="test",
            level=2,
            working_memory={"key": "value"},
        )
        
        serialized = serializer.to_string(ctx)
        restored = serializer.from_string(serialized)
        
        assert restored.agent_id == ctx.agent_id
        assert restored.level == ctx.level
        assert restored.working_memory == ctx.working_memory
    
    def test_context_truncation(self, serializer):
        """Test context truncation (simulating context rot)."""
        # Create large context
        large_history = [
            {"role": "user", "content": f"Message {i} " * 50}
            for i in range(20)
        ]
        
        ctx = AgentContext.create(
            agent_id="test",
            level=1,
            conversation_history=large_history,
        )
        
        original_tokens = serializer.to_tokens(ctx)
        assert original_tokens > 500  # Exceeds limit
        
        truncated = serializer.truncate_to_limit(ctx, limit=500)
        
        assert serializer.to_tokens(truncated) <= 500
        assert len(truncated.conversation_history) < len(ctx.conversation_history)
        assert truncated.metadata.get("context_rot_applied") == True


class TestTask:
    """Tests for Task."""
    
    def test_create_task(self):
        """Test task creation."""
        task = Task.create(
            name="Research audiences",
            description="Find relevant audience segments",
            task_type="research",
            created_by="l2_agent",
        )
        
        assert task.task_id is not None
        assert task.status == TaskStatus.PENDING
    
    def test_task_overdue(self):
        """Test overdue detection."""
        task = Task.create(
            name="Test",
            description="Test",
            task_type="test",
            created_by="agent",
            deadline=datetime.utcnow() - timedelta(hours=1),
        )
        
        assert task.is_overdue()


class TestResult:
    """Tests for Result."""
    
    def test_success_result(self):
        """Test success result creation."""
        result = Result.success(
            task_id="task_1",
            output={"segments": ["seg_1", "seg_2"]},
            metrics={"accuracy": 0.95},
            execution_time_ms=1500,
        )
        
        assert result.status == ResultStatus.SUCCESS
        assert len(result.errors) == 0
    
    def test_failure_result(self):
        """Test failure result creation."""
        result = Result.failure(
            task_id="task_1",
            errors=["Timeout exceeded", "API rate limit"],
        )
        
        assert result.status == ResultStatus.FAILURE
        assert len(result.errors) == 2


class TestInterLevelProtocol:
    """Tests for InterLevelProtocol."""
    
    @pytest.fixture
    def l1_protocol(self):
        return InterLevelProtocol(agent_id="l1_agent", agent_level=1)
    
    @pytest.fixture
    def l2_protocol(self):
        return InterLevelProtocol(agent_id="l2_agent", agent_level=2)
    
    @pytest.mark.asyncio
    async def test_delegate_down(self, l1_protocol):
        """Test task delegation L1 -> L2."""
        l1_protocol.register_subordinate("l2_agent", level=2)
        
        task = Task.create(
            name="Analyze campaign",
            description="Analyze campaign performance",
            task_type="analysis",
            created_by="l1_agent",
        )
        
        result = await l1_protocol.delegate_down(
            from_level=1,
            to_level=2,
            task=task,
        )
        
        assert result.success
        assert result.assigned_agent == "l2_agent"
        assert task.task_id in [t.task_id for t in l1_protocol._pending_delegations.values()]
    
    @pytest.mark.asyncio
    async def test_invalid_delegation_path(self, l1_protocol):
        """Test rejection of invalid delegation path."""
        task = Task.create(
            name="Test",
            description="Test",
            task_type="test",
            created_by="l1_agent",
        )
        
        # L1 -> L3 is invalid (must go through L2)
        result = await l1_protocol.delegate_down(
            from_level=1,
            to_level=3,
            task=task,
        )
        
        assert not result.success
        assert "Invalid delegation path" in result.rejection_reason
    
    @pytest.mark.asyncio
    async def test_report_up(self, l2_protocol):
        """Test result reporting L2 -> L1."""
        result = Result.success(
            task_id="task_1",
            output={"status": "completed"},
        )
        
        ack = await l2_protocol.report_up(
            from_level=2,
            to_level=1,
            result=result,
        )
        
        assert ack.acknowledged
        assert ack.result_id == result.result_id
    
    @pytest.mark.asyncio
    async def test_report_failure_feedback(self, l2_protocol):
        """Test feedback on failed result."""
        result = Result.failure(
            task_id="task_1",
            errors=["Something went wrong"],
        )
        
        ack = await l2_protocol.report_up(
            from_level=2,
            to_level=1,
            result=result,
        )
        
        assert ack.acknowledged
        assert ack.feedback is not None
        assert "failed" in ack.feedback.lower()
    
    def test_delegation_stats(self, l1_protocol):
        """Test delegation statistics."""
        stats = l1_protocol.get_delegation_stats()
        
        assert "pending_count" in stats
        assert "completed_count" in stats
        assert stats["total_delegations"] == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestProtocolIntegration:
    """Integration tests combining multiple protocols."""
    
    @pytest.mark.asyncio
    async def test_ucp_aa_integration(self):
        """Test UCP and AA protocol integration."""
        ucp = UCPProtocol(dimension=128)
        aa = AAProtocol(ucp_protocol=ucp)
        
        # Create brief
        brief = CampaignBrief.create(
            advertiser="Test Brand",
            objective="reach",
            budget=100000,
            target_audience="sports fans who travel frequently",
        )
        
        # Discover segments
        segments = await aa.discover_segments(brief)
        
        # Validate
        if segments:
            validation = await aa.validate_targeting(
                segments[:3],
                inventory_id="test_inv",
            )
            
            # Estimate reach
            estimate = await aa.estimate_reach(
                segments[:3],
                budget=brief.budget,
            )
            
            assert validation is not None
            assert estimate.total_reach > 0
    
    @pytest.mark.asyncio
    async def test_inter_level_with_context(self):
        """Test inter-level communication with context passing."""
        l1 = InterLevelProtocol(agent_id="l1", agent_level=1)
        l2 = InterLevelProtocol(agent_id="l2", agent_level=2)
        
        l1.register_subordinate("l2", level=2)
        
        # Create context with history
        ctx = AgentContext.create(
            agent_id="l1",
            level=1,
            conversation_history=[
                {"role": "user", "content": "Plan a campaign for auto buyers"},
            ],
            working_memory={"campaign_type": "awareness"},
        )
        
        # Create task with context
        task = Task.create(
            name="Find audiences",
            description="Find relevant audience segments",
            task_type="research",
            created_by="l1",
            context=ctx,
        )
        
        # Delegate
        delegation = await l1.delegate_down(1, 2, task)
        assert delegation.success
        
        # L2 processes and reports back
        result = Result.success(
            task_id=task.task_id,
            output={"segments": ["auto_intenders", "luxury_buyers"]},
        )
        
        ack = await l2.report_up(2, 1, result)
        assert ack.acknowledged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
