"""Tests for Redis bus."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.infrastructure.redis_bus import RedisBus, create_redis_bus
from src.infrastructure.message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    STREAMS,
    CONSUMER_GROUPS,
)


class TestRedisBusUnit:
    """Unit tests for RedisBus (mocked Redis)."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.xadd = AsyncMock(return_value="1234567890-0")
        mock.xreadgroup = AsyncMock(return_value=[])
        mock.xack = AsyncMock(return_value=1)
        mock.xgroup_create = AsyncMock()
        mock.xinfo_stream = AsyncMock(return_value={"length": 0})
        mock.xpending = AsyncMock(return_value={"pending": 0})
        mock.xtrim = AsyncMock(return_value=0)
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    async def bus_with_mock(self, mock_redis):
        """Create a RedisBus with mocked client."""
        bus = RedisBus(url="redis://localhost:6379")
        bus._client = mock_redis
        bus._consumer_id = "test-consumer"
        return bus

    @pytest.mark.asyncio
    async def test_connect(self, mock_redis):
        """Test connecting to Redis."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            bus = RedisBus()
            await bus.connect(consumer_id="test-consumer")

            mock_redis.ping.assert_called_once()
            assert bus._consumer_id == "test-consumer"

    @pytest.mark.asyncio
    async def test_disconnect(self, bus_with_mock, mock_redis):
        """Test disconnecting from Redis."""
        await bus_with_mock.disconnect()

        mock_redis.close.assert_called_once()
        assert bus_with_mock._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_redis):
        """Test async context manager."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with RedisBus() as bus:
                assert bus._client is not None

    @pytest.mark.asyncio
    async def test_publish_bid_request(self, bus_with_mock, mock_redis):
        """Test publishing a bid request."""
        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=15.0,
        )

        msg_id = await bus_with_mock.publish_bid_request(request)

        assert msg_id == "1234567890-0"
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == STREAMS["bid_requests"]

    @pytest.mark.asyncio
    async def test_publish_bid_response(self, bus_with_mock, mock_redis):
        """Test publishing a bid response."""
        response = BidResponse(
            request_id="req-123",
            seller_id="seller-001",
            offered_cpm=12.0,
            available_impressions=8000,
        )

        msg_id = await bus_with_mock.publish_bid_response(response)

        assert msg_id == "1234567890-0"
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == STREAMS["bid_responses"]

    @pytest.mark.asyncio
    async def test_publish_deal(self, bus_with_mock, mock_redis):
        """Test publishing a deal confirmation."""
        deal = DealConfirmation(
            request_id="req-123",
            buyer_id="buyer-001",
            seller_id="seller-001",
            impressions=10000,
            cpm=15.0,
            total_cost=150.0,
            scenario="A",
        )

        msg_id = await bus_with_mock.publish_deal(deal)

        assert msg_id == "1234567890-0"
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == STREAMS["deals"]

    @pytest.mark.asyncio
    async def test_publish_event(self, bus_with_mock, mock_redis):
        """Test publishing a generic event."""
        msg_id = await bus_with_mock.publish_event(
            "agent_started",
            {"agent_id": "buyer-001", "type": "buyer"},
        )

        assert msg_id == "1234567890-0"
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == STREAMS["events"]

    @pytest.mark.asyncio
    async def test_ensure_consumer_group_creates(self, bus_with_mock, mock_redis):
        """Test creating a consumer group."""
        await bus_with_mock.ensure_consumer_group("test-stream", "test-group")

        mock_redis.xgroup_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_consumer_group_handles_existing(self, bus_with_mock, mock_redis):
        """Test handling existing consumer group."""
        from redis.exceptions import ResponseError
        mock_redis.xgroup_create.side_effect = ResponseError("BUSYGROUP Consumer Group name already exists")

        # Should not raise
        await bus_with_mock.ensure_consumer_group("test-stream", "test-group")

    @pytest.mark.asyncio
    async def test_read_bid_requests(self, bus_with_mock, mock_redis):
        """Test reading bid requests."""
        # Setup mock response
        mock_redis.xreadgroup.return_value = [
            (
                STREAMS["bid_requests"],
                [
                    (
                        "1234567890-0",
                        {
                            "request_id": "req-123",
                            "message_type": "bid_request",
                            "buyer_id": "buyer-001",
                            "campaign_id": "camp-001",
                            "channel": "display",
                            "impressions_requested": "10000",
                            "max_cpm": "15.0",
                            "targeting": "{}",
                            "timestamp": "2025-01-29T12:00:00",
                        },
                    )
                ],
            )
        ]

        results = await bus_with_mock.read_bid_requests()

        assert len(results) == 1
        msg_id, request = results[0]
        assert msg_id == "1234567890-0"
        assert isinstance(request, BidRequest)
        assert request.buyer_id == "buyer-001"

    @pytest.mark.asyncio
    async def test_ack_messages(self, bus_with_mock, mock_redis):
        """Test acknowledging messages."""
        count = await bus_with_mock.ack(
            STREAMS["bid_requests"],
            CONSUMER_GROUPS["sellers"],
            "1234567890-0",
            "1234567890-1",
        )

        assert count == 1
        mock_redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_seller(self, bus_with_mock, mock_redis):
        """Test routing to specific seller."""
        request = BidRequest(
            buyer_id="buyer-001",
            campaign_id="camp-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=15.0,
        )

        await bus_with_mock.route_to_seller(request, "seller-001")

        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "rtb:seller:seller-001:requests"

    @pytest.mark.asyncio
    async def test_route_to_buyer(self, bus_with_mock, mock_redis):
        """Test routing to specific buyer."""
        response = BidResponse(
            request_id="req-123",
            seller_id="seller-001",
            offered_cpm=12.0,
            available_impressions=8000,
        )

        await bus_with_mock.route_to_buyer(response, "buyer-001")

        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "rtb:buyer:buyer-001:responses"

    @pytest.mark.asyncio
    async def test_get_stream_info(self, bus_with_mock, mock_redis):
        """Test getting stream info."""
        mock_redis.xinfo_stream.return_value = {"length": 100}

        info = await bus_with_mock.get_stream_info("test-stream")

        assert info["length"] == 100

    @pytest.mark.asyncio
    async def test_get_pending_count(self, bus_with_mock, mock_redis):
        """Test getting pending message count."""
        mock_redis.xpending.return_value = {"pending": 5}

        count = await bus_with_mock.get_pending_count("test-stream", "test-group")

        assert count == 5

    @pytest.mark.asyncio
    async def test_clear_stream(self, bus_with_mock, mock_redis):
        """Test clearing a stream."""
        mock_redis.xtrim.return_value = 100

        count = await bus_with_mock.clear_stream("test-stream")

        assert count == 100
        mock_redis.xtrim.assert_called_with("test-stream", maxlen=0)


class TestRedisBusFactory:
    """Tests for create_redis_bus factory."""

    @pytest.mark.asyncio
    async def test_create_redis_bus(self):
        """Test factory function creates connected bus."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.xgroup_create = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            bus = await create_redis_bus(
                url="redis://localhost:6379",
                consumer_id="test-consumer",
                setup_groups=False,
            )

            assert bus._consumer_id == "test-consumer"
            mock_redis.ping.assert_called_once()


# Integration tests (require actual Redis)
# Run with: pytest -m integration tests/infrastructure/test_redis_bus.py

@pytest.mark.integration
class TestRedisBusIntegration:
    """Integration tests requiring actual Redis connection."""

    @pytest.fixture
    async def bus(self):
        """Create a connected RedisBus."""
        bus = RedisBus(url="redis://localhost:6379")
        try:
            await bus.connect(consumer_id="test-integration")
            yield bus
        except Exception:
            pytest.skip("Redis not available")
        finally:
            await bus.disconnect()

    @pytest.mark.asyncio
    async def test_full_message_flow(self, bus):
        """Test complete message flow: publish, read, ack."""
        # Clear streams first
        await bus.clear_all_streams()

        # Create and publish request
        request = BidRequest(
            buyer_id="buyer-int-001",
            campaign_id="camp-int-001",
            channel="display",
            impressions_requested=10000,
            max_cpm=15.0,
        )
        msg_id = await bus.publish_bid_request(request)
        assert msg_id is not None

        # Read request
        results = await bus.read_bid_requests(
            group="test-group",
            count=1,
            block_ms=100,
        )
        assert len(results) == 1
        read_id, read_request = results[0]
        assert read_request.buyer_id == "buyer-int-001"

        # Ack message
        acked = await bus.ack_bid_requests("test-group", read_id)
        assert acked == 1

    @pytest.mark.asyncio
    async def test_agent_routing(self, bus):
        """Test agent-specific routing."""
        # Clear any previous messages
        await bus.clear_all_streams()

        # Route to specific seller
        request = BidRequest(
            buyer_id="buyer-route-001",
            campaign_id="camp-route-001",
            channel="video",
            impressions_requested=5000,
            max_cpm=25.0,
        )
        await bus.route_to_seller(request, "seller-route-001")

        # Read as that seller
        results = await bus.subscribe_as_seller(
            "seller-route-001",
            count=1,
            block_ms=100,
        )
        assert len(results) == 1
        _, read_request = results[0]
        assert read_request.buyer_id == "buyer-route-001"
