"""
Redis Streams wrapper for A2A message routing in RTB simulation.

Provides async pub/sub capabilities for agent-to-agent communication using
Redis Streams with consumer groups for reliable message delivery.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import AsyncIterator, Callable, Optional, TypeVar, Union
import structlog
import redis.asyncio as redis
from redis.exceptions import ResponseError

from .message_schemas import (
    BidRequest,
    BidResponse,
    DealConfirmation,
    MessageType,
    STREAMS,
    CONSUMER_GROUPS,
)

logger = structlog.get_logger()
T = TypeVar("T", BidRequest, BidResponse, DealConfirmation)


class RedisBus:
    """
    Redis Streams wrapper for A2A message routing.

    Supports:
    - Publishing messages to streams
    - Reading messages with consumer groups
    - Agent-specific routing
    - Message acknowledgment
    """

    def __init__(
        self,
        url: Optional[str] = None,
        max_stream_length: int = 10000,
    ):
        """
        Initialize Redis bus.

        Args:
            url: Redis connection URL (default: from REDIS_URL env var)
            max_stream_length: Maximum messages to keep in each stream
        """
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.max_stream_length = max_stream_length
        self._client: Optional[redis.Redis] = None
        self._consumer_id: Optional[str] = None
        self._initialized_groups: set[tuple[str, str]] = set()

    @property
    def client(self) -> redis.Redis:
        """Get Redis client, creating if necessary."""
        if self._client is None:
            raise RuntimeError("RedisBus not connected. Call connect() first.")
        return self._client

    async def connect(self, consumer_id: Optional[str] = None) -> "RedisBus":
        """
        Connect to Redis.

        Args:
            consumer_id: Unique identifier for this consumer (default: generated)

        Returns:
            Self for chaining
        """
        self._client = redis.from_url(self.url, decode_responses=True)
        self._consumer_id = consumer_id or f"consumer-{os.getpid()}-{id(self)}"

        # Test connection
        await self._client.ping()
        logger.info("redis_bus.connected", url=self.url, consumer_id=self._consumer_id)

        return self

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("redis_bus.disconnected")

    async def __aenter__(self) -> "RedisBus":
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    async def publish_bid_request(self, request: BidRequest) -> str:
        """
        Publish a bid request to the requests stream.

        Args:
            request: The bid request to publish

        Returns:
            Message ID from Redis
        """
        return await self._publish(STREAMS["bid_requests"], request.to_stream_data())

    async def publish_bid_response(self, response: BidResponse) -> str:
        """
        Publish a bid response to the responses stream.

        Args:
            response: The bid response to publish

        Returns:
            Message ID from Redis
        """
        return await self._publish(STREAMS["bid_responses"], response.to_stream_data())

    async def publish_deal(self, deal: DealConfirmation) -> str:
        """
        Publish a deal confirmation to the deals stream.

        Args:
            deal: The deal confirmation to publish

        Returns:
            Message ID from Redis
        """
        return await self._publish(STREAMS["deals"], deal.to_stream_data())

    async def publish_event(self, event_type: str, data: dict) -> str:
        """
        Publish a generic event to the events stream.

        Args:
            event_type: Type of event (e.g., "agent_started", "campaign_completed")
            data: Event data

        Returns:
            Message ID from Redis
        """
        event_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": json.dumps(data),
        }
        return await self._publish(STREAMS["events"], event_data)

    async def _publish(self, stream: str, data: dict) -> str:
        """
        Internal publish method with trimming.

        Args:
            stream: Stream name
            data: Message data

        Returns:
            Message ID
        """
        msg_id = await self.client.xadd(
            stream,
            data,
            maxlen=self.max_stream_length,
        )
        logger.debug(
            "redis_bus.published",
            stream=stream,
            message_id=msg_id,
            message_type=data.get("message_type"),
        )
        return msg_id

    # -------------------------------------------------------------------------
    # Consumer Groups
    # -------------------------------------------------------------------------

    async def ensure_consumer_group(
        self,
        stream: str,
        group: str,
        start_id: str = "0",
    ) -> None:
        """
        Ensure a consumer group exists, creating if necessary.

        Args:
            stream: Stream name
            group: Consumer group name
            start_id: Starting message ID for new consumers
        """
        cache_key = (stream, group)
        if cache_key in self._initialized_groups:
            return

        try:
            await self.client.xgroup_create(
                stream,
                group,
                id=start_id,
                mkstream=True,
            )
            logger.info("redis_bus.group_created", stream=stream, group=group)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            # Group already exists
            logger.debug("redis_bus.group_exists", stream=stream, group=group)

        self._initialized_groups.add(cache_key)

    async def setup_default_groups(self) -> None:
        """Set up default consumer groups for all streams."""
        for stream_key, stream_name in STREAMS.items():
            for group_key, group_name in CONSUMER_GROUPS.items():
                await self.ensure_consumer_group(stream_name, group_name)

    # -------------------------------------------------------------------------
    # Reading
    # -------------------------------------------------------------------------

    async def read_bid_requests(
        self,
        group: str = CONSUMER_GROUPS["sellers"],
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, BidRequest]]:
        """
        Read bid requests from the stream.

        Args:
            group: Consumer group name
            count: Maximum messages to read
            block_ms: Block timeout in milliseconds

        Returns:
            List of (message_id, BidRequest) tuples
        """
        await self.ensure_consumer_group(STREAMS["bid_requests"], group)
        return await self._read_typed(
            STREAMS["bid_requests"],
            group,
            BidRequest.from_stream_data,
            count,
            block_ms,
        )

    async def read_bid_responses(
        self,
        group: str = CONSUMER_GROUPS["buyers"],
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, BidResponse]]:
        """
        Read bid responses from the stream.

        Args:
            group: Consumer group name
            count: Maximum messages to read
            block_ms: Block timeout in milliseconds

        Returns:
            List of (message_id, BidResponse) tuples
        """
        await self.ensure_consumer_group(STREAMS["bid_responses"], group)
        return await self._read_typed(
            STREAMS["bid_responses"],
            group,
            BidResponse.from_stream_data,
            count,
            block_ms,
        )

    async def read_deals(
        self,
        group: str = CONSUMER_GROUPS["analytics"],
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, DealConfirmation]]:
        """
        Read deal confirmations from the stream.

        Args:
            group: Consumer group name
            count: Maximum messages to read
            block_ms: Block timeout in milliseconds

        Returns:
            List of (message_id, DealConfirmation) tuples
        """
        await self.ensure_consumer_group(STREAMS["deals"], group)
        return await self._read_typed(
            STREAMS["deals"],
            group,
            DealConfirmation.from_stream_data,
            count,
            block_ms,
        )

    async def _read_typed(
        self,
        stream: str,
        group: str,
        parser: Callable[[dict], T],
        count: int,
        block_ms: int,
    ) -> list[tuple[str, T]]:
        """
        Internal typed read method.

        Args:
            stream: Stream name
            group: Consumer group name
            parser: Function to parse message data
            count: Maximum messages
            block_ms: Block timeout

        Returns:
            List of (message_id, parsed_message) tuples
        """
        messages = await self.client.xreadgroup(
            group,
            self._consumer_id,
            {stream: ">"},  # Only new messages
            count=count,
            block=block_ms,
        )

        results = []
        for stream_name, stream_messages in messages:
            for msg_id, data in stream_messages:
                try:
                    parsed = parser(data)
                    results.append((msg_id, parsed))
                except Exception as e:
                    logger.error(
                        "redis_bus.parse_error",
                        stream=stream,
                        message_id=msg_id,
                        error=str(e),
                    )
        return results

    async def read_raw(
        self,
        stream: str,
        group: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, dict]]:
        """
        Read raw messages from a stream.

        Args:
            stream: Stream name
            group: Consumer group name
            count: Maximum messages
            block_ms: Block timeout

        Returns:
            List of (message_id, data) tuples
        """
        await self.ensure_consumer_group(stream, group)
        messages = await self.client.xreadgroup(
            group,
            self._consumer_id,
            {stream: ">"},
            count=count,
            block=block_ms,
        )

        results = []
        for stream_name, stream_messages in messages:
            for msg_id, data in stream_messages:
                results.append((msg_id, data))
        return results

    # -------------------------------------------------------------------------
    # Acknowledgment
    # -------------------------------------------------------------------------

    async def ack(self, stream: str, group: str, *message_ids: str) -> int:
        """
        Acknowledge messages as processed.

        Args:
            stream: Stream name
            group: Consumer group name
            message_ids: Message IDs to acknowledge

        Returns:
            Number of messages acknowledged
        """
        if not message_ids:
            return 0
        count = await self.client.xack(stream, group, *message_ids)
        logger.debug(
            "redis_bus.acked",
            stream=stream,
            group=group,
            count=count,
        )
        return count

    async def ack_bid_requests(self, group: str, *message_ids: str) -> int:
        """Acknowledge bid request messages."""
        return await self.ack(STREAMS["bid_requests"], group, *message_ids)

    async def ack_bid_responses(self, group: str, *message_ids: str) -> int:
        """Acknowledge bid response messages."""
        return await self.ack(STREAMS["bid_responses"], group, *message_ids)

    async def ack_deals(self, group: str, *message_ids: str) -> int:
        """Acknowledge deal confirmation messages."""
        return await self.ack(STREAMS["deals"], group, *message_ids)

    # -------------------------------------------------------------------------
    # A2A Routing Helpers
    # -------------------------------------------------------------------------

    async def route_to_seller(
        self,
        request: BidRequest,
        seller_id: str,
    ) -> str:
        """
        Route a bid request to a specific seller.

        Uses a seller-specific stream for targeted routing.

        Args:
            request: Bid request to route
            seller_id: Target seller ID

        Returns:
            Message ID
        """
        stream = f"rtb:seller:{seller_id}:requests"
        return await self._publish(stream, request.to_stream_data())

    async def route_to_buyer(
        self,
        response: BidResponse,
        buyer_id: str,
    ) -> str:
        """
        Route a bid response to a specific buyer.

        Uses a buyer-specific stream for targeted routing.

        Args:
            response: Bid response to route
            buyer_id: Target buyer ID

        Returns:
            Message ID
        """
        stream = f"rtb:buyer:{buyer_id}:responses"
        return await self._publish(stream, response.to_stream_data())

    async def subscribe_as_seller(
        self,
        seller_id: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, BidRequest]]:
        """
        Subscribe to requests for a specific seller.

        Args:
            seller_id: Seller ID
            count: Maximum messages
            block_ms: Block timeout

        Returns:
            List of (message_id, BidRequest) tuples
        """
        stream = f"rtb:seller:{seller_id}:requests"
        group = f"seller-{seller_id}"
        await self.ensure_consumer_group(stream, group)
        return await self._read_typed(
            stream,
            group,
            BidRequest.from_stream_data,
            count,
            block_ms,
        )

    async def subscribe_as_buyer(
        self,
        buyer_id: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, BidResponse]]:
        """
        Subscribe to responses for a specific buyer.

        Args:
            buyer_id: Buyer ID
            count: Maximum messages
            block_ms: Block timeout

        Returns:
            List of (message_id, BidResponse) tuples
        """
        stream = f"rtb:buyer:{buyer_id}:responses"
        group = f"buyer-{buyer_id}"
        await self.ensure_consumer_group(stream, group)
        return await self._read_typed(
            stream,
            group,
            BidResponse.from_stream_data,
            count,
            block_ms,
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    async def get_stream_info(self, stream: str) -> dict:
        """
        Get information about a stream.

        Args:
            stream: Stream name

        Returns:
            Stream information dict
        """
        try:
            info = await self.client.xinfo_stream(stream)
            return dict(info)
        except ResponseError:
            return {"exists": False}

    async def get_pending_count(self, stream: str, group: str) -> int:
        """
        Get count of pending (unacknowledged) messages.

        Args:
            stream: Stream name
            group: Consumer group name

        Returns:
            Number of pending messages
        """
        try:
            info = await self.client.xpending(stream, group)
            return info["pending"] if info else 0
        except ResponseError:
            return 0

    async def clear_stream(self, stream: str) -> int:
        """
        Clear all messages from a stream.

        Args:
            stream: Stream name

        Returns:
            Number of messages deleted
        """
        count = await self.client.xtrim(stream, maxlen=0)
        logger.info("redis_bus.stream_cleared", stream=stream, deleted=count)
        return count

    async def clear_all_streams(self) -> dict[str, int]:
        """
        Clear all RTB streams.

        Returns:
            Dict of stream name -> deleted count
        """
        results = {}
        for stream_name in STREAMS.values():
            try:
                results[stream_name] = await self.clear_stream(stream_name)
            except ResponseError:
                results[stream_name] = 0
        return results


# -------------------------------------------------------------------------
# Convenience factory
# -------------------------------------------------------------------------

async def create_redis_bus(
    url: Optional[str] = None,
    consumer_id: Optional[str] = None,
    setup_groups: bool = True,
) -> RedisBus:
    """
    Create and connect a Redis bus.

    Args:
        url: Redis URL (default: from REDIS_URL env)
        consumer_id: Consumer identifier
        setup_groups: Whether to set up default consumer groups

    Returns:
        Connected RedisBus instance
    """
    bus = RedisBus(url=url)
    await bus.connect(consumer_id=consumer_id)
    if setup_groups:
        await bus.setup_default_groups()
    return bus
