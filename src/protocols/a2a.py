"""
A2A (Agent-to-Agent) Protocol.

Natural language communication protocol for inter-agent messaging.
Allows agents to communicate via natural language queries rather
than structured tool calls.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NegotiationStatus(Enum):
    """Status of a negotiation."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTER_OFFER = "counter_offer"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class A2AMessage:
    """
    Message passed between agents.
    
    Attributes:
        sender: Agent ID of the sender
        recipient: Agent ID of the recipient
        content: Natural language message content
        context: Additional context for the message
        timestamp: When the message was created
        message_id: Unique identifier for the message
        priority: Message priority level
        reply_to: ID of message this is replying to
        conversation_id: ID grouping related messages
    """
    sender: str
    recipient: str
    content: str
    context: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: MessagePriority = MessagePriority.NORMAL
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Serialize message to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "priority": self.priority.value,
            "reply_to": self.reply_to,
            "conversation_id": self.conversation_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> A2AMessage:
        """Deserialize message from dictionary."""
        return cls(
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"],
            priority=MessagePriority(data.get("priority", "normal")),
            reply_to=data.get("reply_to"),
            conversation_id=data.get("conversation_id"),
        )


@dataclass
class A2AResponse:
    """
    Response to an A2A message.
    
    Attributes:
        success: Whether the message was processed successfully
        response: Natural language response content
        data: Structured data accompanying the response
        original_message_id: ID of the message being responded to
        processing_time_ms: How long processing took
    """
    success: bool
    response: str
    data: Optional[dict] = None
    original_message_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Serialize response to dictionary."""
        return {
            "success": self.success,
            "response": self.response,
            "data": self.data,
            "original_message_id": self.original_message_id,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class Offer:
    """
    Offer in a negotiation.
    
    Attributes:
        offer_id: Unique identifier
        offerer: Agent making the offer
        recipient: Agent receiving the offer
        terms: Structured terms of the offer
        description: Natural language description
        valid_until: Expiration timestamp
        parent_offer_id: ID of offer this counters
    """
    offer_id: str
    offerer: str
    recipient: str
    terms: dict
    description: str
    valid_until: Optional[datetime] = None
    parent_offer_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(
        cls,
        offerer: str,
        recipient: str,
        terms: dict,
        description: str,
        valid_for_seconds: int = 300,
        parent_offer_id: Optional[str] = None,
    ) -> Offer:
        """Create a new offer with auto-generated ID."""
        from datetime import timedelta
        return cls(
            offer_id=str(uuid.uuid4()),
            offerer=offerer,
            recipient=recipient,
            terms=terms,
            description=description,
            valid_until=datetime.utcnow() + timedelta(seconds=valid_for_seconds),
            parent_offer_id=parent_offer_id,
        )
    
    def is_expired(self) -> bool:
        """Check if offer has expired."""
        if self.valid_until is None:
            return False
        return datetime.utcnow() > self.valid_until
    
    def to_dict(self) -> dict:
        """Serialize offer to dictionary."""
        return {
            "offer_id": self.offer_id,
            "offerer": self.offerer,
            "recipient": self.recipient,
            "terms": self.terms,
            "description": self.description,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "parent_offer_id": self.parent_offer_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class NegotiationResult:
    """
    Result of a negotiation.
    
    Attributes:
        status: Final status of the negotiation
        final_terms: Agreed terms (if accepted)
        offer_history: Chronological list of offers made
        rounds: Number of negotiation rounds
        duration_ms: Total negotiation time
        rejection_reason: Reason if rejected
    """
    status: NegotiationStatus
    final_terms: Optional[dict] = None
    offer_history: list[Offer] = field(default_factory=list)
    rounds: int = 0
    duration_ms: Optional[float] = None
    rejection_reason: Optional[str] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if negotiation resulted in agreement."""
        return self.status == NegotiationStatus.ACCEPTED
    
    def to_dict(self) -> dict:
        """Serialize result to dictionary."""
        return {
            "status": self.status.value,
            "final_terms": self.final_terms,
            "offer_history": [o.to_dict() for o in self.offer_history],
            "rounds": self.rounds,
            "duration_ms": self.duration_ms,
            "rejection_reason": self.rejection_reason,
        }


class A2AProtocol:
    """
    A2A (Agent-to-Agent) natural language protocol.
    
    Allows agents to communicate via natural language queries
    rather than structured tool calls.
    
    Features:
    - Asynchronous message passing
    - Message queuing per agent
    - Multi-turn negotiation support
    - Conversation threading
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize A2A protocol handler.
        
        Args:
            agent_id: ID of the agent using this protocol
        """
        self.agent_id = agent_id
        self._message_queue: asyncio.Queue[A2AMessage] = asyncio.Queue()
        self._message_handlers: dict[str, Callable] = {}
        self._pending_negotiations: dict[str, Offer] = {}
        self._conversations: dict[str, list[A2AMessage]] = {}
    
    async def send_message(
        self,
        recipient: str,
        message: str,
        context: Optional[dict] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        conversation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> A2AResponse:
        """
        Send natural language message to another agent.
        
        Args:
            recipient: Agent ID of the recipient
            message: Natural language message content
            context: Additional context dictionary
            priority: Message priority level
            conversation_id: ID to group related messages
            reply_to: ID of message being replied to
            
        Returns:
            A2AResponse with the recipient's response
        """
        msg = A2AMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=message,
            context=context or {},
            priority=priority,
            conversation_id=conversation_id or str(uuid.uuid4()),
            reply_to=reply_to,
        )
        
        # Track in conversation
        if msg.conversation_id not in self._conversations:
            self._conversations[msg.conversation_id] = []
        self._conversations[msg.conversation_id].append(msg)
        
        # In a real implementation, this would route to the recipient
        # For simulation, we return a placeholder response
        return A2AResponse(
            success=True,
            response=f"Message delivered to {recipient}",
            original_message_id=msg.message_id,
        )
    
    async def receive_message(self, timeout: Optional[float] = None) -> A2AMessage:
        """
        Receive incoming message from queue.
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
            
        Returns:
            The next A2AMessage in the queue
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        if timeout is not None:
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=timeout,
            )
        return await self._message_queue.get()
    
    async def deliver_message(self, message: A2AMessage) -> None:
        """
        Deliver a message to this agent's queue.
        
        Called by the message router to deliver incoming messages.
        
        Args:
            message: The message to deliver
        """
        await self._message_queue.put(message)
    
    def register_handler(
        self,
        message_type: str,
        handler: Callable[[A2AMessage], A2AResponse],
    ) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type identifier (stored in message context)
            handler: Async function to handle messages of this type
        """
        self._message_handlers[message_type] = handler
    
    async def negotiate(
        self,
        initial_offer: Offer,
        max_rounds: int = 10,
        timeout_seconds: float = 60.0,
    ) -> NegotiationResult:
        """
        Multi-turn negotiation protocol.
        
        Args:
            initial_offer: Starting offer for negotiation
            max_rounds: Maximum negotiation rounds before failure
            timeout_seconds: Total timeout for negotiation
            
        Returns:
            NegotiationResult with final status and terms
        """
        import time
        start_time = time.time()
        
        offer_history = [initial_offer]
        current_offer = initial_offer
        rounds = 0
        
        self._pending_negotiations[initial_offer.offer_id] = initial_offer
        
        # Send initial offer
        await self.send_message(
            recipient=initial_offer.recipient,
            message=initial_offer.description,
            context={
                "type": "negotiation_offer",
                "offer": initial_offer.to_dict(),
            },
            priority=MessagePriority.HIGH,
        )
        
        try:
            while rounds < max_rounds:
                rounds += 1
                elapsed = time.time() - start_time
                
                if elapsed > timeout_seconds:
                    return NegotiationResult(
                        status=NegotiationStatus.EXPIRED,
                        offer_history=offer_history,
                        rounds=rounds,
                        duration_ms=(time.time() - start_time) * 1000,
                        rejection_reason="Negotiation timeout",
                    )
                
                if current_offer.is_expired():
                    return NegotiationResult(
                        status=NegotiationStatus.EXPIRED,
                        offer_history=offer_history,
                        rounds=rounds,
                        duration_ms=(time.time() - start_time) * 1000,
                        rejection_reason="Offer expired",
                    )
                
                # Wait for response
                try:
                    response_msg = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=timeout_seconds - elapsed,
                    )
                except asyncio.TimeoutError:
                    return NegotiationResult(
                        status=NegotiationStatus.EXPIRED,
                        offer_history=offer_history,
                        rounds=rounds,
                        duration_ms=(time.time() - start_time) * 1000,
                        rejection_reason="Response timeout",
                    )
                
                # Process response
                response_type = response_msg.context.get("type")
                
                if response_type == "negotiation_accept":
                    return NegotiationResult(
                        status=NegotiationStatus.ACCEPTED,
                        final_terms=current_offer.terms,
                        offer_history=offer_history,
                        rounds=rounds,
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                
                elif response_type == "negotiation_reject":
                    return NegotiationResult(
                        status=NegotiationStatus.REJECTED,
                        offer_history=offer_history,
                        rounds=rounds,
                        duration_ms=(time.time() - start_time) * 1000,
                        rejection_reason=response_msg.content,
                    )
                
                elif response_type == "negotiation_counter":
                    counter_offer_data = response_msg.context.get("offer", {})
                    counter_offer = Offer(
                        offer_id=counter_offer_data.get("offer_id", str(uuid.uuid4())),
                        offerer=response_msg.sender,
                        recipient=self.agent_id,
                        terms=counter_offer_data.get("terms", {}),
                        description=response_msg.content,
                        parent_offer_id=current_offer.offer_id,
                    )
                    offer_history.append(counter_offer)
                    current_offer = counter_offer
                    
                    # Auto-continue for simulation (real impl would invoke decision logic)
                    
        finally:
            self._pending_negotiations.pop(initial_offer.offer_id, None)
        
        return NegotiationResult(
            status=NegotiationStatus.EXPIRED,
            offer_history=offer_history,
            rounds=rounds,
            duration_ms=(time.time() - start_time) * 1000,
            rejection_reason="Max rounds exceeded",
        )
    
    async def respond_to_offer(
        self,
        offer_id: str,
        accept: bool,
        counter_terms: Optional[dict] = None,
        message: str = "",
    ) -> A2AResponse:
        """
        Respond to a negotiation offer.
        
        Args:
            offer_id: ID of the offer being responded to
            accept: True to accept, False to reject or counter
            counter_terms: New terms if making counter-offer
            message: Natural language response message
            
        Returns:
            A2AResponse confirming the response was sent
        """
        if accept:
            response_type = "negotiation_accept"
        elif counter_terms:
            response_type = "negotiation_counter"
        else:
            response_type = "negotiation_reject"
        
        context: dict[str, Any] = {
            "type": response_type,
            "original_offer_id": offer_id,
        }
        
        if counter_terms:
            counter_offer = Offer.create(
                offerer=self.agent_id,
                recipient="",  # Will be filled by message routing
                terms=counter_terms,
                description=message,
                parent_offer_id=offer_id,
            )
            context["offer"] = counter_offer.to_dict()
        
        # This would be routed back to the original offerer
        return A2AResponse(
            success=True,
            response=f"Response sent: {response_type}",
            data=context,
        )
    
    def get_conversation(self, conversation_id: str) -> list[A2AMessage]:
        """
        Get all messages in a conversation.
        
        Args:
            conversation_id: The conversation to retrieve
            
        Returns:
            List of messages in chronological order
        """
        return self._conversations.get(conversation_id, [])
    
    def get_pending_messages_count(self) -> int:
        """Get number of messages waiting in queue."""
        return self._message_queue.qsize()
