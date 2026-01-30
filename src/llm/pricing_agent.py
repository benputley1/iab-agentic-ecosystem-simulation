#!/usr/bin/env python3
"""
Real LLM Pricing Agent using Anthropic Claude API.

This module provides actual Claude API calls for pricing decisions,
replacing the simulated hallucination approach with real LLM behavior.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class PricingDecision:
    """Result of a pricing decision from Claude."""
    bid_cpm: float
    raw_response: str
    is_hallucination: bool
    hallucination_reason: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class HallucinationMetrics:
    """Aggregated hallucination metrics for analysis."""
    total_decisions: int = 0
    hallucinations: int = 0
    hallucination_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_api_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    hallucination_types: Dict[str, int] = field(default_factory=dict)
    
    def update(self, decision: PricingDecision):
        """Update metrics with a new decision."""
        self.total_decisions += 1
        self.total_input_tokens += decision.input_tokens
        self.total_output_tokens += decision.output_tokens
        
        if decision.is_hallucination:
            self.hallucinations += 1
            reason = decision.hallucination_reason or "unknown"
            self.hallucination_types[reason] = self.hallucination_types.get(reason, 0) + 1
        
        self.hallucination_rate = self.hallucinations / self.total_decisions if self.total_decisions > 0 else 0.0
        
        # Update average latency
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.total_decisions - 1) + decision.latency_ms) 
            / self.total_decisions
        )
        
        # Calculate API cost (Haiku pricing: $0.25/1M input, $1.25/1M output)
        self.total_api_cost_usd = (
            (self.total_input_tokens * 0.25 / 1_000_000) +
            (self.total_output_tokens * 1.25 / 1_000_000)
        )


class PricingAgent:
    """
    AI Ad Buyer Agent that makes real pricing decisions via Claude API.
    
    Features:
    - Real Anthropic API calls using claude-sonnet-4 per IAB specs
    - Context history tracking to simulate agent memory
    - Context window limiting to simulate context rot
    - Hallucination detection by comparing decisions against ground truth bounds
    - Full API cost and token tracking
    - Per-agent isolation with unique agent_id
    - Deal recording from agent's perspective for reconciliation tracking
    """
    
    # Haiku pricing per 1M tokens
    INPUT_COST_PER_1M = 0.25
    OUTPUT_COST_PER_1M = 1.25
    
    def __init__(
        self,
        agent_id: str = "default",
        agent_type: str = "buyer",  # "buyer" or "seller"
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_context_history: int = 100,
    ):
        """
        Initialize the Pricing Agent.
        
        Args:
            agent_id: Unique identifier for this agent (e.g., "buyer_1", "seller_2")
            agent_type: Type of agent - "buyer" or "seller"
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: Haiku for cost efficiency)
            max_context_history: Maximum transaction history to keep
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Set it in .env or pass explicitly.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_context_history = max_context_history
        
        # Agent memory - tracks past decisions (ISOLATED per agent instance)
        self.context_history: List[Dict[str, Any]] = []
        
        # Deal records - agent's own view of completed deals (for reconciliation)
        self.deal_records: Dict[str, dict] = {}
        
        # Metrics tracking
        self.metrics = HallucinationMetrics()
        
        # Decision log for analysis
        self.decision_log: List[PricingDecision] = []
    
    def add_to_history(self, deal: Dict[str, Any], decision: PricingDecision):
        """Add a completed decision to the agent's memory."""
        self.context_history.append({
            "deal": deal,
            "bid_cpm": decision.bid_cpm,
            "was_hallucination": decision.is_hallucination,
            "timestamp": decision.timestamp,
        })
        
        # Prune old history if needed
        if len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history:]
    
    def make_pricing_decision(
        self,
        deal_opportunity: Dict[str, Any],
        context_limit: Optional[int] = None,
        include_ground_truth: bool = False,
    ) -> PricingDecision:
        """
        Ask Claude to make a pricing decision based on deal parameters.
        
        Args:
            deal_opportunity: Dict with deal details:
                - channel: str (display, video, ctv)
                - impressions: int
                - cpm_floor: float (minimum acceptable CPM)
                - cpm_max: float (maximum reasonable CPM)
                - seller_id: str
                - buyer_id: str
            context_limit: How much history to include (None = all, simulates context rot)
            include_ground_truth: If True, include floor/max in prompt (reduces hallucinations)
        
        Returns:
            PricingDecision with bid_cpm, hallucination status, and metrics
        """
        # Get context based on limit (simulates context rot)
        if context_limit is not None:
            recent_context = self.context_history[-context_limit:] if context_limit > 0 else []
        else:
            recent_context = self.context_history
        
        # Build the prompt
        prompt = self._build_prompt(deal_opportunity, recent_context, include_ground_truth)
        
        # Make API call with timing
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            raw_response = response.content[0].text.strip()
            
            # Parse the bid from response
            bid_cpm, parse_error = self._parse_bid_response(raw_response)
            
            # Detect hallucinations
            is_hallucination, hallucination_reason = self._detect_hallucination(
                bid_cpm=bid_cpm,
                deal=deal_opportunity,
                parse_error=parse_error,
                raw_response=raw_response,
            )
            
            decision = PricingDecision(
                bid_cpm=bid_cpm,
                raw_response=raw_response,
                is_hallucination=is_hallucination,
                hallucination_reason=hallucination_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            # API error - treat as hallucination
            latency_ms = (time.time() - start_time) * 1000
            decision = PricingDecision(
                bid_cpm=0.0,
                raw_response=f"API_ERROR: {str(e)}",
                is_hallucination=True,
                hallucination_reason="api_error",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
            )
        
        # Update metrics and history
        self.metrics.update(decision)
        self.decision_log.append(decision)
        self.add_to_history(deal_opportunity, decision)
        
        return decision
    
    def _build_prompt(
        self,
        deal: Dict[str, Any],
        context: List[Dict[str, Any]],
        include_ground_truth: bool,
    ) -> str:
        """Build the pricing decision prompt."""
        
        # Format deal opportunity
        deal_str = f"""
Channel: {deal.get('channel', 'display')}
Impressions: {deal.get('impressions', 0):,}
Seller: {deal.get('seller_id', 'unknown')}
Buyer: {deal.get('buyer_id', 'unknown')}"""
        
        if include_ground_truth:
            deal_str += f"""
CPM Floor (minimum): ${deal.get('cpm_floor', 5.0):.2f}
CPM Ceiling (maximum): ${deal.get('cpm_max', 25.0):.2f}"""
        
        # Format context history (if any)
        context_str = ""
        if context:
            context_str = "\n\nYour recent transaction history:\n"
            for i, entry in enumerate(context[-10:], 1):  # Show last 10 max
                ctx_deal = entry.get('deal', {})
                context_str += f"  {i}. {ctx_deal.get('channel', '?')}: bid ${entry.get('bid_cpm', 0):.2f} CPM\n"
        
        prompt = f"""You are an AI ad buyer agent participating in programmatic advertising.

Based on this deal opportunity:
{deal_str}
{context_str}
What CPM (cost per mille) bid should you offer?

Consider:
- Market rates for {deal.get('channel', 'display')} typically range $5-25 CPM
- Video and CTV command higher CPMs than display
- Balance competitive bidding with cost efficiency

Respond with ONLY a number representing your CPM bid (e.g., "12.50"). No other text."""

        return prompt
    
    def _parse_bid_response(self, raw_response: str) -> Tuple[float, Optional[str]]:
        """
        Parse the bid CPM from Claude's response.
        
        Returns:
            Tuple of (bid_cpm, parse_error_or_none)
        """
        # Try to extract a number from the response
        # Handle various formats: "12.50", "$12.50", "12.50 CPM", etc.
        
        # Remove common prefixes/suffixes
        cleaned = raw_response.lower().replace("$", "").replace("cpm", "").strip()
        
        # Try direct float parsing
        try:
            bid = float(cleaned)
            return bid, None
        except ValueError:
            pass
        
        # Try regex to find first number
        match = re.search(r'(\d+\.?\d*)', raw_response)
        if match:
            try:
                bid = float(match.group(1))
                return bid, None
            except ValueError:
                pass
        
        # Failed to parse
        return 0.0, f"parse_failed: {raw_response[:50]}"
    
    def _detect_hallucination(
        self,
        bid_cpm: float,
        deal: Dict[str, Any],
        parse_error: Optional[str],
        raw_response: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if the pricing decision is a hallucination.
        
        Hallucination criteria:
        1. Parse error (couldn't extract a number)
        2. Bid > 2x the max_cpm (way too high)
        3. Bid < 0.5x the floor (way too low)
        4. Negative bid
        5. Absurdly high bid (> $100 CPM)
        
        Returns:
            Tuple of (is_hallucination, reason_or_none)
        """
        cpm_floor = deal.get('cpm_floor', 5.0)
        cpm_max = deal.get('cpm_max', 25.0)
        
        # Check for parse errors
        if parse_error:
            return True, "parse_error"
        
        # Check for negative bids
        if bid_cpm < 0:
            return True, "negative_bid"
        
        # Check for absurdly high bids (> $100)
        if bid_cpm > 100:
            return True, "absurd_high"
        
        # Check for bids way above max (> 2x ceiling)
        if bid_cpm > cpm_max * 2:
            return True, "above_2x_max"
        
        # Check for bids way below floor (< 0.5x floor)
        if bid_cpm < cpm_floor * 0.5:
            return True, "below_half_floor"
        
        # Not a hallucination
        return False, None
    
    def reset_context(self):
        """Clear the agent's context history (simulates memory wipe)."""
        self.context_history = []
        self.deal_records = {}
    
    def record_deal(self, deal_record: dict) -> None:
        """
        Record a completed deal from this agent's perspective.
        
        This is the agent's "memory" of the deal - may differ from counterparty's
        memory due to context rot, hallucinations, or interpretation differences.
        
        Args:
            deal_record: Dict containing:
                - deal_id: str - Unique deal identifier
                - counterparty_id: str - The other party's agent_id
                - agreed_cpm: float - What this agent believes was agreed
                - impressions: int - Impression count this agent recorded
                - channel: str - Ad channel type
                - day: int - Simulation day
                - timestamp: str - When recorded
                - my_bid: float - What this agent bid/accepted
                - their_bid: Optional[float] - What counterparty bid (if known)
                - raw_response: str - LLM response that led to this decision
                - notes: Optional[str] - Any additional context
        """
        deal_id = deal_record.get("deal_id")
        if not deal_id:
            raise ValueError("deal_record must include 'deal_id'")
        
        # Add agent metadata
        deal_record["recorded_by"] = self.agent_id
        deal_record["agent_type"] = self.agent_type
        deal_record["recorded_at"] = datetime.utcnow().isoformat()
        
        self.deal_records[deal_id] = deal_record
    
    def get_deal_record(self, deal_id: str) -> Optional[dict]:
        """Retrieve a deal record by ID."""
        return self.deal_records.get(deal_id)
    
    def get_state_summary(self) -> dict:
        """Get a summary of this agent's current state for monitoring."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "context_entries": len(self.context_history),
            "deal_records": len(self.deal_records),
            "total_tokens": self.metrics.total_input_tokens + self.metrics.total_output_tokens,
            "total_decisions": self.metrics.total_decisions,
            "hallucinations": self.metrics.hallucinations,
            "hallucination_rate": self.metrics.hallucination_rate,
            "total_cost_usd": self.metrics.total_api_cost_usd,
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get a summary of API costs and token usage."""
        return {
            "total_decisions": self.metrics.total_decisions,
            "total_input_tokens": self.metrics.total_input_tokens,
            "total_output_tokens": self.metrics.total_output_tokens,
            "total_api_cost_usd": round(self.metrics.total_api_cost_usd, 6),
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
            "cost_per_decision": round(
                self.metrics.total_api_cost_usd / max(self.metrics.total_decisions, 1), 6
            ),
        }
    
    def get_hallucination_summary(self) -> Dict[str, Any]:
        """Get a summary of hallucination metrics."""
        return {
            "total_decisions": self.metrics.total_decisions,
            "hallucinations": self.metrics.hallucinations,
            "hallucination_rate": round(self.metrics.hallucination_rate * 100, 2),
            "hallucination_types": self.metrics.hallucination_types,
        }


# Quick test function
def test_pricing_agent():
    """Test the pricing agent with a sample deal."""
    print("Testing PricingAgent with real Claude API calls...\n")
    
    agent = PricingAgent()
    
    # Test deals
    test_deals = [
        {
            "channel": "display",
            "impressions": 100000,
            "cpm_floor": 5.0,
            "cpm_max": 15.0,
            "seller_id": "seller-001",
            "buyer_id": "buyer-001",
        },
        {
            "channel": "video",
            "impressions": 50000,
            "cpm_floor": 10.0,
            "cpm_max": 25.0,
            "seller_id": "seller-002",
            "buyer_id": "buyer-001",
        },
        {
            "channel": "ctv",
            "impressions": 25000,
            "cpm_floor": 15.0,
            "cpm_max": 35.0,
            "seller_id": "seller-003",
            "buyer_id": "buyer-002",
        },
    ]
    
    print("Making pricing decisions...")
    for i, deal in enumerate(test_deals, 1):
        print(f"\n--- Deal {i}: {deal['channel']} ---")
        decision = agent.make_pricing_decision(deal, context_limit=None)
        print(f"  Raw response: {decision.raw_response}")
        print(f"  Bid CPM: ${decision.bid_cpm:.2f}")
        print(f"  Is hallucination: {decision.is_hallucination}")
        if decision.is_hallucination:
            print(f"  Hallucination reason: {decision.hallucination_reason}")
        print(f"  Tokens: {decision.input_tokens} in, {decision.output_tokens} out")
        print(f"  Latency: {decision.latency_ms:.0f}ms")
    
    # Test with degraded context (simulating context rot)
    print("\n\n--- Testing with degraded context (context_limit=0) ---")
    decision = agent.make_pricing_decision(test_deals[0], context_limit=0)
    print(f"  Raw response: {decision.raw_response}")
    print(f"  Bid CPM: ${decision.bid_cpm:.2f}")
    print(f"  Is hallucination: {decision.is_hallucination}")
    
    # Print summaries
    print("\n\n=== COST SUMMARY ===")
    cost = agent.get_cost_summary()
    print(f"  Total decisions: {cost['total_decisions']}")
    print(f"  Total tokens: {cost['total_input_tokens']} in, {cost['total_output_tokens']} out")
    print(f"  Total API cost: ${cost['total_api_cost_usd']:.6f}")
    print(f"  Cost per decision: ${cost['cost_per_decision']:.6f}")
    print(f"  Avg latency: {cost['avg_latency_ms']:.0f}ms")
    
    print("\n=== HALLUCINATION SUMMARY ===")
    hall = agent.get_hallucination_summary()
    print(f"  Total decisions: {hall['total_decisions']}")
    print(f"  Hallucinations: {hall['hallucinations']}")
    print(f"  Hallucination rate: {hall['hallucination_rate']:.1f}%")
    print(f"  Types: {hall['hallucination_types']}")
    
    return agent


if __name__ == "__main__":
    test_pricing_agent()
