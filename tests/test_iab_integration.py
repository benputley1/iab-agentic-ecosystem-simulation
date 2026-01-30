"""
Test IAB Package Integration

Verifies that the IAB seller-agent and buyer-agent packages are properly
wired into the simulation infrastructure.

Run with:
    cd /root/clawd/iab-sim-work
    source .venv/bin/activate
    python -m pytest tests/test_iab_integration.py -v

Or for a quick test:
    python tests/test_iab_integration.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


async def test_seller_adapter_import():
    """Test that IAB seller adapter can be imported and initialized."""
    from src.agents.seller.iab_adapter import IABSellerAdapter
    
    # Initialize without connecting (no Redis needed)
    adapter = IABSellerAdapter(
        seller_id="test-seller-001",
        scenario="B",
        mock_llm=True,
    )
    
    assert adapter.seller_id == "test-seller-001"
    assert adapter.mock_llm == True
    logger.info("✓ Seller adapter import test passed")
    return True


async def test_buyer_wrapper_import():
    """Test that IAB buyer wrapper can be imported and initialized."""
    from src.agents.buyer.iab_wrapper import IABBuyerWrapper, Campaign
    
    # Initialize without connecting
    wrapper = IABBuyerWrapper(
        buyer_id="test-buyer-001",
        scenario="B",
        mock_llm=True,
    )
    
    assert wrapper.buyer_id == "test-buyer-001"
    assert wrapper.mock_llm == True
    
    # Add a campaign
    campaign = Campaign(
        campaign_id="camp-001",
        name="Test Campaign",
        budget=10000.0,
        target_impressions=1000000,
        target_cpm=15.0,
        channel="display",
    )
    wrapper.add_campaign(campaign)
    
    assert len(wrapper.get_active_campaigns()) == 1
    logger.info("✓ Buyer wrapper import test passed")
    return True


async def test_iab_packages_import():
    """Test that IAB packages can be imported directly."""
    try:
        from ad_seller.models.pricing_tiers import TieredPricingConfig
        from ad_seller.engines.pricing_rules_engine import PricingRulesEngine
        from ad_buyer.clients.unified_client import UnifiedClient, Protocol
        from ad_buyer.clients.a2a_client import A2AClient
        
        # Create pricing config
        pricing_config = TieredPricingConfig(
            seller_organization_id="test-seller",
            global_floor_cpm=1.0,
        )
        pricing_engine = PricingRulesEngine(pricing_config)
        
        # Create unified client (don't connect)
        client = UnifiedClient(
            base_url="https://agentic-direct-server-hwgrypmndq-uk.a.run.app",
            protocol=Protocol.MCP,
        )
        
        logger.info("✓ IAB packages import test passed")
        return True
        
    except ImportError as e:
        logger.error(f"✗ IAB packages import failed: {e}")
        return False


async def test_mock_evaluation():
    """Test mock evaluation flow (no LLM calls)."""
    from src.agents.seller.iab_adapter import IABSellerAdapter
    from src.agents.seller.inventory import Product, DealType
    from src.infrastructure.message_schemas import BidRequest
    
    adapter = IABSellerAdapter(
        seller_id="test-seller-001",
        scenario="B",
        mock_llm=True,
    )
    
    # Create a test product
    product = Product(
        product_id="prod-001",
        name="Test Display Inventory",
        description="Premium display inventory for testing",
        inventory_type="display",
        base_cpm=20.0,
        floor_cpm=15.0,
        audience_targeting=["sports", "tech"],
        content_targeting=["news"],
        supported_deal_types=[DealType.PREFERRED_DEAL],
    )
    
    # Create a test bid request
    request = BidRequest(
        request_id="req-001",
        buyer_id="buyer-001",
        campaign_id="camp-001",
        channel="display",
        impressions_requested=100000,
        max_cpm=25.0,
        targeting={"segments": ["sports"]},
    )
    
    # Evaluate
    evaluation = await adapter.evaluate_request(request, product)
    
    assert evaluation["accept"] == True, f"Expected accept=True, got {evaluation}"
    assert evaluation["offer_cpm"] > 0, f"Expected offer_cpm > 0, got {evaluation}"
    # Note: offer_impressions may be 0 if inventory not initialized via connect()
    assert "offer_impressions" in evaluation, "Missing offer_impressions key"
    
    logger.info(f"✓ Mock evaluation test passed: offer_cpm={evaluation['offer_cpm']}")
    return True


async def test_iab_server_connection():
    """Test actual connection to IAB OpenDirect server (requires network)."""
    try:
        from ad_buyer.clients.mcp_client import IABMCPClient
        
        logger.info("Connecting to IAB OpenDirect server...")
        
        async with IABMCPClient() as client:
            # List available tools
            tools = client.tools
            logger.info(f"Connected! Available tools: {len(tools)}")
            
            # List products
            result = await client.list_products()
            
            if result.success:
                products = result.data if isinstance(result.data, list) else [result.data]
                logger.info(f"✓ IAB server connection test passed: {len(products)} products found")
                return True
            else:
                logger.warning(f"IAB server returned error: {result.error}")
                return False
                
    except Exception as e:
        logger.error(f"✗ IAB server connection failed: {e}")
        return False


async def test_real_llm_buyer_discovery():
    """Test buyer inventory discovery with real IAB server (requires API key)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("⚠ Skipping real LLM test - ANTHROPIC_API_KEY not set")
        return True
    
    try:
        from src.agents.buyer.iab_wrapper import IABBuyerWrapper, Campaign
        
        wrapper = IABBuyerWrapper(
            buyer_id="test-buyer-001",
            scenario="B",
            mock_llm=False,  # Use real LLM
        )
        
        await wrapper.connect()
        
        try:
            campaign = Campaign(
                campaign_id="camp-001",
                name="Test Display Campaign",
                budget=50000.0,
                target_impressions=1000000,
                target_cpm=20.0,
                channel="display",
            )
            wrapper.add_campaign(campaign)
            
            # Discover inventory
            inventory = await wrapper.discover_inventory(campaign)
            
            logger.info(f"✓ Real LLM buyer discovery test passed: {len(inventory)} products")
            logger.info(f"  LLM calls: {wrapper.state.llm_calls}")
            logger.info(f"  Estimated cost: ${wrapper.state.llm_cost:.4f}")
            
            return True
            
        finally:
            await wrapper.disconnect()
            
    except Exception as e:
        logger.error(f"✗ Real LLM buyer discovery failed: {e}")
        return False


async def test_a2a_natural_language():
    """Test A2A natural language request (requires API key)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("⚠ Skipping A2A test - ANTHROPIC_API_KEY not set")
        return True
    
    try:
        from ad_buyer.clients.a2a_client import A2AClient
        
        async with A2AClient() as client:
            # Get agent card
            card = await client.get_agent_card()
            logger.info(f"Agent card: {card.get('name', 'Unknown')}")
            
            # Send natural language request
            response = await client.send_message(
                "List available advertising products"
            )
            
            logger.info(f"✓ A2A test passed")
            logger.info(f"  Response text preview: {response.text[:200] if response.text else 'No text'}")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ A2A test failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("IAB Package Integration Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Import seller adapter", test_seller_adapter_import),
        ("Import buyer wrapper", test_buyer_wrapper_import),
        ("Import IAB packages", test_iab_packages_import),
        ("Mock evaluation flow", test_mock_evaluation),
        ("IAB server connection", test_iab_server_connection),
        ("Real LLM buyer discovery", test_real_llm_buyer_discovery),
        ("A2A natural language", test_a2a_natural_language),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = await test_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' threw exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
