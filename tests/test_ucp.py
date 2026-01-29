"""Tests for the UCP (User Context Protocol) module."""

import pytest
from datetime import datetime

from src.agents.ucp import (
    # Embeddings
    ContextType,
    EmbeddingFormat,
    UserContextEmbedding,
    UCPExchange,
    UCPExchangeRequest,
    UCPExchangeResponse,
    # Hallucination
    InjectionType,
    Severity,
    HallucinationInjector,
    HallucinationDetector,
    HallucinationManager,
)


class TestUserContextEmbedding:
    """Tests for UserContextEmbedding dataclass."""

    def test_create_segment_embedding(self):
        """Create embedding with segment IDs."""
        emb = UserContextEmbedding(
            context_type=ContextType.BEHAVIORAL,
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["IAB1", "IAB2-3", "custom_tech"],
            source="dmp",
            confidence=0.9,
        )

        assert emb.format == EmbeddingFormat.SEGMENT_IDS
        assert len(emb.segment_ids) == 3
        assert emb.confidence == 0.9
        assert emb.embedding_id.startswith("emb-")

    def test_create_vector_embedding(self):
        """Create embedding with dense vector."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        emb = UserContextEmbedding(
            context_type=ContextType.INTENT,
            format=EmbeddingFormat.DENSE_VECTOR,
            vector=vector,
            source="ml_model",
        )

        assert emb.format == EmbeddingFormat.DENSE_VECTOR
        assert emb.vector_dim == 5
        assert emb.vector == vector

    def test_vector_embedding_requires_vector(self):
        """Dense vector format requires non-empty vector."""
        with pytest.raises(ValueError, match="requires non-empty vector"):
            UserContextEmbedding(
                format=EmbeddingFormat.DENSE_VECTOR,
                vector=[],
            )

    def test_create_sparse_embedding(self):
        """Create embedding with sparse features."""
        emb = UserContextEmbedding(
            format=EmbeddingFormat.SPARSE_FEATURES,
            features={
                "tech_interest": 0.8,
                "sports_interest": 0.3,
                "income_high": 0.6,
            },
        )

        assert emb.format == EmbeddingFormat.SPARSE_FEATURES
        assert len(emb.features) == 3
        assert emb.features["tech_interest"] == 0.8

    def test_fingerprint_uniqueness(self):
        """Different embeddings should have different fingerprints."""
        emb1 = UserContextEmbedding(
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["A", "B"],
        )
        emb2 = UserContextEmbedding(
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["A", "C"],
        )

        assert emb1.fingerprint() != emb2.fingerprint()

    def test_fingerprint_determinism(self):
        """Same content should produce same fingerprint."""
        emb1 = UserContextEmbedding(
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["A", "B"],
        )
        emb2 = UserContextEmbedding(
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["B", "A"],  # Different order, same segments
        )

        # Fingerprint normalizes order
        assert emb1.fingerprint() == emb2.fingerprint()

    def test_serialization_roundtrip(self):
        """Embedding should survive serialization roundtrip."""
        original = UserContextEmbedding(
            context_type=ContextType.DEMOGRAPHIC,
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["IAB1", "IAB2"],
            source="test",
            confidence=0.85,
        )

        data = original.to_dict()
        restored = UserContextEmbedding.from_dict(data)

        assert restored.context_type == original.context_type
        assert restored.format == original.format
        assert restored.segment_ids == original.segment_ids
        assert restored.confidence == original.confidence

    def test_merge_segment_embeddings(self):
        """Merging segment embeddings produces union."""
        emb1 = UserContextEmbedding(
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["A", "B"],
            confidence=0.8,
        )
        emb2 = UserContextEmbedding(
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["B", "C"],
            confidence=0.9,
        )

        merged = emb1.merge(emb2)

        assert set(merged.segment_ids) == {"A", "B", "C"}
        assert merged.confidence == 0.9  # Max of two

    def test_merge_vector_embeddings(self):
        """Merging vector embeddings produces weighted average."""
        emb1 = UserContextEmbedding(
            format=EmbeddingFormat.DENSE_VECTOR,
            vector=[1.0, 0.0],
            confidence=0.5,
        )
        emb2 = UserContextEmbedding(
            format=EmbeddingFormat.DENSE_VECTOR,
            vector=[0.0, 1.0],
            confidence=0.5,
        )

        merged = emb1.merge(emb2)

        # Equal confidence means simple average
        assert merged.vector[0] == pytest.approx(0.5)
        assert merged.vector[1] == pytest.approx(0.5)

    def test_cannot_merge_different_formats(self):
        """Cannot merge embeddings of different formats."""
        emb1 = UserContextEmbedding(
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=["A"],
        )
        emb2 = UserContextEmbedding(
            format=EmbeddingFormat.DENSE_VECTOR,
            vector=[0.5, 0.5],
        )

        with pytest.raises(ValueError, match="Cannot merge different formats"):
            emb1.merge(emb2)


class TestUCPExchange:
    """Tests for UCPExchange class."""

    def test_create_exchange(self):
        """Create UCP exchange handler."""
        exchange = UCPExchange(agent_id="buyer-001", agent_type="buyer")

        assert exchange.agent_id == "buyer-001"
        assert exchange.agent_type == "buyer"

    def test_create_segment_embedding(self):
        """Create segment embedding via exchange."""
        exchange = UCPExchange(agent_id="buyer-001", agent_type="buyer")

        emb = exchange.create_segment_embedding(
            segment_ids=["IAB1", "IAB2"],
            context_type=ContextType.BEHAVIORAL,
            confidence=0.95,
        )

        assert emb.format == EmbeddingFormat.SEGMENT_IDS
        assert len(emb.segment_ids) == 2
        assert emb.confidence == 0.95

    def test_create_vector_embedding(self):
        """Create vector embedding via exchange."""
        exchange = UCPExchange(agent_id="seller-001", agent_type="seller")

        emb = exchange.create_vector_embedding(
            vector=[0.1, 0.2, 0.3],
            context_type=ContextType.INTENT,
        )

        assert emb.format == EmbeddingFormat.DENSE_VECTOR
        assert emb.vector_dim == 3

    def test_create_exchange_request(self):
        """Create exchange request."""
        exchange = UCPExchange(agent_id="buyer-001", agent_type="buyer")
        emb = exchange.create_segment_embedding(segment_ids=["IAB1"])

        request = exchange.create_exchange_request(
            receiver_id="seller-001",
            embedding=emb,
            requested_types=[ContextType.DEMOGRAPHIC],
        )

        assert request.sender_id == "buyer-001"
        assert request.receiver_id == "seller-001"
        assert request.embedding is not None
        assert ContextType.DEMOGRAPHIC in request.requested_context_types

    def test_create_exchange_response(self):
        """Create exchange response."""
        buyer_exchange = UCPExchange(agent_id="buyer-001", agent_type="buyer")
        seller_exchange = UCPExchange(agent_id="seller-001", agent_type="seller")

        request = buyer_exchange.create_exchange_request(receiver_id="seller-001")
        seller_emb = seller_exchange.create_segment_embedding(
            segment_ids=["tech_enthusiasts"]
        )

        response = seller_exchange.create_exchange_response(
            request=request,
            embeddings=[seller_emb],
            match_score=0.85,
        )

        assert response.request_id == request.request_id
        assert response.sender_id == "seller-001"
        assert len(response.embeddings) == 1
        assert response.match_score == 0.85

    def test_calculate_match_score_segments(self):
        """Calculate match score for segment embeddings."""
        exchange = UCPExchange(agent_id="test", agent_type="test")

        buyer_emb = exchange.create_segment_embedding(
            segment_ids=["A", "B", "C"]
        )
        seller_emb = exchange.create_segment_embedding(
            segment_ids=["B", "C", "D"]
        )

        # Jaccard: {B,C} / {A,B,C,D} = 2/4 = 0.5
        score = exchange.calculate_match_score(buyer_emb, seller_emb)
        assert score == pytest.approx(0.5)

    def test_calculate_match_score_vectors(self):
        """Calculate match score for vector embeddings."""
        exchange = UCPExchange(agent_id="test", agent_type="test")

        buyer_emb = exchange.create_vector_embedding(vector=[1.0, 0.0])
        seller_emb = exchange.create_vector_embedding(vector=[1.0, 0.0])

        # Identical vectors should have score 1.0
        score = exchange.calculate_match_score(buyer_emb, seller_emb)
        assert score == pytest.approx(1.0)

    def test_cache_embedding(self):
        """Cache and retrieve embeddings."""
        exchange = UCPExchange(agent_id="test", agent_type="test")
        emb = exchange.create_segment_embedding(segment_ids=["A", "B"])

        exchange.cache_embedding(emb)
        retrieved = exchange.get_cached_embedding(emb.fingerprint())

        assert retrieved is not None
        assert retrieved.segment_ids == emb.segment_ids


class TestHallucinationInjector:
    """Tests for HallucinationInjector class."""

    def test_injection_with_seed(self):
        """Injection is deterministic with seed."""
        injector1 = HallucinationInjector(injection_rate=1.0, seed=42)
        injector2 = HallucinationInjector(injection_rate=1.0, seed=42)

        result1, _ = injector1.maybe_corrupt_inventory({"display": 1000}, "agent-1")
        result2, _ = injector2.maybe_corrupt_inventory({"display": 1000}, "agent-1")

        assert result1 == result2

    def test_no_injection_at_zero_rate(self):
        """No injection when rate is 0."""
        injector = HallucinationInjector(injection_rate=0.0)

        inventory = {"display": 1000, "video": 500}
        result, record = injector.maybe_corrupt_inventory(inventory, "agent-1")

        assert result == inventory
        assert record is None

    def test_always_inject_at_full_rate(self):
        """Always inject when rate is 1.0."""
        injector = HallucinationInjector(injection_rate=1.0, seed=123)

        inventory = {"display": 1000}
        result, record = injector.maybe_corrupt_inventory(inventory, "agent-1")

        assert result != inventory
        assert record is not None
        assert record.injection_type == InjectionType.INVENTORY
        assert record.injection_factor > 1.0  # Inflation

    def test_price_corruption_deflates(self):
        """Price corruption typically deflates prices."""
        injector = HallucinationInjector(injection_rate=1.0, seed=456)

        real_price = 10.0
        corrupted, record = injector.maybe_corrupt_price(real_price, "agent-1")

        assert corrupted < real_price  # Deflated
        assert record is not None
        assert record.injection_type == InjectionType.PRICE
        assert record.injection_factor < 1.0  # Deflation factor

    def test_history_fabrication(self):
        """Fabricated history contains expected fields."""
        injector = HallucinationInjector(injection_rate=1.0, seed=789)

        history, record = injector.maybe_fabricate_history("agent-1")

        assert history is not None
        assert history["fabricated"] is True
        assert "fake_deal_id" in history
        assert history["fake_deal_id"].startswith("FAKE-")
        assert record is not None
        assert record.injection_type == InjectionType.HISTORY

    def test_delivery_corruption(self):
        """Delivery corruption inflates impressions."""
        injector = HallucinationInjector(injection_rate=1.0, seed=101)

        result, record = injector.maybe_corrupt_delivery(
            real_impressions=10000,
            real_clicks=100,
            agent_id="agent-1",
        )

        assert result["impressions"] > 10000  # Inflated
        assert result["clicks"] == 100  # Unchanged
        assert record is not None
        assert record.injection_type == InjectionType.DELIVERY

    def test_injections_tracked(self):
        """All injections are tracked."""
        injector = HallucinationInjector(injection_rate=1.0, seed=202)

        injector.maybe_corrupt_inventory({"a": 100}, "agent-1")
        injector.maybe_corrupt_price(10.0, "agent-2")
        injector.maybe_fabricate_history("agent-3")

        injections = injector.get_injections()
        assert len(injections) == 3


class TestHallucinationDetector:
    """Tests for HallucinationDetector class (without DB)."""

    def test_create_detector(self):
        """Create detector without DB connection."""
        detector = HallucinationDetector()
        assert detector._db_connection is None

    def test_track_verifications(self):
        """Verifications are tracked in memory."""
        detector = HallucinationDetector()

        # Without DB, verifications are tracked but can't query ground truth
        assert detector.get_verifications() == []
        assert detector.get_hallucination_count() == 0
        assert detector.get_hallucination_rate() == 0.0


class TestHallucinationManager:
    """Tests for HallucinationManager class."""

    def test_scenario_b_injects(self):
        """Scenario B enables hallucination injection."""
        manager = HallucinationManager(scenario="B", injection_rate=1.0)

        # With 100% injection rate, inventory should be corrupted
        inventory = {"display": 1000}
        result = manager.process_inventory_data(
            real_inventory=inventory,
            agent_id="agent-1",
            agent_type="buyer",
            publisher_id="pub-1",
        )

        assert result != inventory

    def test_scenario_a_no_injection(self):
        """Scenario A disables injection."""
        manager = HallucinationManager(scenario="A", injection_rate=1.0)

        inventory = {"display": 1000}
        result = manager.process_inventory_data(
            real_inventory=inventory,
            agent_id="agent-1",
            agent_type="buyer",
            publisher_id="pub-1",
        )

        assert result == inventory  # No corruption

    def test_scenario_c_no_injection(self):
        """Scenario C disables injection."""
        manager = HallucinationManager(scenario="C", injection_rate=1.0)

        price = 10.0
        result = manager.process_price_data(
            real_price=price,
            agent_id="agent-1",
            agent_type="seller",
            publisher_id="pub-1",
        )

        assert result == price  # No corruption

    def test_summary_statistics(self):
        """Manager provides summary statistics."""
        manager = HallucinationManager(scenario="B", injection_rate=0.5)

        summary = manager.get_summary()

        assert summary["scenario"] == "B"
        assert "total_verifications" in summary
        assert "hallucinations_detected" in summary
        assert "total_injections" in summary
        assert "severity_breakdown" in summary


class TestSeverityLevels:
    """Tests for severity classification."""

    def test_severity_values(self):
        """Severity enum has expected values."""
        assert Severity.MINOR.value == "minor"
        assert Severity.MODERATE.value == "moderate"
        assert Severity.SEVERE.value == "severe"


class TestInjectionTypes:
    """Tests for injection type classification."""

    def test_injection_type_values(self):
        """InjectionType enum has expected values."""
        assert InjectionType.INVENTORY.value == "inventory"
        assert InjectionType.PRICE.value == "price"
        assert InjectionType.HISTORY.value == "history"
        assert InjectionType.DELIVERY.value == "delivery"
        assert InjectionType.AUDIENCE.value == "audience"
