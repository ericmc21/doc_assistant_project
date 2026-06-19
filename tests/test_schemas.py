import pytest
from pydantic import ValidationError
from datetime import datetime

from schemas import (
    DocumentChunk,
    AnswerResponse,
    UserIntent,
    SummarizationResponse,
    CalculationResponse,
    UpdateMemoryResponse,
    SessionState,
)


class TestDocumentChunk:
    def test_required_fields(self):
        chunk = DocumentChunk(doc_id="INV-001", content="test content")
        assert chunk.doc_id == "INV-001"
        assert chunk.content == "test content"

    def test_metadata_defaults_to_empty_dict(self):
        chunk = DocumentChunk(doc_id="INV-001", content="test")
        assert chunk.metadata == {}
        assert isinstance(chunk.metadata, dict)

    def test_metadata_instances_are_independent(self):
        c1 = DocumentChunk(doc_id="INV-001", content="test")
        c2 = DocumentChunk(doc_id="INV-002", content="test")
        c1.metadata["new_key"] = "value"
        assert "new_key" not in c2.metadata

    def test_relevance_score_defaults_to_zero(self):
        chunk = DocumentChunk(doc_id="INV-001", content="test")
        assert chunk.relevance_score == 0.0

    def test_custom_metadata_accepted(self):
        chunk = DocumentChunk(doc_id="INV-001", content="test", metadata={"client": "Acme"})
        assert chunk.metadata["client"] == "Acme"

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            DocumentChunk(doc_id="INV-001")


class TestAnswerResponse:
    def test_valid_response(self):
        r = AnswerResponse(question="Q?", answer="A", sources=["INV-001"], confidence=0.9)
        assert r.confidence == 0.9
        assert r.sources == ["INV-001"]

    def test_confidence_boundary_zero(self):
        r = AnswerResponse(question="Q?", answer="A", sources=[], confidence=0.0)
        assert r.confidence == 0.0

    def test_confidence_boundary_one(self):
        r = AnswerResponse(question="Q?", answer="A", sources=[], confidence=1.0)
        assert r.confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            AnswerResponse(question="Q?", answer="A", sources=[], confidence=1.1)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            AnswerResponse(question="Q?", answer="A", sources=[], confidence=-0.1)

    def test_timestamp_auto_populated(self):
        r = AnswerResponse(question="Q?", answer="A", sources=[], confidence=0.5)
        assert isinstance(r.timestamp, datetime)

    def test_empty_sources_list_accepted(self):
        r = AnswerResponse(question="Q?", answer="A", sources=[], confidence=0.5)
        assert r.sources == []


class TestUserIntent:
    def test_qa_intent(self):
        intent = UserIntent(intent_type="qa", confidence=0.9, reasoning="question asked")
        assert intent.intent_type == "qa"

    def test_summarization_intent(self):
        intent = UserIntent(intent_type="summarization", confidence=0.9, reasoning="summary requested")
        assert intent.intent_type == "summarization"

    def test_calculation_intent(self):
        intent = UserIntent(intent_type="calculation", confidence=0.9, reasoning="math needed")
        assert intent.intent_type == "calculation"

    def test_unknown_intent(self):
        intent = UserIntent(intent_type="unknown", confidence=0.4, reasoning="unclear")
        assert intent.intent_type == "unknown"

    def test_invalid_intent_type_raises(self):
        with pytest.raises(ValidationError):
            UserIntent(intent_type="chat", confidence=0.9, reasoning="test")

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            UserIntent(intent_type="qa", confidence=1.5, reasoning="test")

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            UserIntent(intent_type="qa", confidence=-0.5, reasoning="test")


class TestSummarizationResponse:
    def test_document_ids_defaults_to_empty_list(self):
        r = SummarizationResponse(original_length=100, summary="summary", key_points=["p1"])
        assert r.document_ids == []
        assert isinstance(r.document_ids, list)

    def test_document_ids_instances_are_independent(self):
        r1 = SummarizationResponse(original_length=100, summary="s", key_points=[])
        r2 = SummarizationResponse(original_length=200, summary="s", key_points=[])
        r1.document_ids.append("INV-001")
        assert r2.document_ids == []

    def test_required_fields_accepted(self):
        r = SummarizationResponse(
            original_length=500,
            summary="Short summary",
            key_points=["Point A", "Point B"],
            document_ids=["CON-001"],
        )
        assert r.original_length == 500
        assert len(r.key_points) == 2


class TestCalculationResponse:
    def test_required_fields(self):
        r = CalculationResponse(expression="1+1", result=2.0, explanation="simple addition")
        assert r.result == 2.0
        assert r.units is None

    def test_optional_units(self):
        r = CalculationResponse(expression="50*2", result=100.0, explanation="mul", units="USD")
        assert r.units == "USD"


class TestUpdateMemoryResponse:
    def test_document_ids_defaults_to_empty_list(self):
        r = UpdateMemoryResponse(summary="summary text")
        assert r.document_ids == []
        assert isinstance(r.document_ids, list)

    def test_document_ids_instances_are_independent(self):
        r1 = UpdateMemoryResponse(summary="s1")
        r2 = UpdateMemoryResponse(summary="s2")
        r1.document_ids.append("INV-001")
        assert r2.document_ids == []


class TestSessionState:
    def test_conversation_history_defaults_to_empty_list(self):
        state = SessionState(session_id="abc", user_id="user1")
        assert state.conversation_history == []
        assert isinstance(state.conversation_history, list)

    def test_document_context_defaults_to_empty_list(self):
        state = SessionState(session_id="abc", user_id="user1")
        assert state.document_context == []
        assert isinstance(state.document_context, list)

    def test_timestamps_auto_populated(self):
        state = SessionState(session_id="abc", user_id="user1")
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_updated, datetime)
