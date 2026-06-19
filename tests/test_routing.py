import pytest
from unittest.mock import MagicMock

from agent import classify_intent, should_continue
from schemas import UserIntent


def _config_with_intent(intent_type: str) -> dict:
    """Build a fake config whose LLM always classifies as intent_type."""
    mock_llm = MagicMock()
    intent = UserIntent(
        intent_type=intent_type,
        confidence=0.9,
        reasoning=f"classified as {intent_type}",
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = intent
    mock_llm.with_structured_output.return_value = mock_structured
    return {"configurable": {"llm": mock_llm}}


def _base_state() -> dict:
    return {"user_input": "test message", "messages": []}


class TestShouldContinue:
    def test_routes_qa_agent(self):
        assert should_continue({"next_step": "qa_agent"}) == "qa_agent"

    def test_routes_summarization_agent(self):
        assert should_continue({"next_step": "summarization_agent"}) == "summarization_agent"

    def test_routes_calculation_agent(self):
        assert should_continue({"next_step": "calculation_agent"}) == "calculation_agent"

    def test_routes_end(self):
        assert should_continue({"next_step": "end"}) == "end"

    def test_defaults_to_end_when_key_missing(self):
        assert should_continue({}) == "end"


class TestClassifyIntent:
    def test_qa_intent_sets_qa_agent_next_step(self):
        result = classify_intent(_base_state(), _config_with_intent("qa"))
        assert result["next_step"] == "qa_agent"

    def test_summarization_intent_sets_summarization_agent(self):
        result = classify_intent(_base_state(), _config_with_intent("summarization"))
        assert result["next_step"] == "summarization_agent"

    def test_calculation_intent_sets_calculation_agent(self):
        result = classify_intent(_base_state(), _config_with_intent("calculation"))
        assert result["next_step"] == "calculation_agent"

    def test_unknown_intent_falls_back_to_qa_agent(self):
        result = classify_intent(_base_state(), _config_with_intent("unknown"))
        assert result["next_step"] == "qa_agent"

    def test_intent_object_stored_in_state(self):
        result = classify_intent(_base_state(), _config_with_intent("qa"))
        assert isinstance(result["intent"], UserIntent)
        assert result["intent"].intent_type == "qa"

    def test_classify_intent_recorded_in_actions_taken(self):
        result = classify_intent(_base_state(), _config_with_intent("qa"))
        assert "classify_intent" in result["actions_taken"]

    def test_structured_output_invoked_with_user_intent_schema(self):
        config = _config_with_intent("qa")
        classify_intent(_base_state(), config)
        mock_llm = config["configurable"]["llm"]
        mock_llm.with_structured_output.assert_called_once_with(UserIntent)
