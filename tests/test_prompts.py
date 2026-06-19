import pytest
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate

from prompts import (
    get_chat_prompt_template,
    get_intent_classification_prompt,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    CALCULATION_SYSTEM_PROMPT,
)


def _system_content(intent_type: str) -> str:
    template = get_chat_prompt_template(intent_type)
    return template.messages[0].prompt.template


class TestGetChatPromptTemplate:
    def test_returns_chat_prompt_template(self):
        assert isinstance(get_chat_prompt_template("qa"), ChatPromptTemplate)

    def test_qa_uses_qa_system_prompt(self):
        assert _system_content("qa") == QA_SYSTEM_PROMPT

    def test_summarization_uses_summarization_system_prompt(self):
        assert _system_content("summarization") == SUMMARIZATION_SYSTEM_PROMPT

    def test_calculation_uses_calculation_system_prompt(self):
        assert _system_content("calculation") == CALCULATION_SYSTEM_PROMPT

    def test_unknown_falls_back_to_qa(self):
        assert _system_content("unknown") == QA_SYSTEM_PROMPT

    def test_arbitrary_type_falls_back_to_qa(self):
        assert _system_content("something_else") == QA_SYSTEM_PROMPT

    def test_template_includes_input_variable(self):
        template = get_chat_prompt_template("qa")
        assert "input" in template.input_variables

    def test_template_includes_chat_history_variable(self):
        template = get_chat_prompt_template("qa")
        assert "chat_history" in template.input_variables

    def test_first_message_is_system_message(self):
        template = get_chat_prompt_template("qa")
        assert isinstance(template.messages[0], SystemMessagePromptTemplate)


class TestGetIntentClassificationPrompt:
    def test_returns_prompt_template(self):
        assert isinstance(get_intent_classification_prompt(), PromptTemplate)

    def test_has_user_input_variable(self):
        prompt = get_intent_classification_prompt()
        assert "user_input" in prompt.input_variables

    def test_has_conversation_history_variable(self):
        prompt = get_intent_classification_prompt()
        assert "conversation_history" in prompt.input_variables

    def test_formats_variables_into_output(self):
        prompt = get_intent_classification_prompt()
        formatted = prompt.format(
            user_input="What is the total?",
            conversation_history="User: Hello",
        )
        assert "What is the total?" in formatted
        assert "User: Hello" in formatted

    def test_prompt_mentions_all_intent_types(self):
        prompt = get_intent_classification_prompt()
        template_text = prompt.template
        for intent in ("qa", "summarization", "calculation", "unknown"):
            assert intent in template_text
