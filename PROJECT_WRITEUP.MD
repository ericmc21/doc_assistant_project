# DocDacity Document Assistant — Project Writeup

## Architecture and Routing Decisions

The Document Assistant was built using a **LangeGraph state graph** with five notes that run in sequence per request:

```
classify_intent -> [qa_agent, summarization_agent, calculation_agent] -> update_memory -> END

```

Why LangGraph? - Because it is a LanGraph course and each user turn needs 3 components

1. Determine the intent
2. Call the right agent
3. Remember what happened on the turn

LanGraph's StateGraph has a lot of this built-in.

Intent classification as the entry point. Every message must got through classify_intent function first to determine which agent should be called or named as the next_step. classify_intent uses llm.with_structured_output(UserIntent) to get an object containing the intent_type, confidence, and reasoning. The should_continue router reads from state state["next_step"] to get the next agent to send a request to. If no intent is recognized, it falls to qa_agent.

**Three agents, one per intent type.** Each intent gets its own system prompt from prompts.py`
