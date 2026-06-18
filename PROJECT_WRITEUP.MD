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

LanGraph's StateGraph has a lot of this built-in with workflow.add_node, workflow.set_entry_point, worfflow.add_conditional_edges.

Intent classification as the entry point. Every message must got through `classify_intent` function first to determine which agent should be called or named as the next_step. `classify_intent` uses llm.with_structured_output(UserIntent) to get an object containing the intent_type, confidence, and reasoning. The should_continue router reads from state state["next_step"] to get the next agent to send a request to. If no intent is recognized, it falls to qa_agent.

**Three agents, one per intent type.** Each intent gets its own system prompt from prompts.py`

- `QA_SYSTEM_PROMPT` - search and read documents, cite sources and give precise answers.
- `SUMMARIZATION_SYSTEM_PROMPT` - extract key info, organize summaries, include document ID in summary, keep summaries short but complete
- `CALCULATION_SYSTEM_PROMPT` - User document reader tool to get the right document then determine the calculation that the user needs.

Each agent runs via `invoke_react_agent`, which creates a `create_react_agent` instance with the task-specific Pydantic schema passed as `response_format`. The agent's final answer comes back as a typed object, not free-form text. **Note** create_react_agent is deprecated. I just left it as is because it was part of the sample code from the workspace.

\*\*Every turn ends with `update_memory`. so that the conversation summary and active document list get updated on every turn.

## State & Memory

### Turn state: `AgentState`

`AgentState` is a `TypeDict` that carries all data through the graph within a single turn:

You have a trailing space issue — Markdown requires **two spaces** at the end of a line for a line break, or a blank line between lines for paragraph breaks.

Your lines probably have inconsistent or no trailing spaces. Easiest fix: use a blank line between each row, or better yet, just make this a proper Markdown table:

```markdown
| Field | Type | Description |
|-------|------|-------------|
| `user_input` | `str` | Raw user message for this turn |
| `messages` | `List[BaseMessage]` | LangChain message history |
| `intent` | `UserIntent` | Result from `classify_intent` |
| `next_step` | `str` | Route returned to `should_continue` |
| `conversation_summary` | `str` | Summary of prior turns |
| `active_documents` | `List[str]` | Document IDs from the conversation |
| `current_response` | `Dict` | Output from the most recent agent |
| `tools_used` | `List[str]` | Names of tools used during the turn |
| `actions_taken` | `List[str]` | Log of nodes visited during the turn |
```