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
| Field                  | Type                | Description                          |
| ---------------------- | ------------------- | ------------------------------------ |
| `user_input`           | `str`               | Raw user message for this turn       |
| `messages`             | `List[BaseMessage]` | LangChain message history            |
| `intent`               | `UserIntent`        | Result from `classify_intent`        |
| `next_step`            | `str`               | Route returned to `should_continue`  |
| `conversation_summary` | `str`               | Summary of prior turns               |
| `active_documents`     | `List[str]`         | Document IDs from the conversation   |
| `current_response`     | `Dict`              | Output from the most recent agent    |
| `tools_used`           | `List[str]`         | Names of tools used during the turn  |
| `actions_taken`        | `List[str]`         | Log of nodes visited during the turn |
```

The `messages` field uses LangGraph's `add_messages` reducer, so new messages are appended. The `actions_taken` field uses `operator.add` for the same reason.

## Turn persistance: `InMemorySaver` checkpointer

The workflow is compiled with `checkpointer=InMemorySaver()`. LanGraph saves the full `AgentState` after every node runs. Each node run is identified by thread_id, which is set to the session_id in the config passed into the workflow.

On the next turn, the previous turn's final state is available with all messages and the conversation_summary. This gives agents the full context of the conversation.

## Memory: update_memory

The `update_memory` function has the LLM summarize the full message and return an UpdateMemoryResponse. The UpdateMemoryResponse includes a short string and list of the document IDs references. This gets stored back to `conversation_summary`. This approach keeps the context window manageable for subsequent agent calls.

## Persistance: JSON files

`DocumentAssistant` writes each session state log to `./sessions/{session_id}.json` after every turn. Whe the flow starts the user can provide an existing `session_id` to resume the flow from disk. This is separate from `InMemorySaver`, which is in memory graph state.

## Structured output

Structured output is used so that the LLM returns results according to the desired schema. It is used in three different cases

### 1. Intent Classification

In `classify_intent`, the LLM must used teh `UserIntent` schema.

### 2. Agent responses

Each agent response requires its own response that is passed into `create_react_agent` as `response_format`

**`AnswerResponse`** — for Q&A:

- Requires `sources: List[str]`, so the model has to name which documents it used.
- Requires `confidence: float` to surface uncertainty.

**`SummarizationResponse`** — for summarization:

- Requires `key_points: List[str]`, so the model breaks findings into a list rather than a wall of text.
- Requires `document_ids: List[str]` to track which documents were summarized.

**`CalculationResponse`** — for calculations:

- Requires `expression: str`, so the formula is visible, not just the answer.
- Requires `result: float` as a typed number, not a string.

### 3. Memory update

`update_memory` uses the same pattern to pull a structured `summary` string and a `document_ids` list out of the message history, so the memory state is always typed rather than raw text.

## Example Conversations

### Q&A: Document lookup

```
Enter Message: what are the payment terms on invoice INV-001

The payment terms on invoice INV-001 are "Net 30 days."

INTENT: qa
TOOLS USED: document_reader
CONVERSATION SUMMARY: The payment terms on invoice INV-001 are "Net 30 days."
```

### Summarization: Summarize all contracts

```
Enter Message: summarize all contracts in the system

### Summary of Contract CON-001

- **Title:** Service Agreement
- **Parties Involved:**
  - **Provider:** DocDacity Solutions Inc.
  - **Client:** Healthcare Partners LLC
- **Date:** January 1, 2024
- **Services Provided:**
  1. Document Processing Platform Access
  2. 24/7 Technical Support
  3. Monthly Data Analytics Reports
  4. Compliance Monitoring
- **Duration:** 12 months
- **Financial Details:**
  - **Monthly Fee:** $15,000
  - **Total Contract Value:** $180,000
- **Termination Clause:** Either party may terminate the agreement with 60 days written notice.

INTENT: summarization
TOOLS USED: document_reader, document_search, document_reader
CONVERSATION SUMMARY: The payment terms on invoice INV-001 are "Net 30 days."
```

### Calculation: Sum invoice totals

```
Enter Message: calculate the sum of all invoice totals

The sum of all invoice totals is $303,800.

INTENT: calculation
TOOLS USED: document_reader, document_search, document_reader, document_reader,
            document_search, document_reader, document_reader, calculator

CONVERSATION SUMMARY: The user inquired about the payment terms on invoice INV-001,
  which were found to be "Net 30 days." They also requested a summary of all contracts,
  leading to the retrieval of contract CON-001, a service agreement between DocDacity
  Solutions Inc. and Healthcare Partners LLC, valued at $180,000 over 12 months.
  Additionally, the sum of all invoice totals was calculated to be $303,800.
```
