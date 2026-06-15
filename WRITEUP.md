# DocDacity Document Assistant — Implementation Writeup

## Architecture & Routing Decisions

The assistant is built as a **LangGraph state graph** with five nodes that run in sequence per request:

```
classify_intent → qa_agent          ┐
               → summarization_agent ├→ update_memory → END
               → calculation_agent  ┘
```

**Why LangGraph?** Each user turn needs to do three things: figure out what the user wants, call the right tools, and remember what happened. LangGraph's `StateGraph` handles all three without a lot of glue code.

**Intent classification as the entry point.** Every message goes through `classify_intent` first. It calls `llm.with_structured_output(UserIntent)` to get back a typed object with `intent_type`, `confidence`, and `reasoning`. The `should_continue` router reads `state["next_step"]` and sends the request to one of three agents. If the intent isn't recognized, it falls back to `qa_agent`.

**Three agents, one per intent type.** Each intent type gets its own system prompt in `prompts.py`:

- `QA_SYSTEM_PROMPT` — focused on citing sources and giving precise answers.
- `SUMMARIZATION_SYSTEM_PROMPT` — focused on structure and pulling out key points.
- `CALCULATION_SYSTEM_PROMPT` — tells the model to always use the `calculator` tool, never compute mentally.

Each agent runs via `invoke_react_agent`, which creates a `create_react_agent` instance with the task-specific Pydantic schema passed as `response_format`. The agent's final answer comes back as a typed object, not free-form text.

**Every turn ends at `update_memory`.** After any agent finishes, the graph always runs `update_memory` before stopping — the conversation summary and active document list get updated on every turn no matter which agent ran.

---

## State & Memory

### Per-turn state: `AgentState`

`AgentState` is a `TypedDict` that carries all data through the graph within a single turn:

| Field                  | Type                              | Role                                         |
| ---------------------- | --------------------------------- | -------------------------------------------- |
| `user_input`           | `str`                             | Raw user message for this turn               |
| `messages`             | `List[BaseMessage]` (append-only) | Full LangChain message history               |
| `intent`               | `UserIntent`                      | Classification result from `classify_intent` |
| `next_step`            | `str`                             | Routing signal read by `should_continue`     |
| `conversation_summary` | `str`                             | Rolling summary of prior turns               |
| `active_documents`     | `List[str]`                       | Document IDs referenced in the conversation  |
| `current_response`     | `Dict`                            | Raw output from the most recent agent        |
| `tools_used`           | `List[str]`                       | Names of tools called this turn              |
| `actions_taken`        | `List[str]`                       | Audit log of nodes visited this turn         |

The `messages` field uses LangGraph's `add_messages` reducer, so nodes add new messages rather than replacing the list.

### Cross-turn persistence: `InMemorySaver` checkpointer

The workflow is compiled with `checkpointer=InMemorySaver()`. LangGraph saves the full `AgentState` after every node runs, keyed by `thread_id`. The `thread_id` is set to the `session_id` in the config passed to each `workflow.invoke` call.

On the next turn, the previous turn's final state is available, including the accumulated `messages` and `conversation_summary`, so agents have the full conversation context.

### Rolling summary: `update_memory`

Rather than passing the full message list every time, `update_memory` asks the LLM to produce an `UpdateMemoryResponse` — a short summary string plus a list of referenced document IDs. This gets stored back into `conversation_summary` and shows up in the CLI as "CONVERSATION SUMMARY." This keeps context window usage from growing unbounded; without it, long sessions would keep adding to the message list sent to every subsequent agent call.

### Session persistence: JSON files

`DocumentAssistant` writes each session's state to `./sessions/<session_id>.json` after every turn. On startup a user can provide an existing `session_id` to resume from disk. This is separate from the LangGraph checkpointer — one is application-level file storage, the other is in-memory graph state.

---

## Structured Output

Structured output is used in three places.

### 1. Intent classification — `llm.with_structured_output()`

In `classify_intent`, the LLM is bound to the `UserIntent` schema before being called:

```python
structured_llm = llm.with_structured_output(UserIntent)
intent = structured_llm.invoke(prompt)   # returns a UserIntent instance
```

`UserIntent` uses a `Literal` type to restrict `intent_type` to four valid values, and `Field(ge=0.0, le=1.0)` to keep `confidence` in range. If the model returns something that doesn't match, it retries automatically.

### 2. Agent responses — `response_format` in `create_react_agent`

Each agent passes a schema class as `response_format` to `create_react_agent`. The three schemas each enforce different things:

**`AnswerResponse`** — for Q&A:
- Requires `sources: List[str]`, so the model has to name which documents it used.
- Requires `confidence: float` to surface uncertainty.

**`SummarizationResponse`** — for summarization:
- Requires `key_points: List[str]`, so the model breaks findings into a list rather than a wall of text.
- Requires `document_ids: List[str]` to track which documents were summarized.

**`CalculationResponse`** — for calculations:
- Requires `expression: str`, so the formula is visible, not just the answer.
- Requires `result: float` as a typed number, not a string.

### 3. Memory update — `llm.with_structured_output(UpdateMemoryResponse)`

`update_memory` uses the same pattern to pull a structured `summary` string and a `document_ids` list out of the message history, so the memory state is always typed rather than raw text.

---

## Example Conversations

The following examples show all three intent paths. Output lines beginning with `>` are assistant responses; metadata labels (INTENT, TOOLS USED, SOURCES) are printed by the CLI.

---

### Example 1 — Q&A: Specific document lookup

```
Enter Message: What is the total amount due on invoice INV-002?

INTENT: qa
TOOLS USED: document_reader
SOURCES: INV-002

> Invoice INV-002 (Invoice #12346, client TechStart Inc., dated 2024-02-20)
> has a total due of $69,300. This reflects a subtotal of $70,000,
> a 10% discount of -$7,000, and 10% tax of $6,300.
> Payment terms are Net 45 days.

CONVERSATION SUMMARY: User asked about the total amount on INV-002.
  The invoice for TechStart Inc. totals $69,300 after discount and tax.
```

---

### Example 2 — Summarization: Summarize all contracts

```
Enter Message: Summarize all contracts in the system.

INTENT: summarization
TOOLS USED: document_search, document_reader
SOURCES: CON-001

> Summary of Contracts:
>
> CON-001 — Service Agreement (DocDacity Solutions Inc. & Healthcare Partners LLC)
> • Duration: 12 months starting January 1, 2024
> • Monthly fee: $15,000 | Total contract value: $180,000
> • Services: Document processing platform, 24/7 support, analytics reports,
>   compliance monitoring
> • Termination: 60 days written notice by either party
>
> Key Points:
> - 1 active contract found
> - Total committed value: $180,000
> - Healthcare sector client

CONVERSATION SUMMARY: User requested a summary of all contracts.
  One contract found: CON-001 with Healthcare Partners LLC, valued at $180,000.
```

---

### Example 3 — Calculation: Sum invoice totals

```
Enter Message: Calculate the sum of all invoice totals.

INTENT: calculation
TOOLS USED: document_search, document_reader, calculator
SOURCES: INV-001, INV-002, INV-003

> I retrieved all three invoices and used the calculator for each step.
>
> INV-001 (Acme Corporation):    $22,000   (subtotal $20,000 + 10% tax)
> INV-002 (TechStart Inc.):      $69,300   (after discount and tax)
> INV-003 (Global Corp):        $214,500   (subtotal $195,000 + 10% tax)
>
> Expression: 22000 + 69300 + 214500
> Total: $305,800

CONVERSATION SUMMARY: User asked for the sum of all invoice totals.
  Three invoices found; combined total is $305,800.
```
