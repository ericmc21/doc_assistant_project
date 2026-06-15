# DocDacity Document Assistant — Implementation Writeup

## Architecture & Routing Decisions

The assistant is built as a **LangGraph state graph** with five nodes that run in sequence per request:

```
classify_intent → qa_agent          ┐
               → summarization_agent ├→ update_memory → END
               → calculation_agent  ┘
```

**Why LangGraph?** Each user turn needs to do three things: figure out what the user wants, call the right tools, and remember what happened. LangGraph's `StateGraph` handles all three without a lot of glue code.

**Intent classification as the entry point.** Every message goes through `classify_intent` first (`agent.py:93`). It calls `llm.with_structured_output(UserIntent)` to get back a typed object with `intent_type`, `confidence`, and `reasoning`. The `should_continue` router (`agent.py:244`) reads `state["next_step"]` and sends the request to one of three agents. If the intent isn't recognized, it falls back to `qa_agent`.

**Three agents, one per intent type.** Each intent type gets its own system prompt in `prompts.py`:

- `QA_SYSTEM_PROMPT` — focused on citing sources and giving precise answers.
- `SUMMARIZATION_SYSTEM_PROMPT` — focused on structure and pulling out key points.
- `CALCULATION_SYSTEM_PROMPT` — tells the model to always use the `calculator` tool, never compute mentally.

Each specialist node calls `invoke_react_agent` (`agent.py:71`), which creates a `create_react_agent` instance with the task-specific Pydantic schema passed as `response_format`. The agent's final answer comes back as a typed object, not free-form text.

One thing worth noting: `tools_used` in the state accumulates tool calls from prior turns because `result["messages"]` from `create_react_agent` includes the full chat history passed in as input. It's a known quirk of how LangGraph's `MessagesState` merges the initial messages with newly generated ones.

**Every turn ends at `update_memory`.** After any agent finishes, the graph always runs `update_memory` before stopping — the conversation summary and active document list get updated on every turn no matter which agent ran.

**Tool set.** Four tools are registered in `tools.py`:

| Tool                  | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| `calculator`          | Safe `eval()` of arithmetic expressions; required for all math   |
| `document_search`     | Keyword, type, amount, and range queries over the document store |
| `document_reader`     | Gets the full content of a document by ID                        |
| `document_statistics` | Aggregate counts and financial totals across the collection      |

All tools log every call through `ToolLogger` to JSON files under `./logs/`.

---

## State & Memory

### Per-turn state: `AgentState`

`AgentState` (`agent.py:42`) is a `TypedDict` that carries all data through the graph within a single turn:

| Field                  | Type                              | Role                                                                                             |
| ---------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------ |
| `user_input`           | `str`                             | Raw user message for this turn                                                                   |
| `messages`             | `List[BaseMessage]` (append-only) | Full LangChain message history                                                                   |
| `intent`               | `UserIntent`                      | Classification result from `classify_intent`                                                     |
| `next_step`            | `str`                             | Routing signal read by `should_continue`                                                         |
| `conversation_summary` | `str`                             | Rolling summary of prior turns                                                                   |
| `active_documents`     | `List[str]`                       | Document IDs referenced in the conversation                                                      |
| `current_response`     | `Dict`                            | Raw output from the most recent agent                                                            |
| `tools_used`           | `List[str]`                       | Names of tools called this turn                                                                  |
| `actions_taken`        | `List[str]` (accumulate)          | Audit log of nodes visited; uses `operator.add` reducer so each node appends without overwriting |

The `messages` field uses LangGraph's `add_messages` reducer, so nodes add new messages rather than replacing the list. The `actions_taken` field uses `operator.add` for the same reason.

### Cross-turn persistence: `InMemorySaver` checkpointer

The workflow is compiled with `checkpointer=InMemorySaver()` (`agent.py:288`). LangGraph saves the full `AgentState` after every node runs, keyed by `thread_id`. The `thread_id` is set to the `session_id` in the config passed to each `workflow.invoke` call (`assistant.py:122`).

On the next turn, `workflow.get_state(config)` returns the previous turn's final state, including the accumulated `messages` and `conversation_summary`. The assistant loads these at the start of each turn (`assistant.py:96–112`) so agents have the full conversation context.

### Rolling summary: `update_memory`

Rather than passing the full message list every time, `update_memory` (`agent.py:214`) asks the LLM to produce an `UpdateMemoryResponse` — a short summary string plus a list of referenced document IDs. This gets stored back into `conversation_summary` and shows up in the CLI as "CONVERSATION SUMMARY." I chose this approach to keep context window usage from growing unbounded; without it, long sessions would keep adding to the message list sent to every subsequent agent call.

### Session persistence: JSON files

`DocumentAssistant` writes each session's state to `./sessions/<session_id>.json` after every turn (`assistant.py:81`). On startup a user can provide an existing `session_id` to resume from disk. This is separate from the LangGraph checkpointer — one is application-level file storage, the other is in-memory graph state.

---

## Structured Output

Structured output is used in three places.

### 1. Intent classification — `llm.with_structured_output()`

In `classify_intent`, the LLM is bound to the `UserIntent` schema before being called:

```python
structured_llm = llm.with_structured_output(UserIntent)
intent = structured_llm.invoke(prompt)   # returns a UserIntent instance
```

`UserIntent` (`schemas.py:70`) uses a `Literal` type to restrict `intent_type` to four valid values, and `Field(ge=0.0, le=1.0)` to keep `confidence` in range. If the model returns something that doesn't match, it retries automatically.

### 2. Agent responses — `response_format` in `create_react_agent`

Each agent passes a schema class to `invoke_react_agent`, which forwards it as `response_format` to `create_react_agent` (`agent.py:79`). The three schemas each enforce different things:

**`AnswerResponse`** (`schemas.py:23`) — for Q&A:

- Requires `sources: List[str]`, so the model has to name which documents it used.
- Requires `confidence: float` to surface uncertainty.

**`SummarizationResponse`** (`schemas.py:35`) — for summarization:

- Requires `key_points: List[str]`, so the model breaks findings into a list rather than a wall of text.
- Requires `document_ids: List[str]` to track which documents were summarized.

**`CalculationResponse`** (`schemas.py:47`) — for calculations:

- Requires `expression: str`, so the formula is visible, not just the answer.
- Requires `result: float` as a typed number, not a string.

### 3. Memory update — `llm.with_structured_output(UpdateMemoryResponse)`

`update_memory` uses the same pattern to pull a structured `summary` string and a `document_ids` list out of the message history (`agent.py:234`), so the memory state is always typed rather than raw text.

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

---

### Example 4 — Calculation: Amount filter query

Note: the intent classifier routes filter queries like this to `calculation_agent` because the prompt defines `calculation` broadly as anything that "may require calculations." A pure filter with no aggregation would arguably fit better under `qa`, but the classification still works since `calculation_agent` can handle search-and-report queries fine.

```
Enter Message: Find documents with amounts over $50,000 and tell me their total.

INTENT: calculation
TOOLS USED: document_search, calculator
SOURCES: CON-001, INV-002, INV-003

> Documents with amounts over $50,000:
>   CON-001 — $180,000
>   INV-002 — $69,300
>   INV-003 — $214,500
>
> Expression: 180000 + 69300 + 214500
> Combined total: $463,800

CONVERSATION SUMMARY: User queried documents over $50,000 and asked for
  their combined total. Three documents matched; combined value is $463,800.
```

---

### Example 5 — Multi-turn: Context carries across turns

```
Enter Message: What is claim CLM-001 about?

INTENT: qa
TOOLS USED: document_reader
SOURCES: CLM-001

> Claim CLM-001 is a medical expense reimbursement claim filed by John Doe
> (policy POL-456789). The incident date is 2024-02-10. Claimed expenses
> include a hospital visit ($1,200), diagnostic tests ($800), medication ($150),
> and a follow-up consultation ($300), for a total of $2,450. Status: Under Review.

---

Enter Message: What percentage of that total is the hospital visit?

INTENT: calculation
TOOLS USED: calculator
SOURCES: CLM-001

> (The assistant uses context from the previous turn — CLM-001 is already active.)
>
> Expression: 1200 / 2450 * 100
> The hospital visit ($1,200) represents approximately 48.98% of the
> total claim amount ($2,450).

CONVERSATION SUMMARY: User first asked about CLM-001 (medical claim, $2,450 total,
  status Under Review). Then asked what percentage the hospital visit represents —
  answer is ~48.98%.
```
