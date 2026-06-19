# DocDacity Document Assistant

An AI-powered document assistant that answers questions, summarizes documents, and performs calculations on financial and healthcare records. Built with LangChain and LangGraph.

## What it does

Each user message is classified into one of three intents and routed to the appropriate agent:

- **Q&A** — searches the document store and answers questions with source citations
- **Summarization** — extracts key points and structure from one or more documents
- **Calculation** — retrieves relevant documents and uses a safe calculator tool to compute results

After each turn, a memory node summarizes the conversation and tracks referenced document IDs so context carries over.

## Architecture

```
classify_intent → [qa_agent | summarization_agent | calculation_agent] → update_memory → END
```

The workflow is a LangGraph `StateGraph` compiled with `InMemorySaver`, so state persists across turns within a session. See [PROJECT_WRITEUP.MD](PROJECT_WRITEUP.MD) for a detailed walkthrough of design decisions.

## Setup

**Prerequisites:** Python 3.9+, OpenAI API key

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

## Running

```bash
python main.py
```

## Commands

| Command | Description |
|---------|-------------|
| `/docs` | List all documents in the system |
| `/help` | Show commands and example queries |
| `/quit` | Exit |

## Example queries

```
What are the payment terms on invoice INV-001?
Summarize all contracts
Calculate the sum of all invoice totals
Find documents over $50,000
```

## Running tests

```bash
pytest
```

## Project structure

```
doc_assistant_project/
├── src/
│   ├── schemas.py      # Pydantic models
│   ├── retrieval.py    # Simulated document retrieval
│   ├── tools.py        # Agent tools (calculator, document search/read/stats)
│   ├── prompts.py      # Prompt templates
│   ├── agent.py        # LangGraph workflow and node functions
│   └── assistant.py    # DocumentAssistant session manager
├── tests/
│   ├── test_schemas.py
│   ├── test_calculator.py
│   ├── test_prompts.py
│   └── test_routing.py
├── sessions/           # Persisted session state (JSON)
├── logs/               # Tool usage logs (JSON)
├── main.py             # CLI entry point
├── requirements.txt
└── PROJECT_WRITEUP.MD  # Architecture and design notes
```
