---
title: Cato Agent Modernization — Architecture Design Spec
date: 2026-03-18
status: agreed
authors: [engineering]
---

# Cato Agent Modernization — Architecture Design Spec

## Table of Contents

1. [Project Context](#1-project-context)
2. [Goals](#2-goals)
3. [Overall Architecture](#3-overall-architecture)
4. [LangGraph State Machine](#4-langgraph-state-machine)
5. [Hybrid RAG Pipeline](#5-hybrid-rag-pipeline)
6. [Memory Architecture](#6-memory-architecture)
7. [Observability and Streaming](#7-observability-and-streaming)
8. [Complete config.yaml Reference](#8-complete-configyaml-reference)
9. [What Is Not Changing](#9-what-is-not-changing)
10. [Upgrade Summary](#10-upgrade-summary)

---

## 1. Project Context

Cato Agent is a multi-agent AI system that helps people cash in on their home equity. The agent qualifies users by checking FICO score, equity percentage, property type, and bankruptcy status. It handles objections, answers informational questions, and books appointments.

The current implementation is built on LangChain with a basic RAG retriever (cosine similarity, k=3) and an if/else routing strategy in `master.py`. This design spec defines the target architecture that replaces those components with a production-grade, provider-agnostic, fully observable system.

---

## 2. Goals

| Goal | Description |
|---|---|
| Replace framework | Move from LangChain to LangGraph + LiteLLM |
| Provider-agnostic | All models and embeddings swappable via `config.yaml` only — no code changes |
| Upgrade RAG | Replace cosine similarity k=3 with Hybrid RAG (BM25 + vector) + cross-encoder re-ranking |
| Replace routing | Replace if/else routing with a LangGraph `StateGraph` |
| 3-tier memory | Working memory, episodic memory (rolling summary), and semantic memory (UserProfile) |
| Observability | Langfuse integration for traces, token costs, latency per node, and conversation replay |
| Streaming | SSE streaming via FastAPI |
| Fully autonomous | No human-in-the-loop (HITL) at any point — the agent makes autonomous qualification decisions |

---

## 3. Overall Architecture

### 3.1 Directory Layout

```
cato-agent/
├── config.yaml              # Single source of truth: models, RAG params, Redis, Langfuse
├── app/
│   ├── core/
│   │   ├── config.py        # Loads config.yaml → typed Settings object
│   │   ├── llm.py           # LiteLLM wrapper (all agent nodes call this)
│   │   └── embeddings.py    # LiteLLM embedding wrapper
│   ├── graph/
│   │   ├── state.py         # CatoState TypedDict (the shared graph state)
│   │   ├── graph.py         # LangGraph StateGraph definition and edge wiring
│   │   └── nodes/           # One file per agent node
│   │       ├── classifier.py
│   │       ├── qualifier.py
│   │       ├── objection.py
│   │       ├── booking.py
│   │       └── info.py
│   ├── rag/
│   │   ├── retriever.py     # Hybrid BM25 + vector search with RRF merge
│   │   ├── reranker.py      # Cross-encoder re-ranking (Cohere API or local model)
│   │   └── indexer.py       # Document ingestion pipeline
│   └── memory/
│       ├── working.py       # Redis-backed chat history (sliding window)
│       ├── episodic.py      # Rolling LLM-generated conversation summary
│       └── profile.py       # Structured UserProfile (FICO, address, equity, etc.)
```

### 3.2 Key Principles

- **`config.yaml` is the only file that changes to swap models or embeddings.** No code changes required to move between providers (e.g., OpenAI → Anthropic → local Ollama).
- **Every agent node is isolated.** One file, one responsibility, independently testable. Nodes communicate only through `CatoState`.
- **`CatoState` flows through the entire graph.** No globals, no singletons, no shared mutable state outside of the typed state object.
- **LangGraph owns all routing logic.** The if/else intent dispatch currently in `master.py` is fully replaced by LangGraph conditional edges.
- **Redis is the sole infrastructure dependency.** It serves as the vector store, session checkpointer, working memory store, and BM25 index — no additional services required.

---

## 4. LangGraph State Machine

### 4.1 CatoState

The shared state object passed through every node in the graph. Defined in `app/graph/state.py`.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from app.memory.profile import UserProfile

class CatoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intent: str                          # classified each turn by classify_intent node
    user_profile: UserProfile            # persisted FICO, address, equity, name, etc.
    qualification_result: str | None     # "qualified" | "unqualified" | "pending"
    conversation_summary: str            # rolling episodic memory injected as system context
```

### 4.2 Graph Topology

Defined in `app/graph/graph.py`. Each conversation turn is one full graph execution. State persists between turns via LangGraph's built-in Redis checkpointer (keyed on `session_id`).

```
START
  └─→ [classify_intent]
          ├─→ "objection"  →  [handle_objection]  →  END
          ├─→ "qualify"    →  [qualify]
          │                       ├─→ "qualified"   →  [book_appointment]  →  END
          │                       ├─→ "unqualified" →  END  (autonomous, no HITL)
          │                       └─→ "pending"     →  END  (ask follow-up next turn)
          ├─→ "book"       →  [book_appointment]  →  END
          └─→ "info"       →  [handle_info]  →  END
```

### 4.3 Node Responsibilities

| Node | File | Responsibility |
|---|---|---|
| `classify_intent` | `nodes/classifier.py` | Classifies user message into: `objection`, `qualify`, `book`, `info` |
| `qualify` | `nodes/qualifier.py` | Runs CoT reasoning on UserProfile; returns `qualified`, `unqualified`, or `pending` |
| `handle_objection` | `nodes/objection.py` | Responds to objections using RAG-retrieved rebuttal content |
| `book_appointment` | `nodes/booking.py` | Collects scheduling info and confirms appointment |
| `handle_info` | `nodes/info.py` | Answers informational questions using Hybrid RAG |

### 4.4 Edge Routing

Routing is implemented as LangGraph conditional edges — no if/else chains in application code.

- After `classify_intent`: route on `state["intent"]`
- After `qualify`: route on `state["qualification_result"]`
  - `"qualified"` → `book_appointment`
  - `"unqualified"` → `END` (autonomous disqualification, no human review)
  - `"pending"` → `END` (Cato asked a follow-up; partial UserProfile persists in Redis for next turn)
- After `handle_objection`: always `END` — on the next user turn, `classify_intent` re-routes naturally
- After `book_appointment`, `handle_info`: always `END`

### 4.5 Qualifier Internal Chain-of-Thought

The qualifier node runs an internal CoT reasoning pass before generating its user-facing response. This reasoning is never shown to the user; it drives the structured decision.

```python
reasoning_prompt = """
Given the conversation so far, determine:
1. What qualification fields are still missing?
2. Is there enough information to make a final decision?
3. If yes — does the user qualify based on:
   - FICO score >= 620
   - Equity percentage >= 25%
   - Property type is eligible (SFR, condo, or qualifying multi-family)
   - No active bankruptcy
4. If no — what single question should Cato ask next?

Respond in structured JSON only:
{
  "status": "qualified" | "unqualified" | "pending",
  "decision": "<brief rationale>",
  "next_question": "<question to ask if status is pending, else null>",
  "reasoning": "<internal step-by-step reasoning>"
}
"""
```

The qualifier checks `state["user_profile"]` first. It only asks for fields that are still `None`. It never re-asks for information already captured in the profile.

---

## 5. Hybrid RAG Pipeline

### 5.1 Overview

Replaces the current cosine similarity retriever (k=3) with a 3-stage pipeline that combines keyword search, vector search, rank fusion, and cross-encoder re-ranking.

```
Query
  └─→ [Stage 1: Dual Retrieval]
          ├─→ BM25 keyword search  → top 10 docs
          └─→ Vector similarity    → top 10 docs
                    └─→ [Stage 2: RRF Merge]  → top 15 deduplicated docs
                                └─→ [Stage 3: Re-ranker]  → top 3 returned to agent node
```

### 5.2 Stage 1 — Dual Retrieval

Both retrievers run in parallel against the same Redis instance.

- **Vector search**: `RedisVectorStore` (existing infrastructure, unchanged)
- **BM25 keyword search**: `FT.SEARCH` against a full-text field added to the existing Redis index schema. No new infrastructure required — this is an additive schema change on the existing index. **Note:** adding the full-text field requires a one-time re-index of all existing documents via `indexer.py`. New documents are indexed correctly on ingestion.

Configuration:
```yaml
rag:
  retrieval_k: 10    # top-k for each retrieval method independently
```

### 5.3 Stage 2 — Reciprocal Rank Fusion (RRF)

Merges and deduplicates results from both retrieval methods. Implemented in `app/rag/retriever.py`.

```python
from collections import defaultdict
from langchain_core.documents import Document

def reciprocal_rank_fusion(
    result_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            scores[doc.id] += 1 / (k + rank + 1)
            doc_map[doc.id] = doc

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:15]]
```

The `k=60` constant is standard RRF. The merge output is capped at 15 documents before passing to the re-ranker.

### 5.4 Stage 3 — Cross-Encoder Re-Ranking

Implemented in `app/rag/reranker.py`. Config-driven — switching between Cohere and local re-ranking requires only a `config.yaml` change.

| `reranker` value | Backend | Notes |
|---|---|---|
| `"cohere"` | Cohere Rerank API | Requires `COHERE_API_KEY` env var |
| `"local"` | `bge-reranker-base` via `sentence-transformers` | No external API, runs on CPU |

Configuration:
```yaml
rag:
  rerank_top_k: 3      # final documents passed to the agent node
  reranker: "cohere"   # or "local"
```

The re-ranker scores each of the 15 fused documents against the original query and returns the top `rerank_top_k` documents. These are injected into the agent node's LLM context.

---

## 6. Memory Architecture

### 6.1 Overview

3-tier memory system backed entirely by the existing Redis instance. No new infrastructure required.

| Tier | Name | Storage | Purpose |
|---|---|---|---|
| 1 | Working Memory | Redis list | Last N messages — raw conversation context |
| 2 | Episodic Memory | Redis string | LLM-generated rolling summary of older turns |
| 3 | Semantic Memory | Redis JSON | Structured `UserProfile` persisted across sessions |

### 6.2 Tier 1 — Working Memory

Implemented in `app/memory/working.py`.

- Stores the last `working_window` messages (default: 20) as a Redis list.
- Injected directly into every LLM call as the conversation history.
- Key pattern: `cato:working:{session_id}`

### 6.3 Tier 2 — Episodic Memory

Implemented in `app/memory/episodic.py`.

- When the message count reaches `summary_threshold` (default: 16), the oldest half of messages is compressed into a single LLM-generated summary and those messages are dropped from the working memory list.
- This compression repeats on every subsequent turn once the threshold is hit — it is a rolling operation, not a one-time trigger. Each turn that pushes the count back to or above the threshold triggers another compression pass.
- The summary is prepended to every subsequent LLM call as a system context message.
- Keeps token usage bounded without losing conversational coherence.
- Key pattern: `cato:summary:{session_id}`

### 6.4 Tier 3 — Semantic Memory (UserProfile)

Implemented in `app/memory/profile.py`.

```python
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str | None = None
    property_address: str | None = None
    property_type: str | None = None    # "SFR" | "condo" | "multi-family"
    estimated_value: float | None = None
    mortgage_balance: float | None = None
    fico_score: int | None = None
    has_bankruptcy: bool | None = None

    @property
    def equity_pct(self) -> float | None:
        if self.estimated_value and self.mortgage_balance:
            return (self.estimated_value - self.mortgage_balance) / self.estimated_value
        return None
```

- Persisted as JSON in Redis with a configurable TTL.
- The qualifier node reads `UserProfile` at the start of every turn and only asks for fields still `None`.
- Key pattern: `cato:profile:{session_id}`

### 6.5 Redis Key Summary

| Key Pattern | Tier | Content |
|---|---|---|
| `cato:working:{session_id}` | Working | Raw message list (last N turns) |
| `cato:summary:{session_id}` | Episodic | LLM-compressed summary string |
| `cato:profile:{session_id}` | Semantic | Serialized `UserProfile` JSON |

### 6.6 Memory Configuration

```yaml
memory:
  working_window: 20       # number of messages to retain verbatim
  summary_threshold: 16    # message count that triggers episodic compression
  profile_ttl_days: 30     # Redis TTL for UserProfile keys
```

---

## 7. Observability and Streaming

### 7.1 Langfuse Observability

Langfuse is integrated via the LiteLLM callback interface. The integration is a single-line addition in `app/core/llm.py` and is controlled by `config.yaml`.

What Langfuse captures automatically:
- Every agent node's LLM call: prompt, response, model name
- Token counts and cost per model per session
- Latency per graph node
- Qualification decisions and the reasoning that produced them
- Full conversation replay for any `session_id`

```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key=settings.langfuse.public_key,
    secret_key=settings.langfuse.secret_key,
)

# Used in every LiteLLM call:
await litellm.acompletion(
    model=settings.llm.model,
    messages=messages,
    callbacks=[langfuse_handler],
)
```

Langfuse can be self-hosted via Docker or used as a cloud service. No vendor lock-in.

Configuration:
```yaml
langfuse:
  enabled: true
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
```

If `enabled: false`, the callback is not registered and no traces are emitted.

### 7.2 FastAPI SSE Streaming

The FastAPI endpoint in `app/main.py` streams Cato's responses token-by-token using Server-Sent Events (SSE).

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.graph.graph import cato_graph
from app.graph.state import CatoState

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    state = CatoState(
        session_id=request.session_id,
        messages=[HumanMessage(content=request.message)],
        intent="",                      # set by classify_intent node
        user_profile=UserProfile(),     # all fields None until populated
        qualification_result=None,
        conversation_summary="",        # empty until first episodic compression
    )

    async def event_stream():
        async for chunk in cato_graph.astream(state):
            if chunk.get("content"):
                yield f"data: {chunk['content']}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

Streaming is controlled by the `llm.streaming` config flag. When `false`, the endpoint falls back to a standard JSON response.

Configuration:
```yaml
llm:
  model: "openai/gpt-4o"
  streaming: true
  temperature: 0.7
```

---

## 8. Complete config.yaml Reference

This is the single source of truth for all runtime configuration. Swapping any model or provider requires editing only this file.

```yaml
llm:
  model: "openai/gpt-4o"      # Any LiteLLM-supported model string
  temperature: 0.7
  streaming: true

embeddings:
  model: "openai/text-embedding-3-large"   # Any LiteLLM-supported embedding model

rag:
  retrieval_k: 10        # Top-k docs retrieved by each method (BM25 and vector)
  rerank_top_k: 3        # Final docs passed to the agent after re-ranking
  reranker: "cohere"     # "cohere" (API) or "local" (bge-reranker-base)

memory:
  working_window: 20     # Number of verbatim messages kept in working memory
  summary_threshold: 16  # Message count that triggers episodic compression
  profile_ttl_days: 30   # Redis TTL for UserProfile records

redis:
  url: "redis://localhost:6379"

langfuse:
  enabled: true
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
```

**Provider swap example:** To move from OpenAI to Anthropic Claude, change:
```yaml
llm:
  model: "anthropic/claude-3-5-sonnet-20241022"

embeddings:
  model: "cohere/embed-english-v3.0"
```

No other file changes are needed.

---

## 9. What Is Not Changing

The following components are intentionally preserved. The modernization is additive and replacive, not a rewrite from scratch.

| Component | Reason Preserved |
|---|---|
| Redis as infrastructure backbone | Already serves session state, memory, and vector store. Extending it for BM25 avoids new infrastructure. |
| Core qualification rules | Business rules are fixed: FICO >= 620, equity >= 25%, eligible property type, no active bankruptcy. |
| Multi-agent structure | Classifier, qualifier, objection handler, booking agent, and info agent are all retained as distinct nodes. |
| Python | Implementation language unchanged. |

---

## 10. Upgrade Summary

| Area | Current State | Target State |
|---|---|---|
| Framework | LangChain | LangGraph + LiteLLM |
| Routing | if/else intent classifier in `master.py` | LangGraph `StateGraph` with conditional edges |
| RAG | Cosine similarity, k=3 | Hybrid (BM25 + vector) + Cross-encoder re-ranking |
| Memory | Last 6 messages | 3-tier: working (20 msg window) + episodic (rolling summary) + semantic (UserProfile) |
| Agent reasoning | Direct LLM action | Internal CoT scratchpad before user-facing response |
| Observability | None | Langfuse: traces, token cost, latency, conversation replay |
| UX delivery | Full-string response | Streaming SSE via FastAPI |
| Human-in-the-loop | N/A | None — fully autonomous qualification and disqualification |
| Provider lock-in | OpenAI-specific | Provider-agnostic via LiteLLM + config.yaml |
