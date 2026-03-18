# Cato Agent Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LangChain if/else routing system with a LangGraph + LiteLLM multi-agent system featuring hybrid RAG, 3-tier memory, Langfuse observability, and FastAPI SSE streaming.

**Architecture:** A `StateGraph` built in LangGraph routes each conversation turn through isolated agent nodes (`classifier → qualifier/objection/booking/info`). All nodes share a typed `CatoState` object and communicate exclusively through it. A 3-tier Redis-backed memory system (working/episodic/semantic) replaces the old 6-message window. Hybrid RAG (BM25 + vector + cross-encoder) replaces the old cosine k=3 retriever.

**Tech Stack:** LangGraph, LiteLLM, FastAPI, Redis Stack, rank-bm25, sentence-transformers or Cohere Rerank, Langfuse, pytest + fakeredis

---

## File Map

### New files to create
| File | Responsibility |
|---|---|
| `config.yaml` | Single source of truth: models, RAG params, memory, Redis, Langfuse |
| `app/core/config.py` | Replace old Settings — loads `config.yaml` → typed `Settings` dataclass |
| `app/core/llm.py` | LiteLLM async completion wrapper with optional Langfuse callback |
| `app/core/embeddings.py` | LiteLLM embedding functions + `CatoEmbeddings` LangChain-compatible adapter |
| `app/memory/profile.py` | `UserProfile` Pydantic model with `equity_pct` property |
| `app/memory/working.py` | Redis-backed sliding window chat history |
| `app/memory/episodic.py` | LLM-generated rolling conversation summary |
| `app/graph/state.py` | `CatoState` TypedDict — shared across all nodes |
| `app/graph/nodes/classifier.py` | LLM classifies user message → intent string |
| `app/graph/nodes/qualifier.py` | CoT qualification reasoning → updates UserProfile + qualification_result |
| `app/graph/nodes/objection.py` | RAG-backed objection rebuttal |
| `app/graph/nodes/booking.py` | Appointment scheduling dialogue |
| `app/graph/nodes/info.py` | RAG-backed informational Q&A |
| `app/graph/graph.py` | LangGraph `StateGraph` — wires all nodes and conditional edges |
| `app/rag/retriever.py` | Dual retrieval (BM25 + vector) + RRF merge |
| `app/rag/reranker.py` | Cross-encoder re-ranking (Cohere API or local bge-reranker-base) |
| `app/rag/indexer.py` | Document ingestion: chunks, embeds, indexes BM25 + vector |
| `app/main.py` | FastAPI app with `/chat` SSE streaming endpoint |
| `tests/test_config.py` | Config loading unit tests |
| `tests/test_profile.py` | UserProfile unit tests |
| `tests/test_working_memory.py` | Working memory sliding window tests |
| `tests/test_retriever.py` | RRF merge logic unit tests |
| `tests/test_episodic.py` | Episodic compression trigger tests |
| `tests/test_graph.py` | End-to-end graph integration tests (mocked LLM) |

### Files to modify
| File | Change |
|---|---|
| `../pyproject.toml` | Add: langgraph, litellm, langfuse, rank-bm25, sentence-transformers, cohere, pyyaml, fastapi, uvicorn, sse-starlette, fakeredis (dev) |

### Files NOT to touch (old code preserved)
- `app/core/agent/` — old LangChain agents stay in place until the graph is validated
- `app/db/redis_client.py` — kept; new memory modules will use it directly
- `app/schemas/` — kept as-is; old CatoState referenced only by old agents

---

## Phase 1: Foundation

### Task 1: Add dependencies to pyproject.toml

**Why first:** Everything else imports from these packages. If they aren't installed, no test can run.

**Files:**
- Modify: `../pyproject.toml` (at `c:\Users\sudar\OneDrive\Desktop\ai-consolidated\pyproject.toml`)

- [ ] **Step 1: Open pyproject.toml and add the new dependencies**

Add to the `dependencies` list:
```toml
"langgraph>=0.2.0",
"litellm>=1.0.0",
"langfuse>=2.0.0",
"rank-bm25>=0.2.2",
"sentence-transformers>=3.0.0",
"cohere>=5.0.0",
"pyyaml>=6.0.0",
"fastapi>=0.115.0",
"uvicorn>=0.30.0",
"sse-starlette>=2.0.0",
"fakeredis>=2.20.0",
"pytest>=8.0.0",
"pytest-asyncio>=0.23.0",
```

- [ ] **Step 2: Install updated dependencies**

Run from the `ai-consolidated` root:
```bash
uv sync
```
Expected: All packages resolve and install successfully. No version conflicts.

- [ ] **Step 3: Verify langgraph and litellm are importable**

```bash
python -c "import langgraph; import litellm; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**
```bash
git add ../pyproject.toml ../uv.lock
git commit -m "feat: add langgraph, litellm, langfuse, hybrid-rag deps"
```

---

### Task 2: Create `config.yaml` and `app/core/config.py`

**Why:** Every subsequent component reads from `Settings`. The config module must exist and be tested before anything else is built.

**Files:**
- Create: `config.yaml`
- Modify: `app/core/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_config.py`:
```python
import pytest
from pathlib import Path

def test_config_loads_yaml(tmp_path, monkeypatch):
    """Settings should load values from a config.yaml file."""
    yaml_content = """
llm:
  model: "openai/gpt-4o"
  temperature: 0.7
  streaming: true
embeddings:
  model: "openai/text-embedding-3-large"
rag:
  retrieval_k: 10
  rerank_top_k: 3
  reranker: "local"
memory:
  working_window: 20
  summary_threshold: 16
  profile_ttl_days: 30
redis:
  url: "redis://localhost:6379"
langfuse:
  enabled: false
  public_key: "test-pub"
  secret_key: "test-sec"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    monkeypatch.chdir(tmp_path)

    # Import after chdir so it picks up the tmp config
    import importlib
    import app.core.config as config_module
    importlib.reload(config_module)
    s = config_module.settings

    assert s.llm.model == "openai/gpt-4o"
    assert s.llm.temperature == 0.7
    assert s.llm.streaming is True
    assert s.rag.retrieval_k == 10
    assert s.rag.reranker == "local"
    assert s.memory.working_window == 20
    assert s.redis.url == "redis://localhost:6379"
    assert s.langfuse.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**
```bash
cd cato-agent && pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError` or `AttributeError`

- [ ] **Step 3: Create `config.yaml`** at `cato-agent/config.yaml`:
```yaml
llm:
  model: "openai/gpt-4o"
  temperature: 0.7
  streaming: true

embeddings:
  model: "openai/text-embedding-3-large"

rag:
  retrieval_k: 10
  rerank_top_k: 3
  reranker: "local"   # "cohere" requires COHERE_API_KEY; "local" uses bge-reranker-base

memory:
  working_window: 20
  summary_threshold: 16
  profile_ttl_days: 30

redis:
  url: "redis://localhost:6379"

langfuse:
  enabled: false
  public_key: ""
  secret_key: ""
```

- [ ] **Step 4: Rewrite `app/core/config.py`**

Replace existing content entirely:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os


@dataclass
class LLMConfig:
    model: str
    temperature: float
    streaming: bool


@dataclass
class EmbeddingsConfig:
    model: str


@dataclass
class RAGConfig:
    retrieval_k: int
    rerank_top_k: int
    reranker: str  # "cohere" | "local"


@dataclass
class MemoryConfig:
    working_window: int
    summary_threshold: int
    profile_ttl_days: int


@dataclass
class RedisConfig:
    url: str


@dataclass
class LangfuseConfig:
    enabled: bool
    public_key: str
    secret_key: str


@dataclass
class Settings:
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    rag: RAGConfig
    memory: MemoryConfig
    redis: RedisConfig
    langfuse: LangfuseConfig


def _load_settings() -> Settings:
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if not config_path.exists():
        # Fallback: look in cwd (useful for tests using monkeypatch.chdir)
        config_path = Path.cwd() / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    def _env(val: str) -> str:
        """Expand ${ENV_VAR} references."""
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            return os.environ.get(val[2:-1], "")
        return val

    lf = raw.get("langfuse", {})
    return Settings(
        llm=LLMConfig(**raw["llm"]),
        embeddings=EmbeddingsConfig(**raw["embeddings"]),
        rag=RAGConfig(**raw["rag"]),
        memory=MemoryConfig(**raw["memory"]),
        redis=RedisConfig(**raw["redis"]),
        langfuse=LangfuseConfig(
            enabled=lf.get("enabled", False),
            public_key=_env(lf.get("public_key", "")),
            secret_key=_env(lf.get("secret_key", "")),
        ),
    )


settings = _load_settings()
```

- [ ] **Step 5: Run test to verify it passes**
```bash
pytest tests/test_config.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add config.yaml app/core/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add config.yaml + typed Settings loader"
```

---

### Task 3: Create `app/core/llm.py` (LiteLLM wrapper)

**Why:** All 5 agent nodes call this wrapper for LLM completions. It centralises Langfuse tracing and the model config. Build and test it before building any nodes.

**Files:**
- Create: `app/core/llm.py`
- No test file: LiteLLM calls are integration-level (require API keys). We test observable behaviour in the graph integration tests (Task 18). Skip unit test here.

- [ ] **Step 1: Create `app/core/llm.py`**
```python
from __future__ import annotations
from typing import Any
import litellm
from app.core.config import settings


def _get_callbacks() -> list:
    if not settings.langfuse.enabled:
        return []
    from langfuse.callback import CallbackHandler
    return [CallbackHandler(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
    )]


async def chat_completion(messages: list[dict], **kwargs: Any) -> str:
    """
    Async LiteLLM chat completion.
    Returns the assistant message content string.
    """
    response = await litellm.acompletion(
        model=settings.llm.model,
        messages=messages,
        temperature=kwargs.get("temperature", settings.llm.temperature),
        callbacks=_get_callbacks(),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )
    return response.choices[0].message.content


async def chat_completion_json(messages: list[dict], **kwargs: Any) -> str:
    """
    Same as chat_completion but requests JSON output (response_format).
    Used by classifier and qualifier nodes.
    """
    return await chat_completion(
        messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
```

- [ ] **Step 2: Verify import is clean**
```bash
python -c "from app.core.llm import chat_completion; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/core/llm.py
git commit -m "feat: add LiteLLM async wrapper with Langfuse callback support"
```

---

### Task 4: Create `app/core/embeddings.py`

**Why:** The RAG retriever and indexer both need embeddings. Build once, used everywhere.

**Files:**
- Create: `app/core/embeddings.py`

- [ ] **Step 1: Create `app/core/embeddings.py`**
```python
from __future__ import annotations
from typing import Sequence
import litellm
from app.core.config import settings


async def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured embeddings model."""
    response = await litellm.aembedding(
        model=settings.embeddings.model,
        input=list(texts),
    )
    return [item["embedding"] for item in response.data]


async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([text])
    return results[0]
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.core.embeddings import embed_query; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/core/embeddings.py
git commit -m "feat: add LiteLLM embeddings wrapper"
```

---

## Phase 2: Memory System

### Task 5: Create `app/memory/profile.py` (UserProfile)

**Why:** `UserProfile` is read by the qualifier node every single turn to decide what to ask next. It must exist before we build the state or qualifier.

**Files:**
- Create: `app/memory/__init__.py`
- Create: `app/memory/profile.py`
- Create: `tests/test_profile.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_profile.py`:
```python
import pytest
from app.memory.profile import UserProfile


def test_equity_pct_computed_correctly():
    profile = UserProfile(estimated_value=400_000, mortgage_balance=300_000)
    assert profile.equity_pct == pytest.approx(0.25)


def test_equity_pct_none_when_missing_value():
    profile = UserProfile(mortgage_balance=300_000)
    assert profile.equity_pct is None


def test_equity_pct_none_when_missing_balance():
    profile = UserProfile(estimated_value=400_000)
    assert profile.equity_pct is None


def test_profile_all_none_by_default():
    profile = UserProfile()
    assert profile.name is None
    assert profile.fico_score is None
    assert profile.has_bankruptcy is None
    assert profile.equity_pct is None
```

- [ ] **Step 2: Run test to verify it fails**
```bash
pytest tests/test_profile.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `app/memory/__init__.py`** (empty) and `app/memory/profile.py`**:
```python
from __future__ import annotations
from pydantic import BaseModel, computed_field


class UserProfile(BaseModel):
    name: str | None = None
    property_address: str | None = None
    property_type: str | None = None       # "SFR" | "condo" | "multi-family"
    estimated_value: float | None = None
    mortgage_balance: float | None = None
    fico_score: int | None = None
    has_bankruptcy: bool | None = None

    @computed_field
    @property
    def equity_pct(self) -> float | None:
        if self.estimated_value and self.mortgage_balance:
            return (self.estimated_value - self.mortgage_balance) / self.estimated_value
        return None
```

- [ ] **Step 4: Run test to verify it passes**
```bash
pytest tests/test_profile.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**
```bash
git add app/memory/__init__.py app/memory/profile.py tests/test_profile.py
git commit -m "feat: add UserProfile semantic memory model"
```

---

### Task 6: Create `app/memory/working.py` (Working Memory)

**Why:** Working memory provides the raw message window injected into every LLM call. It's the simplest memory tier — build and test it before episodic (which depends on it).

**Files:**
- Create: `app/memory/working.py`
- Note: Tests use `fakeredis` to avoid needing a live Redis instance.

- [ ] **Step 1: Write the failing test**

Create `tests/test_working_memory.py`:
```python
import pytest
import fakeredis.aioredis as fakeredis
from unittest.mock import AsyncMock, patch
from app.memory.working import WorkingMemory


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.mark.asyncio
async def test_add_and_get_messages(fake_redis):
    mem = WorkingMemory(redis=fake_redis, window=3)
    await mem.add_message("session1", "human", "Hello")
    await mem.add_message("session1", "assistant", "Hi there")
    messages = await mem.get_messages("session1")
    assert len(messages) == 2
    assert messages[0] == {"role": "human", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi there"}


@pytest.mark.asyncio
async def test_window_enforced(fake_redis):
    mem = WorkingMemory(redis=fake_redis, window=2)
    for i in range(4):
        await mem.add_message("session2", "human", f"msg{i}")
    messages = await mem.get_messages("session2")
    assert len(messages) == 2
    assert messages[0]["content"] == "msg2"
    assert messages[1]["content"] == "msg3"


@pytest.mark.asyncio
async def test_message_count(fake_redis):
    mem = WorkingMemory(redis=fake_redis, window=20)
    for i in range(5):
        await mem.add_message("session3", "human", f"msg{i}")
    assert await mem.count("session3") == 5
```

- [ ] **Step 2: Run test to verify it fails**
```bash
pytest tests/test_working_memory.py -v
```
Expected: FAIL

- [ ] **Step 3: Create `app/memory/working.py`**
```python
from __future__ import annotations
import json
from redis.asyncio import Redis
from app.core.config import settings


class WorkingMemory:
    def __init__(self, redis: Redis, window: int | None = None):
        self._redis = redis
        self._window = window or settings.memory.working_window

    def _key(self, session_id: str) -> str:
        return f"cato:working:{session_id}"

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        await self._redis.rpush(key, json.dumps({"role": role, "content": content}))
        # Trim to window from the right (keep newest messages)
        await self._redis.ltrim(key, -self._window, -1)

    async def get_messages(self, session_id: str) -> list[dict]:
        key = self._key(session_id)
        raw = await self._redis.lrange(key, 0, -1)
        return [json.loads(m) for m in raw]

    async def count(self, session_id: str) -> int:
        return await self._redis.llen(self._key(session_id))

    async def trim_oldest(self, session_id: str, n: int) -> list[dict]:
        """Remove and return the oldest n messages. Used by episodic memory."""
        key = self._key(session_id)
        pipe = self._redis.pipeline()
        for _ in range(n):
            pipe.lpop(key)
        results = await pipe.execute()
        return [json.loads(r) for r in results if r is not None]
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
pytest tests/test_working_memory.py -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**
```bash
git add app/memory/working.py tests/test_working_memory.py
git commit -m "feat: add Redis-backed working memory with sliding window"
```

---

### Task 7: Create `app/memory/episodic.py` (Rolling Summary)

**Why:** Episodic memory keeps token usage bounded for long conversations. The logic — "when count >= threshold, compress oldest half" — needs careful testing to get right.

**Files:**
- Create: `app/memory/episodic.py`
- Create: `tests/test_episodic.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_episodic.py`:
```python
import pytest
import fakeredis.aioredis as fakeredis
from unittest.mock import AsyncMock, patch
from app.memory.episodic import EpisodicMemory
from app.memory.working import WorkingMemory


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.mark.asyncio
async def test_no_compression_below_threshold(fake_redis):
    working = WorkingMemory(redis=fake_redis, window=20)
    episodic = EpisodicMemory(redis=fake_redis, threshold=4)
    for i in range(3):
        await working.add_message("s1", "human", f"msg{i}")
    # Should not trigger compression
    with patch("app.memory.episodic.chat_completion", new_callable=AsyncMock) as mock_llm:
        await episodic.maybe_compress("s1", working)
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_compression_triggered_at_threshold(fake_redis):
    working = WorkingMemory(redis=fake_redis, window=20)
    episodic = EpisodicMemory(redis=fake_redis, threshold=4)
    for i in range(4):
        await working.add_message("s1", "human", f"msg{i}")
    with patch("app.memory.episodic.chat_completion", new_callable=AsyncMock, return_value="Summary text") as mock_llm:
        await episodic.maybe_compress("s1", working)
        mock_llm.assert_called_once()
    # After compression: half of 4 messages trimmed (2), 2 remain
    assert await working.count("s1") == 2


@pytest.mark.asyncio
async def test_summary_stored_and_retrieved(fake_redis):
    working = WorkingMemory(redis=fake_redis, window=20)
    episodic = EpisodicMemory(redis=fake_redis, threshold=4)
    for i in range(4):
        await working.add_message("s1", "human", f"msg{i}")
    with patch("app.memory.episodic.chat_completion", new_callable=AsyncMock, return_value="The user asked 4 things."):
        await episodic.maybe_compress("s1", working)
    summary = await episodic.get_summary("s1")
    assert summary == "The user asked 4 things."


@pytest.mark.asyncio
async def test_empty_summary_when_no_compression(fake_redis):
    episodic = EpisodicMemory(redis=fake_redis, threshold=4)
    assert await episodic.get_summary("new_session") == ""
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_episodic.py -v
```
Expected: FAIL

- [ ] **Step 3: Create `app/memory/episodic.py`**
```python
from __future__ import annotations
import redis.asyncio as redis_module
from app.core.config import settings
from app.core.llm import chat_completion


SUMMARY_PROMPT = """You are summarizing an AI sales conversation for context continuity.
Summarize the following messages in 3-5 concise sentences. Focus on:
- What the user told us about their property and finances
- Any objections or concerns they raised
- The current qualification status

Messages to summarize:
{messages}

Summary:"""


class EpisodicMemory:
    def __init__(self, redis: redis_module.Redis, threshold: int | None = None):
        self._redis = redis
        self._threshold = threshold or settings.memory.summary_threshold

    def _key(self, session_id: str) -> str:
        return f"cato:summary:{session_id}"

    async def get_summary(self, session_id: str) -> str:
        val = await self._redis.get(self._key(session_id))
        return val.decode() if val else ""

    async def maybe_compress(self, session_id: str, working) -> None:
        """If working memory is at or above threshold, compress the oldest half."""
        count = await working.count(session_id)
        if count < self._threshold:
            return

        # Compress oldest half
        n_to_compress = count // 2
        old_messages = await working.trim_oldest(session_id, n_to_compress)

        formatted = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old_messages
        )
        # Prepend existing summary if one exists
        existing = await self.get_summary(session_id)
        if existing:
            formatted = f"[Prior summary]: {existing}\n\n[New messages]:\n{formatted}"

        new_summary = await chat_completion([
            {"role": "user", "content": SUMMARY_PROMPT.format(messages=formatted)}
        ], temperature=0.3)

        ttl = settings.memory.profile_ttl_days * 86400
        await self._redis.set(self._key(session_id), new_summary, ex=ttl)
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
pytest tests/test_episodic.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**
```bash
git add app/memory/episodic.py tests/test_episodic.py
git commit -m "feat: add episodic memory with rolling LLM compression"
```

---

## Phase 3: Hybrid RAG Pipeline

### Task 8: Create `app/rag/retriever.py` (BM25 + Vector + RRF)

**Why:** Objection and info nodes both call the retriever. The RRF merge function has non-trivial logic that must be unit tested.

**Files:**
- Create: `app/rag/__init__.py`
- Create: `app/rag/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retriever.py`:
```python
from langchain_core.documents import Document
from app.rag.retriever import reciprocal_rank_fusion


def _doc(id: str, content: str = "") -> Document:
    d = Document(page_content=content)
    d.id = id
    return d


def test_rrf_preserves_all_unique_docs():
    list_a = [_doc("a"), _doc("b"), _doc("c")]
    list_b = [_doc("b"), _doc("d")]
    result = reciprocal_rank_fusion([list_a, list_b])
    ids = [d.id for d in result]
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_ranks_common_doc_higher():
    list_a = [_doc("x"), _doc("shared"), _doc("y")]
    list_b = [_doc("z"), _doc("shared")]
    result = reciprocal_rank_fusion([list_a, list_b])
    ids = [d.id for d in result]
    assert ids.index("shared") < ids.index("x")


def test_rrf_caps_output_at_15():
    big_list = [_doc(str(i)) for i in range(20)]
    result = reciprocal_rank_fusion([big_list])
    assert len(result) == 15


def test_rrf_single_list_preserves_order():
    docs = [_doc(str(i)) for i in range(5)]
    result = reciprocal_rank_fusion([docs])
    assert [d.id for d in result] == [d.id for d in docs]
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_retriever.py -v
```
Expected: FAIL

- [ ] **Step 3: Create `app/rag/__init__.py`** (empty) and **`app/rag/retriever.py`**:
```python
from __future__ import annotations
from collections import defaultdict
from langchain_core.documents import Document
from langchain_redis import RedisVectorStore
from rank_bm25 import BM25Okapi
from app.core.config import settings
from app.core.embeddings import embed_query


def reciprocal_rank_fusion(
    result_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    """Merge multiple ranked document lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            scores[doc.id] += 1 / (k + rank + 1)
            doc_map[doc.id] = doc

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:15]]


class HybridRetriever:
    """
    Retrieves documents using BM25 keyword search and vector similarity in parallel,
    then merges results with RRF.

    BM25 index is loaded once from Redis on first use (lazy init).
    Vector search uses the existing RedisVectorStore.
    """

    def __init__(self, embeddings, bm25_corpus: list[Document]):
        self._embeddings = embeddings
        # Build BM25 index from tokenized corpus
        tokenized = [doc.page_content.lower().split() for doc in bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus = bm25_corpus
        self._vector_store = RedisVectorStore(
            embeddings,
            redis_url=settings.redis.url,
            index_name="cato_hei_index",
            schema=None,
        )

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._corpus[i] for i in top_indices]

    async def retrieve(self, query: str) -> list[Document]:
        k = settings.rag.retrieval_k
        bm25_results = self._bm25_search(query, k)
        vector_results = await self._vector_store.asimilarity_search(query, k=k)
        return reciprocal_rank_fusion([bm25_results, vector_results])
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
pytest tests/test_retriever.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**
```bash
git add app/rag/__init__.py app/rag/retriever.py tests/test_retriever.py
git commit -m "feat: add hybrid BM25+vector retriever with RRF merge"
```

---

### Task 9: Create `app/rag/reranker.py` (Cross-Encoder Re-Ranking)

**Why:** The reranker is the final stage of the RAG pipeline. It's config-driven (Cohere API vs. local model). No unit test — both backends require external dependencies (network or model download). Verified via the end-to-end graph tests.

**Files:**
- Create: `app/rag/reranker.py`

- [ ] **Step 1: Create `app/rag/reranker.py`**
```python
from __future__ import annotations
from langchain_core.documents import Document
from app.core.config import settings


async def rerank(query: str, docs: list[Document]) -> list[Document]:
    """
    Re-rank documents using the configured reranker.
    Returns top rerank_top_k documents.
    """
    top_k = settings.rag.rerank_top_k
    if settings.rag.reranker == "cohere":
        return await _cohere_rerank(query, docs, top_k)
    return _local_rerank(query, docs, top_k)


async def _cohere_rerank(
    query: str, docs: list[Document], top_k: int
) -> list[Document]:
    import cohere
    import os
    co = cohere.AsyncClient(api_key=os.environ["COHERE_API_KEY"])
    results = await co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[d.page_content for d in docs],
        top_n=top_k,
    )
    return [docs[r.index] for r in results.results]


def _local_rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-base")  # as specified in the design doc
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.rag.reranker import rerank; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/rag/reranker.py
git commit -m "feat: add cross-encoder reranker (Cohere API + local fallback)"
```

---

### Task 10: Create `app/rag/indexer.py` (Document Ingestion)

**Why:** The BM25 corpus in `HybridRetriever` must be built from indexed documents. The indexer loads documents from files and stores them in Redis.

**Files:**
- Create: `app/rag/indexer.py`

- [ ] **Step 1: Create `app/rag/indexer.py`**
```python
from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_redis import RedisVectorStore
from app.core.config import settings
import json


async def index_documents(source_path: Path, embeddings) -> list[Document]:
    """
    Load documents from a JSON file, chunk them, embed them, and store in Redis.
    Also returns the full document list for BM25 index construction.

    Expected JSON format: [{"content": "...", "metadata": {...}}, ...]
    """
    raw = json.loads(source_path.read_text())
    docs = [
        Document(page_content=item["content"], metadata=item.get("metadata", {}))
        for item in raw
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)

    # Assign stable IDs based on content hash
    for i, chunk in enumerate(chunks):
        chunk.id = f"chunk_{i}_{hash(chunk.page_content) & 0xFFFFFF:06x}"

    store = RedisVectorStore(
        embeddings,
        redis_url=settings.redis.url,
        index_name="cato_hei_index",
        schema=None,
    )
    await store.aadd_documents(chunks)
    return chunks
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.rag.indexer import index_documents; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/rag/indexer.py
git commit -m "feat: add document indexer for hybrid RAG ingestion pipeline"
```

---

## Phase 4: LangGraph State Machine

### Task 11: Create `app/graph/state.py` (CatoState)

**Why:** `CatoState` is the single contract all nodes depend on. Define it first. No test — it's a TypedDict with no logic to test.

**Files:**
- Create: `app/graph/__init__.py`
- Create: `app/graph/state.py`

- [ ] **Step 1: Create `app/graph/__init__.py`** (empty) and **`app/graph/state.py`**:
```python
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from app.memory.profile import UserProfile


class CatoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intent: str                           # set by classify_intent; "" until first turn
    user_profile: UserProfile             # persisted across sessions via Redis
    qualification_result: str | None      # "qualified" | "unqualified" | "pending" | None
    conversation_summary: str             # rolling episodic summary; "" until first compression
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.graph.state import CatoState; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/graph/__init__.py app/graph/state.py
git commit -m "feat: add CatoState TypedDict for LangGraph"
```

---

### Task 12: Create `app/graph/nodes/classifier.py`

**Why:** The classifier runs first on every turn. All routing depends on the `intent` it writes into state.

**Files:**
- Create: `app/graph/nodes/__init__.py`
- Create: `app/graph/nodes/classifier.py`

- [ ] **Step 1: Create `app/graph/nodes/__init__.py`** (empty) and **`app/graph/nodes/classifier.py`**:
```python
from __future__ import annotations
import json
from app.core.llm import chat_completion_json
from app.graph.state import CatoState

CLASSIFIER_PROMPT = """\
Classify the user's message into exactly one of these intents:
- "objection": User expresses skepticism, concern, distrust, or a scam worry.
- "qualify": User provides or asks about financial data (FICO, home value, equity, debt, bankruptcy).
- "book": User wants to speak with someone or schedule a call.
- "info": User asks how the program works, about fees, eligibility, or general product questions.
- "general": Greetings, pleasantries, or off-topic messages.

Also extract any named entities if present:
- "name": the user's first name if mentioned
- "address": property address if mentioned

Respond ONLY with valid JSON:
{"intent": "<one of the 5 values>", "name": "<string or null>", "address": "<string or null>"}

User message: "{message}"
"""


async def classify_intent(state: CatoState) -> dict:
    """
    Reads the last human message, classifies intent, extracts entities.
    Returns partial state update: intent, and optionally name/address in user_profile.
    """
    last_message = state["messages"][-1].content
    raw = await chat_completion_json([
        {"role": "user", "content": CLASSIFIER_PROMPT.format(message=last_message)}
    ], temperature=0)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"intent": "general", "name": None, "address": None}

    intent = parsed.get("intent", "general").lower()

    # Merge any extracted entities into the existing UserProfile
    profile = state["user_profile"]
    if parsed.get("name") and profile.name is None:
        profile = profile.model_copy(update={"name": parsed["name"]})
    if parsed.get("address") and profile.property_address is None:
        profile = profile.model_copy(update={"property_address": parsed["address"]})

    return {"intent": intent, "user_profile": profile}
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.graph.nodes.classifier import classify_intent; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/graph/nodes/__init__.py app/graph/nodes/classifier.py
git commit -m "feat: add classify_intent graph node"
```

---

### Task 13: Create `app/graph/nodes/qualifier.py`

**Why:** Most complex node. Implements the internal CoT reasoning pass defined in the spec. Must read `user_profile`, run structured JSON reasoning, update profile, and set `qualification_result`.

**Files:**
- Create: `app/graph/nodes/qualifier.py`

- [ ] **Step 1: Create `app/graph/nodes/qualifier.py`**
```python
from __future__ import annotations
import json
from app.core.llm import chat_completion_json
from app.graph.state import CatoState
from app.memory.profile import UserProfile

QUALIFIER_SYSTEM = """\
You are the Cato Qualifier. Your ONLY job is to collect the four qualification fields and make a decision.

Qualification criteria:
- FICO score >= 620
- Equity percentage >= 25% (equity = (value - mortgage) / value)
- Property type is eligible: SFR, condo, or qualifying multi-family
- No active bankruptcy

Current UserProfile:
{profile}

Conversation summary (prior turns):
{summary}

Think step by step:
1. Which fields are still missing (null)?
2. Is there enough information to make a final QUALIFIED or UNQUALIFIED decision?
3. If yes — does the user meet ALL four criteria?
4. If no — what single question should Cato ask next?

Respond ONLY with valid JSON:
{{
  "status": "qualified" | "unqualified" | "pending",
  "decision": "<brief rationale, 1 sentence>",
  "next_question": "<question if pending, else null>",
  "message_to_user": "<the actual message Cato should send to the user>",
  "reasoning": "<internal step-by-step, never shown to user>",
  "extracted": {{
    "name": "<string or null>",
    "fico_score": "<integer or null>",
    "estimated_value": "<float or null>",
    "mortgage_balance": "<float or null>",
    "property_type": "<SFR|condo|multi-family or null>",
    "property_address": "<string or null>",
    "has_bankruptcy": "<true|false or null>"
  }}
}}
"""


async def qualify(state: CatoState) -> dict:
    profile = state["user_profile"]
    summary = state.get("conversation_summary", "")
    messages = state["messages"]

    system_msg = QUALIFIER_SYSTEM.format(
        profile=profile.model_dump_json(indent=2),
        summary=summary or "None",
    )
    # Build message list: system context + full conversation
    llm_messages = [{"role": "system", "content": system_msg}]
    for m in messages:
        role = "user" if m.type == "human" else "assistant"
        llm_messages.append({"role": role, "content": m.content})

    raw = await chat_completion_json(llm_messages, temperature=0)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "status": "pending",
            "message_to_user": "Could you tell me your FICO score and home value?",
            "extracted": {},
        }

    status = parsed.get("status", "pending")
    message = parsed.get("message_to_user", "")

    # Merge newly extracted profile fields (only overwrite None fields)
    extracted = parsed.get("extracted", {}) or {}
    profile_updates = {
        k: v for k, v in extracted.items()
        if v is not None and getattr(profile, k, None) is None
    }
    if profile_updates:
        profile = profile.model_copy(update=profile_updates)

    from langchain_core.messages import AIMessage
    return {
        "qualification_result": status,
        "user_profile": profile,
        "messages": [AIMessage(content=message)],
    }
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.graph.nodes.qualifier import qualify; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/graph/nodes/qualifier.py
git commit -m "feat: add qualify graph node with internal CoT reasoning"
```

---

### Task 14: Create `app/graph/nodes/objection.py`

**Why:** Handles user trust concerns using RAG-retrieved rebuttal examples, replacing the old `objection_handler.py`.

**Files:**
- Create: `app/graph/nodes/objection.py`

- [ ] **Step 1: Create `app/graph/nodes/objection.py`**
```python
from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

OBJECTION_SYSTEM = """\
You are Cato, a specialist at Home.LLC. Your tone is empathetic, professional, and disarming.

Conversation rules:
- Keep your response to 2-3 sentences max.
- No bullet points or long paragraphs.
- Answer the specific concern, then end with a soft follow-up question.
- You are a helpful peer, not a technical manual.

Relevant knowledge base content:
{context}
"""


async def handle_objection(state: CatoState, retriever=None) -> dict:
    """
    Retrieves rebuttal content from the knowledge base and generates an empathetic response.
    `retriever` is injected at graph construction time.
    """
    last_message = state["messages"][-1].content
    summary = state.get("conversation_summary", "")

    context = ""
    if retriever:
        from app.rag.reranker import rerank
        candidates = await retriever.retrieve(last_message)
        top_docs = await rerank(last_message, candidates)
        context = "\n\n".join(d.page_content for d in top_docs)

    system = OBJECTION_SYSTEM.format(context=context or "No specific context retrieved.")
    messages_payload = [{"role": "system", "content": system}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.3)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.graph.nodes.objection import handle_objection; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/graph/nodes/objection.py
git commit -m "feat: add handle_objection graph node with RAG context"
```

---

### Task 15: Create `app/graph/nodes/booking.py`

**Files:**
- Create: `app/graph/nodes/booking.py`

- [ ] **Step 1: Create `app/graph/nodes/booking.py`**
```python
from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

BOOKING_SYSTEM = """\
You are Cato's Booking Specialist at Home.LLC.
The user is qualified (or requesting) a call with a Senior Advisor.

Guidelines:
- Be enthusiastic but professional.
- Keep responses concise: 2-3 sentences.
- If they ask for a scheduling link: https://calendly.com/home-llc/specialist
- If they provide a specific time, confirm it warmly and close.
"""


async def book_appointment(state: CatoState) -> dict:
    summary = state.get("conversation_summary", "")
    messages_payload = [{"role": "system", "content": BOOKING_SYSTEM}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.2)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.graph.nodes.booking import book_appointment; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/graph/nodes/booking.py
git commit -m "feat: add book_appointment graph node"
```

---

### Task 16: Create `app/graph/nodes/info.py`

**Files:**
- Create: `app/graph/nodes/info.py`

- [ ] **Step 1: Create `app/graph/nodes/info.py`**
```python
from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

INFO_SYSTEM = """\
You are Cato, an expert on Home.LLC's Home Equity Investment (HEI) product.
Answer questions clearly and concisely. Keep responses to 3-4 sentences.
End with a gentle qualifier follow-up if appropriate.

Relevant product knowledge:
{context}
"""


async def handle_info(state: CatoState, retriever=None) -> dict:
    last_message = state["messages"][-1].content
    summary = state.get("conversation_summary", "")

    context = ""
    if retriever:
        from app.rag.reranker import rerank
        candidates = await retriever.retrieve(last_message)
        top_docs = await rerank(last_message, candidates)
        context = "\n\n".join(d.page_content for d in top_docs)

    system = INFO_SYSTEM.format(context=context or "No specific context retrieved.")
    messages_payload = [{"role": "system", "content": system}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.5)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.graph.nodes.info import handle_info; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/graph/nodes/info.py
git commit -m "feat: add handle_info graph node with RAG context"
```

---

### Task 17: Create `app/graph/graph.py` (LangGraph StateGraph)

**Why:** This wires all nodes together with conditional edges, replacing `master.py`. This is the core of the modernization.

**Files:**
- Create: `app/graph/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_graph.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage
from app.graph.state import CatoState
from app.memory.profile import UserProfile


@pytest.fixture
def base_state() -> CatoState:
    return CatoState(
        messages=[HumanMessage(content="Hello")],
        session_id="test-session",
        intent="",
        user_profile=UserProfile(),
        qualification_result=None,
        conversation_summary="",
    )


@pytest.mark.asyncio
async def test_general_intent_routes_to_info(base_state):
    """A general/greeting message should route to handle_info."""
    with patch("app.graph.nodes.classifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value='{"intent": "general", "name": null, "address": null}'), \
         patch("app.graph.nodes.info.chat_completion",
               new_callable=AsyncMock,
               return_value="Hi! I'm Cato from Home.LLC..."):
        from app.graph.graph import build_graph
        graph = build_graph()
        result = await graph.ainvoke(base_state)
    assert result["intent"] == "general"
    assert len(result["messages"]) > 1  # original + AI response


@pytest.mark.asyncio
async def test_objection_intent_routes_to_objection_handler(base_state):
    base_state["messages"] = [HumanMessage(content="This sounds like a scam")]
    with patch("app.graph.nodes.classifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value='{"intent": "objection", "name": null, "address": null}'), \
         patch("app.graph.nodes.objection.chat_completion",
               new_callable=AsyncMock,
               return_value="I understand your concern..."):
        from app.graph.graph import build_graph
        graph = build_graph()
        result = await graph.ainvoke(base_state)
    assert result["intent"] == "objection"


@pytest.mark.asyncio
async def test_qualify_qualified_routes_to_booking(base_state):
    """A qualified user should be routed to book_appointment."""
    base_state["messages"] = [HumanMessage(content="My FICO is 750, home worth 500k, owe 200k")]
    qualify_response = '{"status": "qualified", "decision": "All criteria met.", "next_question": null, "message_to_user": "Great news, you qualify!", "reasoning": "..."}'
    with patch("app.graph.nodes.classifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value='{"intent": "qualify", "name": null, "address": null}'), \
         patch("app.graph.nodes.qualifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value=qualify_response), \
         patch("app.graph.nodes.booking.chat_completion",
               new_callable=AsyncMock,
               return_value="Let's get you scheduled!"):
        from app.graph.graph import build_graph
        graph = build_graph()
        result = await graph.ainvoke(base_state)
    assert result["qualification_result"] == "qualified"
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_graph.py -v
```
Expected: FAIL — `ModuleNotFoundError: app.graph.graph`

- [ ] **Step 3: Create `app/graph/graph.py`**
```python
from __future__ import annotations
from functools import partial
from langgraph.graph import StateGraph, START, END
from app.graph.state import CatoState
from app.graph.nodes.classifier import classify_intent
from app.graph.nodes.qualifier import qualify
from app.graph.nodes.objection import handle_objection
from app.graph.nodes.booking import book_appointment
from app.graph.nodes.info import handle_info


def _route_after_classify(state: CatoState) -> str:
    intent = state.get("intent", "general")
    if intent == "objection":
        return "handle_objection"
    if intent in ("qualify",):
        return "qualify"
    if intent == "book":
        return "book_appointment"
    # "info" and "general" (greetings/pleasantries) both route to handle_info.
    # The info node's system prompt handles graceful fallback for greetings:
    # if no relevant context is retrieved, it introduces Cato naturally.
    return "handle_info"


def _route_after_qualify(state: CatoState) -> str:
    result = state.get("qualification_result")
    if result == "qualified":
        return "book_appointment"
    return END  # unqualified or pending → END (Cato asked follow-up or disqualified)


def build_graph(retriever=None) -> StateGraph:
    """
    Build and compile the CatoState graph.
    `retriever` is an optional HybridRetriever injected into objection/info nodes.
    """
    builder = StateGraph(CatoState)

    # Register nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("qualify", qualify)
    builder.add_node("handle_objection", partial(handle_objection, retriever=retriever))
    builder.add_node("book_appointment", book_appointment)
    builder.add_node("handle_info", partial(handle_info, retriever=retriever))

    # Entry point
    builder.add_edge(START, "classify_intent")

    # Routing after classify
    builder.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "handle_objection": "handle_objection",
            "qualify": "qualify",
            "book_appointment": "book_appointment",
            "handle_info": "handle_info",
        },
    )

    # Routing after qualify
    builder.add_conditional_edges(
        "qualify",
        _route_after_qualify,
        {
            "book_appointment": "book_appointment",
            END: END,
        },
    )

    # Terminal edges
    builder.add_edge("handle_objection", END)
    builder.add_edge("book_appointment", END)
    builder.add_edge("handle_info", END)

    return builder.compile()


# Module-level compiled graph (used by main.py)
cato_graph = build_graph()
```

- [ ] **Step 4: Run integration tests**
```bash
pytest tests/test_graph.py -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**
```bash
git add app/graph/graph.py tests/test_graph.py
git commit -m "feat: add LangGraph StateGraph replacing master.py if/else routing"
```

---

## Phase 5: API Layer

### Task 18: Create `app/core/embeddings.py` LangChain-compatible adapter

**Why:** `RedisVectorStore` from `langchain_redis` requires an object implementing the LangChain `Embeddings` interface (with `.embed_documents()` / `.embed_query()` methods), not raw async functions. This adapter wraps the LiteLLM functions so `HybridRetriever` can be constructed and the RAG pipeline actually runs. Without this, every retrieval call returns "No specific context retrieved."

**Files:**
- Modify: `app/core/embeddings.py`

- [ ] **Step 1: Add the `CatoEmbeddings` adapter class to `app/core/embeddings.py`**

Append to the existing file:
```python
import asyncio
from langchain_core.embeddings import Embeddings as LCEmbeddings


class CatoEmbeddings(LCEmbeddings):
    """LangChain-compatible Embeddings adapter backed by LiteLLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return asyncio.get_event_loop().run_until_complete(embed_texts(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.get_event_loop().run_until_complete(embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await embed_query(text)
```

- [ ] **Step 2: Verify import**
```bash
python -c "from app.core.embeddings import CatoEmbeddings; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add app/core/embeddings.py
git commit -m "feat: add LangChain-compatible CatoEmbeddings adapter for HybridRetriever"
```

---

### Task 19: Create `app/main.py` (FastAPI SSE)

**Why:** The final integration point — the FastAPI endpoint wraps the graph and streams responses token-by-token via SSE. Also responsible for loading working/episodic memory per turn.

**Files:**
- Create: `app/main.py`

- [ ] **Step 1: Create `app/main.py`**
```python
from __future__ import annotations
from pathlib import Path
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.core.embeddings import CatoEmbeddings
from app.graph.graph import build_graph
from app.graph.state import CatoState
from app.memory.profile import UserProfile
from app.memory.working import WorkingMemory
from app.memory.episodic import EpisodicMemory
from app.rag.retriever import HybridRetriever
from app.rag.indexer import index_documents

app = FastAPI(title="Cato Agent API")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.on_event("startup")
async def startup():
    app.state.redis = aioredis.from_url(settings.redis.url, decode_responses=True)
    # Build embeddings adapter and load the BM25 corpus from indexed documents
    embeddings = CatoEmbeddings()
    data_path = Path(__file__).parent.parent / "data" / "hei_knowledge.json"
    corpus = await index_documents(data_path, embeddings) if data_path.exists() else []
    retriever = HybridRetriever(embeddings=embeddings, bm25_corpus=corpus)
    # Compile a new graph instance with the wired retriever
    app.state.graph = build_graph(retriever=retriever)


@app.on_event("shutdown")
async def shutdown():
    await app.state.redis.aclose()


@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    r = req.app.state.redis
    graph = req.app.state.graph
    working = WorkingMemory(redis=r)
    episodic = EpisodicMemory(redis=r)

    # Load or initialise UserProfile from Redis
    profile_key = f"cato:profile:{request.session_id}"
    raw_profile = await r.get(profile_key)
    profile = UserProfile.model_validate_json(raw_profile) if raw_profile else UserProfile()

    # Load episodic summary
    summary = await episodic.get_summary(request.session_id)

    # Build working memory message list for the graph
    prior_messages = await working.get_messages(request.session_id)

    # Record the new human message
    await working.add_message(request.session_id, "human", request.message)
    await episodic.maybe_compress(request.session_id, working)

    # Reconstruct message history for the graph
    from langchain_core.messages import AIMessage
    history = []
    for m in prior_messages:
        if m["role"] == "human":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    initial_state = CatoState(
        messages=history + [HumanMessage(content=request.message)],
        session_id=request.session_id,
        intent="",
        user_profile=profile,
        qualification_result=None,
        conversation_summary=summary,
    )

    if settings.llm.streaming:
        async def event_stream():
            response_parts: list[str] = []
            final_state: dict = {}
            async for chunk in graph.astream(initial_state):
                # astream yields node output dicts; extract AI message content
                for node_output in chunk.values():
                    final_state.update(node_output)
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "content") and msg.content:
                            response_parts.append(msg.content)
                            yield f"data: {msg.content}\n\n"

            # Persist the complete AI response and updated profile after streaming
            full_response = "".join(response_parts)
            if full_response:
                await working.add_message(request.session_id, "assistant", full_response)
            ttl = settings.memory.profile_ttl_days * 86400
            updated_profile = final_state.get("user_profile", profile)
            await r.set(profile_key, updated_profile.model_dump_json(), ex=ttl)

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming fallback
    result = await graph.ainvoke(initial_state)
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    response_text = ai_messages[-1].content if ai_messages else ""

    # Save AI response and updated profile to Redis
    await working.add_message(request.session_id, "assistant", response_text)
    ttl = settings.memory.profile_ttl_days * 86400
    await r.set(profile_key, result["user_profile"].model_dump_json(), ex=ttl)

    return {"response": response_text, "session_id": request.session_id}
```

- [ ] **Step 2: Verify the app starts**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Expected: `Application startup complete.` (Redis must be running — use `docker-compose up redis`)

- [ ] **Step 3: Smoke test the endpoint**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-001", "message": "Hi, what is this?"}'
```
Expected: SSE stream of tokens ending with `data: [DONE]`

- [ ] **Step 4: Commit**
```bash
git add app/main.py
git commit -m "feat: add FastAPI SSE /chat endpoint wrapping LangGraph"
```

---

## Final Verification

- [ ] **Run the full test suite**
```bash
pytest tests/ -v
```
Expected: All tests PASS

- [ ] **Run the old demo script to confirm backward compatibility**
```bash
python scripts/cato_demo.py
```
The old `master.py` is untouched — this should still work.

- [ ] **Final commit**
```bash
git add .
git commit -m "feat: complete Cato Agent modernization — LangGraph + LiteLLM"
```
