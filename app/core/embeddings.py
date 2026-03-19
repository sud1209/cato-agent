from __future__ import annotations
from typing import Sequence
import litellm
from app.core.config import settings
from langchain_core.embeddings import Embeddings as LCEmbeddings


def _embed_texts_sync(texts: Sequence[str]) -> list[list[float]]:
    """Synchronous embedding via LiteLLM (safe inside a running event loop)."""
    response = litellm.embedding(
        model=settings.embeddings.model,
        input=list(texts),
    )
    return [item["embedding"] for item in response.data]


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


class CatoEmbeddings(LCEmbeddings):
    """LangChain-compatible Embeddings adapter backed by LiteLLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _embed_texts_sync(texts)

    def embed_query(self, text: str) -> list[float]:
        return _embed_texts_sync([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await embed_query(text)
