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
    return await _local_rerank(query, docs, top_k)


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


_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    return _cross_encoder


async def _local_rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    import asyncio
    model = _get_cross_encoder()
    pairs = [(query, d.page_content) for d in docs]
    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, model.predict, pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
