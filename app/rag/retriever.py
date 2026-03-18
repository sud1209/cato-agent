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
    """

    def __init__(self, embeddings, bm25_corpus: list[Document]):
        self._embeddings = embeddings
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
