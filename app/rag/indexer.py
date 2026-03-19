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

    for i, chunk in enumerate(chunks):
        chunk.id = f"chunk_{i}_{hash(chunk.page_content) & 0xFFFFFF:06x}"

    store = RedisVectorStore(
        embeddings,
        redis_url=settings.redis.url,
        index_name="cato_hei_index",
    )
    await store.aadd_documents(chunks)
    return chunks
