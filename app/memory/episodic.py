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

        n_to_compress = count // 2
        old_messages = await working.trim_oldest(session_id, n_to_compress)

        formatted = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old_messages
        )
        existing = await self.get_summary(session_id)
        if existing:
            formatted = f"[Prior summary]: {existing}\n\n[New messages]:\n{formatted}"

        new_summary = await chat_completion([
            {"role": "user", "content": SUMMARY_PROMPT.format(messages=formatted)}
        ], temperature=0.3)

        ttl = settings.memory.profile_ttl_days * 86400
        await self._redis.set(self._key(session_id), new_summary, ex=ttl)
