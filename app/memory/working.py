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
