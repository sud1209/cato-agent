import redis.asyncio as redis
import json
from redis.asyncio import ConnectionPool
from langchain_redis import RedisVectorStore, RedisChatMessageHistory
from app.core.config import settings

class RedisManager:
    def __init__(self):
        # Connection Pooling
        self.pool = ConnectionPool.from_url(
            settings.REDIS_URL, 
            decode_responses=True,
            max_connections=20
        )
        self.client = redis.Redis(connection_pool=self.pool)

    # Session & Flag Management
    async def get_cato_state(self, session_id: str) -> dict:
        """Fetches the conversation flags (booking_intent, etc.) safely using JSON"""
        data = await self.client.get(f"cato:state:{session_id}")
        if not data:
            return {}
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # Fallback for old records that might still be str(dict)
            return {}

    async def update_cato_state(self, session_id: str, updates: dict):
        """Updates logs and flags with a TTL for auto-cleanup"""
        current = await self.get_cato_state(session_id)
        current.update(updates)
        
        # Use json.dumps for safe storage
        await self.client.set(
            f"cato:state:{session_id}", 
            json.dumps(current), 
            ex=settings.DEFAULT_SESSION_TTL
        )

    def get_vector_store(self, embeddings):
        return RedisVectorStore(
            embeddings,
            redis_url=settings.REDIS_URL,
            index_name=settings.REDIS_VECTOR_INDEX,
            schema=None
        )

    async def close(self):
        await self.client.aclose()
        await self.pool.disconnect()

redis_manager = RedisManager()