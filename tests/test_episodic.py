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
