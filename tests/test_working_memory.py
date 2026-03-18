import pytest
import fakeredis.aioredis as fakeredis
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
