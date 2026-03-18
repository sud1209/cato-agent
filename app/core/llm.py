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
