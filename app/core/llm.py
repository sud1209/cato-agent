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
    """Async LiteLLM chat completion using the main model."""
    response = await litellm.acompletion(
        model=settings.llm.model,
        messages=messages,
        temperature=kwargs.get("temperature", settings.llm.temperature),
        callbacks=_get_callbacks(),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )
    return response.choices[0].message.content


async def chat_completion_fast(messages: list[dict], **kwargs: Any) -> str:
    """Async LiteLLM chat completion using the fast/cheap model (gpt-4o-mini)."""
    response = await litellm.acompletion(
        model=settings.llm.fast_model,
        messages=messages,
        temperature=kwargs.get("temperature", settings.llm.temperature),
        callbacks=_get_callbacks(),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )
    return response.choices[0].message.content


async def chat_completion_json(messages: list[dict], **kwargs: Any) -> str:
    """JSON-mode completion using the fast model (classifier, qualifier)."""
    return await chat_completion_fast(
        messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
