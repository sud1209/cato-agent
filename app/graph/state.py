from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from app.memory.profile import UserProfile


class CatoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intent: str                           # set by classify_intent; "" until first turn
    user_profile: UserProfile             # persisted across sessions via Redis
    qualification_result: str | None      # "qualified" | "unqualified" | "pending" | None
    conversation_summary: str             # rolling episodic summary; "" until first compression
