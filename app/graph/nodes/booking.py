from __future__ import annotations
from app.core.llm import chat_completion_fast as chat_completion
from app.graph.state import CatoState

BOOKING_SYSTEM = """\
You are Cato at Shire.LLC, helping a homeowner book a call with an advisor.

Tone rules — this is non-negotiable:
- ONE sentence per response.
- Casual, warm, direct — like texting a friend.
- No filler phrases ("Awesome!", "Great!", "Absolutely!").
- No bullet points. No lists.

If they need a scheduling link: https://calendly.com/shirellc/advisor
If they give you a specific time, confirm it in one short sentence and close.
"""


async def book_appointment(state: CatoState) -> dict:
    summary = state.get("conversation_summary", "")
    messages_payload = [{"role": "system", "content": BOOKING_SYSTEM}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.2)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
