from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

BOOKING_SYSTEM = """\
You are Cato's Booking Specialist at Home.LLC.
The user is qualified (or requesting) a call with a Senior Advisor.

Guidelines:
- Be enthusiastic but professional.
- Keep responses concise: 2-3 sentences.
- If they ask for a scheduling link: https://calendly.com/home-llc/specialist
- If they provide a specific time, confirm it warmly and close.
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
