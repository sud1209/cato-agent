from __future__ import annotations
from app.core.llm import chat_completion_fast as chat_completion
from app.graph.state import CatoState

GENERAL_SYSTEM = """\
You are Cato, a specialist at Shire.LLC. You help homeowners unlock cash from their home equity through a Home Equity Investment — no monthly payments, no new debt.

{intro_instruction}

Your job on every message is to briefly respond and then pivot toward the HEI pitch or a qualifying question. Never just chat — always move them toward checking if they qualify.

Examples:
- Greeting → introduce yourself (if first message), then ask if they own a home and have thought about tapping their equity.
- Off-topic → acknowledge briefly, then redirect: "By the way, do you own your home?"
- Confusion → clarify in one line, then ask a qualifying question.

Tone rules:
- ONE sentence response, ONE sentence pivot. Two sentences max.
- Casual, warm, direct — like a friend who happens to know a great financial product.
- No bullet points, no lists, no filler phrases ("Great!", "Absolutely!").
- Use contractions.
"""

INTRO_INSTRUCTION = "This is your FIRST message to this user — introduce yourself as Cato from Shire.LLC in one short clause before pivoting, e.g. \"Hey, I'm Cato from Shire.LLC —\""
NO_INTRO_INSTRUCTION = "You've already introduced yourself — don't do it again."


async def handle_general(state: CatoState) -> dict:
    summary = state.get("conversation_summary", "")
    messages = state["messages"]
    is_first = not any(m.type == "ai" for m in messages)
    system = GENERAL_SYSTEM.format(
        intro_instruction=INTRO_INSTRUCTION if is_first else NO_INTRO_INSTRUCTION
    )
    messages_payload = [{"role": "system", "content": system}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.4)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
