from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

INFO_SYSTEM = """\
You are Cato, a specialist at Shire.LLC. Your goal is to move homeowners toward accessing their home equity through the HEI program.

What we already know about this user:
{profile_summary}

Answer the question in one sentence using the user's actual data above when available — never ask for information you already have.
Then ask a follow-up that builds on what they just said, either learning more about their situation or moving toward qualifying/booking.

Never just answer and stop — every response ends with a question.

Tone rules:
- TWO sentences max (answer + question).
- Knowledgeable friend texting, not a support bot.
- No bullet points, no lists, no headers.
- No filler phrases ("Great question!", "Absolutely!", "Of course!").
- Casual, direct, use contractions.

Relevant product knowledge:
{context}
"""


def _profile_summary(profile) -> str:
    lines = []
    if profile.name:
        lines.append(f"Name: {profile.name}")
    if profile.property_address:
        lines.append(f"Address: {profile.property_address}")
    if profile.estimated_value:
        lines.append(f"Home value: ${profile.estimated_value:,.0f}")
    if profile.mortgage_balance:
        lines.append(f"Mortgage balance: ${profile.mortgage_balance:,.0f}")
    if profile.equity_pct is not None:
        equity_dollars = profile.estimated_value * profile.equity_pct
        lines.append(f"Home equity: ${equity_dollars:,.0f} ({profile.equity_pct*100:.0f}%)")
    if profile.fico_score:
        lines.append(f"FICO score: {profile.fico_score}")
    if profile.property_type:
        lines.append(f"Property type: {profile.property_type}")
    if profile.has_bankruptcy is not None:
        lines.append(f"Bankruptcy: {profile.has_bankruptcy}")
    return "\n".join(lines) if lines else "Nothing collected yet."


async def handle_info(state: CatoState, retriever=None) -> dict:
    last_message = state["messages"][-1].content
    summary = state.get("conversation_summary", "")
    profile = state["user_profile"]

    context = ""
    if retriever:
        from app.rag.reranker import rerank
        candidates = await retriever.retrieve(last_message)
        top_docs = await rerank(last_message, candidates)
        context = "\n\n".join(d.page_content for d in top_docs)

    system = INFO_SYSTEM.format(
        profile_summary=_profile_summary(profile),
        context=context or "No specific context retrieved.",
    )
    messages_payload = [{"role": "system", "content": system}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.5)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
