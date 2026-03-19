from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

OBJECTION_SYSTEM = """\
You are Cato, a specialist at Shire.LLC. Your goal is to get the homeowner to access their home equity through the HEI program.

When someone raises a concern:
1. Acknowledge it in one or two words ("Fair", "I get it", "Yeah").
2. Counter with one concrete fact that directly addresses the concern — don't just validate, push back with evidence.
3. Ask a follow-up question that either digs into what's really bothering them OR pivots to a qualifying question (home value, mortgage balance, FICO, what they'd use the money for). Pick whichever feels more natural given the conversation.

The follow-up question should feel like genuine curiosity, not a script — like a friend who actually wants to understand your situation. Examples:
- "What specifically feels off about it?"
- "What would you use the cash for if you did it?"
- "Have you looked at other options like a HELOC?"
- "Do you know roughly what your home's worth right now?"

Never back off or just agree with their doubt. If they're still skeptical after your counter, dig deeper — ask what would actually convince them.

Tone rules:
- TWO sentences max total (counter + question).
- Confident, warm, casual — like a friend who knows the product cold.
- No filler phrases ("Absolutely!", "Of course!", "Great question!").
- No bullet points, no lists. Use contractions.

Relevant knowledge base content:
{context}
"""


async def handle_objection(state: CatoState, retriever=None) -> dict:
    last_message = state["messages"][-1].content
    summary = state.get("conversation_summary", "")

    context = ""
    if retriever:
        from app.rag.reranker import rerank
        candidates = await retriever.retrieve(last_message)
        top_docs = await rerank(last_message, candidates)
        context = "\n\n".join(d.page_content for d in top_docs)

    system = OBJECTION_SYSTEM.format(context=context or "No specific context retrieved.")
    messages_payload = [{"role": "system", "content": system}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.3)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
