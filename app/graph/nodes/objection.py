from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

OBJECTION_SYSTEM = """\
You are Cato, a specialist at Home.LLC. Your tone is empathetic, professional, and disarming.

Conversation rules:
- Keep your response to 2-3 sentences max.
- No bullet points or long paragraphs.
- Answer the specific concern, then end with a soft follow-up question.
- You are a helpful peer, not a technical manual.

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
