from __future__ import annotations
from app.core.llm import chat_completion
from app.graph.state import CatoState

INFO_SYSTEM = """\
You are Cato, an expert on Home.LLC's Home Equity Investment (HEI) product.
Answer questions clearly and concisely. Keep responses to 3-4 sentences.
End with a gentle qualifier follow-up if appropriate.

Relevant product knowledge:
{context}
"""


async def handle_info(state: CatoState, retriever=None) -> dict:
    last_message = state["messages"][-1].content
    summary = state.get("conversation_summary", "")

    context = ""
    if retriever:
        from app.rag.reranker import rerank
        candidates = await retriever.retrieve(last_message)
        top_docs = await rerank(last_message, candidates)
        context = "\n\n".join(d.page_content for d in top_docs)

    system = INFO_SYSTEM.format(context=context or "No specific context retrieved.")
    messages_payload = [{"role": "system", "content": system}]
    if summary:
        messages_payload.append({"role": "system", "content": f"Conversation so far: {summary}"})
    for m in state["messages"]:
        role = "user" if m.type == "human" else "assistant"
        messages_payload.append({"role": role, "content": m.content})

    response = await chat_completion(messages_payload, temperature=0.5)

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}
