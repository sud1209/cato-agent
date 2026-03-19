from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # picks up OPENAI_API_KEY from .env
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.core.embeddings import CatoEmbeddings
from app.graph.graph import build_graph
from app.graph.state import CatoState
from app.memory.profile import UserProfile
from app.memory.working import WorkingMemory
from app.memory.episodic import EpisodicMemory
from app.rag.retriever import HybridRetriever
from app.rag.indexer import index_documents

app = FastAPI(title="Cato Agent API")

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
async def index():
    return FileResponse(_static / "index.html")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.on_event("startup")
async def startup():
    app.state.redis = aioredis.from_url(settings.redis.url, decode_responses=True)
    # Build embeddings adapter and load the BM25 corpus from indexed documents
    embeddings = CatoEmbeddings()
    data_path = Path(__file__).parent.parent / "data" / "hei_knowledge.json"
    corpus = await index_documents(data_path, embeddings) if data_path.exists() else []
    retriever = HybridRetriever(embeddings=embeddings, bm25_corpus=corpus)
    # Compile a new graph instance with the wired retriever
    app.state.graph = build_graph(retriever=retriever)


@app.on_event("shutdown")
async def shutdown():
    await app.state.redis.aclose()


@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    r = req.app.state.redis
    graph = req.app.state.graph
    working = WorkingMemory(redis=r)
    episodic = EpisodicMemory(redis=r)

    # Load or initialise UserProfile from Redis
    profile_key = f"cato:profile:{request.session_id}"
    raw_profile = await r.get(profile_key)
    profile = UserProfile.model_validate_json(raw_profile) if raw_profile else UserProfile()

    # Load episodic summary
    summary = await episodic.get_summary(request.session_id)

    # Build working memory message list for the graph
    prior_messages = await working.get_messages(request.session_id)

    # Record the new human message
    await working.add_message(request.session_id, "human", request.message)
    await episodic.maybe_compress(request.session_id, working)

    # Reconstruct message history for the graph
    from langchain_core.messages import AIMessage
    history = []
    for m in prior_messages:
        if m["role"] == "human":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    initial_state = CatoState(
        messages=history + [HumanMessage(content=request.message)],
        session_id=request.session_id,
        intent="",
        user_profile=profile,
        qualification_result=None,
        conversation_summary=summary,
    )

    if settings.llm.streaming:
        async def event_stream():
            response_parts: list[str] = []
            final_state: dict = {}
            async for chunk in graph.astream(initial_state):
                for node_output in chunk.values():
                    final_state.update(node_output)
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "content") and msg.content:
                            response_parts.append(msg.content)
                            yield f"data: {msg.content}\n\n"

            # Persist the complete AI response and updated profile after streaming
            full_response = "".join(response_parts)
            if full_response:
                await working.add_message(request.session_id, "assistant", full_response)
            ttl = settings.memory.profile_ttl_days * 86400
            updated_profile = final_state.get("user_profile", profile)
            await r.set(profile_key, updated_profile.model_dump_json(), ex=ttl)

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming fallback
    result = await graph.ainvoke(initial_state)
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    response_text = ai_messages[-1].content if ai_messages else ""

    # Save AI response and updated profile to Redis
    await working.add_message(request.session_id, "assistant", response_text)
    ttl = settings.memory.profile_ttl_days * 86400
    await r.set(profile_key, result["user_profile"].model_dump_json(), ex=ttl)

    return {"response": response_text, "session_id": request.session_id}
