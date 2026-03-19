import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage
from app.graph.state import CatoState
from app.memory.profile import UserProfile


@pytest.fixture
def base_state() -> CatoState:
    return CatoState(
        messages=[HumanMessage(content="Hello")],
        session_id="test-session",
        intent="",
        user_profile=UserProfile(),
        qualification_result=None,
        conversation_summary="",
    )


@pytest.mark.asyncio
async def test_general_intent_routes_to_info(base_state):
    """A general/greeting message should route to handle_info."""
    with patch("app.graph.nodes.classifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value='{"intent": "general", "name": null, "address": null}'), \
         patch("app.graph.nodes.info.chat_completion",
               new_callable=AsyncMock,
               return_value="Hi! I'm Cato from Home.LLC..."):
        from app.graph.graph import build_graph
        graph = build_graph()
        result = await graph.ainvoke(base_state)
    assert result["intent"] == "general"
    assert len(result["messages"]) > 1


@pytest.mark.asyncio
async def test_objection_intent_routes_to_objection_handler(base_state):
    base_state["messages"] = [HumanMessage(content="This sounds like a scam")]
    with patch("app.graph.nodes.classifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value='{"intent": "objection", "name": null, "address": null}'), \
         patch("app.graph.nodes.objection.chat_completion",
               new_callable=AsyncMock,
               return_value="I understand your concern..."):
        from app.graph.graph import build_graph
        graph = build_graph()
        result = await graph.ainvoke(base_state)
    assert result["intent"] == "objection"


@pytest.mark.asyncio
async def test_qualify_qualified_routes_to_booking(base_state):
    """A qualified user should be routed to book_appointment."""
    base_state["messages"] = [HumanMessage(content="My FICO is 750, home worth 500k, owe 200k")]
    qualify_response = '{"status": "qualified", "decision": "All criteria met.", "next_question": null, "message_to_user": "Great news, you qualify!", "reasoning": "...", "extracted": {}}'
    with patch("app.graph.nodes.classifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value='{"intent": "qualify", "name": null, "address": null}'), \
         patch("app.graph.nodes.qualifier.chat_completion_json",
               new_callable=AsyncMock,
               return_value=qualify_response), \
         patch("app.graph.nodes.booking.chat_completion",
               new_callable=AsyncMock,
               return_value="Let's get you scheduled!"):
        from app.graph.graph import build_graph
        graph = build_graph()
        result = await graph.ainvoke(base_state)
    assert result["qualification_result"] == "qualified"
