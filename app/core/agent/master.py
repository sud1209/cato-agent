import json
from app.core.agent.model_factory import get_model
from app.db.redis_client import redis_manager
from app.core.agent.qualifier import qualifier_agent
from app.core.agent.objection_handler import get_objection_response
from app.core.agent.unqualified_agent import handle_unqualified_user # New Import
from app.core.agent.booking_agent import handle_booking
from app.schemas.state import CatoState
from app.schemas.agent_responses import QualificationResponse
from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.core.config import settings
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

llm = get_model(temperature=0)

class IntentAnalysis(BaseModel):
    intent: str = Field(description="OBJECTION, QUALIFICATION, BOOKING, INFO, or GENERAL")
    user_name: Optional[str] = Field(None, description="The user's name if mentioned")
    address: Optional[str] = Field(None, description="The property address if mentioned")

async def run_master_agent(session_id: str, user_input: str):
    #print(f"DEBUG: Cato is talking to Redis at {settings.REDIS_URL} for session {session_id}")
    history = RedisChatMessageHistory(
            session_id=f"chat_{session_id}", 
            url=settings.REDIS_URL
        )
    memory_messages = history.messages[-6:] if history.messages else []

    # 1. Fetch current state from Redis
    raw_state = await redis_manager.get_cato_state(session_id)
    state_dict = raw_state if raw_state else {}
    state_dict["session_id"] = session_id
    state = CatoState(**state_dict)

    # 2. Intent Classification (The Router)
    classifier = llm.with_structured_output(IntentAnalysis)
    
    intent_prompt = f"""
    Classify the user intent and extract any entities:
    - 'GENERAL': Simple greetings (Hi, Hello), social pleasantries, or introductions.
    - 'OBJECTION': Skepticism, scam concerns, or 'why should I trust you?'.
    - 'QUALIFICATION': Providing financial data (FICO, Value, Debt).
    - 'BOOKING': Wants to talk to a person or schedule.
    - 'INFO': Questions about the product, fees, or "how it works".
    
    User message: "{user_input}"
    """
    
    analysis = await classifier.ainvoke(intent_prompt)
    intent = analysis.intent.strip().upper()

    state_updates = {}
    if analysis.user_name:
        state.user_name = analysis.user_name
        state_updates["user_name"] = analysis.user_name
    if analysis.address:
        state.address = analysis.address
        state_updates["address"] = analysis.address
    
    if state_updates:
        await redis_manager.update_cato_state(session_id, state_updates)

    # 3. Routing Logic
    
    # CASE A: User has concerns
    if "OBJECTION" in intent:
        response_text = await get_objection_response(user_input, chat_history=memory_messages)        
        history.add_user_message(user_input)
        history.add_ai_message(response_text)
        await redis_manager.update_cato_state(session_id, {"last_agent": "ObjectionHandler"})
        return response_text

    # CASE B: User is providing data for the HEI
    if "QUALIFICATION" in intent or state.address:
        print(f"Routing to QUALIFIER for {session_id}")
        full_messages = memory_messages + [("user", user_input)]
        # 1. Invoke the agent
        agent_result = await qualifier_agent.ainvoke({
            "messages": full_messages
            })
        
        # 2. Extract the structured response
        # In the new spec, the Pydantic object is in 'structured_response'
        data = agent_result["structured_response"]
        final_msg=""

        # 3. VERDICT CHECK: This is where we short-circuit
        if data.qual_status == "UNQUALIFIED":
            final_msg = await handle_unqualified_user(data.reason, state.model_dump())
            await redis_manager.update_cato_state(session_id, {
                "qual_status": "UNQUALIFIED",
                "last_agent": "UnqualifiedAgent"
            })
        else:
            final_msg = data.message_to_user
            await redis_manager.update_cato_state(session_id, {
                "qual_status": data.qual_status,
                "last_agent": "Qualifier"
            })

        history.add_user_message(user_input)
        history.add_ai_message(final_msg)
        return final_msg
    # CASE C: User is ready to book
    if "BOOKING" in intent or (state.qual_status == "QUALIFIED" and "YES" in user_input.upper()):
        print(f"Routing to BOOKING_AGENT for {session_id}")
        booking_msg = await handle_booking(user_input, state.model_dump())
        
        history.add_user_message(user_input)
        history.add_ai_message(booking_msg)

        await redis_manager.update_cato_state(session_id, {
            "booking_intent": "TRUE",
            "last_agent": "BookingAgent"
        })
        return booking_msg

    await redis_manager.update_cato_state(session_id, {
            "last_agent": "Greeter",
            "session_id": session_id
        })
    # CASE D: User wants to learn more
    if "INFO" in intent or "GENERAL" in intent:
        print(f"Routing to INFO_HANDLER for {session_id}")
        
        # We can leverage the Objection Handler because it pulls from your HEI knowledge base
        response_text = await get_objection_response(user_input, chat_history=memory_messages)
        
        # Save to history so she remembers explaining it
        history.add_user_message(user_input)
        history.add_ai_message(response_text)
        
        await redis_manager.update_cato_state(session_id, {"last_agent": "InfoProvider"})
        return response_text
    
    # CASE E: Fallback
    fallback_msg = "Hi! I'm Cato from Home.LLC. I help homeowners unlock their equity without the stress of monthly payments. Before we dive in, may I ask who I'm speaking with?"
    history.add_user_message(user_input)
    history.add_ai_message(fallback_msg)
    
    await redis_manager.update_cato_state(session_id, {"last_agent": "Greeter"})
    return fallback_msg