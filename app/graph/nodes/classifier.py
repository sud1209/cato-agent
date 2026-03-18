from __future__ import annotations
import json
from app.core.llm import chat_completion_json
from app.graph.state import CatoState

CLASSIFIER_PROMPT = """\
Classify the user's message into exactly one of these intents:
- "objection": User expresses skepticism, concern, distrust, or a scam worry.
- "qualify": User provides or asks about financial data (FICO, home value, equity, debt, bankruptcy).
- "book": User wants to speak with someone or schedule a call.
- "info": User asks how the program works, about fees, eligibility, or general product questions.
- "general": Greetings, pleasantries, or off-topic messages.

Also extract any named entities if present:
- "name": the user's first name if mentioned
- "address": property address if mentioned

Respond ONLY with valid JSON:
{{"intent": "<one of the 5 values>", "name": "<string or null>", "address": "<string or null>"}}

User message: "{message}"
"""


async def classify_intent(state: CatoState) -> dict:
    """
    Reads the last human message, classifies intent, extracts entities.
    Returns partial state update: intent, and optionally name/address in user_profile.
    """
    last_message = state["messages"][-1].content
    raw = await chat_completion_json([
        {"role": "user", "content": CLASSIFIER_PROMPT.format(message=last_message)}
    ], temperature=0)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"intent": "general", "name": None, "address": None}

    intent = parsed.get("intent", "general").lower()

    profile = state["user_profile"]
    if parsed.get("name") and profile.name is None:
        profile = profile.model_copy(update={"name": parsed["name"]})
    if parsed.get("address") and profile.property_address is None:
        profile = profile.model_copy(update={"property_address": parsed["address"]})

    return {"intent": intent, "user_profile": profile}
