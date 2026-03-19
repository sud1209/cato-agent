from __future__ import annotations
import json
from app.core.llm import chat_completion_json
from app.graph.state import CatoState

CLASSIFIER_PROMPT = """\
Classify the user's message into exactly one of these intents:

- "objection": User expresses ANY skepticism, doubt, distrust, hesitation, or concern about the program.
  Examples: "sounds too good to be true", "this feels fishy", "I'm not convinced", "what's the catch",
  "sounds like a scam", "I don't trust this", "seems risky", "I have doubts", "not sure about this".
  When in doubt between objection and info, choose objection.

- "qualify": User provides or asks about their own financial data (FICO score, home value, mortgage,
  equity, debt, bankruptcy, property type) OR asks to look up their home/property details.

- "book": User wants to speak with someone, schedule a call, or book an appointment.

- "info": User asks neutral questions about how the program works, fees, eligibility rules, or
  general product mechanics — with no skepticism.

- "general": Pure greetings, off-topic messages, or one-word filler ("ok", "sure", "thanks").

Also extract any named entities if present:
- "name": the user's full name if mentioned
- "address": property address if mentioned

Respond ONLY with valid JSON:
{{"intent": "<one of the 5 values>", "name": "<string or null>", "address": "<string or null>"}}

User message: "{message}"
"""


async def classify_intent(state: CatoState) -> dict:
    from app.db.property_lookup import lookup_property

    last_message = state["messages"][-1].content

    # Run classification and any pending DB lookup concurrently
    profile = state["user_profile"]
    new_name = None
    new_address = None

    import asyncio
    raw, _ = await asyncio.gather(
        chat_completion_json([
            {"role": "user", "content": CLASSIFIER_PROMPT.format(message=last_message)}
        ], temperature=0),
        asyncio.sleep(0),  # placeholder to keep gather structure
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"intent": "general", "name": None, "address": None}

    intent = parsed.get("intent", "general").lower()

    extracted_name = parsed.get("name") if profile.name is None else None
    extracted_address = parsed.get("address") if profile.property_address is None else None

    # If we got new name or address, look up the property DB immediately
    record = None
    if extracted_name or extracted_address:
        record = await lookup_property(
            name=extracted_name,
            address=extracted_address,
        )

    # Build profile updates
    updates = {}
    if extracted_name:
        updates["name"] = extracted_name
    if extracted_address:
        updates["property_address"] = extracted_address

    if record:
        # Pre-fill qualification fields from DB — only set fields still null
        if profile.fico_score is None:
            updates["fico_score"] = record.fico_score
        if profile.estimated_value is None:
            updates["estimated_value"] = record.home_value
        if profile.mortgage_balance is None:
            updates["mortgage_balance"] = record.mortgage_balance
        if profile.property_address is None:
            updates["property_address"] = record.address
        if profile.name is None and not extracted_name:
            updates["name"] = record.full_name

    if updates:
        profile = profile.model_copy(update=updates)

    return {"intent": intent, "user_profile": profile}
