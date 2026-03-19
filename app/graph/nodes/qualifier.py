from __future__ import annotations
import json
from app.core.llm import chat_completion_json
from app.graph.state import CatoState
from app.memory.profile import UserProfile

QUALIFIER_SYSTEM = """\
You are the Cato Qualifier at Shire.LLC. Your ONLY job is to collect qualification fields and make a decision.

Qualification criteria:
- FICO score >= 500
- Equity percentage >= 30% (equity = (value - mortgage) / value)
- Property type is eligible: SFR, condo, or qualifying multi-family
- No active bankruptcy
- Property located in California, Colorado, Florida, or Arizona

Current UserProfile:
{profile}

Conversation summary (prior turns):
{summary}

IMPORTANT — question priority order:
1. If name AND address are both missing, ask for full name and address FIRST so we can look them up in our system. Do not ask about FICO or home value yet.
2. If name or address is present but home_value is still null, it means the DB lookup didn't find them — ask for home value and mortgage balance directly.
3. If home_value and fico_score are already filled (from DB lookup), skip those questions and only ask for anything still missing (property_type, bankruptcy).
4. Never ask for information that's already in the profile.

For message_to_user: ONE sentence, casual and direct — like texting a friend.
No filler phrases. No bullet points. Use contractions.

Respond ONLY with valid JSON:
{{
  "status": "qualified" | "unqualified" | "pending",
  "decision": "<brief rationale, 1 sentence>",
  "next_question": "<question if pending, else null>",
  "message_to_user": "<the actual message Cato sends — 1 sentence, casual>",
  "reasoning": "<internal step-by-step, never shown to user>",
  "extracted": {{
    "name": "<string or null>",
    "fico_score": "<integer or null>",
    "estimated_value": "<float or null>",
    "mortgage_balance": "<float or null>",
    "property_type": "<SFR|condo|multi-family or null>",
    "property_address": "<string or null>",
    "has_bankruptcy": "<true|false or null>"
  }}
}}
"""


async def qualify(state: CatoState) -> dict:
    profile = state["user_profile"]
    summary = state.get("conversation_summary", "")
    messages = state["messages"]

    system_msg = QUALIFIER_SYSTEM.format(
        profile=profile.model_dump_json(indent=2),
        summary=summary or "None",
    )
    llm_messages = [{"role": "system", "content": system_msg}]
    for m in messages:
        role = "user" if m.type == "human" else "assistant"
        llm_messages.append({"role": role, "content": m.content})

    raw = await chat_completion_json(llm_messages, temperature=0)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "status": "pending",
            "message_to_user": "What's your FICO score and roughly how much is your home worth?",
            "extracted": {},
        }

    status = parsed.get("status", "pending")
    message = parsed.get("message_to_user", "")

    extracted = parsed.get("extracted", {}) or {}
    profile_updates = {
        k: v for k, v in extracted.items()
        if v is not None and getattr(profile, k, None) is None
    }
    if profile_updates:
        profile = profile.model_copy(update=profile_updates)

    from langchain_core.messages import AIMessage
    return {
        "qualification_result": status,
        "user_profile": profile,
        "messages": [AIMessage(content=message)],
    }
