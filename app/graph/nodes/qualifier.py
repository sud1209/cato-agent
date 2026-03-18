from __future__ import annotations
import json
from app.core.llm import chat_completion_json
from app.graph.state import CatoState
from app.memory.profile import UserProfile

QUALIFIER_SYSTEM = """\
You are the Cato Qualifier. Your ONLY job is to collect the four qualification fields and make a decision.

Qualification criteria:
- FICO score >= 620
- Equity percentage >= 25% (equity = (value - mortgage) / value)
- Property type is eligible: SFR, condo, or qualifying multi-family
- No active bankruptcy

Current UserProfile:
{profile}

Conversation summary (prior turns):
{summary}

Think step by step:
1. Which fields are still missing (null)?
2. Is there enough information to make a final QUALIFIED or UNQUALIFIED decision?
3. If yes — does the user meet ALL four criteria?
4. If no — what single question should Cato ask next?

Respond ONLY with valid JSON:
{{
  "status": "qualified" | "unqualified" | "pending",
  "decision": "<brief rationale, 1 sentence>",
  "next_question": "<question if pending, else null>",
  "message_to_user": "<the actual message Cato should send to the user>",
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
            "message_to_user": "Could you tell me your FICO score and home value?",
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
