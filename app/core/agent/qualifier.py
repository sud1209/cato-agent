from langchain.agents import create_agent
from app.core.agent.model_factory import get_model
from app.core.tools.property_tool import fetch_user_property_profile
from app.core.tools.rag_tool import check_qualification_rules
from app.schemas.agent_responses import QualificationResponse
# 1. Agnostic Model
llm = get_model(temperature=0)

# 2. Define the Prompt (Modern Placeholder Style)
QUALIFIER_PROMPT = """
You are the Cato Qualifier Agent. Your ONLY goal is to determine HEI qualification.

### IMMEDIATE DISQUALIFICATION (HARD FAILS):
If the user provides a FICO score below 620, you must IMMEDIATELY set:
- "qual_status": "UNQUALIFIED"
- "reason": "FICO score of [SCORE] is below the 620 minimum."
DO NOT ask for more information. DO NOT call any other tools. STOP immediately.

### STANDARD PROCESS:
1. Use 'fetch_user_property_profile' for data.
2. Use 'check_qualification_rules' for criteria.
3. If data is missing (and no Hard Fail is present), set 'qual_status' to 'PENDING'.
4. Only set 'QUALIFIED' if ALL criteria (FICO, Equity, Liens) are confirmed.

### OUTPUT FORMAT:
Return a JSON object matching the QualificationResponse schema.
"""

# 3. Initialize Agent with the custom prompt
qualifier_agent = create_agent(
    model=llm,
    tools=[fetch_user_property_profile, check_qualification_rules],
    system_prompt=QUALIFIER_PROMPT,
    response_format=QualificationResponse # Structured Output preserved
)