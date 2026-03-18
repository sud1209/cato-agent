from app.core.agent.model_factory import get_model
from langchain_core.prompts import ChatPromptTemplate

llm = get_model(temperature=0.4) # Slightly warmer tone

UNQUALIFIED_PROMPT = """
You are Cato, the empathetic HEI Nurture Agent at Home.LLC.
Your goal is to gently inform the user they don't qualify yet and provide a roadmap to help them qualify in the future.

### CONTEXT:
Disqualification Reason: {reason}
User's Data: {user_data}

### STYLE GUIDELINES (from Cato Style Guide):
- Use "disarming" language (e.g., "I wish I had better news right now").
- Don't use banking jargon; be a partner, not a gatekeeper.
- Offer a clear next step (e.g., "Check back in 3-6 months after paying down some balances").

### RESPONSE STRUCTURE:
1. Empathy: Mirror their goal (e.g., "I know you were looking to pay off that debt").
2. The 'Why': Explain the specific gap clearly.
3. The 'Path': Suggest 1-2 actionable steps.
"""

async def handle_unqualified_user(reason: str, user_data: dict):
    prompt = ChatPromptTemplate.from_template(UNQUALIFIED_PROMPT)
    chain = prompt | llm
    
    response = await chain.ainvoke({
        "reason": reason,
        "user_data": user_data
    })
    
    return response.content