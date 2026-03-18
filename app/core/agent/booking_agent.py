from app.core.agent.model_factory import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

llm = get_model(temperature=0.2)

# Tool for actual scheduling (Placeholder for Calendly/Email API)
@tool
def schedule_specialist_call(session_id: str, preferred_time: str):
    """Call this when the user agrees to a specific time for a call."""
    # Logic to send invite or update DB
    return f"Success: Meeting scheduled for {preferred_time}."

BOOKING_PROMPT = """
You are Cato's Booking Specialist at Home.LLC.
The user is QUALIFIED for a Home Equity Investment. Your sole mission is to get them on the phone with a Senior Advisor.

### GUIDELINES:
- Be enthusiastic but professional.
- Use the 'disarming' style from our training: "I'd love to make this easy for you..."
- If they ask for a link, provide the Calendly link: https://calendly.com/home-llc/specialist
- If they give a time, use the 'schedule_specialist_call' tool.

### STYLE:
Match the tone from the transcript:
User: "Let's talk."
Cato: "That’s great news! I’ve seen your profile and you’re a fantastic candidate for this. Would tomorrow at 10 AM work, or would you prefer a link to pick your own time?"
"""

async def handle_booking(user_input: str, state_data: dict):
    prompt = ChatPromptTemplate.from_template(BOOKING_PROMPT)
    # We use an agent executor if we want it to actually use the 'schedule' tool, 
    # or a simple chain if we just want it to talk.
    chain = prompt | llm
    response = await chain.ainvoke({"input": user_input, "state": state_data})
    return response.content