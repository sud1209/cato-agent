import asyncio
import uuid
from app.core.agent.master import run_master_agent
from app.db.redis_client import redis_manager
from app.schemas.state import CatoState

# Define your Test Cases
test_scenarios = [
    {
        "name": "Objection Handling Style Check",
        "input": "How do I know this isn't a scam? I've never heard of Home LLC.",
        "expected_agent": "ObjectionHandler",
        "must_include": ["understand", "verify"] 
    },
    {
        "name": "General Information Check",
        "input": "What exactly is this Home Equity Investment thing?",
        "expected_agent": "InfoProvider",
        "must_include": ["equity", "repay", "monthly"] 
    },
    {
        "name": "Qualification Data Entry",
        "input": "My house is worth 900,000 and I owe about 400,000 on it.",
        "expected_agent": "Qualifier",
        "expected_qual_status": "PENDING"
    },
    {
        "name": "Disqualification Logic (Low FICO)",
        "input": "My FICO score is 520.",
        "expected_agent": "UnqualifiedAgent",
        "expected_qual_status": "UNQUALIFIED"
    },
    {
        "name": "Booking Intent",
        "input": "This sounds great, let's set up a time to talk to a specialist.",
        "expected_agent": "BookingAgent",
        "expected_booking_intent": "TRUE"
    }
]

async def run_tests():
    print(f"{'='*20} STARTING CATO TEST SUITE {'='*20}\n")
    
    for case in test_scenarios:
        session_id = f"test_{uuid.uuid4()}"
        print(f"RUNNING: {case['name']}")
        print(f"INPUT:   {case['input']}")

        try:
            # 1. Run the Master Agent
            response = await run_master_agent(session_id, case['input'])
            
            # 2. Fetch resulting state
            raw_state = await redis_manager.get_cato_state(session_id)
            state = CatoState(**raw_state)

            # 3. Validation Logic
            passed = True
            
            # Check Agent Routing
            if state.last_agent != case['expected_agent']:
                print(f"FAILED: Wrong agent. Expected {case['expected_agent']}, got {state.last_agent}")
                passed = False

            # Check for specific state updates (Qual Status)
            if "expected_qual_status" in case and state.qual_status != case['expected_qual_status']:
                print(f"FAILED: Wrong Qual Status. Expected {case['expected_qual_status']}, got {state.qual_status}")
                passed = False

            # Check Keyword Presence (Essential for RAG validation)
            if "must_include" in case:
                found_keywords = [word for word in case["must_include"] if word.lower() in response.lower()]
                if not found_keywords:
                    print(f"FAILED: Response did not include expected info keywords. Output: {response[:50]}...")
                    passed = False

            if passed:
                print(f"PASSED")
                print(f"   CATO: \"{response[:100]}...\"\n")

        except Exception as e:
            print(f"ERROR during test '{case['name']}': {e}\n")

    print(f"{'='*20} TEST SUITE COMPLETE {'='*20}")

if __name__ == "__main__":
    asyncio.run(run_tests())