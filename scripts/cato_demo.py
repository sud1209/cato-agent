import asyncio
import uuid
from app.core.agent.master import run_master_agent

async def chat():
    session_id = f"vibe-check-{uuid.uuid4().hex[:6]}"
    print(f"--- Cato Vibe Check (Session: {session_id}) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            response = await run_master_agent(session_id, user_input)
            print(f"\nCato: {response}\n")
        except Exception as e:
            print(f"\n[ERROR]: {e}\n")

if __name__ == "__main__":
    asyncio.run(chat())