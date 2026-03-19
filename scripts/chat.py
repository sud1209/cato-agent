#!/usr/bin/env python3
"""Terminal chat client for Cato. Requires httpx (already in .venv)."""
from __future__ import annotations
import sys
import json
import random
import string
import httpx

BASE_URL = "http://localhost:8000"


def new_session() -> str:
    return "demo-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=7))


def chat(session_id: str, message: str) -> None:
    with httpx.Client(timeout=60) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": message},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            print("\033[36mCato:\033[0m ", end="", flush=True)
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                print(data, end="", flush=True)
            print()  # newline after response


def main() -> None:
    session_id = new_session()
    print(f"\033[90mSession: {session_id}  |  Type 'exit' or Ctrl-C to quit\033[0m\n")

    while True:
        try:
            user_input = input("\033[33mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        try:
            chat(session_id, user_input)
        except httpx.ConnectError:
            print(f"\033[31mError: Could not connect to {BASE_URL}. Is the server running?\033[0m")
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")

        print()


if __name__ == "__main__":
    main()
