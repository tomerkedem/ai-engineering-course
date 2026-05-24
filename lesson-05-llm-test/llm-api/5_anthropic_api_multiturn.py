"""Multi-turn chat: read user lines from stdin, append user + assistant turns to messages, call the API each time."""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1000


def complete_turn(
    client: anthropic.Anthropic,
    messages: list[dict[str, str]],
    user_text: str,
) -> str:
    messages.append({"role": "user", "content": user_text})
    response = client.messages.create(
        system="You are a helpful assistant that can answer questions and help with tasks.",
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        messages=messages,
    )
    assistant_text = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text


def run_chat_loop(client: anthropic.Anthropic) -> None:
    messages: list[dict[str, str]] = []
    print("Multi-turn chat (empty line to exit).")
    while True:
        try:
            user_line = input("You: ")
        except EOFError:
            print()
            break
        user_text = user_line.strip()
        if not user_text:
            break

        assistant_text = complete_turn(client, messages, user_text)
        print(f"Assistant: {assistant_text}\n")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    run_chat_loop(client)


if __name__ == "__main__":
    main()
