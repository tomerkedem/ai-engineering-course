import os

import anthropic
from dotenv import load_dotenv


load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

messages = []


def send_message_to_model(user_input: str) -> str:
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        temperature=0,
        messages=messages,
    )

    assistant_text = response.content[0].text

    messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
        }
    )

    return assistant_text


def main() -> None:
    print("Chat started. Type 'exit' to stop.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        assistant_response = send_message_to_model(user_input)

        print(f"Claude: {assistant_response}")


if __name__ == "__main__":
    main()