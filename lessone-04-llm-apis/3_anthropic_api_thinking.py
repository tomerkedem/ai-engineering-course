import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[
        {
            "role": "user",
            "content": "Are there an infinite number of prime numbers such that n mod 4 == 3?",
        }
    ],
)

print(
    f"input_tokens={response.usage.input_tokens}, "
    f"output_tokens={response.usage.output_tokens}"
)

# The response contains summarized thinking blocks and text blocks
for block in response.content:
    if block.type == "thinking":
        print(f"\n\n==========Thinking summary: {block.thinking}==========")
    elif block.type == "text":
        print(f"\n\n==========Response: {block.text}==========")
        