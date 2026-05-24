import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    max_tokens=1024,
    messages=[{"role": "user", "content": "write 500 words about the history of the internet"}],
    model="claude-haiku-4-5",
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)