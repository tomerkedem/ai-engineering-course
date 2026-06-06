"""Call a local Ollama model via the Anthropic Messages API compatibility layer.

Requires Ollama v0.14.0+ and the anthropic package:
    pip install anthropic

Docs: https://docs.ollama.com/api/anthropic-compatibility
"""

import os

import anthropic

client = anthropic.Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL", "http://localhost:11434"),
    api_key=os.getenv("ANTHROPIC_AUTH_TOKEN", "ollama"),  # required but ignored by Ollama
)

message = client.messages.create(
    model="gemma3:1b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)

print(message.content[0].text)
