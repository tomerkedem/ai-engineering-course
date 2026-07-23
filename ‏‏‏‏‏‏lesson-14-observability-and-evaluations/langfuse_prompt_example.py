"""
Langfuse prompt management example: fetch a prompt from Langfuse, run it, and trace.
Loads API keys from the repo-root .env (LANGFUSE_*, ANTHROPIC_API_KEY).
Uses prompt 'prompt_1' (latest version).
user_id / session_id group traces under Users and Sessions in the Langfuse UI.
"""
from pathlib import Path
from dotenv import load_dotenv
from langfuse import observe, propagate_attributes, get_client
from langfuse.model import ChatPromptClient
from anthropic import Anthropic
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

# Load the repo-root .env (shared across lessons)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Capture all Anthropic calls (including token usage) into the trace
AnthropicInstrumentor().instrument()

langfuse = get_client()
client = Anthropic()

PROMPT_NAME = "prompt_1"


def call_llm(messages, *, system: str | None = None):
    kwargs = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def messages_from_prompt(prompt):
    """Compile Langfuse prompt (text or chat) into Anthropic messages + optional system."""
    compiled = prompt.compile()

    if isinstance(prompt, ChatPromptClient):
        system = next(
            (m["content"] for m in compiled if m.get("role") == "system"),
            None,
        )
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in compiled
            if m.get("role") in ("user", "assistant")
        ]
        return messages, system

    # Text prompt → single user message
    return [{"role": "user", "content": compiled}], None


@observe()
def agent_workflow(*, user_id: str, session_id: str):
    # Latest version of the managed prompt (label "latest" is maintained by Langfuse)
    prompt = langfuse.get_prompt(PROMPT_NAME, label="latest")
    messages, system = messages_from_prompt(prompt)

    # Attach user/session + link this prompt version to the Anthropic generation
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        trace_name="prompt-workflow",
        prompt=prompt,
    ):
        return call_llm(messages, system=system)


if __name__ == "__main__":
    result = agent_workflow(user_id="demo-user-1", session_id="demo-prompt-chat-1")
    print("Response:", result)
