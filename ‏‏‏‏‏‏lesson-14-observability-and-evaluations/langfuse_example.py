"""
Simple Langfuse example: trace LLM calls and see token usage.
Loads API keys from .env (LANGFUSE_*, ANTHROPIC_API_KEY).
AnthropicInstrumentor captures each Anthropic call as a Generation with token usage.
user_id / session_id group traces under Users and Sessions in the Langfuse UI.
"""
from pathlib import Path
from dotenv import load_dotenv
from langfuse import observe, propagate_attributes
from anthropic import Anthropic
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

# Load .env from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")

# Capture all Anthropic calls (including token usage) into the trace
AnthropicInstrumentor().instrument()

client = Anthropic()


def call_llm(user_input):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_input}],
    )
    return response.content[0].text

@observe()
def agent_workflow(query, *, user_id: str, session_id: str):
    # Attach user/session to this span and all nested observations (incl. Anthropic)
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        trace_name="agent-workflow",
    ):
        return call_llm(query)


if __name__ == "__main__":
    user_id = "demo-user-1"
    session_id = "demo-chat-1"  # reuse for every turn in the same conversation

    # Two turns share session_id → one Session with two traces in Langfuse
    for question in (
        "Is Italy the capital of Rome?",
    ):
        result = agent_workflow(question, user_id=user_id, session_id=session_id)
        print("Response:", result)
