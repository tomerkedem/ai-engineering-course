"""
Langfuse example with user feedback scoring.
Same flow as langfuse_example.py, plus a prompt for response quality
that is written as a Langfuse score (visible under Scores in the UI).

All turns share one pre-created trace_id via langfuse_trace_id.
"""
from pathlib import Path
from dotenv import load_dotenv
from langfuse import observe, propagate_attributes, get_client
from anthropic import Anthropic
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

# Load .env from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")

# Capture all Anthropic calls (including token usage) into the trace
AnthropicInstrumentor().instrument()

client = Anthropic()
langfuse = get_client()


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
        trace_name="agent-workflow-with-feedback",
    ):
        return call_llm(query)


def collect_and_score_feedback(trace_id: str) -> None:
    """Ask the user for a 1–5 quality rating and record it in Langfuse."""
    while True:
        raw = input("Rate response quality (1=poor … 5=excellent): ").strip()
        try:
            score = int(raw)
            if 1 <= score <= 5:
                break
        except ValueError:
            pass
        print("Please enter an integer from 1 to 5.")

    comment = input("Optional comment (press Enter to skip): ").strip() or None

    langfuse.create_score(
        trace_id=trace_id,
        name="user-feedback",
        value=score,
        data_type="NUMERIC",
        comment=comment,
    )
    print(f"Scored trace {trace_id} with user-feedback={score}")


if __name__ == "__main__":
    user_id = "demo-user-1"
    session_id = "demo-chat-feedback-1"
    # Must be 32 lowercase hex chars; reuse so every turn/LLM call shares one trace
    trace_id = langfuse.create_trace_id()
    print(f"Using shared Langfuse trace_id: {trace_id}")

    for question in (
        "Is Italy the capital of Rome?",
    ):
        result = agent_workflow(
            question,
            user_id=user_id,
            session_id=session_id,
            langfuse_trace_id=trace_id,
        )
        print("Response:", result)
        collect_and_score_feedback(trace_id)

    langfuse.flush()
