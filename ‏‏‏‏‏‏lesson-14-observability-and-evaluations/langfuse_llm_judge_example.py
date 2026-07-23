"""
Langfuse example with LLM-as-a-judge scoring.
Agent is instructed to answer in Spanish; a second LLM call judges whether
the answer is in Spanish and writes the result as a Langfuse score.

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

AGENT_SYSTEM = (
    "You are a helpful assistant. Always answer the user's question in Spanish. "
    "Do not use any other language."
)

JUDGE_PROMPT = """\
You are evaluating whether an assistant response is written in the same language as the user question.

User question:
{question}

Assistant response:
{answer}

Reply with exactly one line in this format:
SCORE: <0 or 1>
REASON: <brief explanation>

SCORE must be 1 if the response is primarily in the same language as the user question, otherwise 0.
"""


def call_llm(user_input: str, *, system: str | None = None) -> str:
    kwargs = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": user_input}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


@observe()
def agent_workflow(query: str, *, user_id: str, session_id: str) -> str:
    # Attach user/session to this span and all nested observations (incl. Anthropic)
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        trace_name="agent-workflow-with-llm-judge",
    ):
        return call_llm(query, system=AGENT_SYSTEM)


def judge_language(question: str, answer: str) -> tuple[int, str]:
    """LLM-as-a-judge: 1 if answer is in Spanish, else 0."""
    raw = call_llm(JUDGE_PROMPT.format(question=question, answer=answer))
    score = 0
    reason = raw.strip()
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            value = line.split(":", 1)[1].strip()
            score = 1 if value.startswith("1") else 0
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


def score_with_llm_judge(trace_id: str, question: str, answer: str) -> None:
    """Run the language judge and record the score in Langfuse."""
    score, reason = judge_language(question, answer)
    langfuse.create_score(
        trace_id=trace_id,
        name="language-match-spanish",
        value=score,
        data_type="BOOLEAN",
        comment=reason,
    )
    print(f"Judge scored language-match-spanish={score} ({reason})")


if __name__ == "__main__":
    user_id = "demo-user-1"
    session_id = "demo-chat-llm-judge-1"
    # Must be 32 lowercase hex chars; reuse so every turn/LLM call shares one trace
    trace_id = langfuse.create_trace_id()
    print(f"Using shared Langfuse trace_id: {trace_id}")

    for question in (
        "What is the capital of Italy?",
        "¿Cuál es la capital de Italia?",
    ):
        result = agent_workflow(
            question,
            user_id=user_id,
            session_id=session_id,
            langfuse_trace_id=trace_id,
        )
        print("Response:", result)
        score_with_llm_judge(trace_id, question, result)

    langfuse.flush()
