"""
Run the Spanish-language agent + LLM judge from langfuse_llm_judge_example
as a Langfuse experiment on dataset `test_ds`.
"""
from langfuse import Evaluation, get_client, observe

# Importing this module loads .env and instruments Anthropic for tracing.
from langfuse_llm_judge_example import agent_workflow, call_llm, judge_language

DATASET_NAME = "test_ds"
EXPERIMENT_NAME = "spanish-agent-llm-judge"

langfuse = get_client()

CLOSENESS_JUDGE_PROMPT = """\
You are evaluating whether an assistant response is close enough to the expected answer.

Expected answer:
{expected}

Actual assistant response:
{answer}

Judge semantic / factual closeness, not exact wording. SCORE 1 if the actual response
captures the same core facts and meaning as the expected answer (extra detail or
different phrasing is fine). SCORE 0 if key facts are missing, wrong, or contradicted.

Reply with exactly one line in this format:
SCORE: <0 or 1>
REASON: <brief explanation>
"""


def _text_from_message(message: dict) -> str | None:
    """Extract plain text from a chat message dict (content or parts)."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    parts = message.get("parts")
    if isinstance(parts, list):
        texts = [
            part["content"]
            for part in parts
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("content"), str)
        ]
        if texts:
            return "\n".join(texts)
    return None


def _text_from_chat_or_plain(raw, *, preferred_roles: tuple[str, ...]) -> str:
    """Normalize plain text, dict, or chat-message list to a string."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("question", "input", "query", "text", "answer", "content"):
            value = raw.get(key)
            if isinstance(value, str):
                return value
        text = _text_from_message(raw)
        if text:
            return text
        raise ValueError(f"Unsupported dict keys: {list(raw.keys())}")
    if isinstance(raw, list):
        for message in reversed(raw):
            if isinstance(message, dict) and message.get("role") in preferred_roles:
                text = _text_from_message(message)
                if text:
                    return text
        # Fallback: last message with any extractable text
        for message in reversed(raw):
            if isinstance(message, dict):
                text = _text_from_message(message)
                if text:
                    return text
        raise ValueError("No text found in chat-style value")
    raise TypeError(f"Unsupported type: {type(raw)}")


def _question_from_input(raw) -> str:
    """Normalize dataset item input to a question string."""
    return _text_from_chat_or_plain(raw, preferred_roles=("user",))


def _answer_from_expected(raw) -> str:
    """Normalize dataset item expected_output to an answer string."""
    return _text_from_chat_or_plain(raw, preferred_roles=("assistant",))


def _parse_score_reason(raw: str) -> tuple[int, str]:
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


def judge_closeness(answer: str, expected: str) -> tuple[int, str]:
    """LLM-as-a-judge: 1 if answer is close enough to expected, else 0."""
    raw = call_llm(CLOSENESS_JUDGE_PROMPT.format(expected=expected, answer=answer))
    return _parse_score_reason(raw)


def run_agent_task(*, item, **kwargs) -> str:
    """Experiment task: run the Spanish-answer agent on one dataset item."""
    question = _question_from_input(item.input)
    return agent_workflow(
        question,
        user_id="experiment-user",
        session_id=f"experiment-{DATASET_NAME}",
    )


def language_match_evaluator(*, input, output, expected_output=None, **kwargs):
    """Item evaluator: LLM judge for language match (same as the example scoring)."""
    question = _question_from_input(input)
    score, reason = judge_language(question, output)
    return Evaluation(
        name="language-match-spanish",
        value=score,
        data_type="BOOLEAN",
        comment=reason,
    )


def closeness_evaluator(*, input, output, expected_output=None, **kwargs):
    """Item evaluator: LLM judge for closeness to expected_output."""
    if expected_output is None:
        return Evaluation(
            name="answer-closeness",
            value=0,
            data_type="BOOLEAN",
            comment="No expected_output on dataset item",
        )
    expected = _answer_from_expected(expected_output)
    score, reason = judge_closeness(output, expected)
    return Evaluation(
        name="answer-closeness",
        value=score,
        data_type="BOOLEAN",
        comment=reason,
    )


def _is_failure(value) -> bool:
    """Treat 0 / False as a failed boolean-style score."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    return False


def failure_counts_run_evaluator(*, item_results, **kwargs):
    """Run-level: count failures per item evaluator (logged on the dataset run)."""
    totals: dict[str, int] = {}
    failures: dict[str, int] = {}
    for item_result in item_results:
        for evaluation in item_result.evaluations:
            name = evaluation.name
            totals[name] = totals.get(name, 0) + 1
            if _is_failure(evaluation.value):
                failures[name] = failures.get(name, 0) + 1

    return [
        Evaluation(
            name=f"{name}-failures",
            value=failures.get(name, 0),
            data_type="NUMERIC",
            comment=f"{failures.get(name, 0)}/{totals.get(name, 0)} items failed",
        )
        for name in totals
    ]


def print_failure_summary(item_results) -> None:
    """Local console summary of failed items per evaluator."""
    totals: dict[str, int] = {}
    failures: dict[str, int] = {}
    for item_result in item_results:
        for evaluation in item_result.evaluations:
            name = evaluation.name
            totals[name] = totals.get(name, 0) + 1
            if _is_failure(evaluation.value):
                failures[name] = failures.get(name, 0) + 1

    print("Failures per evaluator:")
    for name in totals:
        failed = failures.get(name, 0)
        print(f"  • {name}: {failed}/{totals[name]} failed")


if __name__ == "__main__":
    dataset = langfuse.get_dataset(DATASET_NAME)
    print(f"Loaded dataset {DATASET_NAME!r} with {len(dataset.items)} item(s)")

    result = dataset.run_experiment(
        name=EXPERIMENT_NAME,
        description=(
            "Agent answers in Spanish; LLM judges score language match and "
            "closeness to expected_output."
        ),
        task=run_agent_task,
        evaluators=[language_match_evaluator, closeness_evaluator],
        run_evaluators=[failure_counts_run_evaluator],
        max_concurrency=1,
        metadata={"source": "langfuse_llm_judge_example"},
    )

    print(result.format(include_item_results=True))
    print_failure_summary(result.item_results)
    if result.dataset_run_url:
        print(f"View experiment in Langfuse: {result.dataset_run_url}")

    langfuse.flush()
