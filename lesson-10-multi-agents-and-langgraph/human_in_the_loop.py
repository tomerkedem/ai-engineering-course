"""
LangGraph human-in-the-loop example.

Demonstrates LangGraph's built-in HITL primitives:
  - interrupt() to pause and surface payloads to the caller
  - Command(resume=...) to resume with human input
  - Command(goto=...) to route after approval/rejection
  - stream_events(..., version="v3") with stream.interrupted / stream.interrupts
  - InMemorySaver checkpointer (required for interrupts)

Workflow: submit an expense → manager approves or rejects → on reject, collect reason and loop back to approval until approved.

Usage:
    python human_in_the_loop.py
"""

import uuid
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ExpenseState(TypedDict):
    employee: str
    amount: float
    description: str
    status: str
    rejection_reason: str


def submit_expense(state: ExpenseState) -> dict:
    """Build the expense request (no LLM — keeps the example self-contained)."""
    summary = (
        f"{state['employee']} requests ${state['amount']:.2f} "
        f"for: {state['description']}"
    )
    print("\n--- Expense submitted ---")
    print(summary)
    return {"status": "pending"}


def manager_approval(state: ExpenseState) -> Command[Literal["process", "reject"]]:
    """Pause for manager approval; route with Command(goto=...) after resume."""
    payload: dict = {
        "type": "approval",
        "question": "Approve this expense? (y/n)",
        "expense": {
            "employee": state["employee"],
            "amount": state["amount"],
            "description": state["description"],
        },
    }
    if state.get("rejection_reason"):
        payload["rejection_reason"] = state["rejection_reason"]
    approved = interrupt(payload)
    if approved:
        return Command(goto="process")
    return Command(goto="reject")


def process_payment(state: ExpenseState) -> dict:
    print(f"\n--- Payment processed for {state['employee']} ---")
    return {"status": "approved"}


def reject_expense(state: ExpenseState) -> dict:
    """Collect rejection reason, then loop back to approval via graph edge."""
    reason = interrupt(
        {
            "type": "feedback",
            "question": "Expense rejected. Enter a reason for the employee:",
            "expense": {
                "employee": state["employee"],
                "amount": state["amount"],
                "description": state["description"],
            },
        }
    )
    return {"status": "pending", "rejection_reason": reason or ""}


def build_graph():
    builder = StateGraph(ExpenseState)
    builder.add_node("submit", submit_expense)
    builder.add_node("approval", manager_approval)
    builder.add_node("process", process_payment)
    builder.add_node("reject", reject_expense)

    builder.add_edge(START, "submit")
    builder.add_edge("submit", "approval")
    builder.add_edge("process", END)
    builder.add_edge("reject", "approval")

    return builder.compile(checkpointer=InMemorySaver())


def _parse_approval(response: str) -> bool:
    return response.strip().lower() in {"y", "yes", "approve", "true", "1"}


def _prompt_for_interrupt(payload: dict) -> str | bool:
    """Read terminal input for a single interrupt payload."""
    print("\n--- Human input required ---")
    if payload.get("type") == "approval":
        expense = payload["expense"]
        print(payload["question"])
        print(
            f"  Employee: {expense['employee']}\n"
            f"  Amount:   ${expense['amount']:.2f}\n"
            f"  For:      {expense['description']}"
        )
        if payload.get("rejection_reason"):
            print(f"  Prior rejection reason: {payload['rejection_reason']}")
        return _parse_approval(input("Your decision (y/n): "))
    if payload.get("type") == "feedback":
        print(payload["question"])
        return input("Reason: ").strip()
    print(payload)
    return input("Your response: ").strip()


def run_with_human_input(graph, initial_state: ExpenseState, thread_id: str) -> ExpenseState:
    """
    Drive the graph using LangGraph's recommended stream_events v3 loop.

    Checks stream.interrupted after each run and resumes with Command(resume=...).
    """
    config = {"configurable": {"thread_id": thread_id}}
    stream_input: dict | Command = initial_state

    # start the loop
    while True:
        # Stream the events from the graph execution.
        # This runs the workflow up until any human input (interrupt) is required, or until completion.
        # The stream object yields intermediate states and interruptions for human-in-the-loop workflows.
        stream = graph.stream_events(stream_input, config=config, version="v3")

        # Access the current output of the graph (after handling any events in this step).
        # This output is the updated state produced by the latest graph execution.
        # We assign it to "_" because it's mainly used to drive the loop or for checking completion below.
        _ = stream.output

        # Check if the stream has been interrupted.
        if not stream.interrupted:
            return stream.output

        # This graph is sequential — at most one interrupt per step.
        answer = _prompt_for_interrupt(stream.interrupts[0].value)
        stream_input = Command(resume=answer)


def main() -> None:
    print("Expense reimbursement (human-in-the-loop demo)\n")
    employee = input("Employee name: ").strip()
    amount_raw = input("Amount (USD): ").strip()
    description = input("Description: ").strip()

    if not employee or not amount_raw or not description:
        print("All fields are required.")
        return

    try:
        amount = float(amount_raw)
    except ValueError:
        print("Amount must be a number.")
        return

    graph = build_graph()
    thread_id = str(uuid.uuid4())

    result = run_with_human_input(
        graph,
        {
            "employee": employee,
            "amount": amount,
            "description": description,
            "status": "",
            "rejection_reason": "",
        },
        thread_id,
    )

    print(f"\nDone — status: {result['status']}")


if __name__ == "__main__":
    main()
