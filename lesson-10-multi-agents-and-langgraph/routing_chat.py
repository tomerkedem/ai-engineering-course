"""
LangGraph routing example: route to a formal (politics) or comedian (other) node.
Uses Claude Haiku 4.5. Politics questions get a professional answer; other topics get a comedic answer.

Usage:
  Set ANTHROPIC_API_KEY, then:
    python routing_chat.py "What is the role of the electoral college?"
    python routing_chat.py "Why is the sky blue?"
    python routing_chat.py                    # interactive loop
"""

import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from langgraph.graph.message import add_messages

load_dotenv()


# --- State ---
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# --- Model (Claude Haiku 4.5) ---
def get_model():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY in your environment.")
        sys.exit(1)
    return init_chat_model(
        "anthropic:claude-haiku-4-5-20251001",
        temperature=0,
        api_key=api_key,
    )


def _last_user_text(state: ChatState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def make_router(model):
    """Returns a routing function: politics -> formal node, other -> comedian node."""

    ROUTER_PROMPT = """You are a classifier. Given the user's question, decide if it is about POLITICS (government, elections, law, policy, political parties, politicians, voting, legislation, etc.) or something else.

Answer with exactly one word: politics or other. No explanation."""

    def route(state: ChatState) -> str:
        text = _last_user_text(state)
        if not text:
            return "comedian"
        response = model.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=text),
            ]
        )
        content = (response.content or "").strip().lower()
        if "politics" in content:
            return "formal"
        return "comedian"

    return route


def make_formal_node(model):
    """Answers in a formal, professional manner (for politics)."""

    SYSTEM = (
        "You are a knowledgeable, formal expert. Answer the user's question in a serious, "
        "professional, and objective manner. Be clear and informative. Do not use humor or jokes."
    )

    def node(state: ChatState):
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    return node


def make_comedian_node(model):
    """Answers as a comedian (for non-politics topics)."""

    SYSTEM = (
        "You are a comedian. Answer the user's question in a funny, witty, and entertaining way. "
        "Use humor, jokes, puns, and a light tone. Still be somewhat informative, but make it fun."
    )

    def node(state: ChatState):
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    return node


def main():
    # Avoid UnicodeEncodeError on Windows when printing model output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model = get_model()

    router = make_router(model)
    formal_node = make_formal_node(model)
    comedian_node = make_comedian_node(model)

    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("formal", formal_node)
    graph_builder.add_node("comedian", comedian_node)
    graph_builder.add_conditional_edges(START, router, path_map={"formal": "formal", "comedian": "comedian"})
    graph_builder.add_edge("formal", END)
    graph_builder.add_edge("comedian", END)
    graph = graph_builder.compile()


    print(
        "Routing chat (Claude Haiku 4.5): politics -> formal, other -> comedian. Type your query, or 'quit' to exit.\n"
    )
    messages = []
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        messages.append(HumanMessage(content=query))
        result = graph.invoke({"messages": messages})
        messages = result["messages"]
        last = messages[-1]
        print("Assistant:", last.content, "\n")


if __name__ == "__main__":
    main()
