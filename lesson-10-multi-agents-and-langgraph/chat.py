"""
Simple LangGraph chat: one node, command-line prompt, response from Claude Haiku 4.5.

Usage:
  Set ANTHROPIC_API_KEY, then:
    python chat.py                    # interactive: type a query, get a response
    python chat.py "Your question"    # one-shot: pass query as argument
"""

import os
import sys

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import Annotated, TypedDict


from dotenv import load_dotenv
load_dotenv()



# --- State: we only keep the message list ---
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# --- Build model (Claude Haiku 4.5) ---
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


# --- Single node: call the LLM with current messages ---
def chat_node(state: ChatState, model):
    response = model.invoke(
        [
            SystemMessage(content="You are a helpful assistant. Answer concisely."),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def main():
    model = get_model()

    # One node: user messages -> LLM -> assistant message
    def node(state: ChatState):
        return chat_node(state, model)

    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chat", node)
    graph_builder.add_edge(START, "chat")
    graph_builder.add_edge("chat", END)
    graph = graph_builder.compile()

    # Interactive loop
    print("LangGraph chat (Claude Haiku 4.5). Type your query, or 'quit' to exit.\n")
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
