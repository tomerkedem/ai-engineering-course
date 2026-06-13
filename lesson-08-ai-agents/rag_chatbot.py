import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from build_rag_db import load_vectorstore


load_dotenv()


MODEL_NAME = "claude-haiku-4-5-20251001"
RETRIEVER_K = 4
MAX_HISTORY_MESSAGES = 10


def build_llm() -> ChatAnthropic:
    """
    Build the LLM client.

    The API key is read from the ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Please set it before running rag_chatbot.py."
        )

    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=0,
    )


def build_rag_chain(vectorstore, llm):
    """
    Build a RAG chain that retrieves relevant context
    and sends it to the language model.
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful assistant.

Answer the user's question based only on the following context.

If the context does not contain relevant information,
say that you do not have enough information in the provided documents.

Keep the answer concise and clear.

Context:
{context}
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def get_context(inputs):
        docs = retriever.invoke(inputs["question"])
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(context=get_context)
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def convert_history_to_messages(chat_history: list[dict]):
    """
    Convert a simple list of dictionaries into LangChain message objects.
    """
    messages = []

    for message in chat_history[-MAX_HISTORY_MESSAGES:]:
        role = message["role"]
        content = message["content"]

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def main():
    print("Loading RAG chatbot...")

    llm = build_llm()
    vectorstore = load_vectorstore()
    rag_chain = build_rag_chain(vectorstore, llm)

    chat_history = []

    print("RAG chatbot is ready.")
    print("Ask a question about the documents.")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not user_input:
            print("Please enter a question.")
            continue

        try:
            messages_for_prompt = convert_history_to_messages(chat_history)

            answer = rag_chain.invoke(
                {
                    "question": user_input,
                    "chat_history": messages_for_prompt,
                }
            )

            print(f"\nAssistant: {answer}")

            chat_history.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )
            chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()