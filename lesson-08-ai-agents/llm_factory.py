import os

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


def build_llm():
    """
    Build an LLM client based on environment configuration.

    Supported providers:
    - anthropic
    - ollama
    """
    provider = os.getenv("MODEL_PROVIDER", "anthropic").strip().lower()
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Set it or use MODEL_PROVIDER=ollama."
            )

        model_name = os.getenv(
            "ANTHROPIC_MODEL",
            "claude-haiku-4-5-20251001",
        )

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
        )

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "gemma3:1b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER: {provider}. "
        "Use 'anthropic' or 'ollama'."
    )