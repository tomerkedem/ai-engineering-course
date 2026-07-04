"""
LiteLLM proxy client: send chat requests through a running LiteLLM proxy server.

The proxy exposes an OpenAI-compatible API. Point this client at the proxy URL
and use a model alias defined in the proxy config (not the upstream provider name).

Requires: pip install litellm httpx
Env:
  LITELLM_PROXY_URL   — proxy base URL (default: http://localhost:4000)
  LITELLM_MASTER_KEY  — proxy master key (must match proxy config)
  LITELLM_PROXY_MODEL — model alias on the proxy (optional; uses first /v1/models entry if unset)

Start the proxy first (from this directory, with .env loaded):
  litellm --config config.yaml --port 4000

On Windows, if startup fails with UnicodeEncodeError, set: $env:PYTHONIOENCODING='utf-8'

Available model aliases in config.yaml: claude-haiku, gemini-flash
"""
import os

import httpx
from litellm import completion

PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000").rstrip("/")
PROXY_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
PROXY_MODEL = os.getenv("LITELLM_PROXY_MODEL")


def _proxy_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {PROXY_KEY}"}


def check_proxy_health() -> dict:
    """Return the proxy /health/liveliness payload, or raise on connection failure."""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{PROXY_URL}/health/liveliness")
        response.raise_for_status()
        return response.json()


def list_models() -> list[str]:
    """Return model aliases exposed by the proxy (/v1/models)."""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{PROXY_URL}/v1/models", headers=_proxy_headers())
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", [])]


def chat(prompt: str, *, model: str | None = None, max_tokens: int = 256) -> str:
    """Send one user message through the proxy and return the assistant reply."""
    response = completion(
        model=model or resolve_model(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        api_base=PROXY_URL,
        api_key=PROXY_KEY,
        custom_llm_provider="litellm_proxy",
    )
    return response.choices[0].message.content or ""


def resolve_model() -> str:
    """Pick the model alias from env or the first entry returned by the proxy."""
    if PROXY_MODEL:
        return PROXY_MODEL

    models = list_models()
    if not models:
        raise RuntimeError(
            "No models on the proxy. Add a model_list to config.yaml and restart the proxy, "
            "or set LITELLM_PROXY_MODEL to a configured alias."
        )
    return models[0]


def main():
    print(f"Proxy URL: {PROXY_URL}\n")

    try:
        health = check_proxy_health()
        print("Proxy health:", health)
    except httpx.HTTPError as exc:
        print(f"Proxy not reachable at {PROXY_URL}: {exc}")
        print("Start the proxy first, then rerun this script.")
        return

    try:
        model = resolve_model()
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"Could not resolve model: {exc}")
        return

    print(f"Model alias: {model}")

    reply = chat("Say hello in one short sentence.", model=model)
    print("\nReply:", reply)


if __name__ == "__main__":
    main()
