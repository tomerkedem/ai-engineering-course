"""
Load balancing example: distribute requests across two models via LiteLLM Router.

Both deployments share the same model alias; the router picks one per request
(default strategy: simple-shuffle). Optional `weight` in litellm_params skews traffic.

Requires: pip install litellm
Env: ANTHROPIC_API_KEY, GEMINI_API_KEY
"""
from litellm import Router

MODEL_ALIAS = "fast-chat"

router = Router(
    model_list=[
        {
            "model_name": MODEL_ALIAS,
            "litellm_params": {
                "model": "anthropic/claude-haiku-4-5",
                "weight": 1,
            },
        },
        {
            "model_name": MODEL_ALIAS,
            "litellm_params": {
                "model": "gemini/gemini-3.5-flash",
                "weight": 1,
            },
        },
    ],
    routing_strategy="simple-shuffle",
)

MESSAGES = [{"role": "user", "content": "Reply with exactly one word: hello."}]


def main():
    print(f"Sending 6 requests through alias '{MODEL_ALIAS}' (50/50 load balance):\n")
    for i in range(1, 7):
        response = router.completion(
            model=MODEL_ALIAS,
            messages=MESSAGES,
            max_tokens=32,
        )
        text = (response.choices[0].message.content or "").strip()
        print(f"  Request {i}: {response.model!r} -> {text!r}")


if __name__ == "__main__":
    main()
