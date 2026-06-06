import time
import ollama


PROMPT = "Explain what quantization is in local LLMs in one short paragraph."

MODELS = [
    "llama3:8b-instruct-q4_K_M",
    "gemma3:4b",
]


def measure_model_speed(model_name: str) -> dict:
    start_time = time.perf_counter()

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ],
    )

    end_time = time.perf_counter()

    return {
        "model": model_name,
        "elapsed_seconds": end_time - start_time,
        "response": response["message"]["content"]
    }


def main() -> None:
    for model_name in MODELS:
        print(f"Testing model: {model_name}")

        result = measure_model_speed(model_name)

        print(f"Time: {result['elapsed_seconds']:.2f} seconds")
        print("Response:")
        print(result["response"])
        print("-" * 80)


if __name__ == "__main__":
    main()
