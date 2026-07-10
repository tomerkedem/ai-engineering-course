import asyncio
import os
from pathlib import Path

import aioboto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import time

INPUT_FILE = Path(__file__).resolve().parent / "input.txt"
MODEL_ID = "us.amazon.nova-2-lite-v1:0"
REGION = "us-east-1"


def setup_env() -> None:
    load_dotenv()
    if not os.environ.get("AWS_SECRET_ACCESS_KEY") and os.environ.get(
        "AWS_ACCESS_SECRET_KEY"
    ):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_ACCESS_SECRET_KEY"]


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path.name} is empty.")
    return text


async def summarize(text: str) -> str:
    session = aioboto3.Session()
    async with session.client("bedrock-runtime", region_name=REGION) as client:
        response = await client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "Summarize the following text in 2-3 concise sentences:\n\n"
                                f"{text}"
                            )
                        }
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.3,
            },
        )

    output = response["output"]["message"]["content"]
    for block in output:
        if "text" in block:
            return block["text"].strip()

    raise RuntimeError("No text returned from the model.")


async def main() -> None:
    setup_env()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Expected text file at {INPUT_FILE}")

    text = load_text(INPUT_FILE)

    try:
        time_start = time.time()
        summary = await summarize(text)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
    except ClientError as exc:
        raise SystemExit(f"Bedrock request failed: {exc}") from exc

    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
