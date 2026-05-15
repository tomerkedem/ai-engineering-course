"""Read text files from input/, classify and summarize via one Anthropic call each, write to output/<topic>/."""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output"

TOPICS = frozenset({"cars", "sport", "music"})

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["topic", "summary"],
    "properties": {
        "topic": {
            "type": "string",
            "enum": ["cars", "sport", "music"],
            "description": "Which single topic best matches the document: cars, sport, or music.",
        },
        "summary": {
            "type": "string",
            "description": "A summary of the document in exactly 20 words (count words; stay at 20 if possible).",
        },
    },
}


def ensure_input_and_topic_dirs() -> None:
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Input directory not found: {INPUT_DIR}")
    for topic in TOPICS:
        (OUTPUT_DIR / topic).mkdir(parents=True, exist_ok=True)


def classify_and_summarize(client: anthropic.Anthropic, document: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    "You classify documents and summarize them.\n\n"
                    "Rules:\n"
                    "- Choose exactly one topic: cars, sport, or music.\n"
                    "- Write summary as exactly 20 words when possible.\n\n"
                    "Document:\n"
                    f"{document}"
                ),
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": OUTPUT_SCHEMA,
            }
        },
    )
    return json.loads(response.content[0].text)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    ensure_input_and_topic_dirs()

    client = anthropic.Anthropic(api_key=api_key)

    for path in sorted(INPUT_DIR.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        result = classify_and_summarize(client, text)
        topic = result["topic"]
        summary = result["summary"]

        if topic not in TOPICS:
            raise ValueError(f"Unexpected topic from model: {topic!r}")

        out_path = OUTPUT_DIR / topic / path.name
        body = f"Summary:\n{summary}\n\n---\n\n{text}"
        out_path.write_text(body, encoding="utf-8")
        print(f"{path.name} -> {topic}/")


if __name__ == "__main__":
    main()
