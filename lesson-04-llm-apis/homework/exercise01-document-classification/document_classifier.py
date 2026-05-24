import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv


load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
SUMMARIES_DIR = BASE_DIR / "summaries"

ALLOWED_CLASSES = ["cars", "sport", "music"]


def classify_and_summarize_document(document_text: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": f"""
You are a document classification and summarization assistant.

Classify the following document into exactly one of these classes:
cars, sport, music.

Return only valid JSON in this exact format:

{{
  "class_name": "cars | sport | music",
  "summary": "short summary of the document"
}}

Rules:
1. class_name must be exactly one of: cars, sport, music.
2. summary must be clear, concise, and based only on the document.
3. Do not add explanations outside the JSON.
4. Do not wrap the JSON in markdown.

Document:
{document_text}
"""
            }
        ],
    )

    return json.loads(response.content[0].text)


def save_summary(file_name: str, class_name: str, summary: str) -> None:
    if class_name not in ALLOWED_CLASSES:
        raise ValueError(
            f"Invalid class name returned by model: {class_name}"
        )

    target_dir = SUMMARIES_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file = target_dir / f"{Path(file_name).stem}_summary.txt"

    output_file.write_text(summary, encoding="utf-8")


def main() -> None:
    SUMMARIES_DIR.mkdir(exist_ok=True)

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        document_text = file_path.read_text(encoding="utf-8")

        result = classify_and_summarize_document(document_text)

        class_name = result["class_name"]
        summary = result["summary"]

        save_summary(
            file_name=file_path.name,
            class_name=class_name,
            summary=summary
        )

        print(f"Processed {file_path.name} -> {class_name}")


if __name__ == "__main__":
    main()