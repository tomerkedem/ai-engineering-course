"""
Practice 1: Formatted response with LiteLLM.

Goal:
Ask the model to list 3 programming languages with one-word traits,
and return a valid JSON array with objects that contain:
- name
- trait

Requires:
    pip install litellm pydantic

Environment:
    ANTHROPIC_API_KEY
    or another provider key, depending on the selected model.
"""

import json
import os
from typing import Any

from litellm import completion
from pydantic import BaseModel, ValidationError, field_validator


class ProgrammingLanguageTrait(BaseModel):
    name: str
    trait: str

    @field_validator("name", "trait")
    @classmethod
    def must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty")
        return value

    @field_validator("trait")
    @classmethod
    def trait_must_be_one_word(cls, value: str) -> str:
        value = value.strip()
        if len(value.split()) != 1:
            raise ValueError("Trait must be exactly one word")
        return value


def strip_json_markdown(text: str) -> str:
    """
    Some models may wrap JSON with ```json ... ```.
    This helper removes common markdown fences if they appear.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def parse_and_validate_languages(raw_text: str) -> list[ProgrammingLanguageTrait]:
    """
    Parse model output as JSON and validate that it matches the required schema.
    """
    cleaned_text = strip_json_markdown(raw_text)

    try:
        data: Any = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array")

    if len(data) != 3:
        raise ValueError("Expected exactly 3 programming languages")

    try:
        return [ProgrammingLanguageTrait.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"JSON structure is invalid: {exc}") from exc


def main() -> None:
    model = os.getenv("LITELLM_MODEL", "anthropic/claude-haiku-4-5")

    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You return only valid JSON. "
                    "Do not include markdown, explanations, or extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    "List 3 programming languages with one-word traits. "
                    "Return only a JSON array. "
                    "Each item must contain exactly these keys: name, trait. "
                    "The trait value must be exactly one word."
                ),
            },
        ],
        max_tokens=256,
    )

    raw_text = response.choices[0].message.content or ""
    languages = parse_and_validate_languages(raw_text)

    result = [item.model_dump() for item in languages]

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
