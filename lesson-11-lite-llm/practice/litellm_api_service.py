"""
Practice 2: API service with LiteLLM.

Goal:
Upgrade an API-based LLM service to use LiteLLM while keeping:
- Async LLM API calls
- Formatted response
- Schema validation
- Retries
- Exception handling
- Fallback to Gemini if Anthropic is unavailable

Requires:
    pip install fastapi uvicorn litellm pydantic

Environment:
    ANTHROPIC_API_KEY
    GEMINI_API_KEY

Run:
    uvicorn litellm_api_service:app --reload
"""

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from litellm import acompletion
from pydantic import BaseModel, Field, ValidationError, field_validator


PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "anthropic/claude-haiku-4-5")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini/gemini-3.5-flash")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


app = FastAPI(
    title="LiteLLM Practice API",
    description="Async API service using LiteLLM with formatted response, validation, retries and fallback.",
    version="1.0.0",
)


class LanguageTraitsRequest(BaseModel):
    prompt: str = Field(
        default="List 3 programming languages with one-word traits.",
        min_length=3,
        description="Prompt to send to the LLM",
    )


class LanguageTrait(BaseModel):
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


class LanguageTraitsResponse(BaseModel):
    model_used: str
    items: list[LanguageTrait]


def strip_json_markdown(text: str) -> str:
    """
    Remove common markdown code fences if the model wraps the JSON response.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def parse_and_validate_response(raw_text: str) -> list[LanguageTrait]:
    """
    Parse the LLM response as JSON and validate it with Pydantic.
    """
    cleaned_text = strip_json_markdown(raw_text)

    try:
        data: Any = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("LLM response must be a JSON array")

    if len(data) != 3:
        raise ValueError("LLM response must contain exactly 3 items")

    try:
        return [LanguageTrait.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"LLM response schema validation failed: {exc}") from exc


async def call_litellm_with_retries(prompt: str) -> LanguageTraitsResponse:
    """
    Call LiteLLM asynchronously with retries and fallback.

    The primary model is Anthropic by default.
    The fallback model is Gemini by default.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await acompletion(
                model=PRIMARY_MODEL,
                fallbacks=[FALLBACK_MODEL],
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
                            f"{prompt}\n\n"
                            "Return only a JSON array with exactly 3 items. "
                            "Each item must contain exactly these keys: name, trait. "
                            "The trait value must be exactly one word."
                        ),
                    },
                ],
                max_tokens=256,
            )

            raw_text = response.choices[0].message.content or ""
            validated_items = parse_and_validate_response(raw_text)

            return LanguageTraitsResponse(
                model_used=response.model or PRIMARY_MODEL,
                items=validated_items,
            )

        except Exception as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

    raise RuntimeError(
        f"LLM request failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/languages/traits", response_model=LanguageTraitsResponse)
async def create_language_traits(
    request: LanguageTraitsRequest,
) -> LanguageTraitsResponse:
    try:
        return await call_litellm_with_retries(request.prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc
