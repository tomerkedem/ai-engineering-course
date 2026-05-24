import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anthropic_api_structured import get_structured_data
import uvicorn

app = FastAPI(title="Call structured extraction")

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_SCHEMA_PATH = _DATA_DIR / "call_summary_schema.json"


def _load_default_schema() -> dict:
    if not _DEFAULT_SCHEMA_PATH.is_file():
        raise RuntimeError(f"Default schema not found: {_DEFAULT_SCHEMA_PATH}")
    return json.loads(_DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))


class StructuredExtractRequest(BaseModel):
    text: str = Field(..., description="Transcript or text of the agent–customer interaction")
    output_schema: dict | None = Field(
        default=None,
        alias="schema",
        description="Optional JSON Schema for output; defaults to call_summary_schema.json",
    )

    model_config = {"populate_by_name": True}


@app.get("/schema")
def get_default_schema() -> dict:
    """Return the default JSON Schema used for structured extraction."""
    return _load_default_schema()


@app.post("/structured")
async def structured_extract(body: StructuredExtractRequest) -> dict:
    """
    Run `get_structured_data` on the given text and return the structured object
    (parsed JSON matching the schema).
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    schema = body.output_schema if body.output_schema is not None else _load_default_schema()
    try:
        return await get_structured_data(body.text, schema)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
