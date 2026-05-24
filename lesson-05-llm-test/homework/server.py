import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anthropic_api_structured import get_structured_data

app = FastAPI(title="Smart Support Router")

_DATA_DIR = Path(__file__).parent / "data"
_ROUTING_SCHEMA_PATH = _DATA_DIR / "routing_schema.json"


def load_routing_schema() -> dict:
    if not _ROUTING_SCHEMA_PATH.is_file():
        raise RuntimeError(f"Routing schema not found: {_ROUTING_SCHEMA_PATH}")

    return json.loads(_ROUTING_SCHEMA_PATH.read_text(encoding="utf-8"))


class RouteTicketRequest(BaseModel):
    message: str = Field(
        ...,
        description="Unstructured customer support message to analyze and route",
    )


@app.get("/routing-schema")
def get_routing_schema() -> dict:
    return load_routing_schema()


@app.post("/route-ticket")
async def route_ticket(body: RouteTicketRequest) -> dict:
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    schema = load_routing_schema()

    try:
        return await get_structured_data(body.message, schema)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e