from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from summarize import setup_env, summarize

app = FastAPI()


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)


class SummarizeResponse(BaseModel):
    summary: str


@app.on_event("startup")
async def startup() -> None:
    setup_env()


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    try:
        print(f"\nSummarizing text: {request.text.strip()}")
        summary = await summarize(request.text.strip())
        print(f"\nSummary: {summary}")
    except ClientError as exc:
        raise HTTPException(
            status_code=502, detail=f"Bedrock request failed: {exc}"
        ) from exc

    return SummarizeResponse(summary=summary)


def main() -> None:
    uvicorn.run("server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
