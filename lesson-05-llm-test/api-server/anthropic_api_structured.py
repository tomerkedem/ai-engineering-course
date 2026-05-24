import asyncio
import json
from pathlib import Path

import anthropic
from jsonschema import validate
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

client = anthropic.AsyncAnthropic()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
)
async def get_structured_data(text, schema):
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Extract the key information from this interaction between agent and customer: {text}",
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
    )

    parsed = json.loads(response.content[0].text)
    validate(instance=parsed, schema=schema)
    return parsed

if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"

    text = (data_dir / "call1.txt").read_text(encoding="utf-8")
    schema = json.loads((data_dir / "call_summary_schema.json").read_text(encoding="utf-8"))

    structured_data = asyncio.run(get_structured_data(text, schema))
    print(json.dumps(structured_data, indent=4))
