import json

import anthropic
from jsonschema import validate
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

client = anthropic.AsyncAnthropic()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
)
async def get_structured_data(text: str, schema: dict) -> dict:
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze the following customer support message.\n"
                    "Return ONLY valid JSON. Do not include markdown.\n"
                    "The JSON must match this schema:\n"
                    f"{json.dumps(schema)}\n\n"
                    "Customer message:\n"
                    f"{text}"
                ),
            }
        ],
    )

    
    raw_text = response.content[0].text.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text.removeprefix("```").removesuffix("```").strip()

    parsed = json.loads(raw_text)
    validate(instance=parsed, schema=schema)
    return parsed