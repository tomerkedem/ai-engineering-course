import anthropic
import json
from pathlib import Path

client = anthropic.Anthropic()


def get_structured_data(text, schema):
    response = client.messages.create(
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

    return json.loads(response.content[0].text)

data_dir = Path(__file__).parent / "data"

text = (data_dir / "call1.txt").read_text(encoding="utf-8")
schema = json.loads((data_dir / "call_summary_schema.json").read_text(encoding="utf-8"))

structured_data = get_structured_data(text, schema)
print(json.dumps(structured_data, indent=4))
