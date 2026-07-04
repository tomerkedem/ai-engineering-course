"""
Tool calling example: agent loop with generic tool dispatch via LiteLLM.

1. Send the user question with tool definitions
2. While the model requests tools, run them locally and append results
3. Stop when the model returns a natural-language answer

Requires: pip install litellm httpx
Env: ANTHROPIC_API_KEY

Weather data from Open-Meteo (https://open-meteo.com) — free, no API key.
"""
import json
from collections.abc import Callable

import httpx
from litellm import completion

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WMO_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]

ToolFn = Callable[..., str]
TOOL_REGISTRY: dict[str, ToolFn] = {}


def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Look up current weather via Open-Meteo geocoding + forecast APIs."""
    try:
        with httpx.Client(timeout=10.0) as client:
            geo = client.get(
                GEO_URL,
                params={"name": location, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            results = geo.json().get("results")
            if not results:
                return json.dumps({"error": f"Location not found: {location}"})

            place = results[0]
            display_name = ", ".join(
                part
                for part in (place.get("name"), place.get("admin1"), place.get("country"))
                if part
            )

            weather = client.get(
                WEATHER_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "temperature_unit": unit,
                    "wind_speed_unit": "mph" if unit == "fahrenheit" else "kmh",
                },
            )
            weather.raise_for_status()
            current = weather.json()["current"]
            code = current["weather_code"]

            return json.dumps(
                {
                    "location": display_name,
                    "temperature": current["temperature_2m"],
                    "unit": unit,
                    "condition": WMO_DESCRIPTIONS.get(code, f"weather code {code}"),
                    "wind_speed": current["wind_speed_10m"],
                    "wind_speed_unit": "mph" if unit == "fahrenheit" else "km/h",
                }
            )
    except httpx.HTTPError as exc:
        return json.dumps({"error": f"Weather API request failed: {exc}"})


TOOL_REGISTRY["get_weather"] = get_weather


def execute_tool_call(name: str, arguments: str) -> str:
    """Run a tool by name with JSON-encoded arguments."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid tool arguments: {exc}"})

    try:
        return fn(**args)
    except TypeError as exc:
        return json.dumps({"error": f"Tool argument mismatch for {name}: {exc}"})


def run_agent_loop(
    messages: list[dict],
    *,
    model: str = "anthropic/claude-haiku-4-5",
    tools: list[dict] | None = None,
    max_tokens: int = 256,
    max_turns: int = 10,
) -> str:
    """Call the model in a loop until it stops requesting tools."""
    tools = tools or TOOLS

    for _ in range(max_turns):
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            return assistant_message.content or ""

        messages.append(assistant_message)

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            result = execute_tool_call(name, args)
            print(f"Tool call: {name}({args}) -> {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result,
                }
            )

    raise RuntimeError(f"Agent loop exceeded max_turns ({max_turns})")


def main():
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"},
    ]

    answer = run_agent_loop(messages)
    print("\nFinal answer:", answer)


if __name__ == "__main__":
    main()
