"""
MCP server with streamable HTTP transport.

Uses FastMCP. Exposes tools: calculator, get_time, get_weather.
Run with: python server.py
Connect at: http://127.0.0.1:8765/mcp
"""

import functools
import logging
from datetime import datetime, timezone

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-local-example", host="127.0.0.1", port=8765)


def log_tool_call(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(fn.__name__)
        return fn(*args, **kwargs)

    return wrapper


@mcp.tool()
@log_tool_call
def calculator(expression: str) -> str:
    """
    Evaluate a math expression. Supports +, -, *, /, **, ( ). Example: (2 + 3) * 4

    Args:
        expression: Math expression to evaluate, e.g. '2 + 3 * 4'
    """
    expr = (expression or "").strip()
    if not expr:
        raise ValueError("'expression' is required")
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expr):
        raise ValueError("Only numbers and + - * / ( ) allowed")
    result = eval(expr)
    return f"Result: {result}"


@mcp.tool()
@log_tool_call
def get_time() -> str:
    """
    Get the current date and time in UTC.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


WMO_WEATHER = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


@mcp.tool()
@log_tool_call
def get_weather(city: str) -> str:
    """
    Get current weather for a city.

    Args:
        city: City name, e.g. 'London' or 'New York'
    """
    name = (city or "").strip()
    if not name:
        raise ValueError("'city' is required")

    with httpx.Client(timeout=10.0) as client:
        geo = client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "en", "format": "json"},
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            raise ValueError(f"City not found: {name}")

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        label = ", ".join(
            part for part in (place.get("name"), place.get("country")) if part
        )

        forecast = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        current = forecast.json()["current"]

    temp = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    wind = current["wind_speed_10m"]
    condition = WMO_WEATHER.get(current["weather_code"], "Unknown")
    return (
        f"Weather in {label}: {condition}, {temp}°C, "
        f"humidity {humidity}%, wind {wind} km/h"
    )


if __name__ == "__main__":
    logger.info("MCP server starting at http://127.0.0.1:8765/mcp")
    mcp.run(transport="streamable-http")
