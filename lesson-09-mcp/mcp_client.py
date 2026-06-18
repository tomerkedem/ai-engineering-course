"""
MCP CLI chat: natural-language questions answered via a single MCP server from mcp_servers.yaml.

Connects via streamable HTTP (hosted Supabase MCP or a local server.py instance).

Environment:
  ANTHROPIC_API_KEY         Required for the chat agent
  ANTHROPIC_MODEL           Optional (default: claude-haiku-4-5-20251001)
  SUPABASE_PROJECT_REF      Used in supabase server URL template (mcp_servers.yaml)
  SUPABASE_MCP_URL          Overrides supabase server URL when set
  SUPABASE_ACCESS_TOKEN     Bearer token for supabase when configured in mcp_servers.yaml

Run:
  python supabase_chat.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

CONFIG_PATH = Path(__file__).with_name("mcp_servers.yaml")
MAX_AGENT_ROUNDS = 15


SYSTEM_PROMPT = """You are a helpful assistant connected via MCP.

Use the available MCP tools to answer the user.
When you have enough information, respond with a final plain-text answer (no tool calls)."""


@dataclass
class ServerConfig:
    """Resolved connection settings for a single MCP server from mcp_servers.yaml.

    Attributes:
        name: Server key from the YAML ``servers`` map (e.g. ``"supabase"``).
        url: Fully resolved MCP endpoint URL, including any ``${VAR}`` substitutions.
        access_token: Bearer token for authenticated servers, or ``None`` when auth is not configured.
        description: Optional human-readable label from the YAML entry.
    """

    name: str
    url: str
    access_token: str | None = None
    description: str = ""


def resolve_env_vars(text: str) -> str:
    """Replace ``${VAR_NAME}`` placeholders in *text* with ``os.environ`` values.

    Args:
        text: String that may contain ``${VAR_NAME}`` placeholders to expand.

    Returns:
        *text* with every ``${VAR_NAME}`` replaced by the matching environment value,
        or an empty string when the variable is unset.
    """

    def replacer(match: re.Match[str]) -> str:
        """Look up the environment variable captured by a ``${VAR}`` regex match.

        Args:
            match: Regex match whose group 1 is the variable name inside ``${...}``.

        Returns:
            The value of that environment variable, or ``""`` if it is not set.
        """
        return os.environ.get(match.group(1), "")

    return re.sub(r"\$\{([^}]+)\}", replacer, text)


def load_server_config(config_path: Path) -> ServerConfig:
    """Load and validate the first server entry from *config_path* into a ``ServerConfig``.

    Args:
        config_path: Path to ``mcp_servers.yaml`` (or equivalent) containing a ``servers`` map.

    Returns:
        A ``ServerConfig`` with resolved URL, optional access token, and metadata.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        ValueError: If the file has no servers, an invalid entry, empty URL, or missing auth token.
        yaml.YAMLError: If the file is not valid YAML (propagated from ``yaml.safe_load``).
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"MCP config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    servers_raw = raw.get("servers")
    if not isinstance(servers_raw, dict) or not servers_raw:
        raise ValueError(f"No servers defined in {config_path}")

    name, entry = next(iter(servers_raw.items()))
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid server entry for {name!r}")

    url_env = (entry.get("url_env") or "").strip() or None
    url = (os.environ.get(url_env, "").strip() if url_env else "") or entry.get("url", "").strip()
    url = resolve_env_vars(url)
    if not url:
        raise ValueError(f"Server {name!r}: url is empty (check config and env vars)")

    auth_token_env = (entry.get("auth_token_env") or "").strip() or None
    access_token: str | None = None
    if auth_token_env:
        access_token = os.environ.get(auth_token_env, "").strip()
        if not access_token:
            raise ValueError(
                f"Set {auth_token_env} "
                "(https://supabase.com/dashboard/account/tokens for Supabase PAT)."
            )

    return ServerConfig(
        name=name,
        url=url,
        access_token=access_token,
        description=(entry.get("description") or "").strip(),
    )


def looks_like_project_api_key(token: str) -> bool:
    """Detect whether *token* is a Supabase project API key rather than an account PAT.

    Args:
        token: Credential string from the environment or user input.

    Returns:
        ``True`` if *token* looks like a project secret, publishable, or JWT anon/service key;
        ``False`` otherwise (e.g. a personal access token).
    """
    lowered = token.lower()
    return (
        lowered.startswith("sb_secret_")
        or lowered.startswith("sb_publishable_")
        or lowered.startswith("eyj")  # JWT anon/service keys
    )


def mcp_tools_to_anthropic(tools_result: types.ListToolsResult) -> list[dict[str, Any]]:
    """Convert MCP ``ListToolsResult`` tools into Anthropic ``tools`` API format.

    Args:
        tools_result: Tool list returned by ``ClientSession.list_tools()``.

    Returns:
        A list of dicts with ``name``, ``description``, and ``input_schema`` keys suitable
        for the Anthropic Messages API ``tools`` parameter.
    """
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools_result.tools:
        schema = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {"type": "object", "properties": {}}
        )
        if "type" not in schema:
            schema = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        anthropic_tools.append(
            {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "input_schema": schema,
            }
        )
    return anthropic_tools


def content_to_text(content: list[Any]) -> str:
    """Extract and join text from MCP or Anthropic content blocks.

    Args:
        content: Sequence of content blocks (Pydantic models or dicts with a ``text`` field).

    Returns:
        All ``text`` fields joined with newlines, or ``""`` when *content* is empty.
    """
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts)


def format_tool_result(result: types.CallToolResult) -> str:
    """Format an MCP ``CallToolResult`` as a plain-text string for the agent.

    Args:
        result: Outcome of ``ClientSession.call_tool()``, including error flag and content blocks.

    Returns:
        Tool output text, prefixed with ``"Error: "`` when ``result.isError`` is ``True``.
    """
    if result.isError:
        return f"Error: {content_to_text(result.content)}"
    return content_to_text(result.content)


def content_blocks_to_list(content: Any) -> list[dict[str, Any]]:
    """Normalize Anthropic response content blocks into plain dicts.

    Args:
        content: Anthropic message ``content`` (Pydantic models, dicts, or other block objects).

    Returns:
        A list of serializable dicts representing each block (e.g. ``text`` or ``tool_use``).
    """
    if not content:
        return []
    blocks: list[dict[str, Any]] = []
    for block in content:
        if hasattr(block, "model_dump"):
            blocks.append(block.model_dump())
        elif isinstance(block, dict):
            blocks.append(block)
        else:
            blocks.append(
                {
                    "type": getattr(block, "type", "text"),
                    "text": getattr(block, "text", str(block)),
                }
            )
    return blocks


def text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    """Join stripped text from ``text``-type content blocks into a single answer."""
    parts = [
        block.get("text", "").strip()
        for block in blocks
        if block.get("type") == "text"
    ]
    return "\n".join(parts).strip() or "(No text in response.)"


def parse_tool_input(inp: Any) -> dict[str, Any]:
    """Normalize a tool-use ``input`` field to a dict."""
    if not isinstance(inp, str):
        return inp or {}
    try:
        return json.loads(inp)
    except json.JSONDecodeError:
        return {}


async def execute_tool_uses(
    session: ClientSession,
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run each ``tool_use`` block via MCP and return matching ``tool_result`` blocks."""
    tool_results: list[dict[str, Any]] = []
    for block in content_blocks:
        if block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        inp = parse_tool_input(block.get("input"))
        # Call the tool
        result = await session.call_tool(name, inp)
        # Format the result
        result_str = format_tool_result(result)
        # Print the result
        preview = result_str[:400] + ("..." if len(result_str) > 400 else "")
        print(f"tool result: {preview}")
        # Add the tool result to the list
        tool_results.append({"type": "tool_result", "tool_use_id": block.get("id", ""), "content": result_str})
    # Return the tool results
    return tool_results

async def run_agent_loop(session: ClientSession, anthropic_tools: list[dict[str, Any]], user_query: str, history: list[dict[str, Any]], system_prompt: str) -> str:
    """Run the Claude tool-use loop until a final answer or ``MAX_AGENT_ROUNDS`` is reached.

    Args:
        session: Active MCP client session used to invoke tools on the server.
        anthropic_tools: Tool definitions in Anthropic API format from ``mcp_tools_to_anthropic``.
        user_query: The current user message to answer.
        history: Prior user/assistant turns; updated in place with this exchange on success.
        system_prompt: System instructions passed to the Anthropic Messages API.

    Returns:
        Claude's final plain-text answer, or a fallback message if the round limit is hit.
    """
    # Create a client for the Anthropic API
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # Create the messages list
    messages = history + [{"role": "user", "content": user_query}]

    for round_num in range(1, MAX_AGENT_ROUNDS + 1):
        print("\n--------------------------------")
        print(f"STEP {round_num}")
        print("--------------------------------\n")
        # Call the Anthropic API
        response = await client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=anthropic_tools,
        )
        # Convert the response to a list of content blocks
        content_blocks = content_blocks_to_list(response.content)
        # Check if the response is a final answer
        if response.stop_reason == "end_turn":
            # Extract the text from the content blocks
            answer = text_from_blocks(content_blocks)
            # Add the user query and assistant response to the history
            history.extend([{"role": "user", "content": user_query}, {"role": "assistant", "content": answer}])
            # Return the answer
            return answer

        thinking = text_from_blocks(content_blocks)
        if thinking and thinking != "(No text in response.)":
            print(f"Thinking: {thinking}")

        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            inp = parse_tool_input(block.get("input"))
            print(f"Tool use: {name}")
            print(f"Params: {inp}")

        # Execute the tool uses
        tool_result_blocks = await execute_tool_uses(session, content_blocks)
        # Add the content blocks and tool result blocks to the messages list
        messages.append({"role": "assistant", "content": content_blocks})
        messages.append({"role": "user", "content": tool_result_blocks})

    # Return a fallback message if the round limit is hit
    return "(Reached max tool rounds without a final answer.)"


def open_transport(config: ServerConfig):
    """Return an async context manager for the MCP streamable HTTP transport.

    Args:
        config: Server URL and optional bearer token for the ``Authorization`` header.

    Returns:
        An async context manager yielding ``(read_stream, write_stream, get_session_id)``
        for use with ``ClientSession``.
    """
    headers: dict[str, str] = {}
    if config.access_token:
        headers["Authorization"] = f"Bearer {config.access_token}"
    return streamablehttp_client(
        url=config.url,
        headers=headers or None,
        sse_read_timeout=300,
    )


async def run_chat(config: ServerConfig) -> None:
    """Connect to the MCP server and run an interactive REPL until the user exits.

    Args:
        config: Resolved MCP server connection settings.

    Returns:
        ``None``. Runs until the user types ``exit``/``quit``/``q`` or interrupts input.
    """
    auth_mode = "pat" if config.access_token else "none"
    desc = f" — {config.description}" if config.description else ""
    print(f"MCP server: {config.name}{desc}")
    print(f"URL: {config.url}")
    print(f"Auth mode: {auth_mode}")
    system_prompt = SYSTEM_PROMPT

    # Open the MCP transport
    async with open_transport(config) as (read_stream, write_stream, _get_session_id):
        # Create a client session
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            # List the tools
            tools_result = await session.list_tools()
            # Convert the tools to Anthropic API format
            anthropic_tools = mcp_tools_to_anthropic(tools_result)
            # Print the tool names
            tool_names = [t["name"] for t in anthropic_tools]
            print(f"Connected. MCP tools: {', '.join(tool_names)}")
            print("Type a question or exit/quit.\n")

            history: list[dict[str, Any]] = []

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye.")
                    break

                if not user_input:
                    continue
                lowered = user_input.lower()
                if lowered in {"exit", "quit", "q"}:
                    print("Bye.")
                    break
                print("Thinking...")
                try:
                    # Run the agent loop
                    answer = await run_agent_loop(
                        session,
                        anthropic_tools,
                        user_input,
                        history,
                        system_prompt,
                    )
                except Exception as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    continue
                print(f"\nAssistant:\n{answer}\n")


def _print_auth_failure() -> None:
    """Print guidance when MCP authentication fails with 401/403.

    Returns:
        ``None``. Writes a short help message to ``sys.stderr``.
    """
    print(
        "Authentication failed (401/403). Use a Supabase account personal access token "
        "(https://supabase.com/dashboard/account/tokens), not a project API key.",
        file=sys.stderr,
    )


def _extract_http_status_error(exc: BaseException) -> httpx.HTTPStatusError | None:
    """Return the first ``httpx.HTTPStatusError`` nested inside *exc*, if any.

    Args:
        exc: Exception raised during MCP transport or HTTP calls, possibly wrapped in
            a ``BaseExceptionGroup``.

    Returns:
        The first matching ``httpx.HTTPStatusError``, or ``None`` if none is found.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            found = _extract_http_status_error(sub)
            if found is not None:
                return found
    return None


def main() -> None:
    """Entry point: load env, validate config, and start the chat session.

    Returns:
        ``None``. Exits the process with code 1 on missing API key, config errors, or
        HTTP auth failures; otherwise runs until the chat loop ends.
    """
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY for the chat agent.", file=sys.stderr)
        sys.exit(1)

    # Load the MCP server config
    config = load_server_config(CONFIG_PATH)

    # Run the chat session
    asyncio.run(run_chat(config))

if __name__ == "__main__":
    main()
