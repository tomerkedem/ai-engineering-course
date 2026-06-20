"""
Policy MCP server exercise.

Run with:
    python policy_server.py

Connect at:
    http://127.0.0.1:8766/mcp

Tools:
    get_policies - list available policy names
    get_policy   - read one policy by name
"""

import functools
import logging
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
POLICIES_DIR = BASE_DIR / "policies"

mcp = FastMCP("policy-mcp-exercise", host="127.0.0.1", port=8766)


def log_tool_call(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(fn.__name__)
        return fn(*args, **kwargs)

    return wrapper


def normalize_policy_name(policy_name: str) -> str:
    """Normalize and validate a policy name before reading a file."""
    name = (policy_name or "").strip()

    if name.endswith(".txt"):
        name = name[:-4]

    if not name:
        raise ValueError("'policy_name' is required")

    # Allow only simple policy names such as password_recovery_policy.
    # This prevents path traversal such as ../secrets.
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name):
        raise ValueError("Invalid policy name")

    return name


@mcp.tool()
@log_tool_call
def get_policies() -> str:
    """
    List available company policies.

    Use this tool when the user asks what policies are available,
    or when you need to choose the correct policy before reading it.
    """
    if not POLICIES_DIR.exists():
        return "No policies folder was found."

    policy_names = sorted(path.stem for path in POLICIES_DIR.glob("*.txt") if path.is_file())

    if not policy_names:
        return "No policies are available."

    return "Available policies:\n" + "\n".join(f"- {name}" for name in policy_names)


@mcp.tool()
@log_tool_call
def get_policy(policy_name: str) -> str:
    """
    Read a specific company policy by policy name.

    Args:
        policy_name: The policy name to read, for example 'password_recovery_policy'.
    """
    name = normalize_policy_name(policy_name)
    policy_path = (POLICIES_DIR / f"{name}.txt").resolve()

    # Security check: make sure the resolved file is still inside the policies folder.
    policies_root = POLICIES_DIR.resolve()
    if policies_root not in policy_path.parents:
        raise ValueError("Access outside the policies folder is not allowed")

    if not policy_path.exists() or not policy_path.is_file():
        return f"Policy not found: {name}"

    content = policy_path.read_text(encoding="utf-8").strip()
    if not content:
        return f"Policy is empty: {name}"

    return f"Policy: {name}\n\n{content}"


if __name__ == "__main__":
    logger.info("Policy MCP server starting at http://127.0.0.1:8766/mcp")
    mcp.run(transport="streamable-http")
