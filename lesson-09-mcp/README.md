# MCP Examples

This project demonstrates the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) with two examples:

1. **Local MCP server** (`server.py`) — a small FastMCP server with calculator, time, and weather tools.
2. **Interactive chat client** (`mcp_client.py`) — connects to an MCP server over streamable HTTP and uses Claude to answer questions via tool calls.

The client reads server connection settings from `mcp_servers.yaml`. By default it is configured for the hosted [Supabase MCP](https://supabase.com/docs/guides/getting-started/mcp); you can switch to the local server instead.

## Prerequisites

- Python 3.11+ (recommended)
- An [Anthropic API key](https://console.anthropic.com/) (required for the chat client)
- For the Supabase example: a [Supabase personal access token (PAT)](https://supabase.com/dashboard/account/tokens) and your project reference ID

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure `.env`

Create a `.env` file in the project root (same folder as `mcp_client.py`). The client loads it automatically via `python-dotenv`.

### Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key. Used by `mcp_client.py` to run the Claude agent. |

### Supabase MCP (default config)

These are required when using the `supabase` entry in `mcp_servers.yaml` (the default):

| Variable | Description |
|----------|-------------|
| `SUPABASE_ACCESS_TOKEN` | Supabase **account** personal access token (PAT), not a project API key. Create one at [Account → Access Tokens](https://supabase.com/dashboard/account/tokens). |
| `SUPABASE_PROJECT_REF` | Your Supabase project reference ID (the short ID in the dashboard URL, e.g. `abcdefghijklmnop`). Used to build the MCP URL in `mcp_servers.yaml`. |

### Optional

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_MODEL` | Claude model for the chat agent. Default: `claude-haiku-4-5-20251001`. |
| `SUPABASE_MCP_URL` | If set, overrides the Supabase MCP URL from `mcp_servers.yaml`. |

### Example `.env`

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
SUPABASE_ACCESS_TOKEN=sbp_...
SUPABASE_PROJECT_REF=your-project-ref
```

Do not commit `.env` to version control — it contains secrets.

## MCP server configuration

Server endpoints are defined in `mcp_servers.yaml`. The client uses the **first** server entry in the file.

### Option A — Supabase MCP (default)

Leave the file as shipped: the `supabase` server is active and `local` is commented out. Ensure `SUPABASE_ACCESS_TOKEN` and `SUPABASE_PROJECT_REF` are set in `.env`.

### Option B — Local MCP server

1. Comment out the `supabase` block and uncomment the `local` block in `mcp_servers.yaml`:

```yaml
servers:
  local:
    url: http://127.0.0.1:8765/mcp
    description: Local MCP server (run server.py first)
  # supabase:
  #   ...
```

2. Only `ANTHROPIC_API_KEY` is required in `.env` for this mode (no Supabase variables).

## Run the examples

Activate `.venv` before running any command.

### 1. Local MCP server

Starts a streamable HTTP MCP server on `http://127.0.0.1:8765/mcp` with tools: `calculator`, `get_time`, `get_weather`.

```powershell
python server.py
```

Leave this terminal running. The server does not need `.env` variables.

### 2. Interactive chat client

In a **second** terminal (with the MCP server running if you use the local config):

```powershell
python mcp_client.py
```

You will see the connected MCP server, available tools, and a prompt:

```
You: What is 15 * 23?
```

Type `exit`, `quit`, or `q` to leave.

**Example questions (local server):**

- `What time is it in UTC?`
- `What is (12 + 8) * 3?`
- `What's the weather in London?`

**Example questions (Supabase MCP):**

- `List tables in my database`
- `How many rows are in the users table?`

### VS Code debugging

Use the **Python Debugger: Current File** launch configuration in `.vscode/launch.json`. It runs the active file with the workspace `.venv` interpreter.

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `Set ANTHROPIC_API_KEY for the chat agent.` | Add `ANTHROPIC_API_KEY` to `.env`. |
| `Set SUPABASE_ACCESS_TOKEN ...` | Add a Supabase PAT (not `sb_secret_` or JWT project keys). |
| `Authentication failed (401/403)` | Regenerate a PAT at [supabase.com/dashboard/account/tokens](https://supabase.com/dashboard/account/tokens). |
| Client cannot connect (local) | Start `server.py` first and confirm `mcp_servers.yaml` points to `http://127.0.0.1:8765/mcp`. |
| `url is empty` | Set `SUPABASE_PROJECT_REF` or uncomment/configure the correct server in `mcp_servers.yaml`. |

## Project layout

```
mcp/
├── .env                 # Secrets (create locally, not committed)
├── mcp_servers.yaml     # MCP server URLs and auth settings
├── mcp_client.py        # Claude + MCP interactive chat
├── server.py            # Local FastMCP example server
└── requirements.txt     # Python dependencies
```
