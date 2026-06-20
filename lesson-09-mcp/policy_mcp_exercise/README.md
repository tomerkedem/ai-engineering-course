# Policy MCP Exercise

This exercise creates a local MCP server that exposes company policy files as tools.

## Files

```text
policy_mcp_exercise/
├── policy_server.py
├── mcp_servers.yaml
└── policies/
    ├── data_retention_policy.txt
    ├── remote_work_security_policy.txt
    ├── password_recovery_policy.txt
    └── expense_reimbursement_policy.txt
```

## Run

1. Install the course requirements if needed.
2. Start the policy server:

```bash
python policy_server.py
```

3. In a second terminal, run the existing MCP client:

```bash
python mcp_client.py
```

Make sure the client uses the included `mcp_servers.yaml` or copy the `policy` server block into your existing `mcp_servers.yaml`.

## Test questions

```text
What policies are available?
How do I recover my password?
How can I work at home in a secure way?
What is our expense reimbursement policy?
```
