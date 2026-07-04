---
description: Create a new meeting persona agent in .claude/agents/
argument-hint: name short description of who this persona is
allowed-tools: Read, Write, Glob
---

# New Agent

Create a persona file for the meeting system.

## Input

`$ARGUMENTS`

- **Name** — the first token, **kebab-case, lowercase** (becomes
  `.claude/agents/<name>.md` and the subagent identifier — the name must be
  usable as a Claude Code agent name).
- **Description** — everything after the first space: who this persona is,
  what they own, what perspective they bring.

If either is missing, ask for it. If `.claude/agents/<name>.md` already
exists, show its current content and ask whether to replace, extend, or pick
another name.

## Writing the persona

Write `.claude/agents/<name>.md` in exactly this structure:

```markdown
---
name: <kebab-case-name>
description: <one line — role and what they own; shown when listing personas>
model: <opus for deep-reasoning roles, sonnet otherwise — this is the model
  the persona's agent actually runs on>
---

You are the <Role>. <2-3 sentences: expertise, what they own, their stance.>

**How you think:** <their mental models — e.g. in trade-offs, in user
journeys, in attack surfaces, in unit economics.>

**You always ask:** "<signature question 1>" · "<question 2>" · "<question 3>"

**You push back on:** <the 2-4 things that reliably trigger their skepticism.>
```

Guidelines for a persona that makes meetings better:

- **One sharp lens, not a generalist.** The persona's value is the perspective
  the others lack. If it agrees with everyone, it's dead weight.
- **Opinionated.** Give it defaults, biases, and pet peeves it will voice.
- **Short.** Under ~15 lines. Sharp beats long.
- Match the style and depth of the existing files in `.claude/agents/` — read
  one or two first.

## After creating

Confirm the file path and show usage:
`/run-meeting <name>,<other>,<other> <subject>`
