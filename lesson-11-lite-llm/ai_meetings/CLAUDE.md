# CLAUDE.md

## What this project is

A real multi-agent meeting system for Claude Code: each persona runs as an
**independent subagent with an isolated context and its own model**, so
positions genuinely collide instead of one model staging a debate. Meetings
produce a **decision brief** — that document is the product, not the chat
transcript. There is no application code — the markdown protocols and
personas ARE the product, so treat them with the same care as code.

## Structure

- `.claude/commands/run-meeting.md` — `/run-meeting pm,critic <subject>`:
  frame with the user → Round 1: all personas spawned in parallel, isolated,
  returning grounded positions → synthesis + user steering → Round 2:
  adversarial wave only where positions conflict → decision brief.
- `.claude/commands/review-spec.md` — `/review-spec <path>`: same engine for
  spec review; verdict + severity-ranked gap table; findings found
  independently by 2+ agents are marked verified.
- `.claude/commands/end-meeting.md` — finalizes the brief (status טיוטה →
  מאושר), files action items to `meetings/action-items.md`, appends decisions
  to `meetings/decisions.md` (the institutional memory — meetings read it
  instead of all past briefs).
- `.claude/commands/new-agent.md` — scaffolds a new persona agent.
- `.claude/agents/*.md` — the persona library, registered as real Claude Code
  subagents. Kebab-case `name` (it's the subagent identifier), `model` is the
  model the persona actually runs on. Each is one sharp lens: how-you-think /
  always-ask / push-back-on.
- `meetings/` — decision briefs, created on demand, named
  `YYYY-MM-DD-<slug>.md`.
- `specs/notifications-spec.md` — the example spec the README walkthrough
  runs on. It contains **intentional gaps** (missing states, undefined
  "real-time", no privacy handling) as demo material for meetings — do not
  helpfully "fix" them.
- `README.md` — the user-facing guide, in Hebrew. Keep it in sync when
  commands or personas change.

## Conventions

- **Anti-generic rule:** agents must ground every claim in a quote/section of
  the material and are forbidden from advice that would be true for any
  project. This rule is the core of the product's value — never weaken it.
- **Personas:** one opinionated lens each, under ~15 lines, exact structure
  from `new-agent.md`. Read an existing persona before writing a new one.
  A persona that agrees with everyone is dead weight — don't add it.
- **Token discipline:** two agent waves per meeting by default; Round 2 only
  for genuinely conflicting positions. Never re-spawn an agent that has
  nothing to defend.
- **Language:** meeting output is entirely in Hebrew — briefs, headers,
  summaries, action items. Only file names stay in English/Latin characters.
  Command and persona files (the system itself) stay in English.
- **Lean by policy:** no files that don't earn their place — no speculative
  docs, no superseded drafts. When something is replaced, delete it in the
  same change.
- **No `template_workflow/`:** this project began as a copy of
  `agent-manager-template`; all references to its `template_workflow/` folder
  were removed because that folder doesn't exist here. Never reintroduce them.

## Working style — senior developer, not order-taker

- **Understand before changing.** Read every file you're about to edit, in
  full — these files are short, there's no excuse not to.
- **Challenge the requirement first.** Ask what decision or problem it serves.
  If there's a simpler way or it conflicts with the existing design, say so
  *before* implementing, with a concrete alternative.
- **State trade-offs out loud.** When you pick an approach, give one line on
  what you rejected and why.
- **Smallest change that works.** No speculative features, no abstractions
  for futures nobody asked for.
- **Verify your own work.** After editing a command or persona, check that
  every path, agent name, and cross-reference in it actually exists.
- **Report honestly.** What changed, what you skipped, what's risky. A noted
  doubt is worth more than fake confidence — same rule as in the meetings.
