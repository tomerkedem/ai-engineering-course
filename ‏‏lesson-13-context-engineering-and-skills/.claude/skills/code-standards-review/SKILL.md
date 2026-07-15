---
name: code-standards-review
description: Review a source-code file against the team's coding standards (naming, function length, exception handling, documentation, input validation). Use when the user asks to review code or check it against the coding standards.
---

# code-standards-review

Conduct a code review of a source-code file against the standards below. Suggest
fixes first, and implement them only after the user approves.

## Standards

Check the reviewed file against every rule:

1. **Naming** — all functions and variables must have meaningful, easy-to-read
   names. Flag cryptic, abbreviated, or misleading names.
2. **Function length** — each function should be **shorter than 20 lines**. For
   any longer function, suggest splitting it into smaller, well-named functions.
3. **Exceptions** — exception handling must **always print the stack trace**
   (e.g. `traceback.print_exc()` / logging with `exc_info=True`). Flag any
   `except` block that swallows the error or hides the trace.
4. **Documentation** — **every function must be documented** (docstring
   describing purpose, parameters, and return value). Flag undocumented functions.
5. **Input parameters** — **always validate input parameters**. Flag functions
   that use parameters without checking type/range/None/empty as appropriate.

## Review process

Follow these steps in order.

### 1. Identify files to review
- Look at the source files in the project and **suggest which file(s) to review**,
  briefly explaining why. If the user already named a file, use it.

### 2. Review and suggest fixes
- Go over the selected file and check it against all five standards.
- Present the suggested fixes as a **table** with columns:

  | # | Standard | Location (function / line) | Issue | Suggested fix |
  |---|----------|----------------------------|-------|---------------|

- Do **not** modify any code yet.

### 3. Approval gate (MANDATORY)
- Wait for the user's decision.
- **If the user approves** → implement the fixes (step 4).
- **If the user disapproves** → suggest a **new plan** and wait for additional
  approval. Do not implement anything until approved.

### 4. Implement approved fixes
- Implement only the approved fixes, making the smallest safe changes.
- Keep behavior intact unless a change was explicitly approved.

### 5. Print a summary
- At the end, print a summary of the review, including **how many fixes were
  done** (and, optionally, a breakdown by standard).
