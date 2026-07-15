---
name: solid-code-review
description: Review a specific source-code file with strong emphasis on SOLID principles. Use when the user asks to review, critique, or assess a source file for design quality, defects, or SOLID compliance.
---

# solid-code-review

Review a single source-code file for correctness and design quality, with strong
emphasis on the SOLID principles. Do not modify any code before the findings are
approved. Report findings, then wait for approval before implementing fixes.

## Rules

### Scope and safety
- Review the code **without modifying it** before approval.
- Inspect the reviewed file together with its **direct dependencies and callers**.
- Preserve existing behavior unless a behavior change is explicitly approved.
- Follow the conventions and architecture already used in the repository.
- Do **not** perform broad refactoring outside the reviewed scope.
- Do **not** modify generated files, dependencies, configuration, or public
  contracts without explicit approval.

### What to prioritize
- Prioritize correctness, maintainability, readability, testability, security,
  and performance.
- Do **not** recommend abstractions unless they solve a concrete problem.
- Do **not** over-engineer simple code.
- Distinguish actual defects from optional improvements.

### SOLID
Evaluate compliance with each principle separately:
- **S** — Single Responsibility Principle
- **O** — Open/Closed Principle
- **L** — Liskov Substitution Principle
- **I** — Interface Segregation Principle
- **D** — Dependency Inversion Principle

### Design issues to look for
- Duplicated logic, excessive coupling, hidden dependencies, large functions,
  large classes, unclear naming, and mixed responsibilities.

### Robustness issues to look for
- Missing validation, weak error handling, unsafe assumptions, resource leaks,
  and concurrency risks.

### How to report each finding
- Every finding must have a **clear, explainable title**.
- Every finding must include: **severity, location, explanation, impact, and a
  concrete recommendation**.
- Severity levels: **critical, high, medium, low, suggestion**.
- Include code examples only when they materially clarify the proposed fix.

## Flow

Follow these steps in order.

### 1. Choose the file and scope
- Work only inside the `lesson-13-context-engineering-and-skills` folder, unless
  reviewing a direct dependency requires reading another file.
- If no file is specified, **ask which file should be reviewed**.

### 2. Understand the context
- Examine the selected file and its relevant surrounding context (direct
  dependencies and callers).
- Identify the language, framework, architecture, and repository conventions.
- Check whether related tests already exist.

### 3. Review
- Review the implementation for defects and design issues.
- Evaluate each relevant SOLID principle separately.
- Avoid reporting speculative issues without evidence.
- Group duplicate or closely related findings into a single finding.

### 4. Present findings (do not modify code yet)
Present the findings as a **table** with columns:

| ID | Severity | File & line | Category | Finding | Impact | Recommended fix |
|----|----------|-------------|----------|---------|--------|-----------------|

After the table, provide a **SOLID assessment** table with one row per principle:

| Principle | Status | Evidence | Recommendation |
|-----------|--------|----------|----------------|

Status is one of: **compliant, partially compliant, violated, not applicable**.

Then provide a **prioritized implementation plan**.

### 5. Approval gate (MANDATORY)
- Show the complete list of proposed changes for approval **before modifying any
  code**.
- **Wait for approval.** Do not change anything until the user approves.

### 6. Implement approved findings
- Implement **only the approved** findings.
- Make the **smallest safe changes** required.
- Add or update tests when behavior is affected.

### 7. Validate
- Run the relevant tests, static analysis, formatter, and build checks available
  in the repository.
- Report the **exact files changed** and the validation results.
- Do **not** commit or push unless explicitly requested.
