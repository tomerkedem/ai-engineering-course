"""
Minimal Agent Skills–compatible agent (https://agentskills.io).

Progressive disclosure:
  1. Discovery  — load name + description for each skill
  2. Activation — load full SKILL.md when relevant
  3. Execution  — follow instructions; read/run bundled resources on demand
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 8192
MAX_TURNS = 40

# Generated deliverables are collected here (relative to the workspace root).
OUTPUT_DIR_NAME = "output"
# Artifact suffixes swept into the output directory after a run.
ARTIFACT_SUFFIXES = {".pptx", ".pdf", ".docx", ".xlsx", ".pptm", ".key"}

# Client-native + cross-client skill roots (project then user).
# Project skills override user skills on name collision.
SKILL_ROOT_RELATIVE = (
    Path(".cursor") / "skills",
    Path(".agents") / "skills",
    Path(".claude") / "skills",
)


@dataclass
class Skill:
    name: str
    description: str
    location: Path  # absolute path to SKILL.md
    disable_model_invocation: bool = False
    body: str | None = None

    @property
    def directory(self) -> Path:
        return self.location.parent


@dataclass
class SkillRegistry:
    skills: dict[str, Skill] = field(default_factory=dict)

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def catalog_skills(self) -> list[Skill]:
        return [
            s
            for s in self.skills.values()
            if not s.disable_model_invocation
        ]


def _parse_skill_md(path: Path) -> Skill | None:
    """Parse SKILL.md frontmatter + body. Skip if description is missing."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, re.DOTALL)
    if not match:
        print(f"[warn] no frontmatter: {path}", file=sys.stderr)
        return None

    raw_fm, body = match.group(1), match.group(2).strip()
    try:
        fm = yaml.safe_load(raw_fm) or {}
    except yaml.YAMLError as exc:
        # Soft fallback for unquoted colons in description values
        try:
            fixed = re.sub(
                r"^(description:\s*)(.+)$",
                lambda m: m.group(1) + json.dumps(m.group(2).strip()),
                raw_fm,
                flags=re.MULTILINE,
            )
            fm = yaml.safe_load(fixed) or {}
        except yaml.YAMLError:
            print(f"[warn] bad YAML in {path}: {exc}", file=sys.stderr)
            return None

    if not isinstance(fm, dict):
        print(f"[warn] frontmatter is not a mapping: {path}", file=sys.stderr)
        return None

    name = str(fm.get("name") or "").strip()
    description = str(fm.get("description") or "").strip()
    if not description:
        print(f"[warn] missing description, skipping: {path}", file=sys.stderr)
        return None
    if not name:
        name = path.parent.name

    parent = path.parent.name
    if name != parent:
        print(
            f"[warn] name '{name}' != directory '{parent}' ({path})",
            file=sys.stderr,
        )

    return Skill(
        name=name,
        description=description,
        location=path.resolve(),
        disable_model_invocation=bool(fm.get("disable-model-invocation", False)),
        body=body,
    )


def discover_skills(workspace: Path) -> SkillRegistry:
    """Scan project + user skill directories. Project wins on collisions."""
    registry = SkillRegistry()
    home = Path.home()

    roots: list[tuple[str, Path]] = []
    # User first, then project — so project overwrites on same name
    for rel in SKILL_ROOT_RELATIVE:
        roots.append(("user", home / rel))
    for rel in SKILL_ROOT_RELATIVE:
        roots.append(("project", workspace / rel))

    for scope, root in roots:
        if not root.is_dir():
            continue
        for skill_md in sorted(root.glob("*/SKILL.md")):
            skill = _parse_skill_md(skill_md)
            if skill is None:
                continue
            if skill.name in registry.skills:
                print(
                    f"[info] {scope} skill '{skill.name}' overrides "
                    f"{registry.skills[skill.name].location}",
                    file=sys.stderr,
                )
            registry.skills[skill.name] = skill

    return registry


def list_skill_resources(skill: Skill, limit: int = 50) -> list[str]:
    """List bundled files relative to the skill directory (not eagerly read)."""
    skip = {".git", "__pycache__", "node_modules"}
    files: list[str] = []
    for path in sorted(skill.directory.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "SKILL.md":
            continue
        if any(part in skip for part in path.parts):
            continue
        rel = path.relative_to(skill.directory).as_posix()
        files.append(rel)
        if len(files) >= limit:
            break
    return files


def _shell_guidance() -> list[str]:
    """Platform-specific notes for run_command (Windows uses cmd.exe)."""
    if sys.platform == "win32":
        return [
            "Shell: run_command executes via Windows cmd.exe (not PowerShell, not bash).",
            "Prefer write_file over shell redirects when creating or overwriting files.",
            "For Python, prefer .venv\\Scripts\\python.exe (avoid python3 / Unix activate).",
            "Do not use bash syntax: heredocs (<<), cat/ls/grep/rm/source, or .ps1 activate.",
            "cmd equivalents when needed: dir, type, del, copy; chain with &&.",
        ]
    return [
        "Prefer write_file over shell redirects when creating or overwriting files.",
        "Prefer the project .venv python when running Python "
        "(.venv/bin/python -m …).",
    ]


def build_system_prompt(registry: SkillRegistry, workspace: Path) -> str:
    skills = registry.catalog_skills()
    shell_lines = _shell_guidance()
    if not skills:
        return "\n".join(
            [
                "You are a helpful coding agent. No Agent Skills are currently installed.",
                f"Workspace: {workspace.resolve()}",
                "",
                *shell_lines,
            ]
        )

    lines = [
        "You are a helpful coding agent with Agent Skills support.",
        "Skills give specialized instructions for specific tasks.",
        "",
        "When a task matches a skill's description, call activate_skill with that",
        "skill's name BEFORE doing the work. Follow the loaded instructions.",
        "Relative paths inside a skill resolve against that skill's directory.",
        "Use read_file / write_file / run_command for bundled scripts and references as needed.",
        *shell_lines,
        "",
        f"Workspace: {workspace.resolve()}",
        f"Output directory: {(workspace / OUTPUT_DIR_NAME).resolve()}",
        "IMPORTANT: Save every generated deliverable (.pptx, .pdf, .docx, .xlsx, etc.)",
        "into the 'output' directory shown above, using its ABSOLUTE path — never a",
        "bare filename (a bare filename lands in the workspace root, which is wrong).",
        f"Example: write the deck to \"{(workspace / OUTPUT_DIR_NAME / 'deck.pptx').resolve()}\".",
        "",
        "<available_skills>",
    ]
    for skill in skills:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{skill.name}</name>",
                f"    <description>{skill.description}</description>",
                f"    <location>{skill.location}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def tool_definitions(registry: SkillRegistry) -> list[dict]:
    names = [s.name for s in registry.catalog_skills()]
    tools: list[dict] = []

    if names:
        tools.append(
            {
                "name": "activate_skill",
                "description": (
                    "Load a skill's full SKILL.md instructions into context. "
                    "Call this when a task matches a skill description."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": names,
                            "description": "Skill name from the catalog",
                        }
                    },
                    "required": ["name"],
                },
            }
        )

    tools.extend(
        [
            {
                "name": "read_file",
                "description": (
                    "Read a text file. Prefer absolute paths. For skill resources, "
                    "resolve relative paths against the skill directory."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or workspace-relative path",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": (
                    "Create or overwrite a text file with the given contents. "
                    "Prefer this over shell redirects (cat, heredocs, echo). "
                    "Creates parent directories if needed."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or workspace-relative path",
                        },
                        "contents": {
                            "type": "string",
                            "description": "Full file contents to write",
                        },
                    },
                    "required": ["path", "contents"],
                },
            },
            {
                "name": "run_command",
                "description": (
                    "Run a shell command in the workspace via the OS default shell "
                    "(Windows: cmd.exe). Prefer write_file for creating files. "
                    "On Windows use .venv\\Scripts\\python.exe and cmd syntax "
                    "(dir/type/del/copy, &&); avoid bash heredocs and PowerShell. "
                    "Destructive commands (rm -rf, format, git reset --hard, "
                    "DROP TABLE, etc.) are blocked."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "cmd.exe command on Windows "
                                "(e.g. .venv\\Scripts\\python.exe -m pytest test_bookstore.py)"
                            ),
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "list_dir",
                "description": "List files in a directory (non-recursive).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path",
                        }
                    },
                    "required": ["path"],
                },
            },
        ]
    )
    return tools


def resolve_path(path_str: str, workspace: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


# Patterns that permanently destroy data, wipe disks, or take down the machine.
_DESTRUCTIVE_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*\s--force\b)",
        r"\brm\s+.*\s+(-r|--recursive)\b",
        r"\brmdir\s+/s\b",
        r"\bdel\s+/[sfq]+\b",
        r"\berase\s+/[sfq]+\b",
        r"\bRemove-Item\b.*(-Recurse|-Force|rm\b)",
        r"\b(format|diskpart|mkfs(\.\w+)?)\b",
        r"\bdd\s+.*\bof\s*=",
        r"\b(shutdown|reboot|poweroff|halt)\b",
        r"\bgit\s+push\s+.*--force\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-[a-zA-Z]*f",
        r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b",
        r"\bTRUNCATE\s+TABLE\b",
        r"\breg\s+delete\b",
        r"\bcipher\s+/w\b",
        r">\s*\\\\\.\\PhysicalDrive",
        r">\s*/dev/[sh]d[a-z]",
        r"\bcurl\b.+\|\s*(ba)?sh\b",
        r"\bwget\b.+\|\s*(ba)?sh\b",
    )
)


def _destructive_command_reason(command: str) -> str | None:
    """Return a reason if the command looks destructive; otherwise None."""
    for pattern in _DESTRUCTIVE_COMMAND_PATTERNS:
        if pattern.search(command):
            return f"Blocked destructive command matching /{pattern.pattern}/: {command!r}"
    return None


def handle_tool(
    name: str,
    tool_input: dict,
    registry: SkillRegistry,
    workspace: Path,
    activated: set[str],
) -> str:
    if name == "activate_skill":
        skill_name = tool_input["name"]
        skill = registry.get(skill_name)
        if skill is None:
            return f"Unknown skill: {skill_name}"
        if skill_name in activated:
            return f"Skill '{skill_name}' is already active in this session."

        body = skill.body
        if body is None:
            parsed = _parse_skill_md(skill.location)
            body = parsed.body if parsed else skill.location.read_text(encoding="utf-8")

        resources = list_skill_resources(skill)
        resource_xml = "\n".join(f"  <file>{f}</file>" for f in resources)
        activated.add(skill_name)
        return (
            f'<skill_content name="{skill.name}">\n'
            f"{body}\n\n"
            f"Skill directory: {skill.directory}\n"
            "Relative paths in this skill are relative to the skill directory.\n\n"
            f"<skill_resources>\n{resource_xml}\n</skill_resources>\n"
            f"</skill_content>"
        )

    if name == "read_file":
        path = resolve_path(tool_input["path"], workspace)
        if not path.is_file():
            return f"File not found: {path}"
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Cannot read binary file: {path}"

    if name == "write_file":
        path = resolve_path(tool_input["path"], workspace)
        contents = tool_input.get("contents", "")
        if not isinstance(contents, str):
            return "contents must be a string"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            return f"Wrote {len(contents)} characters to {path}"
        except OSError as exc:
            return f"Failed to write {path}: {exc}"

    if name == "list_dir":
        path = resolve_path(tool_input["path"], workspace)
        if not path.is_dir():
            return f"Not a directory: {path}"
        entries = []
        for child in sorted(path.iterdir()):
            kind = "dir" if child.is_dir() else "file"
            entries.append(f"{kind}\t{child.name}")
        return "\n".join(entries) or "(empty)"

    if name == "run_command":
        command = tool_input["command"]
        blocked = _destructive_command_reason(command)
        if blocked:
            return blocked
        cwd = resolve_path(tool_input["cwd"], workspace) if tool_input.get("cwd") else workspace
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=120,
            )
            out = (result.stdout or "") + (result.stderr or "")
            return f"exit_code={result.returncode}\n{out}".strip()
        except subprocess.TimeoutExpired:
            return "Command timed out after 120s"

    return f"Unknown tool: {name}"


def _snapshot_artifacts(directory: Path) -> dict[str, float]:
    """Map artifact filename -> mtime for artifacts directly in `directory`."""
    snapshot: dict[str, float] = {}
    if not directory.is_dir():
        return snapshot
    for child in directory.iterdir():
        if child.is_file() and child.suffix.lower() in ARTIFACT_SUFFIXES:
            snapshot[child.name] = child.stat().st_mtime
    return snapshot


def _relocate_outputs(workspace: Path, before: dict[str, float]) -> list[Path]:
    """Move newly created/modified artifacts from workspace root into output/.

    The model is asked to write deliverables straight into the output directory;
    this is a deterministic safety net for when it writes a bare filename that
    lands in the workspace root instead.
    """
    output_dir = workspace / OUTPUT_DIR_NAME
    moved: list[Path] = []
    for child in list(workspace.iterdir()):
        if not child.is_file() or child.suffix.lower() not in ARTIFACT_SUFFIXES:
            continue
        # Only touch files this run created or changed.
        if before.get(child.name) == child.stat().st_mtime:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / child.name
        try:
            if dest.exists():
                dest.unlink()
            shutil.move(str(child), str(dest))
            moved.append(dest)
        except OSError as exc:
            print(f"[warn] could not move {child} -> {dest}: {exc}", file=sys.stderr)
    return moved


def run_agent(prompt: str, workspace: Path, registry: SkillRegistry) -> str:
    client = Anthropic()
    system = build_system_prompt(registry, workspace)
    tools = tool_definitions(registry)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    activated: set[str] = set()
    final_text: list[str] = []
    artifacts_before = _snapshot_artifacts(workspace)

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools,
            messages=messages,
        )

        assistant_content = response.content
        messages.append(
            {
                "role": "assistant",
                "content": [block.model_dump(exclude_none=True) for block in assistant_content],
            }
        )

        tool_uses = [b for b in assistant_content if b.type == "tool_use"]
        text_blocks = [b.text for b in assistant_content if b.type == "text" and b.text]

        if response.stop_reason == "end_turn" or not tool_uses:
            final_text.extend(text_blocks)
            break

        if text_blocks:
            print("\n".join(text_blocks))

        tool_results = []
        for block in tool_uses:
            print(f"→ {block.name}({json.dumps(block.input)})")
            result = handle_tool(block.name, block.input, registry, workspace, activated)
            # Keep tool results reasonably sized for context
            if len(result) > 80_000:
                result = result[:80_000] + "\n…[truncated]"
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    for dest in _relocate_outputs(workspace, artifacts_before):
        print(f"[info] moved generated file to {dest}")

    return "\n".join(final_text).strip()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Simple Agent Skills runner (Haiku 4.5)"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="User task. If omitted, starts an interactive REPL.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project root used for skill discovery (default: cwd)",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="Print discovered skills and exit",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    registry = discover_skills(workspace)

    if args.list_skills:
        if not registry.skills:
            print("No skills found.")
            return
        for skill in registry.skills.values():
            flag = " [disabled-model-invocation]" if skill.disable_model_invocation else ""
            print(f"- {skill.name}{flag}")
            print(f"  {skill.description}")
            print(f"  {skill.location}")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY (e.g. in .env)", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {MODEL}")
    print(f"Skills: {len(registry.catalog_skills())} available\n")

    if args.prompt:
        reply = run_agent(args.prompt, workspace, registry)
        if reply:
            print(reply)
        return

    print("Interactive mode. Type a task, or 'quit' to exit.\n")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            break
        reply = run_agent(user, workspace, registry)
        print(f"\nAgent> {reply}\n")


if __name__ == "__main__":
    main()
