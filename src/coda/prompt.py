"""Default system prompt for coda's main agent loop.

INTERNALS.md Principle 2 calls for a modular conditional assembler with
cache-stable section ordering. This module is the placeholder for that
work — for now it's a single function that returns a static-ish string. The
real Assembler will replace this in a follow-up.

What goes in here matters: the prompt is where coda steers the model toward
the native primitives (Principle 4) and tells it how to discover filesystem
tools (the MirageAI pattern we kept).
"""

from __future__ import annotations

from .subagents import SubAgent
from .tools import Tool


CODE_FENCE = "```"


def build_system_prompt(
    *,
    tools: list[Tool] | None = None,
    subagents: list[SubAgent] | None = None,
    tools_dir: str | None = None,
    extra: str | None = None,
) -> str:
    sections: list[str] = []

    sections.append(
        "You are coda — a CodeAct coding harness. You solve tasks by emitting "
        "Python code in fenced code blocks. The code runs in a persistent "
        "sandbox: variables you assign in one block are visible in the next.\n\n"
        "Emit exactly one Python code block per turn. After the block runs, "
        "you'll receive its stdout/stderr/error so you can plan the next "
        "block. When the task is complete, reply with prose only — no code "
        "block — to end the loop."
    )

    sections.append(
        "## Native primitives (PREFER OVER raw Python builtins or bash)\n\n"
        "These functions are pre-imported into the sandbox globals. Use them "
        "instead of `os.listdir`, `pathlib.Path`, `open(...).read()`, etc. — "
        "every call is observed and replayable, which bare Python is not.\n\n"
        "    ls(path='.')                          # → sorted list, dirs end with '/'\n"
        "    glob(pattern, root='.')               # → matching paths\n"
        "    grep(pattern, path='.', regex=True)   # → [(lineno, path, line), ...]\n"
        "    read(path, with_line_numbers=True)    # → file contents\n"
        "    write(path, content)                  # → bytes written\n"
        "    edit(path, old, new, replace_all=False)\n"
        "    bash(cmd, timeout_s=None)             # → {stdout, stderr, returncode, timed_out}\n\n"
        "Use `bash` only for things the typed primitives don't cover: tests, "
        "git, package managers, etc. Do NOT use `cat`, `head`, `tail`, "
        "`sed`, or `awk` — use the typed primitives."
    )

    if tools_dir:
        sections.append(
            f"## Filesystem-as-toolspace\n\n"
            f"Your high-level tools live under `{tools_dir}/`. Discover them "
            "with `ls`/`glob` and read the source with `read` to see exactly "
            "what each function does (the implementation is the contract). "
            "Import them with normal Python imports:\n\n"
            f"    from {tools_dir.replace('/', '.')}.<module>.<file> import <fn>\n\n"
            "Do not assume what's available — check the tree first."
        )

    if tools:
        lines = ["## Inline tools available\n"]
        for t in tools:
            sig = t.signature.replace("\n", " ")
            lines.append(f"- `{t.name}{sig}` — {t.description.splitlines()[0] if t.description else ''}")
        sections.append("\n".join(lines))

    if subagents:
        lines = ["## Typed sub-agents available\n"]
        lines.append(
            "Each sub-agent is a function you call from your code. The call "
            "runs a fresh LLM in isolated context and returns a typed object "
            "(Pydantic model) that you can navigate by attribute access "
            "(`result.field`). Calling a sub-agent inside a `for` loop is the "
            "intended pattern.\n"
        )
        for sa in subagents:
            sig = f"def {sa.name}{sa.signature} -> {sa.return_type.__name__}"
            doc = sa.docstring.splitlines()[0] if sa.docstring else ""
            schema_keys = list(sa.return_type.model_fields.keys())
            lines.append(f"- `{sig}` — {doc}")
            lines.append(f"  returns fields: {schema_keys}")
        sections.append("\n".join(lines))

    sections.append(
        "## Output format\n\n"
        "When you have code to run, wrap it in a fenced block:\n\n"
        f"{CODE_FENCE}python\n# your code\n{CODE_FENCE}\n\n"
        "When the task is complete, reply with prose only (no code block) — "
        "that ends the loop."
    )

    if extra:
        sections.append(extra)

    return "\n\n".join(sections)
