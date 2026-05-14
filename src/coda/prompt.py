"""Modular, conditional, cache-aware system prompt assembler.

INTERNALS.md Principle 2 calls for a dynamic assembler with ~20 sections.
This module is coda's version: a small list of `Section`s, each with:

- `name`         — for debugging / preview()
- `cache_stable` — True if the rendered text doesn't change between runs of
                   the same agent shape. Cache-stable sections render FIRST
                   so they hit Anthropic's prompt cache; cache-breaking
                   sections (those that include user-specific or per-call
                   data) come last and only invalidate their own suffix.
- `condition`    — predicate over the build-time context dict; section is
                   omitted when False
- `render`       — callable returning the section's text

We start with 6 stable + 2 breaking sections, populated by
`default_assembler()`. The list will grow toward Claude Code's ~20 as we
need them; the API stays the same.

# Inspection

`assembler.preview(ctx)` returns a list of dicts describing which sections
fired, whether each is cache-stable, and the rendered char count. Useful
for debugging "why is my prompt cache missing?" without dumping the whole
~5 KB string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .subagents import SubAgent
from .tools import Tool


CODE_FENCE = "```"


# A build context — keys we read are listed below. Extra keys are ignored,
# so user code can stash whatever it wants alongside ours without conflict.
#   tools:      list[Tool]            inline @tool-decorated callables
#   subagents:  list[SubAgent]        @subagent-decorated callables
#   tools_dir:  str | None            filesystem-as-toolspace root
#   extra:      str | None            user-appended prose
Context = dict[str, Any]


@dataclass
class Section:
    name: str
    render: Callable[[Context], str]
    cache_stable: bool = True
    condition: Callable[[Context], bool] = field(
        default=lambda ctx: True, repr=False
    )


class Assembler:
    def __init__(self) -> None:
        self._sections: list[Section] = []

    def add(self, section: Section) -> "Assembler":
        """Append `section`. Returns self for chaining."""
        self._sections.append(section)
        return self

    def remove(self, name: str) -> "Assembler":
        """Drop the section named `name`. Silently no-op if absent."""
        self._sections = [s for s in self._sections if s.name != name]
        return self

    def replace(self, section: Section) -> "Assembler":
        """Replace a section by name; append if not present."""
        for i, s in enumerate(self._sections):
            if s.name == section.name:
                self._sections[i] = section
                return self
        return self.add(section)

    def render(self, context: Context | None = None) -> str:
        """Build the system prompt by evaluating each section.

        Cache-stable sections come first in registration order, then
        cache-breaking sections in registration order. Sections whose
        `condition` returns False are omitted entirely. Empty `render()`
        outputs are dropped so an enabled-but-empty section doesn't leave a
        gap.
        """
        ctx = context or {}
        ordered = [s for s in self._sections if s.cache_stable] + [
            s for s in self._sections if not s.cache_stable
        ]
        parts: list[str] = []
        for s in ordered:
            if not s.condition(ctx):
                continue
            text = s.render(ctx).strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def preview(self, context: Context | None = None) -> list[dict[str, Any]]:
        """Return a debug-friendly list describing each section's contribution.

        Each entry: `{name, cache_stable, included, chars}`. `included` is
        False when the condition failed; `chars` is 0 for excluded sections.
        """
        ctx = context or {}
        out: list[dict[str, Any]] = []
        for s in self._sections:
            included = s.condition(ctx)
            chars = len(s.render(ctx)) if included else 0
            out.append(
                {
                    "name": s.name,
                    "cache_stable": s.cache_stable,
                    "included": included,
                    "chars": chars,
                }
            )
        return out

    def __len__(self) -> int:
        return len(self._sections)


# --- default sections ---------------------------------------------------------


def _intro(ctx: Context) -> str:
    return (
        "You are coda — a CodeAct coding harness. You solve tasks by emitting "
        "Python code in fenced code blocks. The code runs in a persistent "
        "sandbox: variables you assign in one block are visible in the next.\n\n"
        "Emit exactly one Python code block per turn. After the block runs, "
        "you'll receive its stdout/stderr/error so you can plan the next "
        "block. When the task is complete, reply with prose only — no code "
        "block — to end the loop."
    )


def _primitives(ctx: Context) -> str:
    return (
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


def _output_format(ctx: Context) -> str:
    return (
        "## Output format\n\n"
        "When you have code to run, wrap it in a fenced block:\n\n"
        f"{CODE_FENCE}python\n# your code\n{CODE_FENCE}\n\n"
        "When the task is complete, reply with prose only (no code block) — "
        "that ends the loop."
    )


def _filesystem_tools(ctx: Context) -> str:
    td = ctx["tools_dir"]
    return (
        "## Filesystem-as-toolspace\n\n"
        f"Your high-level tools live under `{td}/`. Discover them with "
        "`ls`/`glob` and read the source with `read` to see exactly what "
        "each function does (the implementation is the contract). Import "
        "them with normal Python imports:\n\n"
        f"    from {td.replace('/', '.')}.<module>.<file> import <fn>\n\n"
        "Do not assume what's available — check the tree first."
    )


def _inline_tools(ctx: Context) -> str:
    tools: list[Tool] = ctx["tools"]
    lines = ["## Inline tools available\n"]
    for t in tools:
        sig = t.signature.replace("\n", " ")
        first_line = t.description.splitlines()[0] if t.description else ""
        lines.append(f"- `{t.name}{sig}` — {first_line}")
    return "\n".join(lines)


def _subagents(ctx: Context) -> str:
    subagents: list[SubAgent] = ctx["subagents"]
    lines = [
        "## Typed sub-agents available\n",
        "Each sub-agent is a function you call from your code. The call runs "
        "a fresh LLM in isolated context and returns a typed object "
        "(Pydantic model) that you can navigate by attribute access "
        "(`result.field`). Calling a sub-agent inside a `for` loop is the "
        "intended pattern.\n",
    ]
    for sa in subagents:
        sig = f"def {sa.name}{sa.signature} -> {sa.return_type.__name__}"
        doc = sa.docstring.splitlines()[0] if sa.docstring else ""
        schema_keys = list(sa.return_type.model_fields.keys())
        lines.append(f"- `{sig}` — {doc}")
        lines.append(f"  returns fields: {schema_keys}")
    return "\n".join(lines)


def _extra(ctx: Context) -> str:
    return ctx["extra"]


def default_assembler() -> Assembler:
    """Build the default Assembler with coda's starter sections.

    Cache-stable (registered first, render first):
      1. intro
      2. primitives
      3. filesystem_tools     (when `tools_dir` is set)
      4. output_format

    Cache-breaking (registered last, render last):
      5. inline_tools         (when `tools` is non-empty)
      6. subagents            (when `subagents` is non-empty)
      7. extra                (when `extra` is non-empty)

    Why this split: inline tools and sub-agents change per-Agent — they
    invalidate cache. The intro / primitives / output_format text is
    identical across every coda agent, so putting it first means the cache
    hit covers ~80% of the prompt even when the user adds more tools.
    """
    a = Assembler()
    a.add(Section("intro", render=_intro, cache_stable=True))
    a.add(Section("primitives", render=_primitives, cache_stable=True))
    a.add(
        Section(
            "filesystem_tools",
            render=_filesystem_tools,
            cache_stable=True,
            condition=lambda ctx: bool(ctx.get("tools_dir")),
        )
    )
    a.add(Section("output_format", render=_output_format, cache_stable=True))
    a.add(
        Section(
            "inline_tools",
            render=_inline_tools,
            cache_stable=False,
            condition=lambda ctx: bool(ctx.get("tools")),
        )
    )
    a.add(
        Section(
            "subagents",
            render=_subagents,
            cache_stable=False,
            condition=lambda ctx: bool(ctx.get("subagents")),
        )
    )
    a.add(
        Section(
            "extra",
            render=_extra,
            cache_stable=False,
            condition=lambda ctx: bool(ctx.get("extra")),
        )
    )
    return a


def build_system_prompt(
    *,
    tools: list[Tool] | None = None,
    subagents: list[SubAgent] | None = None,
    tools_dir: str | None = None,
    extra: str | None = None,
    assembler: Assembler | None = None,
) -> str:
    """Shortcut: build the default Assembler and render it against a context.

    Callers who want to customize the prompt can either:
    - Pass `assembler=` with their own Assembler instance, or
    - Construct one with `default_assembler()`, mutate it via `.add()` /
      `.remove()` / `.replace()`, and pass it to `Agent(... )` (when the
      Agent accepts an assembler — pending follow-up).
    """
    ctx: Context = {
        "tools": list(tools or []),
        "subagents": list(subagents or []),
        "tools_dir": tools_dir,
        "extra": extra,
    }
    a = assembler or default_assembler()
    return a.render(ctx)
