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

from .mcp import ServerProxy
from .skills import Skill
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
        "When you have code to run, wrap each chunk in a fenced block:\n\n"
        f"{CODE_FENCE}python\n# your code\n{CODE_FENCE}\n\n"
        "You may emit MULTIPLE code blocks per turn — they all run in "
        "order against the same persistent sandbox, sharing globals. "
        "Prefer one block per turn when you want to react to its output "
        "before the next chunk; emit several when later chunks don't "
        "depend on earlier results (e.g. parallel data gathering).\n\n"
        "After your code runs, the response may contain a "
        "`⚠️ Lint warnings` section. **These are real bugs caught by "
        "static analysis against the typed sub-agent schemas** — e.g. "
        "you wrote `x.severity == 'medium'` but the sub-agent's "
        "`severity` is `Literal['low','high']`, so the branch is dead "
        "code. When you see lint warnings, your NEXT code block must "
        "fix them before continuing the workflow. Do not save such code "
        "as a skill until clean.\n\n"
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


def _mcp(ctx: Context) -> str:
    proxies: dict[str, ServerProxy] = ctx["mcp_proxies"]
    lines = [
        "## MCP servers available\n",
        "Each MCP server is exposed in the sandbox as a Python object. Call its "
        "tools as normal method calls — no JSON tool-use, no schemas. Argument "
        "names match the MCP tool's input schema; pass them as keyword args.\n",
    ]
    for py_name, proxy in proxies.items():
        tools = proxy.list_tools()
        lines.append(f"### `{py_name}` ({len(tools)} tool(s))")
        for tool_name in tools:
            meta = proxy.describe(tool_name)
            doc_first = (meta["description"].splitlines()[0] if meta["description"] else "").strip()
            schema_keys = list((meta["input_schema"].get("properties") or {}).keys())
            args_hint = ", ".join(schema_keys) if schema_keys else ""
            lines.append(f"- `{py_name}.{tool_name}({args_hint})` — {doc_first}")
    lines.append(
        "\nFor full input-schema details on any tool, call "
        "`<server>.describe('<tool>')` from your code."
    )
    return "\n".join(lines)


def _skills(ctx: Context) -> str:
    skills: dict[str, Skill] = ctx["skills"]
    skills_dir = ctx["skills_dir"]
    lines = [
        "## Your skills library\n",
        "Previously-saved workflows live in `" + str(skills_dir) + "`. "
        "Each is a Python function loaded into your sandbox globals at "
        "this run's startup. Call them by name — no import needed.",
    ]
    if skills:
        lines.append("\nCurrently available:")
        for s in skills.values():
            sig = s.signature.replace("\n", " ")
            doc_first = s.docstring.splitlines()[0] if s.docstring else ""
            lines.append(f"- `{s.name}{sig}` — {doc_first}")
    else:
        lines.append("\n(The library is empty so far. Anything you save below will be available in your next run.)")
    lines.append("")
    lines.append(
        "### Saving new skills — IMPORTANT\n"
        "You have `save_skill(code=...)` in your globals. **When you've "
        "just successfully completed a workflow whose code would work "
        "again on a similar task with different inputs, call `save_skill` "
        "to crystallize it.** The harness doesn't do this for you; you "
        "decide what's worth keeping.\n\n"
        "Pass `code` containing EXACTLY ONE top-level `def` whose name "
        "is snake_case. Parameterize anything that was task-specific in "
        "the run you just did (hardcoded names, paths, time windows). "
        "The function body may reference any sandbox global you used "
        "this run (MCP proxies like `gmail`, sub-agents, primitives like "
        "`read`/`write`). Skills are versionless — saving with the same "
        "name overwrites. Save aggressively for useful work, skip "
        "trivial one-offs like `print(ls('.'))`. Example:\n\n"
        "```python\n"
        "save_skill(code='''\n"
        "def brief_vips(vips: list[str], hours: int = 24) -> str:\n"
        "    \"\"\"Return a markdown brief of Gmail messages from VIPs in the last N hours.\"\"\"\n"
        "    msgs = gmail.search(query=f\"from:({' OR '.join(vips)}) newer_than:{hours}h\")\n"
        "    return '\\n'.join(f'- {m[\"from\"]}: {m[\"subject\"]}' for m in msgs)\n"
        "''')\n"
        "```\n\n"
        "After saving, the skill is available in your NEXT run with this "
        "same `skills_dir`. The save returns "
        "`{ok: True, path: ..., name: ...}`. If lint warnings appeared "
        "in your earlier execution results, you should have fixed them "
        "BEFORE saving — saved skills shouldn't carry dead branches.\n\n"
        "You also have a `lint(code: str)` primitive for an explicit "
        "check on any snippet (the same check that runs automatically "
        "on every executed block)."
    )
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
            "mcp",
            render=_mcp,
            cache_stable=False,
            condition=lambda ctx: bool(ctx.get("mcp_proxies")),
        )
    )
    a.add(
        Section(
            "skills",
            render=_skills,
            cache_stable=False,
            condition=lambda ctx: ctx.get("skills_dir") is not None,
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
    mcp_proxies: dict[str, "ServerProxy"] | None = None,
    skills: dict[str, Skill] | None = None,
    skills_dir: Any = None,
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
        "mcp_proxies": dict(mcp_proxies or {}),
        "skills": dict(skills or {}),
        "skills_dir": skills_dir,
    }
    a = assembler or default_assembler()
    return a.render(ctx)
