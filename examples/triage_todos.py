"""TODO triage demo — the typed-sub-agent-inside-a-loop pattern, end to end.

Run with a paid Claude plan (no API budget required pre-2026-06-15):

    python examples/triage_todos.py

What it does:
1. Creates a scratch workspace with a few sample Python files containing
   TODO/FIXME/XXX comments.
2. Spins up a coda Agent backed by `claude -p` (subprocess) so it runs
   against the user's existing Claude plan with no API spend.
3. Gives the Agent a single instruction: find TODOs, triage each via a
   typed sub-agent, and write a markdown report.
4. Writes a JSONL trace of every event to `runs/triage_<ts>.jsonl` so the
   future codetrace UI can replay the run line by line.

The Triage sub-agent is a typed function. The outer Agent calls it from
inside a `for` loop — each call is a fresh-context LLM completion with
forced tool-use schema, returning a validated Pydantic object that the
generated code navigates by attribute (`t.priority`, `t.effort_hours`).
This is the "dual schema flow" coda exists to make ergonomic.

Switch to AnthropicClient after 2026-06-15 (or set ANTHROPIC_API_KEY now)
by changing `make_llm()` below.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

# Add src/ to path when running from the repo without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coda import (  # noqa: E402
    Agent,
    AnthropicClient,
    ClaudeCodeClient,
    HookRegistry,
    LLMClient,
    Sandbox,
    TraceWriter,
    subagent,
)


SAMPLE_FILES = {
    "auth.py": '''
def login(user, password):
    # TODO: rate-limit failed attempts, we got burned by credential stuffing
    return user == "admin"


def signup(user):
    # FIXME: emails are not validated, this lets through obvious garbage
    return {"id": 1, "user": user}
''',
    "billing.py": '''
def charge(card, amount):
    # XXX: this swallows Stripe errors silently — needs proper handling
    try:
        return {"ok": True}
    except Exception:
        return {"ok": False}


# TODO: refactor this module, it has grown organically
''',
    "utils.py": '''
def format_money(cents: int) -> str:
    return f"${cents/100:.2f}"  # TODO: support EUR / GBP


def slugify(s):
    return s.lower().replace(" ", "-")
''',
}


class Triage(BaseModel):
    priority: Literal["low", "medium", "high", "critical"]
    category: Literal["bug", "security", "feature", "refactor", "docs", "tech_debt"]
    effort_hours: float
    rationale: str


@subagent(model="claude-haiku-4-5", max_tokens=512)
def triage_todo(todo: dict) -> Triage:
    """Classify one TODO/FIXME/XXX comment from a code base.

    Be specific in the rationale — reference what the comment says about
    the underlying problem. effort_hours should reflect real engineering
    time including testing, not just the lines-of-code estimate. Prefer
    "security" over "bug" when the comment mentions auth, credentials,
    rate-limiting, or user input. Be decisive.
    """


USER_PROMPT = """\
The workspace contains a few Python files with TODO/FIXME/XXX comments.

1. Use `glob` to find all .py files.
2. Use `grep` with the regex `TODO|FIXME|XXX` to find every comment.
3. For each match, call the `triage_todo` sub-agent with a dict like
   {"path": ..., "line": ..., "text": ..., "context": ...}. Include 1-2
   surrounding lines of code in `context` if useful.
4. Build a markdown report grouped by priority (critical → high → medium → low),
   showing path:line, category, effort, and rationale for each item.
5. Write the report to `triage_report.md`.
6. Print the path of the report and a one-line summary count.

Use the native primitives (ls/glob/grep/read/write). Do not use bash or
os.* — the typed primitives are what we observe.
"""


def setup_workspace() -> Path:
    root = Path(tempfile.mkdtemp(prefix="coda-triage-")).resolve()
    for name, body in SAMPLE_FILES.items():
        (root / name).write_text(body.lstrip())
    return root


def make_llm() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    return ClaudeCodeClient()


def main() -> int:
    workspace = setup_workspace()
    print(f"workspace: {workspace}")

    trace_dir = Path("runs")
    trace_dir.mkdir(exist_ok=True)
    trace_path = trace_dir / f"triage_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    hooks = HookRegistry()
    sandbox = Sandbox(root=workspace, hooks=hooks)

    with TraceWriter(trace_path).attach(hooks) as tracer:
        agent = Agent(
            model="claude-sonnet-4-6",
            llm=make_llm(),
            sandbox=sandbox,
            hooks=hooks,
            subagents=[triage_todo],
            max_turns=12,
        )
        result = agent.run(USER_PROMPT)

    print(f"trace:     {trace_path}  ({tracer.events_written} events)")
    print(f"turns:     {result.turns}")
    print(f"tokens:    in={result.input_tokens} out={result.output_tokens}")
    report = workspace / "triage_report.md"
    if report.exists():
        print(f"report:    {report}")
        print("---")
        print(report.read_text())
    else:
        print("(no report produced — final text:)")
        print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
