"""Scripted CodeAct demo — runnable anywhere, no API key, no network.

Uses MockLLMClient with canned responses to demonstrate the loop end-to-end:
exploration → grep → sub-agent calls inside a `for` loop → file write.

Useful for:
- Docs / screenshots that don't depend on a paid Claude plan
- CI smoke-testing that the wiring is still correct
- Showing the JSONL trace format without real model calls

What this is NOT: a realistic agent run — the model "responses" here are
hardcoded. For the real thing, see `triage_todos.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coda import (  # noqa: E402
    Agent,
    CompletionResponse,
    HookRegistry,
    MockLLMClient,
    Sandbox,
    ToolCall,
    TraceWriter,
    read_trace,
    subagent,
)


class Triage(BaseModel):
    priority: Literal["low", "medium", "high"]
    category: Literal["bug", "feature", "refactor"]
    rationale: str


@subagent()
def triage(todo: dict) -> Triage:
    """Triage one TODO."""


SAMPLE = {
    "alpha.py": "def foo():\n    # TODO: handle empty input\n    return 1\n",
    "beta.py": "# FIXME: refactor\ndef bar():\n    return 2\n",
}

MAIN_TURNS = [
    """\
I'll explore first.

```python
files = glob('**/*.py')
print('files:', files)
hits = grep(r'TODO|FIXME', '.')
print('hits:', hits)
```
""",
    """\
Now I'll triage each match in a loop.

```python
triages = []
for lineno, path, text in hits:
    t = triage(todo={'path': path, 'line': lineno, 'text': text})
    triages.append((path, lineno, t))
    print(path, lineno, t.priority, t.category)

report = ['# TODO triage', '']
for path, lineno, t in triages:
    report.append(f'- **{path}:{lineno}** [{t.priority}/{t.category}] — {t.rationale}')
write('report.md', '\\n'.join(report))
print('wrote', len(triages), 'triages')
```
""",
    "Triage complete — report at `report.md`.",
]


SUBAGENT_PAYLOADS = [
    {"priority": "high", "category": "bug", "rationale": "Missing input validation can crash callers."},
    {"priority": "medium", "category": "refactor", "rationale": "Module shape has grown; deserves a sweep."},
]


def make_responder():
    main_q = list(MAIN_TURNS)
    sub_q = list(SUBAGENT_PAYLOADS)

    def respond(call):
        # Forced tool-use → sub-agent call
        if call.tool_choice and call.tool_choice.name == "return_triage":
            return CompletionResponse(
                text="",
                tool_calls=[
                    ToolCall(id=f"sub_{len(sub_q)}", name="return_triage", input=sub_q.pop(0))
                ],
                stop_reason="tool_use",
            )
        return CompletionResponse(text=main_q.pop(0))

    return respond


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="coda-scripted-")).resolve()
    for name, body in SAMPLE.items():
        (workspace / name).write_text(body)
    print(f"workspace: {workspace}")

    hooks = HookRegistry()
    sandbox = Sandbox(root=workspace, hooks=hooks)
    # Keep the trace OUTSIDE the workspace so it doesn't show up in the
    # agent's own ls/glob/grep calls.
    trace_path = workspace.parent / f"{workspace.name}-trace.jsonl"

    with TraceWriter(trace_path).attach(hooks) as tracer:
        agent = Agent(
            model="mock",
            llm=MockLLMClient(responder=make_responder()),
            sandbox=sandbox,
            hooks=hooks,
            subagents=[triage],
        )
        result = agent.run("Triage TODOs in this workspace.")

    print(f"turns:     {result.turns}")
    print(f"trace:     {trace_path} ({tracer.events_written} events)")
    print(f"report:    {workspace / 'report.md'}")
    print("--- report ---")
    print((workspace / "report.md").read_text())

    events = read_trace(trace_path)
    print(f"--- {len(events)} events captured; first 3:")
    for e in events[:3]:
        print(f"  {e['type']:<22} {e['payload']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
