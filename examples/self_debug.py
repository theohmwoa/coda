"""Demo A — agent self-debugging from its own past trace.

Hand a fresh coda Agent the JSONL trace of a real prior failure
(`runs/framework_compare_20260517_020702.jsonl`) plus access to the
coda source. Ask it: what went wrong, what's the root cause, what
would you change? It writes the answer to `diagnosis.md`.

This exercises coda's actual differentiator: line-level structured
events in the trace. The failure was a SyntaxError that compiled-
voided ~833 lines of legitimate work; with just tool-call JSON logs
the agent would see "agent emitted code → agent claimed success".
With coda's trace it sees `execution_error` with the exact offending
character (U+2014, line 257, col 56) plus the entire 833-line code
block in the `code_emitted` payload — enough to reverse-engineer
what fix would prevent recurrence.

# Ground truth

The fix shipped in commit 8049ca4: agent.py executes blocks ONE AT
A TIME instead of concatenating them, so a SyntaxError in block N
doesn't void blocks 1..N-1. The agent under test should arrive at
this same conclusion from first principles.

# Why I'm not telling the agent that a fix exists

The fair test is: does the agent find the same root cause we did?
Does it propose a fix that matches what we shipped? The launcher
parks the proposed `diagnosis.md` and the actual fix side-by-side
afterward for comparison.
"""

from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coda import (  # noqa: E402
    Agent,
    ClaudeAgentSDKClient,
    HookRegistry,
    Sandbox,
    TraceWriter,
)


CODA_ROOT = Path(__file__).resolve().parent.parent
TARGET_TRACE = CODA_ROOT / "runs" / "framework_compare_20260517_020702.jsonl"

USER_PROMPT = f"""\
A previous coda agent run failed. Investigate.

The trace lives at: `{TARGET_TRACE.relative_to(CODA_ROOT)}`
coda's source lives at: `src/coda/`

The agent run claimed success but the workflow it was supposed to
do (build six framework profiles + post a Discord summary +
produce `analysis.md`) didn't happen. Your job: figure out what
actually went wrong and propose a fix to the **coda runtime**, not
the prompt.

You have ls/glob/grep/read/write/bash. The trace is JSONL — one
JSON object per line. The event types you'll see: `llm_request`,
`llm_response`, `code_emitted`, `code_executed`, `line_executed`,
`tool_called`, `execution_error`, `turn_complete`. Each event's
`payload` is the interesting part.

Investigation order I'd suggest (deviate if you have a better
theory):

1. Read the trace event-by-event. Look for `execution_error`. Note
   the exact error and the `code_emitted` event it came from.
2. The `code_emitted` payload is the literal source the agent
   submitted to the sandbox. Look at it. What was the model trying
   to do?
3. The error itself tells you a Python-level issue. But the more
   interesting question is: WHY did this kill 800+ lines of code
   that look fine on their own? Trace the path from
   `code_emitted` → `sandbox.execute` → the actual compile/exec
   in `src/coda/sandbox.py` → the wrapper in `src/coda/agent.py`
   that called it. Read those.
4. Decide what behavior coda SHOULD have had. Propose a code
   change to make this kind of failure non-catastrophic.

Write your findings to `diagnosis.md` at the repo root with these
sections:

  ## Failure mode (one sentence)
  ## Evidence from the trace
  (event types + payloads + line numbers IN the trace file)
  ## Root cause in coda's source
  (file path + function name + line numbers)
  ## Proposed fix
  (before/after code blocks OR a unified diff. Be specific.)
  ## How you'd test the fix
  (one paragraph)

Do NOT modify any source files. Do NOT run `git`. Just write
`diagnosis.md`. When `diagnosis.md` is in place, end the loop
with a one-line summary of your conclusion.
"""


def main() -> int:
    if not TARGET_TRACE.exists():
        print(f"missing target trace: {TARGET_TRACE}", file=sys.stderr)
        return 1

    # Sandbox rooted at coda repo so the agent can `read` both the
    # trace file under runs/ AND coda's source under src/. It writes
    # diagnosis.md at the repo root — we move it out to runs/ for
    # preservation afterward.
    workspace = CODA_ROOT

    trace_dir = CODA_ROOT / "runs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"self_debug_{stamp}.jsonl"

    print(f"target trace: {TARGET_TRACE.relative_to(CODA_ROOT)}")
    print(f"this run's trace: {trace_path.relative_to(CODA_ROOT)}")
    print()

    hooks = HookRegistry()
    sandbox = Sandbox(root=workspace, hooks=hooks, bash_timeout_s=60)

    t0 = time.monotonic()
    with TraceWriter(trace_path).attach(hooks) as tracer:
        with Agent(
            model="claude-opus-4-7",
            llm=ClaudeAgentSDKClient(cwd=str(workspace)),
            sandbox=sandbox,
            hooks=hooks,
            max_turns=20,
            max_tokens=8192,
        ) as agent:
            result = agent.run(USER_PROMPT)
    elapsed = time.monotonic() - t0

    print()
    print("=== RUN SUMMARY ===")
    print(f"elapsed:    {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"turns:      {result.turns}")
    print(f"tokens:     in={result.input_tokens} out={result.output_tokens}")
    print(f"events:     {tracer.events_written}")
    print(f"trace:      {trace_path}")
    print()

    diag = workspace / "diagnosis.md"
    if diag.exists():
        # Move to runs/ as the demo artifact, keep a copy at the root for now
        kept = trace_dir / f"self_debug_diagnosis_{stamp}.md"
        kept.write_text(diag.read_text())
        # Remove from repo root so it doesn't pollute the working tree
        diag.unlink()
        print(f"diagnosis: {kept.relative_to(CODA_ROOT)}")
        print()
        print("--- diagnosis.md ---")
        print(kept.read_text())
    else:
        print("(no diagnosis.md produced — final text:)")
        print(result.text[-2000:])

    return 0


if __name__ == "__main__":
    sys.exit(main())
