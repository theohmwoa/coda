"""Dogfood demo: coda extends coda.

Spawns a coda Agent (Opus 4.7) against a clean copy of the coda repo in a
tempdir, asks it to implement the missing `coda replay` CLI with tests,
and verifies the result independently after the agent finishes.

The agent's workspace is a copy — never the live repo — so a botched run
just throws away a tempdir. Once the agent reports success, the
launcher re-runs pytest in the workspace from outside the sandbox and
prints what changed. Reviewing the diff and copying the files back into
the real repo is a human step.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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


SOURCE = Path("/Users/theophilushomawoo/coda")
PYTHON = "/opt/homebrew/bin/python3"


USER_PROMPT = """\
You are a contributor to the `coda` library. Your task: add the missing
`coda replay` CLI. Acceptance criteria are precise. Tests must pass.

# Acceptance criteria

1. New file `src/coda/replay.py` exposes a function
   `replay(path: str | Path, filter_types: set[str] | None = None,
           color: bool = True, stream: TextIO | None = None) -> int`
   that reads a JSONL trace file produced by `coda.TraceWriter` and prints
   a human-readable, colored, chronological timeline to `stream` (default
   sys.stdout). Returns the number of events printed.

2. New file `src/coda/__main__.py` wires the CLI so:
       python -m coda replay <path> [--filter <type>,<type>] [--no-color]
   works. Use argparse from stdlib. `--filter` is comma-separated event
   types. Print a one-line summary at the end (total events, by type).

3. New file `tests/test_replay.py` covers, at minimum:
       - ordering: events come out in trace order
       - filter_types: only matching events appear
       - color=False: no ANSI escape codes in output
       - FileNotFoundError when the path doesn't exist
       - CLI behavior via subprocess: `python -m coda replay <fixture>`

4. Existing 114 tests in `tests/` continue to pass.

5. Update `README.md`:
   - Find the line `- [ ] Replay CLI (week 2, part 2)`
   - Replace with `- [x] Replay CLI — \\`python -m coda replay <trace.jsonl>\\``

6. Update `src/coda/__init__.py` to export `replay` from `.replay`, and
   add `"replay"` to `__all__`.

# Conventions (read these BEFORE writing any new code)

   read('src/coda/hooks.py')      # Event, EventType, HookRegistry
   read('src/coda/trace.py')      # read_trace, iter_trace — build on these, don't reimplement
   read('src/coda/__init__.py')   # exports pattern
   read('tests/test_trace.py')    # test style for trace stuff
   read('tests/test_hooks.py')    # test style
   ls('examples/sample_run/')     # there's a real trace file you can use as a fixture

Use the sample trace at `examples/sample_run/trace.jsonl` to develop
against. It contains 1,081 real events from a prior agent run.

# Workflow

1. Read the conventions above so you understand the data model.
2. Sketch the timeline format before writing — what does ONE printed line
   look like for a `code_emitted` event? a `line_executed`? a
   `tool_called`? Keep it terse and grep-friendly. Pick an ANSI palette.
3. Implement `src/coda/replay.py`, then `src/coda/__main__.py`.
4. Write `tests/test_replay.py`. For the CLI subprocess test, use
   `subprocess.run([sys.executable, '-m', 'coda', 'replay', ...])` with
   `cwd` set so the workspace's `src/` is importable.
5. Run pytest. Iterate until everything is green:
       bash("PYTHONPATH=src %s -m pytest tests/ -q --tb=short")
6. Update `src/coda/__init__.py` and `README.md`.
7. Final turn: print a clean summary of what you produced (files + line
   counts), the pytest result, and an example invocation that runs
   `python -m coda replay examples/sample_run/trace.jsonl` against the
   real fixture and shows the first 10 lines of output.

# Notes

- Stay inside the existing code style. Read the files first.
- Do not delete or modify unrelated files. Touch only the paths listed.
- Quality matters more than turn count. Take your time. If pytest is
  slow on a turn, that's fine — wait for it.
""" % (PYTHON,)


def setup_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="coda-dogfood-")).resolve()
    work_root = workspace / "coda"
    shutil.copytree(
        SOURCE,
        work_root,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "runs",
            ".claude",
            ".DS_Store",
            ".venv",
            "build",
            "dist",
        ),
    )
    return work_root


def verify(work_root: Path) -> None:
    print()
    print("--- post-run verification ---")
    new_files = [
        "src/coda/replay.py",
        "src/coda/__main__.py",
        "tests/test_replay.py",
    ]
    for nf in new_files:
        p = work_root / nf
        if p.exists():
            n = sum(1 for _ in p.open())
            print(f"  ✓ {nf} ({n} lines)")
        else:
            print(f"  ✗ {nf} MISSING")

    print()
    print("--- pytest (run from outside the agent's sandbox) ---")
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=str(work_root),
        env={**os.environ, "PYTHONPATH": str(work_root / "src")},
        capture_output=True,
        text=True,
        timeout=180,
    )
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print("STDERR (last 1500):", proc.stderr[-1500:])
    print(f"pytest exit: {proc.returncode}")

    if (work_root / "src" / "coda" / "__main__.py").exists():
        print()
        print("--- python -m coda replay (against the bundled fixture) ---")
        fix = work_root / "examples" / "sample_run" / "trace.jsonl"
        if fix.exists():
            proc = subprocess.run(
                [PYTHON, "-m", "coda", "replay", str(fix)],
                cwd=str(work_root),
                env={**os.environ, "PYTHONPATH": str(work_root / "src")},
                capture_output=True,
                text=True,
                timeout=30,
            )
            for line in proc.stdout.splitlines()[:15]:
                print(line)
            print(f"...({len(proc.stdout.splitlines())} lines total, exit {proc.returncode})")


def main() -> int:
    work_root = setup_workspace()
    print(f"workspace: {work_root}")

    trace_dir = SOURCE / "runs"
    trace_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"dogfood_{stamp}.jsonl"
    print(f"trace:     {trace_path}")
    print()

    hooks = HookRegistry()
    sandbox = Sandbox(
        root=work_root,
        hooks=hooks,
        bash_timeout_s=180,
        max_read_bytes=20_000_000,
    )

    t0 = time.monotonic()
    with TraceWriter(trace_path).attach(hooks) as tracer:
        with Agent(
            model="claude-opus-4-7",
            llm=ClaudeAgentSDKClient(cwd=str(work_root)),
            sandbox=sandbox,
            hooks=hooks,
            max_turns=40,
            max_tokens=8192,
        ) as agent:
            result = agent.run(USER_PROMPT)
    elapsed = time.monotonic() - t0

    print()
    print("=== AGENT RUN SUMMARY ===")
    print(f"elapsed:         {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"coda turns:      {result.turns}")
    print(f"tokens:          in={result.input_tokens} out={result.output_tokens}")
    print(f"events captured: {tracer.events_written}")
    print(f"trace:           {trace_path}")
    print()
    print("--- agent final text (last 2500 chars) ---")
    print(result.text[-2500:])

    verify(work_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
