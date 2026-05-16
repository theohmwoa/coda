"""Real-world OSS PR demo: coda fixes arrow-py/arrow #1259.

End-to-end: clone arrow-py/arrow into a tempdir, set up a venv, hand the
checkout to a coda Agent (Opus 4.7) along with the issue text, and have
the agent reproduce → locate → fix → test → produce a clean diff for
human review. The agent does NOT commit, push, or open a PR; the
launcher captures `git diff HEAD` so the human reviewer has the final
say before anything lands on a public branch.

Verifiability is built in:
- the agent must produce a new test that fails before the fix and
  passes after
- all existing tests must still pass
- the launcher independently re-runs the full test suite from outside
  the sandbox once the agent finishes
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


PYTHON = "/opt/homebrew/bin/python3"
REPO = "arrow-py/arrow"
ISSUE = 1259

ISSUE_TEXT = """\
arrow-py/arrow #1259 — arrow.get() behaviour for tzinfo=None

When `arrow.get()` is invoked with a value, a format and a `tzinfo`
keyword argument that is explicitly None, the behaviour is different
than if the keyword argument were omitted. Minimal repro:

    import arrow
    arrow.get('2025-01-01', 'YYYY-MM-DD', tzinfo=None)

Observed: TypeError: Arrow.__init__() missing 1 required positional
argument: 'day'

Root cause (as the reporter notes): in `arrow/factory.py`, this check:

    if len(kwargs) == 1 and tz is None:
        arg_count = 3

assumes that `tzinfo` being None means it wasn't passed, but the user
DID pass it explicitly. The factory then routes to the 3+ args
datetime-like constructor path and crashes.

Expected: explicit `tzinfo=None` should be treated the same as not
passing `tzinfo` at all. The call should succeed and return an
Arrow object equivalent to `arrow.get('2025-01-01', 'YYYY-MM-DD')`.
"""


USER_PROMPT = f"""\
You are a contributor preparing a pull request for the open-source
project `{REPO}`. The repository checkout is your workspace. Your
goal: fix issue #{ISSUE} cleanly, with tests, ready for a human reviewer
to submit upstream.

# The issue (verbatim — do not look it up, it's all here)

{ISSUE_TEXT}

# Acceptance criteria (verifiable, in order)

1. Reproduce the bug locally. Construct the failing call from the
   issue text, confirm it raises the documented `TypeError`. This is
   step zero — if the bug doesn't reproduce, stop and report that.

2. Locate the offending code. The issue points at `arrow/factory.py`
   and the check `if len(kwargs) == 1 and tz is None`. Read the
   surrounding function fully to understand the routing.

3. Design a minimal fix. The right behaviour is: explicit
   `tzinfo=None` should be treated the same as omitting `tzinfo`.
   Distinguish "kwarg was passed but is None" from "kwarg was not
   passed". The fix should be SMALL (a few lines).

4. Write a regression test in the project's existing test layout.
   Find where the factory's `get()` is already tested — there's
   almost certainly a `tests/factory_tests.py` or similar — and add
   a test that:
     a) verifies the call from the issue no longer crashes
     b) verifies the result equals the same call without `tzinfo=`
   Follow the file's existing test style (pytest, unittest, etc.).

5. Run the FULL test suite and confirm:
     - your new test passes
     - all existing tests still pass
   Do not declare success until both are true.

6. Produce a clean diff. `bash('git diff HEAD')` should show ONLY
   the lines you intended to change — no whitespace noise, no
   unrelated files. If you accidentally touched a file, revert it.

7. Final turn: print, in this order, a self-contained PR description
   for a human reviewer to use verbatim:

     Title: <one line, conventional commit style if appropriate>
     Body:
       ## Summary
       <2-3 sentences on what broke and how the fix works>

       ## Reproduction
       <the failing call from the issue>

       ## Test
       <where the new test lives and what it asserts>

       Fixes #{ISSUE}.

# Environment setup (do this first)

This is a fresh clone with no Python environment. Set one up:

    bash('{PYTHON} -m venv .venv', timeout_s=60)
    bash('.venv/bin/pip install --upgrade pip', timeout_s=60)
    bash('.venv/bin/pip install -e ".[dev]" -q', timeout_s=180)
    bash('.venv/bin/pytest tests/ -q -x --tb=short', timeout_s=180)

After install you should see all tests passing. If they don't,
investigate before touching any code.

Then use `.venv/bin/pytest` for every test run.

# Conventions

- Read existing files in `arrow/` and `tests/` first to learn the
  project's style. Match it.
- Do NOT commit, push, or touch git remotes. Leave the diff
  uncommitted. The human reviewer takes it from there.
- Do NOT change anything not directly required for the fix and its
  test. No drive-by formatting, no import reordering, no version
  bumps.
- If a test fails for a reason you don't understand, READ the
  failure carefully before guessing fixes.

Take your time. Quality over turns.
"""


def setup_workspace() -> Path:
    workspace_parent = Path(tempfile.mkdtemp(prefix="coda-arrow-pr-")).resolve()
    work_root = workspace_parent / "arrow"
    print(f"cloning {REPO}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", f"https://github.com/{REPO}.git", str(work_root)],
        check=True,
        capture_output=True,
    )
    print(f"workspace: {work_root}")
    return work_root


def verify(work_root: Path) -> None:
    print()
    print("=== POST-RUN VERIFICATION (outside the agent sandbox) ===")

    print("--- git diff stat ---")
    diff_stat = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        cwd=str(work_root), capture_output=True, text=True,
    )
    print(diff_stat.stdout or "(no diff)")

    print("--- full git diff ---")
    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=str(work_root), capture_output=True, text=True,
    )
    print(diff.stdout[:6000])
    if len(diff.stdout) > 6000:
        print(f"... ({len(diff.stdout)} bytes total) ...")

    print("--- independent pytest re-run ---")
    venv_pytest = work_root / ".venv" / "bin" / "pytest"
    if not venv_pytest.exists():
        print("(no .venv/bin/pytest — agent likely didn't install)")
        return
    proc = subprocess.run(
        [str(venv_pytest), "tests/", "-q", "--tb=short"],
        cwd=str(work_root), capture_output=True, text=True, timeout=300,
    )
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print("STDERR (tail):", proc.stderr[-1500:])
    print(f"pytest exit: {proc.returncode}")

    # Save the diff for follow-up
    diff_save = Path(__file__).resolve().parent.parent / "runs" / f"arrow_{ISSUE}_diff.patch"
    diff_save.parent.mkdir(exist_ok=True)
    diff_save.write_text(diff.stdout)
    print(f"--- diff saved: {diff_save} ---")


def main() -> int:
    work_root = setup_workspace()

    trace_dir = Path(__file__).resolve().parent.parent / "runs"
    trace_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"arrow_{ISSUE}_{stamp}.jsonl"
    print(f"trace:     {trace_path}")
    print()

    hooks = HookRegistry()
    sandbox = Sandbox(
        root=work_root,
        hooks=hooks,
        bash_timeout_s=300,
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
    print("--- agent final text (last 3500 chars) ---")
    print(result.text[-3500:])

    verify(work_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
