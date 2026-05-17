"""Honest version of the morning brief demo — thin one-line prompt.

The earlier morning_brief.py prescribed almost the entire solution in
its phase-1 prompt (function name, parameter signature, bucket
structure, Discord format, what to parameterize). That validated the
save_skill MECHANISM but not the agent's judgment about WHAT to
crystallize. This launcher is the honest test: ONE LINE of user prompt,
no scaffolding, see what the agent invents.

That's what `coda "morning brief"` would actually feel like as a
product. The agent has access to:
- gmail MCP (search_emails, read_email, send_email, draft_email, ...)
- post_to_discord inline @tool
- triage_email typed sub-agent
- save_skill (skills_dir is set; the system-prompt section tells the
  agent it can save reusable code)

That's all. The user just asks. Whatever the agent decides — what
tools to use, how to format output, whether to parameterize, what to
name a saved skill, whether to save one at all — is on the agent.
"""

from __future__ import annotations

import os
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
    MCPServer,
    Sandbox,
    TraceWriter,
)
# Reuse the sub-agent and Discord tool from the prescribed version
from morning_brief import gmail_mcp, post_to_discord, triage_email  # noqa: E402


SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


# The user prompt. ONE line. No scaffolding. This is what someone
# would actually type into a `coda "..."` CLI.
USER_PROMPT = (
    "Give me a morning brief from my Gmail and post it to Discord. "
    "If you end up writing code you'd want to reuse next time, "
    "save it."
)


def main() -> int:
    if not WEBHOOK_URL:
        print("Set DISCORD_WEBHOOK_URL", file=sys.stderr)
        return 1
    if not (Path.home() / ".gmail-mcp" / "credentials.json").exists():
        print("missing ~/.gmail-mcp/credentials.json", file=sys.stderr)
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="coda-brief-thin-")).resolve()
    trace_dir = Path(__file__).resolve().parent.parent / "runs"
    trace_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"morning_brief_thin_{stamp}.jsonl"

    print(f"workspace: {workspace}")
    print(f"trace:     {trace_path}")
    print(f"skills:    {SKILLS_DIR} ({len(list(SKILLS_DIR.glob('*.py')))} files)")
    print(f"user prompt: {USER_PROMPT!r}")
    print()

    hooks = HookRegistry()
    sandbox = Sandbox(root=workspace, hooks=hooks, bash_timeout_s=120)

    t0 = time.monotonic()
    with TraceWriter(trace_path).attach(hooks) as tracer:
        with Agent(
            model="claude-opus-4-7",
            llm=ClaudeAgentSDKClient(cwd=str(workspace)),
            sandbox=sandbox,
            hooks=hooks,
            subagents=[triage_email],
            tools=[post_to_discord],
            mcp_servers=[gmail_mcp()],
            skills_dir=SKILLS_DIR,
            max_turns=25,
            max_tokens=8192,
        ) as agent:
            result = agent.run(USER_PROMPT)
    elapsed = time.monotonic() - t0

    print()
    print("=== RUN SUMMARY ===")
    print(f"elapsed: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"turns: {result.turns}, tokens in/out: {result.input_tokens}/{result.output_tokens}")
    print(f"events: {tracer.events_written}")
    print()
    print("--- agent final text (last 2000 chars) ---")
    print(result.text[-2000:])

    saved = sorted(SKILLS_DIR.glob("*.py"))
    print(f"\n--- skills/ after run: {[f.name for f in saved]} ---")
    for f in saved:
        if f.name == "__init__.py":
            continue
        print(f"\n>>> {f.name} ({f.stat().st_size} bytes)")
        print(f.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
