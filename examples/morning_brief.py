"""Morning brief demo — Gmail + Discord + skill curation.

Phase 1 (build): agent reads last-24h Gmail, runs a typed sub-agent on
each message (importance / category / needs_reply / one_line summary),
groups by importance, posts a Discord brief, and — explicitly told it
has this capability — calls `save_skill(...)` to crystallize the
workflow as a reusable function in `./skills/`.

Phase 2 (use): a SECOND coda Agent constructed against the same
`skills_dir` boots, the saved skill is auto-loaded into its sandbox
globals, and a one-line prompt ("give me the brief") should make the
agent reach for the skill instead of redoing the whole workflow from
scratch. Expected: turn 1 calls the skill, turn 2 ends the loop.

This is the loop that makes coda accumulate capability across runs.

Requires:
- ~/.gmail-mcp/credentials.json from the OAuth flow
- DISCORD_WEBHOOK_URL env var
- claude CLI logged in (for ClaudeAgentSDKClient subscription auth)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coda import (  # noqa: E402
    Agent,
    ClaudeAgentSDKClient,
    HookRegistry,
    MCPServer,
    Sandbox,
    TraceWriter,
    subagent,
    tool,
)


WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
# Persist skills under the project so both phases see the same library.
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


# --- typed sub-agent ----------------------------------------------------


class EmailTriage(BaseModel):
    importance: Literal["critical", "high", "normal", "low"]
    category: Literal[
        "work", "personal", "newsletter", "promotion",
        "automated", "security", "social", "spam",
    ]
    needs_reply: bool
    one_line: str  # one-sentence summary


@subagent(model="claude-haiku-4-5", max_tokens=400)
def triage_email(email: dict) -> EmailTriage:
    """Classify one Gmail message.

    `importance`:
    - critical: account/security/billing problem; deadline today
    - high: real person asking something or expecting a reply
    - normal: real but not urgent (FYI, scheduled, info)
    - low: bulk / automated / promotion / social notification

    `needs_reply` is True only if a real human is waiting on a response
    from the user. Newsletters and automated notifications are never
    needs_reply, even if they have a "reply" button.

    `one_line`: ONE concise sentence. No filler. State who and what.
    """


# --- inline tool --------------------------------------------------------


@tool
def post_to_discord(message: str) -> dict:
    """Post a message to the announcements Discord channel.

    Args:
        message: text or Discord markdown. Max 1900 chars.

    Returns:
        {"status": int, "ok": bool}. status=204 means accepted.
    """
    import json as _json
    import urllib.error
    import urllib.request

    if not WEBHOOK_URL:
        return {"status": 0, "ok": False, "error": "DISCORD_WEBHOOK_URL not set"}
    payload = _json.dumps({"content": message[:1900]}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "coda-morning-brief/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "ok": 200 <= resp.status < 300}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "ok": False, "error": str(e)}


# --- phase 1 prompt -----------------------------------------------------


PHASE1_PROMPT = """\
You're building a morning brief from the user's Gmail and posting it
to Discord. Then you will crystallize what you did as a reusable
skill so future runs can call it directly.

# Step 1 — fetch

Use the `gmail` MCP. The relevant tools are:
- `gmail.search_emails(query=..., maxResults=...)`
- `gmail.read_email(messageId=...)`

Use a Gmail search query for the last 24 hours, unread or recent:
    query = "newer_than:1d category:primary"
Cap maxResults at 25. The `search_emails` result gives you a list of
message stubs (id + snippet). For each stub, call `read_email` to get
the headers (From, Subject) — the body isn't needed for triage.

# Step 2 — triage each one

For each fetched message, call the typed sub-agent:

    verdict = triage_email(email={
        "from": "...",
        "subject": "...",
        "snippet": "...",   # first ~200 chars
    })

This returns an `EmailTriage` Pydantic. Collect (msg, verdict) pairs.

# Step 3 — group + post Discord

Build a Discord message with this structure:

    **📬 Morning brief — <date>**
    **🔴 Critical:** <n>
    <bullets, one per critical item>
    **🟠 High (needs reply):** <n>
    <bullets, one per needs_reply item>
    **🟢 Normal:** <n>  • **⚪ Low:** <m>

Each bullet: `- <from>: <one_line>`. Keep critical/high bullets;
omit normal/low bullets but show their counts. Cap message at 1800
chars. Call `post_to_discord(message=...)`.

# Step 4 — SAVE THE SKILL (important — this is the point of the demo)

You ARE able to save reusable workflows via `save_skill(code=...)`.
The system prompt explained when and how. Now do it.

Look at the code you just wrote. Generalize it into a reusable
function. Specifically:
- The Gmail search query's time window should be a parameter
  (`hours_back: int = 24`).
- The Discord webhook is handled by the inline `post_to_discord` tool
  which is available in any future run that includes it — assume that.
- The triage sub-agent is `triage_email`. Future runs that include it
  in their `subagents=[...]` list will have it available. Assume that.
- Return the brief STRING so callers can do whatever with it; ALSO
  post to Discord inside the function (the existing behavior).
- Name the function `morning_brief`.

Call `save_skill(code='...')` passing the full function definition.
Verify the response is `{"ok": True, ...}` — if not, fix and retry.

# Step 5 — finish

Print: number of emails triaged, the importance counts, Discord post
status, and the path returned by save_skill. Then prose to end.
"""


# --- phase 2 prompt -----------------------------------------------------


PHASE2_PROMPT = """\
Give me a morning brief.

You have a `morning_brief` skill saved in your skills library — see
your system prompt for its signature. Use it directly; do not redo
the whole workflow by hand. Default the time window to 24 hours.

Print what the skill returned (or its top 20 lines if long), the
Discord status, then prose to end.
"""


# --- runner -------------------------------------------------------------


def setup_workspace(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix)).resolve()


def gmail_mcp() -> MCPServer:
    return MCPServer.stdio(
        "gmail",
        command="npx",
        args=["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
    )


def run_phase(phase_name: str, user_prompt: str) -> dict:
    workspace = setup_workspace(f"coda-{phase_name}-")
    trace_dir = Path(__file__).resolve().parent.parent / "runs"
    trace_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"morning_brief_{phase_name}_{stamp}.jsonl"

    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name}")
    print(f"workspace: {workspace}")
    print(f"trace:     {trace_path}")
    print(f"skills:    {SKILLS_DIR} ({len(list(SKILLS_DIR.glob('*.py'))) if SKILLS_DIR.exists() else 0} files)")
    print(f"{'='*60}\n")

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
            max_turns=20,
            max_tokens=8192,
        ) as agent:
            print(f"[skills loaded for {phase_name}: {list(agent.skills.keys())}]")
            result = agent.run(user_prompt)
    elapsed = time.monotonic() - t0

    print(f"\n--- {phase_name} done ---")
    print(f"elapsed: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"turns: {result.turns}, tokens in/out: {result.input_tokens}/{result.output_tokens}")
    print(f"events: {tracer.events_written}")
    print(f"\n--- final agent text (last 2000 chars) ---")
    print(result.text[-2000:])
    print()
    return {
        "phase": phase_name,
        "elapsed": elapsed,
        "turns": result.turns,
        "tokens_in": result.input_tokens,
        "tokens_out": result.output_tokens,
        "trace": str(trace_path),
        "workspace": str(workspace),
        "executions": len(result.executions),
    }


def main() -> int:
    if not WEBHOOK_URL:
        print("Set DISCORD_WEBHOOK_URL", file=sys.stderr)
        return 1
    creds = Path.home() / ".gmail-mcp" / "credentials.json"
    if not creds.exists():
        print(f"missing {creds} — finish the Gmail OAuth flow first", file=sys.stderr)
        return 1

    phase = sys.argv[1] if len(sys.argv) > 1 else "both"

    if phase in ("phase1", "both"):
        r1 = run_phase("phase1_build", PHASE1_PROMPT)
        # Show what landed in skills/
        saved = sorted(SKILLS_DIR.glob("*.py")) if SKILLS_DIR.exists() else []
        print(f"\n--- skills/ after phase 1: {[f.name for f in saved]} ---")
        for f in saved:
            print(f"\n>>> {f.name} ({f.stat().st_size} bytes)")
            print(f.read_text())

    if phase in ("phase2", "both"):
        # Brief pause so the user can scan phase 1 output
        if phase == "both":
            print("\n[pausing 5s before phase 2 so you can read phase 1 output]")
            time.sleep(5)
        r2 = run_phase("phase2_use", PHASE2_PROMPT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
