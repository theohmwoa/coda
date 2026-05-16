"""On-call triage agent — TheAgentCompany-shaped end-to-end workflow.

What it does: you're the on-call SWE for `anthropics/claude-agent-sdk-python`.
Triage every open PR and open issue updated in the last 7 days. For each one,
classify severity + category + who to ping. Write TRIAGE.md grouped by
severity, then post a summary to a Discord channel.

Why this shape: TAC-style "long-horizon multi-tool branching workflow with a
verifiable outcome". Exercises every piece of coda at once:

- bash + gh CLI for GitHub state
- a typed @subagent (`assess`) called inside a `for` loop — one fresh-context
  LLM call per item, outer context stays clean. The MirageAI gem, dogfooded.
- @tool for the Discord webhook (URL never lands in the LLM input)
- TraceWriter for the full event log → an artifact a recruiter can replay
- ClaudeAgentSDKClient so the run bills to a paid Claude subscription, not
  API credit

Requires:
- gh CLI authenticated (`gh auth status`)
- DISCORD_WEBHOOK_URL in env
- Logged in to Claude Code (`claude login`)

Cost envelope (rough): ~20 sub-agent calls × ~2K input tok each + main
agent ~10 turns. Single-digit cents on Sonnet+Haiku.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coda import (  # noqa: E402
    Agent,
    ClaudeAgentSDKClient,
    HookRegistry,
    Sandbox,
    TraceWriter,
    subagent,
    tool,
)


# --- config -------------------------------------------------------------

TARGET_REPO = "anthropics/claude-agent-sdk-python"
DAYS = 7
SINCE = (datetime.now(timezone.utc) - timedelta(days=DAYS)).strftime("%Y-%m-%d")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


# --- sub-agent ----------------------------------------------------------


class RiskAssessment(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    category: Literal[
        "bug", "security", "stale", "review_needed", "feature", "docs", "info"
    ]
    who_to_ping: str | None
    rationale: str


@subagent(model="claude-haiku-4-5", max_tokens=512)
def assess(item: dict) -> RiskAssessment:
    """Triage one open PR or open issue.

    `severity` reflects on-call urgency:
    - critical: security vuln, production-breaking bug, data loss risk
    - high: significant bug or release blocker
    - medium: needs review but not urgent; stale PR with conflicts
    - low: typo, minor docs, low-priority feature

    `category`:
    - bug:            a defect in the code/behavior
    - security:       anything CVE-adjacent or auth/credential-related
    - stale:          PR open >2 weeks with no recent push, or awaiting author
    - review_needed:  ready PR with no recent reviewer activity
    - feature:        new functionality proposal
    - docs:           documentation only
    - info:           question / RFC / discussion

    `who_to_ping`: a single GitHub username with no leading `@`, or None.
    Usually the PR author for stalled PRs that need their attention, OR None
    for low-priority/info items that don't need a ping.

    Be decisive. Read the body when classifying; one-line PR titles are not
    enough.
    """


# --- inline tool: Discord webhook --------------------------------------


@tool
def post_to_discord(message: str) -> dict:
    """Post a single message to the on-call Discord channel.

    Args:
        message: plain text or Discord markdown (`**bold**`, `` `code` ``).
            Discord's per-message limit is 2000 chars; longer messages are
            truncated.

    Returns:
        `{"status": int, "ok": bool}`. status==204 means the webhook
        accepted the post.
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
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "ok": 200 <= resp.status < 300}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "ok": False, "error": str(e)}


# --- user prompt -------------------------------------------------------

USER_PROMPT = f"""\
You are the on-call SWE for `{TARGET_REPO}`. Triage everything that
needs attention from the last {DAYS} days.

## Step 1 — gather

Use bash + `gh` CLI:

    bash("gh pr list -R {TARGET_REPO} --state open --search 'updated:>={SINCE}' --limit 50 --json number,title,author,createdAt,updatedAt,isDraft,labels,body")
    bash("gh issue list -R {TARGET_REPO} --state open --search 'updated:>={SINCE}' --limit 50 --json number,title,author,createdAt,updatedAt,labels,body")

Parse the JSON (`json.loads(result['stdout'])`). Expect roughly 10-15 PRs
and ~10 issues. Truncate each item's `body` to 800 chars before passing
to the sub-agent — full bodies blow context for no benefit.

## Step 2 — assess each one

For every PR and every issue, call the typed sub-agent:

    verdict = assess(item={{...}})  # fresh-context LLM call, returns RiskAssessment

Pass a dict that includes: number, title, author (just the login string,
not the full nested user object), body (truncated), createdAt, updatedAt,
isDraft (if PR), labels (list of names). Collect `(item, verdict)` pairs.

## Step 3 — write TRIAGE.md

Group by severity (critical → high → medium → low). Format each item:

    ### #NUM TITLE
    - **Category**: <category>
    - **Author**: @<author>
    - **Updated**: <date>
    - **Ping**: @<who_to_ping> (or "no ping")
    - **Rationale**: <one-line>

Put a header at the top with the date, the repo, and the total counts
per severity. Write to TRIAGE.md.

## Step 4 — post Discord summary

Call the inline tool `post_to_discord(message=...)`. The summary should:
- Open with `**🔧 On-call triage — {TARGET_REPO}**`
- Show counts per severity in one line
- List the top 3 items by severity (critical first), each with #num, title,
  and the ping target if any
- Close with a link: `https://github.com/{TARGET_REPO}/pulls`

Keep it under 1900 chars. Use Discord markdown.

## Step 5 — finish

Print: TRIAGE.md path, total items triaged, severity counts, Discord post
status. Then reply with prose (no code block) to end the loop.
"""


# --- main ------------------------------------------------------------


def main() -> int:
    if not WEBHOOK_URL:
        print("Set DISCORD_WEBHOOK_URL", file=sys.stderr)
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="coda-oncall-")).resolve()
    trace_dir = Path("runs")
    trace_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"oncall_triage_{stamp}.jsonl"

    print(f"workspace: {workspace}")
    print(f"trace:     {trace_path}")
    print(f"target:    {TARGET_REPO} (open items updated since {SINCE})")
    print("--- starting agent run ---")
    print()

    hooks = HookRegistry()
    sandbox = Sandbox(root=workspace, hooks=hooks, bash_timeout_s=120)

    t0 = time.monotonic()
    with TraceWriter(trace_path).attach(hooks) as tracer:
        with Agent(
            model="claude-sonnet-4-6",
            llm=ClaudeAgentSDKClient(cwd=str(workspace)),
            sandbox=sandbox,
            hooks=hooks,
            subagents=[assess],
            tools=[post_to_discord],
            max_turns=15,
        ) as agent:
            result = agent.run(USER_PROMPT)
    elapsed = time.monotonic() - t0

    print()
    print("=== RUN SUMMARY ===")
    print(f"elapsed:         {elapsed:.1f}s")
    print(f"coda turns:      {result.turns}")
    print(f"tokens:          in={result.input_tokens} out={result.output_tokens}")
    print(f"events captured: {tracer.events_written}")
    print(f"trace:           {trace_path}")

    triage = workspace / "TRIAGE.md"
    if triage.exists():
        keep = trace_dir / f"oncall_triage_{stamp}.md"
        shutil.copy(triage, keep)
        print(f"report:          {keep}")
        print()
        print("--- TRIAGE.md (first 80 lines) ---")
        print("\n".join(triage.read_text().splitlines()[:80]))
    else:
        print("(no TRIAGE.md was produced)")
        print("final agent text:")
        print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
