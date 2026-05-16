"""Multi-MCP demo: agent-framework convergence/divergence study.

The agent pulls real metrics + READMEs + source code for 6 agent
frameworks (including coda itself), runs a typed sub-agent to extract a
structured profile per framework, then a second sub-agent on every
pair to surface agreements/divergences. Synthesizes a comparison
report and posts a Discord summary.

Three real MCPs run in parallel:
- `fetch`      — pull READMEs / docs / blog URLs as markdown
- `github`     — repo metrics (stars, recent commits, languages, releases)
                 via the official server, authed with `gh auth token`
- `coda_src`   — filesystem server scoped read-only at the coda repo so
                 the agent can read coda's actual source code as the
                 ground-truth source for coda's profile (instead of
                 re-reading the README from the repo it's running in)

Plus:
- `assess` typed sub-agent → FrameworkProfile (11 fields, citations
  required)
- `compare` typed sub-agent → PairComparison (agreements/divergences,
  significance)
- `post_to_discord` inline @tool (URL in closure)
- coda's native primitives (ls/glob/grep/read/write/bash) for the report
  workspace, separate from the filesystem MCP scope

Cost envelope: ~20-30 min on Opus 4.7, single-digit dollars of
subscription credit. Will fail loudly if GITHUB_PERSONAL_ACCESS_TOKEN
isn't set or if `npx` can't pull the MCP servers.
"""

from __future__ import annotations

import os
import subprocess
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


CODA_ROOT = Path("/Users/theophilushomawoo/coda")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


FRAMEWORKS = [
    ("claude-agent-sdk", "anthropics/claude-agent-sdk-python"),
    ("OpenHands",        "All-Hands-AI/OpenHands"),
    ("LangGraph",        "langchain-ai/langgraph"),
    ("AutoGen",          "microsoft/autogen"),
    ("smolagents",       "huggingface/smolagents"),
    ("coda",             "theohmwoa/coda"),
]


# --- typed sub-agents ----------------------------------------------------


class FrameworkProfile(BaseModel):
    name: str
    repo_url: str
    stars: int                    # approximate is fine
    primary_language: str
    agent_loop_style: Literal["react", "codeact", "graph", "swarm", "linear", "hybrid"]
    tool_protocol: Literal["json_tool_use", "code", "mcp_native", "openai_functions", "hybrid"]
    sub_agent_support: bool
    observability: Literal["none", "logs", "structured_events"]
    state_model: str              # one-line: how state persists between turns
    distinguishing_feature: str   # what's unusual / unique
    citations: list[str]          # at least 2 URLs that back the above


class PairComparison(BaseModel):
    framework_a: str
    framework_b: str
    agreements: list[str]         # 0..3 short bullets
    divergences: list[str]        # 0..3 short bullets
    significance: Literal["minor", "notable", "fundamental"]


@subagent(model="claude-haiku-4-5", max_tokens=1500)
def assess(framework_name: str, sources: list[dict]) -> FrameworkProfile:
    """Build a structured profile of an agent framework from primary sources.

    `sources` is a list of {url, kind, content} dicts the caller has
    already fetched (READMEs, docs pages, github metric blobs, source
    code excerpts). DO NOT invent fields not present in the sources —
    if you can't tell from the material, pick the most defensible
    minimum for that field. Always populate `citations` with at least
    two URLs taken from the sources.

    Be terse: state_model and distinguishing_feature are ONE LINE each.
    """


@subagent(model="claude-haiku-4-5", max_tokens=1000)
def compare(profile_a: dict, profile_b: dict) -> PairComparison:
    """Compare two FrameworkProfile dicts. Return short bullets.

    `agreements`: things both frameworks treat similarly (e.g. both use
    json tool-use, both lack sub-agent support).
    `divergences`: things they treat differently.

    `significance` = `fundamental` if the divergences imply a different
    mental model (e.g. CodeAct vs JSON tool-use); `notable` if they're
    real but not paradigm-defining; `minor` for surface differences.
    """


# --- inline tool ---------------------------------------------------------


@tool
def post_to_discord(message: str) -> dict:
    """Post a message to the announcements Discord channel.

    Args:
        message: text or Discord markdown. Truncated to 1900 chars.

    Returns:
        {"status": int, "ok": bool}. status=204 means delivered.
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
            "User-Agent": "coda-research/0.1 (+https://github.com/theohmwoa/coda)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "ok": 200 <= resp.status < 300}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "ok": False, "error": str(e)}


# --- prompt --------------------------------------------------------------


FRAMEWORK_LIST_STR = "\n".join(f"- {n}: https://github.com/{r}" for n, r in FRAMEWORKS)


USER_PROMPT = f"""\
You are conducting a structured comparative study of open-source agent
frameworks as of mid-2026. Goal: produce a `analysis.md` report and a
Discord summary that identifies CONVERGENCE (what the field agrees on)
and DIVERGENCE (where the open design choices live), with cited
evidence for every claim.

# Frameworks (6 total)

{FRAMEWORK_LIST_STR}

# Available MCPs (already connected)

- `fetch.fetch(url, max_length=...)`  — return a URL's content as
  cleaned markdown. Use for READMEs, docs pages, blog posts.

- `github.*`  — official GitHub MCP server. Useful tools include
  `get_file_contents(owner, repo, path)`, `list_commits`,
  `search_code`, `list_releases`. Use this for objective repo
  metrics (stars, recent commit cadence, releases) and for
  reading specific source files from each framework's repo.

- `coda_src.*`  — filesystem MCP scoped READ-ONLY at the local coda
  repo. Use `coda_src.list_directory(path='src/coda')` and
  `coda_src.read_text_file(path=...)` for coda's ground-truth source
  (instead of re-fetching its README). This is how you build coda's
  FrameworkProfile.

Use `ls`/`read`/`write` (coda's native primitives) for THIS workspace
— that's where you'll write `analysis.md`. The filesystem MCP is for
the coda repo only.

# Typed sub-agents

- `assess(framework_name, sources) -> FrameworkProfile`
  Pass a list of {{url, kind, content}} dicts. Each `kind` is one of
  "readme", "doc", "source", "metrics". Sub-agent returns a typed
  profile. Citations REQUIRED (at least 2 URLs from `sources`).

- `compare(profile_a, profile_b) -> PairComparison`
  Pass the dict form of two FrameworkProfile objects
  (`.model_dump()`). Returns agreements/divergences/significance.

# Workflow

Step 1 — gather per framework (6 iterations)
  For each (name, repo) in the list:
    a) Fetch the repo's README (use github.get_file_contents or
       fetch.fetch on the raw URL).
    b) Pull repo metadata (stars, primary language, recent activity)
       via the GitHub MCP — exactly which tools you call is up to you,
       but get the numbers from the API, not from the README.
    c) For coda, additionally use coda_src to read 2-3 key files:
       src/coda/agent.py, src/coda/sandbox.py, src/coda/subagents.py.
       These are the ground truth.
    d) Optionally fetch ONE additional doc page if the README is too
       thin.
    e) Call `assess(framework_name=name, sources=[...])`. Store the
       returned FrameworkProfile.

Step 2 — pairwise comparison (15 iterations)
  For each pair (i < j) of the 6 profiles, call
  `compare(profile_a=p_i.model_dump(), profile_b=p_j.model_dump())`.

Step 3 — synthesize
  Write `analysis.md`:
    # Open-source agent frameworks — convergence/divergence (May 2026)

    ## At a glance
    A 6-row markdown table: name | stars | loop style | tool protocol
    | sub-agents | observability.

    ## Convergence
    3-6 bullets — patterns shared by >=4 of the 6 frameworks, citing
    which frameworks for each.

    ## Divergence
    3-6 bullets — design choices that genuinely split the field.
    Reference the PairComparison `fundamental`-significance items.

    ## Where coda fits
    2-4 paragraphs — be honest. Which design choices coda picked,
    which competing approaches it doesn't try to be. Cite source
    files via coda_src paths.

    ## Citations
    A flat list of every URL referenced.

Step 4 — Discord summary
  Call `post_to_discord(message=...)`. Include:
  - one-line headline of the most fundamental divergence found
  - 3 bullet-point findings
  - link to the local report path
  Use Discord markdown. Max ~1500 chars.

Step 5 — finish
  Print: number of profiles built, number of pairs compared, the
  report path, the Discord status. Then prose to end.

# Quality bar

- Every claim in the report must be traceable to a URL or coda_src
  path that appears in the citations list. Sub-agent enforces this
  for individual profiles; for the synthesis sections it's on you.
- The "Where coda fits" section MUST cite specific lines/functions
  via coda_src — not "coda has X" but "coda's `Agent.run()` at
  src/coda/agent.py implements X".
- If you can't justify a divergence/convergence claim with evidence,
  drop it. Better fewer bullets, all backed.

You have access to: ls/glob/grep/read/write/bash (workspace),
fetch.*, github.*, coda_src.* (MCPs), assess() and compare() (typed
sub-agents), post_to_discord() (inline tool).
"""


def setup_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="coda-framework-")).resolve()
    return workspace


def main() -> int:
    if not WEBHOOK_URL:
        print("set DISCORD_WEBHOOK_URL", file=sys.stderr)
        return 1
    # Need a GH token for the github MCP. Pull it from gh CLI auth.
    gh_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not gh_token:
        try:
            gh_token = subprocess.check_output(
                ["gh", "auth", "token"], text=True, timeout=10
            ).strip()
        except Exception as e:
            print(f"could not get gh token: {e}", file=sys.stderr)
            return 1

    workspace = setup_workspace()
    trace_dir = Path(__file__).resolve().parent.parent / "runs"
    trace_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"framework_compare_{stamp}.jsonl"

    print(f"workspace: {workspace}")
    print(f"trace:     {trace_path}")
    print(f"coda src:  {CODA_ROOT}")
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
            subagents=[assess, compare],
            tools=[post_to_discord],
            mcp_servers=[
                # fetch is Python-based, distributed via PyPI as
                # mcp-server-fetch and run via uvx. There is no npm
                # `@modelcontextprotocol/server-fetch`.
                MCPServer.stdio(
                    "fetch",
                    command="uvx",
                    args=["mcp-server-fetch"],
                ),
                MCPServer.stdio(
                    "github",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-github"],
                    env={"GITHUB_PERSONAL_ACCESS_TOKEN": gh_token},
                ),
                MCPServer.stdio(
                    "coda_src",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", str(CODA_ROOT)],
                ),
            ],
            max_turns=35,
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
    print("--- agent final text (last 3000 chars) ---")
    print(result.text[-3000:])

    # Preserve report
    report = workspace / "analysis.md"
    if report.exists():
        keep = trace_dir / f"framework_compare_{stamp}.md"
        import shutil
        shutil.copy(report, keep)
        print()
        print(f"report kept at: {keep}")
        print()
        print("--- first 100 lines of analysis.md ---")
        print("\n".join(report.read_text().splitlines()[:100]))
    else:
        print("(no analysis.md produced)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
