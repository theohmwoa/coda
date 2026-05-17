# coda

> A Python agent harness with line-level observability, schema-aware lint, agent-curated skills, and a real human-in-the-loop UI. Plug in MCP servers, type a task, watch it happen.

---

## What it does in 30 seconds

You install coda, point it at your Gmail (or any MCP server), run `python -m coda serve`, open `localhost:8000`, and type *"give me a morning brief"*. Coda's agent reads your inbox, classifies each message via a typed sub-agent, drafts a reply when you ask for one, and **pauses for your approval before sending**. Every line of the agent's working Python is captured as a structured event; if it does something dumb, you can replay it. If it does something useful, the agent saves it as a reusable skill that auto-loads next time.

## Status

- **174 tests** passing in ~1s
- Real PR submitted to [`arrow-py/arrow #1259`](https://github.com/arrow-py/arrow/pull/1259) — coda diagnosed and patched a real bug end-to-end
- **Self-debugged its own failure mode** — handed a fresh agent a failed past trace, it identified a bug the maintainer (me) missed in a prior fix, proposed the right patch, the patch landed
- Live Gmail + Discord round-trips working today via a paid Claude plan (no API key, no API budget)
- ~3,200 LoC runtime, MIT, single-user localhost-first

## Install

Python ≥ 3.11. Two runtime deps (`anthropic`, `pydantic`). Optional extras pull in MCP, the Agent SDK, and the live UI server.

```bash
git clone https://github.com/theohmwoa/coda
cd coda
pip install -e ".[dev,mcp,agent-sdk,server]"
pytest                                          # 174 tests, ~1s

# build the UI (vite — Node 18+)
cd app && npm install && npm run build && cd ..

# offline smoke test (no API key, no network)
python examples/scripted_demo.py
```

## Quick start — the live UI

```bash
# subscription auth: log into Claude Code once, then run coda serve
claude login                                    # one-time
unset ANTHROPIC_API_KEY                         # don't shadow the OAuth token
DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...' \
  python -m coda serve --port 8000
```

Open `http://localhost:8000`. You'll see:

- **Header** — model, MCPs, sub-agents, and currently-loaded skills as ambient chips
- **Greeting + skill cards** — the saved workflows the agent has crystallized over past runs; click one to re-run it
- **Chat input** — type a free-form task, `⌘↵` to send
- **Live status pane** — humanized step-by-step ("📧 Reading recent emails", "🤔 Classifying email from Maya", "✍️ Drafting an email"), with timing
- **Inline approval cards** — when the agent calls `approve("send_email", payload)`, a review card lands in the message stream with an email-compose form (vendored from [agent-feedback-ui](https://github.com/theohmwoa/agent-feedback-ui))
- **Result card** — final agent response rendered as markdown
- **Run summary** — turns, tokens, elapsed time

For debugging, `/debug` exposes the raw event timeline with code execution highlighting.

## Quick start — the library

```python
import coda
from typing import Literal
from pydantic import BaseModel


class Triage(BaseModel):
    priority: Literal["low", "medium", "high"]
    category: Literal["bug", "feature", "refactor"]
    rationale: str


@coda.subagent(model="claude-haiku-4-5")
def triage(todo: dict) -> Triage:
    """Classify one TODO comment."""


agent = coda.Agent(
    model="claude-sonnet-4-6",
    llm=coda.ClaudeAgentSDKClient(),     # subscription auth via the official SDK
    subagents=[triage],
)
result = agent.run(
    "Find every TODO/FIXME comment under the current directory and triage "
    "each one. Write a markdown report to triage.md."
)
print(result.text)
```

The agent uses `glob` to find Python files, `grep` to find TODOs, then calls `triage(...)` inside a `for` loop — one fresh-context LLM call per item, each returning a validated `Triage` Pydantic. Finally it calls `write("triage.md", ...)`.

## What makes it different

The substrate, not the agent loop itself. Every framework has an agent loop now. Coda has four things no other open framework combines:

### 1. Line-level observability via `sys.settrace`
Every line of model-emitted Python emits a structured event. The trace JSONL captures emissions, executions, tool calls, sub-agent invocations, line-by-line execution, errors. Replay any past run with `python -m coda replay <trace.jsonl>` — colored timeline, filterable by event type. The trace is also what makes self-debugging possible (see *Demos* below).

### 2. Schema-aware AST lint, every code block
After every code block the agent emits, an AST walker checks comparisons against the connected sub-agents' Pydantic `Literal` schemas. If the agent writes `if t.severity == "medium"` and the schema only allows `low|high`, it's flagged as **dead code** in the next user message and the agent fixes it before continuing. Caught a real bug live during development that text-grep couldn't find — `t["importance"] == "medium"` against `Literal["critical","high","normal","low"]`.

### 3. Agent-curated skills
When a workflow succeeds, the agent calls `save_skill(code=...)` to crystallize what it just did as a reusable Python function in `skills/`. Next run with the same `skills_dir` auto-loads it as a callable in the sandbox globals — same task, ~10× fewer tokens. The harness *learns from itself* over time.

### 4. Human-in-the-loop that the agent can't bypass
Sandbox primitive `approve("send_email", payload)` blocks the agent and surfaces a typed UI component for review. User clicks Send / Edit / Discard; the agent resumes with the (possibly edited) payload. Crucially, `approve()` returns the payload directly — the agent has no other code path to the final action. Skip the approval, lose the destination.

## The CodeAct loop (one paragraph)

The agent emits Python in fenced code blocks; coda execs it against a persistent sandbox; stdout / stderr / lint warnings feed back to the model. Variables in turn N are visible in turn N+1. Native primitives (`ls`, `glob`, `grep`, `read`, `write`, `edit`, `bash`) are pre-imported into the sandbox globals so the model uses them instead of `os.listdir`. MCP servers are exposed as Python objects: the model writes `gmail.search_emails(query="...")` and the call dispatches to the underlying stdio MCP server. Sub-agents are typed: `@coda.subagent` on a function with a Pydantic return type generates an Anthropic forced tool-use schema; the function is callable from inside a `for` loop and each call is a fresh-context LLM completion with a validated typed return.

## Pluggable LLM backends

| backend | how it bills | needs | works today |
|---|---|---|---|
| `MockLLMClient` | — | nothing | tests, offline demos |
| `ClaudeCodeClient` | your Claude plan | `claude` CLI logged in | yes (subprocess per call) |
| `ClaudeAgentSDKClient` | your Claude plan | `claude-agent-sdk` + `claude login` | **yes**, in-process, no fork per call |
| `AnthropicClient` | API key | `ANTHROPIC_API_KEY` | yes |

`ClaudeAgentSDKClient` is the recommended dev backend pre-2026-06-15 — no API budget, no per-call fork.

## MCP servers as Python objects

```python
import coda

with coda.Agent(
    model="claude-sonnet-4-6",
    llm=coda.ClaudeAgentSDKClient(),
    mcp_servers=[
        coda.MCPServer.stdio(
            "gmail",
            command="npx",
            args=["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        ),
        coda.MCPServer.stdio(
            "github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."},
        ),
    ],
) as agent:
    agent.run("Summarize the last 24h of unread Gmail and post the digest to Discord.")
```

Each MCP server connects on `Agent` construction and is auto-included in the prompt's MCP section. Tool names with hyphens (`get-env`) are sanitized to valid Python identifiers (`get_env`). Use the `Agent` as a context manager so stdio subprocesses close cleanly.

## Demos

Each demo ships with its full event trace under `examples/sample_run/` or `runs/`. Open the trace with `python -m coda replay <path>` or read the rendered artifact:

| demo | what it does | artifact |
|---|---|---|
| `examples/scripted_demo.py` | TODO triage with canned `MockLLMClient` responses — runs offline, deterministic | (stdout) |
| `examples/triage_todos.py` | Same workflow, real LLM, real Gmail-style triage | `examples/sample_run/triage.md` |
| `examples/oncall_triage.py` | On-call SWE for `anthropics/claude-agent-sdk-python` — pulls live PRs+issues via `gh`, sub-agent per item, Discord summary | `examples/sample_run/triage.md`, `trace.jsonl` |
| `examples/framework_compare.py` | Comparative architecture study of 6 agent frameworks via 3 MCPs (`fetch` + `github` + `filesystem`), 6 typed `assess` + 15 pairwise `compare` sub-agent calls | `examples/sample_run/framework_compare.md` |
| `examples/dogfood_replay_cli.py` | **Coda built coda's own replay CLI** in 3 turns, 123/123 tests green first try | `examples/sample_run/dogfood_trace.jsonl` |
| `examples/oss_pr_arrow_1259.py` | Coda diagnosed and submitted a real PR to `arrow-py/arrow #1259` | `runs/arrow_1259_diff.patch` |
| `examples/self_debug.py` | Coda read a failed run's trace, found a runtime bug, proposed the fix | `examples/sample_run/self_debug_diagnosis.md` |
| `examples/morning_brief.py` | Two-phase Gmail+Discord brief: phase 1 builds + `save_skill`s the workflow, phase 2 uses the saved skill | `skills/morning_brief_to_discord.py` |

## Architecture

```
src/coda/
  agent.py            # the while-loop, multi-block exec, per-block lint, approval primitive
  sandbox.py          # native primitives, sys.settrace, persistent globals
  hooks.py            # typed event bus
  prompt.py           # cache-aware system-prompt Assembler
  subagents.py        # @coda.subagent — Pydantic return types, forced tool-use, PEP-563 resolution
  tools.py            # @coda.tool — inline callable injection
  skills.py           # load + save_skill, lint-on-save, __globals__ rebinding for sandbox refs
  lint.py             # AST walker, Literal-mismatch checks against live sub-agent schemas
  mcp.py              # stdio MCP servers as Python objects, async-to-sync bridge
  trace.py            # JSONL TraceWriter + read_trace / iter_trace
  replay.py           # python -m coda replay <trace.jsonl>
  server.py           # FastAPI + WebSocket UI server, approval dispatcher
  static/             # bundled UI (debug view + built React app)
  llm/
    base.py           # LLMClient protocol
    mock.py           # MockLLMClient
    anthropic_client.py
    claude_code_client.py
    agent_sdk_client.py  # in-process subscription auth via claude-agent-sdk
  __main__.py         # CLI dispatch (replay / serve)

app/                  # the React end-user UI
  src/
    App.tsx           # composition
    api.ts            # WebSocket client
    components/       # SkillCard, ApprovalCard, LiveStatus, InputDock, …
    lib/
      humanize.ts     # event → human-readable status mapping
      agent-feedback/ # vendored email modal from agent-feedback-ui
```

## Origin

This started at the end of a two-year engineering project — **MirageAI** (Epitech EIP, 2024–2026). We built a production CodeAct backend: an LLM workspace where the model generated `workflow.py` files calling typed `interfaces/` instead of emitting tool-call JSON. Three things kept biting every tool-calling agent we tried:

- **Loops and branches were impossible without context bloat.** Iterating over 200 items with a reasoning step per item meant 200 round-trips through the main agent's context.
- **Composing tool outputs into a single decision was painful.** Every intermediate value had to round-trip through JSON-serialisable tool results.
- **Workflows that needed sub-agents inside loops were impossible.** No way to call a model with fresh, isolated context from inside a loop body.

Writing Python — with the LLM authoring code that ran in a persistent sandbox, called typed interfaces, and could spawn sub-agent calls *inside loops* — solved all three.

Two years later the industry has converged. The original CodeAct paper (Wang et al., ICML 2024) named the pattern. Microsoft Hyperlight productised it. LlamaIndex shipped `CodeActAgent`. Manus, OpenHands, Claude Code, and most of Anthropic's internal patterns all run on it. What's still missing is a clean, framework-agnostic runtime with first-class observability. `coda` is the clean version of Mirage with what we got wrong rebuilt.

## What I think we got wrong in Mirage (rebuilt)

- **Subprocess-based execution.** Mirage ran each generated workflow in a fresh Python subprocess via Claude Code. Clean isolation but expensive, and persistent state across turns was awkward. `coda` uses an in-process sandbox with `sys.settrace` for line-level instrumentation and an explicit globals dict for state — orders of magnitude faster and instrumentable at the bytecode level.
- **Print-style logs instead of structured events.** Mirage emitted human-readable strings. `coda` emits structured events through a `HookRegistry` — every line execution, every tool call, every variable assignment is a typed event that downstream tools (debuggers, audit, replay, the live UI) subscribe to.
- **One-shot whole-workflow generation.** Mirage's `WorkflowGenerator` generated a complete `workflow.py` upfront. `coda` is iterative — the agent writes a block, sees the output, writes the next block, observing intermediate state. Better for exploration and error recovery.
- **Closed runtime.** Mirage was a service. `coda` is a library plus a localhost server — `pip install -e`, import, run. No microservices.

## Cost notes

From **June 15, 2026**, every paid Claude plan gets a dedicated programmatic-usage credit that covers Agent SDK / `claude -p` / third-party SDK use ([Anthropic announcement](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)): $20/mo on Pro, $100/mo on Max 5x, $200/mo on Max 20x. Billed at full API rates from that pool, non-rollover.

Before then (today): coda runs on subscription auth via `ClaudeCodeClient` (subprocess `claude -p`) or `ClaudeAgentSDKClient` (in-process via the official Python SDK). Both route through Claude Code's OAuth token — make sure `ANTHROPIC_API_KEY` is unset or it'll shadow the OAuth path.

Switch to `AnthropicClient` whenever you want API-key billing.

## What's not yet

- **Hosted / multi-user.** Single-user localhost only — `coda serve` is bound to `127.0.0.1`. Multi-tenant requires per-user MCP lifecycle + auth, not in scope.
- **Approval surfaces beyond `send_email`.** `shell_command`, `sql_query`, `slack_message` stubs exist in the UI's fallback path; only the email modal is fully styled.
- **Scheduling.** No cron-style "run morning_brief at 8 AM" yet.
- **Persistent chat history across server restarts.** State is in-memory per WS session.
- **Skill versioning.** Saving a skill with the same name overwrites; no `.v2` history.
- **Counterfactual replay.** Fork a trace at event N, change a variable, watch the rest diverge — designed for, not built yet.
- **Provenance query language.** *"Did the agent ever read `*.env`? Show me the line."* — the trace has the data; the DSL hasn't been written.
- **`pip install coda` from PyPI.** Source-only for now.

## Prior art

- Wang et al. — *Executable Code Actions Elicit Better LLM Agents* (ICML 2024) — [paper](https://arxiv.org/abs/2402.01030)
- [`xingyaoww/code-act`](https://github.com/xingyaoww/code-act) — reference implementation from the paper authors
- LlamaIndex `CodeActAgent`
- Microsoft Hyperlight CodeAct
- HuggingFace `smolagents` — CodeAct with a sandboxed `exec`
- [`agent-feedback-ui`](https://github.com/theohmwoa/agent-feedback-ui) — the human-in-the-loop component layer coda's UI vendors
- **MirageAI EIP** (Epitech, 2024–2026) — where this started

See `INTERNALS.md` for design principles distilled from the leaked Claude Code source map.

## License

MIT.
