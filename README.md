# coda — a CodeAct coding harness with typed sub-agents and line-level observability

> A Claude-Code-shaped coding harness, packaged as a clean Python library. Native filesystem tools, typed sub-agents inside loops, and `sys.settrace` line-by-line observability for every byte of generated code.

## What coda does

- **The CodeAct loop.** The model emits Python in fenced code blocks; coda execs it against a persistent sandbox; stdout/stderr/error are fed back to the model. Variables you assign in turn N are visible in turn N+1.
- **Native filesystem primitives.** `ls`, `glob`, `grep`, `read`, `write`, `edit`, `bash` — pre-imported into the sandbox globals. The agent uses these instead of `os.listdir`, `cat`, `sed`. Every call emits a structured event.
- **Typed sub-agents.** `@coda.subagent` decorates a function with a Pydantic return type. Each call is a fresh-context LLM completion with forced tool-use schema, returning a validated Pydantic object. Designed to be called from inside a `for` loop without bloating the outer context.
- **Line-level observability.** `sys.settrace` captures every line of model-emitted code. A `HookRegistry` lets you subscribe to typed events — `code_emitted`, `line_executed`, `tool_called`, `subagent_called`, `llm_request`, etc.
- **JSONL trace writer.** `coda.TraceWriter` attaches to a `HookRegistry` and streams every event to a JSONL file. The wire format the future debugger UI will replay.
- **Pluggable LLM backend.** Four backends ship: `MockLLMClient` (tests / canned demos), `ClaudeCodeClient` (`claude -p` subprocess — billed to your Claude plan today), `ClaudeAgentSDKClient` (official `claude-agent-sdk` Python package — same subscription billing, no subprocess fork per call), `AnthropicClient` (direct SDK, API-key billed).
- **Cache-aware prompt assembler.** Sections are marked cache-stable or cache-breaking and rendered in that order, so adding a tool invalidates only the prompt suffix.
- **MCP servers as Python objects.** Connect a stdio MCP server (gmail, slack, github, filesystem, anything from the registry) and every tool it exposes becomes a method on a sandbox global — the model writes `gmail.list_unread(label="INBOX")`, not JSON tool-use blocks. The MirageAI "tools are code, not protocol" pattern, adapted to MCP.

## Install

Python ≥ 3.11. Two runtime deps (`anthropic`, `pydantic`).

```bash
git clone https://github.com/theohmwoa/coda
cd coda
pip install -e ".[dev]"
pytest                          # 92 tests, ~0.2s
python examples/scripted_demo.py   # offline, no API key
```

## Quick start

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
    llm=coda.ClaudeCodeClient(),     # free under a paid Claude plan, via subprocess
    subagents=[triage],
)
result = agent.run(
    "Find every TODO/FIXME comment under the current directory and triage "
    "each one. Write a markdown report to triage.md."
)
print(result.text)
```

The agent will use `glob` to find Python files, `grep` to find TODOs, then call `triage(...)` inside a `for` loop — one fresh-context LLM call per item, each returning a validated `Triage`. Finally it calls `write("triage.md", ...)`.

The full runnable version is in [`examples/triage_todos.py`](examples/triage_todos.py). For an offline / CI-safe version with canned responses, see [`examples/scripted_demo.py`](examples/scripted_demo.py).

### A harder end-to-end: on-call PR/issue triage

[`examples/oncall_triage.py`](examples/oncall_triage.py) is a TheAgentCompany-shaped workflow: act as the on-call SWE for a real public repo (`anthropics/claude-agent-sdk-python`), pull every PR/issue updated in the last 7 days via the `gh` CLI, call a typed sub-agent on each one to assess severity / category / who-to-ping, write a grouped `TRIAGE.md`, and post a Discord summary via an inline `@tool` (URL kept in tool closure — never lands in LLM input). Latest sample run is checked in under [`examples/sample_run/`](examples/sample_run/) — `triage.md` is the rendered report and `trace.jsonl` is the full event log a debugger UI can replay.

Verified live: 23 items triaged (1 critical / 9 high / 8 medium / 5 low), 9 main-agent turns, 23 sub-agent calls, 1,081 trace events, ~13 min wall time on a Pro plan via `ClaudeAgentSDKClient`.

### Adding an MCP server

```python
import coda

with coda.Agent(
    model="claude-sonnet-4-6",
    llm=coda.ClaudeAgentSDKClient(),
    mcp_servers=[
        coda.MCPServer.stdio(
            "everything",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
    ],
) as agent:
    result = agent.run("Use everything.get_sum to add 2 and 40, then echo the answer.")
```

Each MCP server connects on `Agent` construction; its tools are listed in the system prompt's MCP section and injected into the sandbox as a Python object. The model writes `everything.echo(message="hi")` — not JSON tool-use. Hyphenated MCP tool names (`get-env`) are sanitized to Python identifiers (`get_env`). Use the `Agent` as a context manager so stdio subprocesses are closed cleanly.

## Origin

This started at the end of a two-year engineering project — **MirageAI** (Epitech EIP, 2024–2026). Our team built a production CodeAct backend: an LLM workspace where the model generated `workflow.py` files calling typed `interfaces/` instead of emitting tool-call JSON. We did this because we kept hitting the wall every tool-calling agent hits:

- **Loops and branches were impossible without context bloat.** Iterating over 200 items with a reasoning step per item meant 200 round-trips through the main agent's context.
- **Composing tool outputs into a single decision was painful.** Every intermediate value had to round-trip through JSON-serialisable tool results.
- **Workflows that needed sub-agents inside loops were impossible.** No way to call a model with fresh, isolated context from inside a loop body.

Writing Python — with the LLM authoring code that ran in a persistent sandbox, called typed interfaces, and could spawn sub-agent calls *inside loops* — solved all three.

Two years later, the industry has converged on this pattern. The original CodeAct paper (Wang et al., ICML 2024) named it. Microsoft Hyperlight productised it. LlamaIndex shipped `CodeActAgent`. Manus, OpenHands, and Anthropic's internal patterns all run on it.

**What's still missing is a clean, minimal, framework-agnostic runtime.** Every existing implementation is either academic ([`xingyaoww/code-act`](https://github.com/xingyaoww/code-act)), framework-coupled (LlamaIndex), or heavy (Hyperlight requires a WASM sandbox). And none of them emit the kind of structured hooks you'd want for a real debugger.

`coda` is the clean version. The clever bits from Mirage, extracted, plus what I think we got wrong, rebuilt.

## The three design moves

### 1. Coding-harness primitives, not raw Python builtins

Every coding agent in production — Claude Code, Aider, OpenHands — exposes a small, well-curated set of filesystem and shell primitives instead of forcing the model to write `import pathlib; for p in pathlib.Path(...)...`. coda does the same:

```
ls(path)              # → sorted entries, dirs end with '/'
glob(pattern)         # → matching paths
grep(pattern, path)   # → [(lineno, path, line), ...]
read(path)            # → file contents, with cat -n line numbers by default
write(path, content)  # → bytes written
edit(path, old, new)  # → find/replace, refuses ambiguous matches
bash(cmd)             # → {stdout, stderr, returncode, timed_out}
```

These are pre-imported into the sandbox globals. The system prompt is borrowed in spirit from the (now-public) Claude Code design: clear, tool-first, no LangChain-style indirection. The agent doesn't write `os.listdir`; it writes `ls("interfaces")`.

### 2. Tools can live as files, discovered by the agent (kept from MirageAI)

Point the agent at a directory like `./interfaces/{gmail,slack,github,stripe,...}/*.py` and don't enumerate them in the system prompt. The agent discovers what's available with its native `ls`/`glob`/`grep` primitives, reads the implementation of what it needs, then imports it as normal Python. **Prompt size is O(1); toolspace is unbounded.** Add 1,000 more `.py` files; the prompt doesn't grow.

```python
agent = coda.Agent(
    model="claude-sonnet-4-6",
    llm=coda.ClaudeCodeClient(),
    tools_dir="interfaces",         # hint in the system prompt; the model discovers from here
)
```

The prompt tells the model: *"Your tools live under `interfaces/`. Discover them with `ls`/`glob`/`grep`. Read the source to see what each function does — the implementation is the contract."* To `import` discovered modules, run from the workspace root (or add it to `sys.path`).

### 3. Typed sub-agents with dual schema flow

A sub-agent isn't just "another tool that returns a string". It's a `@coda.subagent`-decorated Python function with a Pydantic return type. Two things happen:

- The **inner LLM call** is *forced* to produce that shape via Anthropic's tool-use schema enforcement.
- The **outer agent's system prompt** is augmented with the typed signature, so generated code can navigate the return value structurally — `if verdict.is_redeye and verdict.layover_score < 4: ...`.

No parsing, no try/except, no "did the model return what I asked for?" guesswork. Because the sub-agent is a Python function, the LLM can call it inside a `for` loop: each iteration is a fresh-context Claude call, and the outer loop doesn't bloat. **This is the pattern carried over from MirageAI.**

## Also in the box

- **Persistent execution context** — variables in turn N are visible in turn N+1.
- **`sys.settrace` line tracing** — every line of generated code emits a `line_executed` event. Trace scope is filtered to frames whose filename is `<coda>`, so coda's own implementation does NOT pollute the event stream.
- **`HookRegistry`** — register handlers per event type (or `'*'` for everything). The single integration seam for debuggers, audit, replay, model routers.
- **`TraceWriter`** — JSONL writer that subscribes to a HookRegistry. One event per line. `read_trace()` / `iter_trace()` to play it back.
- **`Assembler`** — modular system-prompt builder. Sections declare `cache_stable` so adding tools doesn't invalidate the prompt prelude. `.preview(ctx)` reports what each section contributed.

## Cost notes

From **June 15, 2026**, every paid Claude plan gets a dedicated programmatic-usage credit that covers Agent SDK / `claude -p` / third-party SDK use ([Anthropic announcement](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)): $20/mo on Pro, $100/mo on Max 5x, $200/mo on Max 20x. Billed at full API rates from that pool, non-rollover.

Two ways to bill to your Claude plan today (no API key, no API budget):

- **`ClaudeCodeClient`** — shells out to `claude -p` per call. Zero dependencies beyond the Claude Code CLI; portable; one subprocess fork per coda turn.
- **`ClaudeAgentSDKClient`** — uses the official `claude-agent-sdk` Python package. Same auth path (Claude Code's OAuth token), no fork per call. Requires `pip install claude-agent-sdk` and `claude login`; make sure `ANTHROPIC_API_KEY` is unset or it'll shadow the OAuth token.

Switch to `AnthropicClient` whenever you want API-key billing (the post-June-15 credit pool route, or production where you don't want to depend on a developer's subscription).

## What I think we got wrong in Mirage (rebuilt)

- **Subprocess-based execution.** Mirage ran each generated workflow in a fresh Python subprocess via Claude Code. Clean isolation but expensive, and persistent state across turns was awkward. `coda` uses an in-process sandbox with `sys.settrace` for line-by-line instrumentation and an explicit globals dict for state — orders of magnitude faster and instrumentable at the bytecode level.
- **Print-style logs instead of structured events.** Mirage emitted human-readable strings. `coda` emits structured events through a `HookRegistry` — every line execution, every tool call, every variable assignment is a typed event that downstream tools (debuggers, audit, replay) subscribe to.
- **One-shot whole-workflow generation.** Mirage's `WorkflowGenerator` generated a complete `workflow.py` upfront. `coda` is iterative — the agent writes a block, sees the output, writes the next block, observing intermediate state. Better for exploration and error recovery.
- **Closed runtime.** Mirage was a service. `coda` is a library — `pip install coda`, import, run. No microservices required.

## Status

What's in:

- LLMClient protocol + Mock / Anthropic / ClaudeCode backends.
- Sandbox with native primitives, persistent globals, sys.settrace line tracing.
- HookRegistry with typed events; per-type and wildcard subscribers; capture-buffer mode.
- `@tool` and `@subagent` decorators. `@subagent` enforces Pydantic schema via Anthropic forced tool-use, with PEP-563 forward-ref resolution.
- ~80-LoC Agent loop tying it all together.
- Modular cache-aware prompt Assembler with 7 starter sections; `Assembler.preview(ctx)` for debugging.
- TraceWriter (JSONL) + `read_trace` / `iter_trace`.
- 92 tests; ~0.2s suite time.
- Two runnable examples (`triage_todos.py`, `scripted_demo.py`).

Not yet:

- Replay CLI (read a trace and step through it interactively).
- `codetrace` web UI — the time-traveling debugger this whole library was scaffolding for.
- Approval gates for destructive tools.
- Multi-provider routing (OpenAI / local / per-tool model selection).
- `pip install coda` from PyPI.

## Architecture (one paragraph)

The agent loop alternates between an `LLMClient.complete()` call (the model emits text containing a fenced Python block) and a sandbox execution (in-process `exec` against a persistent globals dict, instrumented with `sys.settrace` so every line of model code emits a structured event). Stdout, stderr, and error tracebacks from each block are formatted into `<execution_result>` tags and appended to the conversation as the next user message. The loop exits when the model emits prose without a code block. Native primitives (`ls`/`glob`/`grep`/`read`/`write`/`edit`/`bash`) live inside the `Sandbox` and emit `tool_called` events; small inline tools registered via `@coda.tool` and typed sub-agents registered via `@coda.subagent` are injected into the sandbox globals by name. **Sub-agents** are special: at registration time, coda inspects the Pydantic return type and generates an Anthropic tool-use schema; at call-time, the inner LLM call is *forced* to produce that schema and coda deserialises into the typed object. The signature flows into the outer agent's system prompt via the prompt `Assembler` so generated code can navigate `verdict.field` structurally. A `HookRegistry` lets external tools subscribe to typed events — `code_emitted`, `code_executed`, `line_executed`, `tool_called`, `subagent_called`, `subagent_returned`, `llm_request`, `llm_response`, `execution_error`, `turn_complete` — and `TraceWriter` serializes the same events to JSONL. The runtime is ~600 lines with two external dependencies (`anthropic`, `pydantic`).

## Why open source

Every agent framework I look at locks you in. `coda` is a library, not a framework: ~600 lines, two external dependencies, no microservices, no orchestration layer, no opinion about your storage. The hook protocol is documented and stable. Want to swap Anthropic for OpenAI, route per-tool to different models, plug in your own sandbox, write your own prompt assembler? There's nothing fighting you.

## Prior art

- Wang et al. — *Executable Code Actions Elicit Better LLM Agents* (ICML 2024) — [paper](https://arxiv.org/abs/2402.01030)
- [`xingyaoww/code-act`](https://github.com/xingyaoww/code-act) — reference implementation from the paper authors
- LlamaIndex `CodeActAgent`
- Microsoft Hyperlight CodeAct
- **MirageAI EIP** (Epitech, 2024–2026) — where this started for me

See also `INTERNALS.md` for design principles distilled from the leaked Claude Code source map.

## License

MIT.
