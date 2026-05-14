# coda — a CodeAct coding harness with typed sub-agents and line-level observability

> A Claude-Code-shaped coding harness, packaged as a clean Python library. Native filesystem tools, typed sub-agents inside loops, and `sys.settrace` line-by-line observability for every byte of generated code.

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
ls(path)              # → directory entries
glob(pattern)         # → matching paths
grep(pattern, path)   # → matching lines, with line numbers
read(path)            # → file contents
write(path, content)  # → write file
edit(path, old, new)  # → find/replace in file
bash(cmd)             # → shell command (timeout-bounded)
```

These are available as Python functions in the sandbox globals. The system prompt is borrowed in spirit from the (now-public) Claude Code design: clear, tool-first, no LangChain-style indirection. The agent doesn't write `os.listdir`; it writes `ls("interfaces")`.

### 2. Tools live as files, discovered by the agent (kept from MirageAI)

Point the agent at a directory like `./interfaces/{gmail,slack,github,stripe,...}/*.py` and don't enumerate them in the system prompt. The agent discovers what's available with its native `ls`/`glob`/`grep` primitives, reads the implementation of what it needs, then imports it as normal Python. **Prompt size is O(1); toolspace is unbounded.** Add 1,000 more `.py` files; the prompt doesn't grow.
### 3. Typed sub-agents with dual schema flow

A sub-agent isn't just "another tool that returns a string". It's a `@subagent`-decorated Python function with a Pydantic return type. Two things happen:

- The **inner Claude call** is *forced* to produce that shape via Anthropic's tool-use schema enforcement
- The **outer agent's system prompt** is augmented with the typed signature, so its generated code can navigate the return value structurally — `if verdict.is_redeye and verdict.layover_score < 4: ...`

No parsing, no try/except, no "did the model return what I asked for?" guesswork. Because the sub-agent is just a Python function, the LLM can call it inside a `for` loop: each iteration is a fresh-context Claude call, and the outer loop doesn't bloat. **This is the single most powerful pattern carried over from MirageAI.**

### Plus

- **Persistent execution context** — variables in turn N are visible in turn N+1.
- **Approval gates as decorators** — destructive tools pause for human confirmation. `@coda.requires_approval` wraps any tool. No special protocol.
- **Line-by-line observability** — `sys.settrace` captures every line of generated code, every variable assignment, every tool call as a typed event. `HookRegistry` lets debuggers, audit tools, and routers subscribe.

## Cost notes

`coda` calls Anthropic's API directly. From **June 15, 2026**, every paid Claude plan gets a dedicated programmatic-usage credit that covers Agent SDK / `claude -p` / third-party SDK use ([Anthropic announcement](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)): $20/mo on Pro, $100/mo on Max 5x, $200/mo on Max 20x. Billed at full API rates from that pool, non-rollover. For solo development that's enough headroom to run real loops.

## What I think we got wrong in Mirage (rebuilt)

- **Subprocess-based execution.** Mirage ran each generated workflow in a fresh Python subprocess via Claude Code. Clean isolation but expensive, and persistent state across turns was awkward. `coda` uses an in-process sandbox with `sys.settrace` for line-by-line instrumentation and an explicit globals dict for state — orders of magnitude faster and instrumentable at the bytecode level.
- **Print-style logs instead of structured events.** Mirage emitted human-readable strings. `coda` emits structured events through a `HookRegistry` — every line execution, every tool call, every variable assignment is a typed event that downstream tools (debuggers, audit, replay) subscribe to.
- **One-shot whole-workflow generation.** Mirage's `WorkflowGenerator` generated a complete `workflow.py` upfront. `coda` is iterative — the agent writes a block, sees the output, writes the next block, observing intermediate state. Better for exploration and error recovery.
- **Closed runtime.** Mirage was a service. `coda` is a library — `pip install coda`, import, run. No microservices required.

## Status

Week 1 of a 4-week solo build:

- [x] Project scaffold
- [x] Agent loop + sandbox + tool registry + hook registry
- [x] First end-to-end demo (week 1 target) — `examples/triage_todos.py`, `examples/scripted_demo.py`
- [x] JSONL trace writer (week 2, part 1) — `coda.TraceWriter` + `read_trace` / `iter_trace`
- [ ] Replay CLI (week 2, part 2)
- [ ] `codetrace` web UI on top — time-traveling debugger (week 3)
- [ ] Docs + launch (week 4)

## Target API (week 1)

**Option A — small inline set** (for quick scripts):

```python
import coda
from pydantic import BaseModel
from typing import Literal

@coda.tool
def search_flights(origin: str, dest: str, max_price: int) -> list[dict]:
    """Return matching flight offers."""
    ...

class FlightVerdict(BaseModel):
    is_redeye: bool
    layover_score: int       # 0 = none, 10 = brutal
    confidence: Literal["low", "medium", "high"]
    reasoning: str

@coda.subagent(model="claude-haiku-4-5")
def evaluate_flight(flight: dict) -> FlightVerdict:
    """Evaluate one flight on redeye + layover dimensions. Be strict."""

agent = coda.Agent(
    model="claude-sonnet-4-6",
    tools=[search_flights],
    subagents=[evaluate_flight],
)
result = agent.run("Find me the best non-redeye flight CDG→JFK under €600 next month.")
```

**Option B — filesystem as the tool registry** (the real pattern, scales to thousands):

```
my-agent/
  interfaces/
    gmail/
      send_email.py
      list_unread.py
      search_threads.py
    slack/
      post_message.py
    github/
      open_pr.py
      list_issues.py
    stripe/
      list_invoices.py
    data/
      query_postgres.py
  subagents/
    evaluate_flight.py
```

```python
agent = coda.Agent(
    model="claude-sonnet-4-6",
    tools_dir="./interfaces",        # agent discovers, doesn't enumerate
    subagents_dir="./subagents",
)
result = agent.run("Triage my unread emails, draft replies to the ones flagged urgent.")
```

System prompt (auto-generated, tiny):
```
Your interfaces live in ./interfaces/. Each file contains one function with a
docstring. Discover what's available with os.listdir, pathlib.Path.glob, or grep.
Import what you need: from interfaces.gmail.send_email import send_email.
...
```

The agent's first code blocks might be:

```python
# Block 1: explore the tree
import pathlib
for p in pathlib.Path("interfaces").iterdir():
    print(p.name, [f.name for f in p.iterdir()])
```
```output
gmail  ['send_email.py', 'list_unread.py', 'search_threads.py']
slack  ['post_message.py']
...
```

```python
# Block 2: read the implementation of the tools it actually needs
print(pathlib.Path("interfaces/gmail/list_unread.py").read_text())
print(pathlib.Path("interfaces/gmail/send_email.py").read_text())
```
```output
def list_unread(label: str = "INBOX", max: int = 50) -> list[Email]:
    """Return up to `max` unread emails in `label`. Ordered by date desc."""
    ...
def send_email(to: str, subject: str, body: str, reply_to: str | None = None):
    """Send an email. Rate-limited: 100/hour. Returns the message_id."""
    ...
```

```python
# Block 3: actually do the work — and notice the typed sub-agent IN the loop
from interfaces.gmail.list_unread import list_unread
from interfaces.gmail.send_email import send_email
from subagents.is_urgent import is_urgent  # typed sub-agent: returns UrgencyVerdict

unread = list_unread(max=100)
drafts = []
for email in unread:
    v = is_urgent(email)                      # fresh-context Claude call per email
    if v.score >= 8 and v.confidence == "high":
        drafts.append((email, v.suggested_reply))
print(f"Drafted {len(drafts)} replies for review.")
```

What this gives you:

- **The outer agent's prompt mentions zero specific tools.** It mentions only the *protocol* for finding them. Add 1,000 more interfaces tomorrow; the prompt stays the same size.
- **The agent sees the actual implementation** of each tool, including edge-case comments (e.g. the "rate-limited: 100/hour" line on `send_email`). No synthesized tool description; the source is the contract.
- **Typed sub-agents inside loops** — each `is_urgent(email)` call runs Claude in fresh context, returns a typed `UrgencyVerdict` the outer agent navigates as `v.score`, `v.confidence`, `v.suggested_reply`. No parsing.
- **No JSON, no schemas in prompts, no MCP server boilerplate.** It's just Python files in a directory.

## Architecture (one paragraph)

The agent loop alternates between an Anthropic call (model emits text, stopping at a code-block boundary) and a sandbox execution (in-process `exec` against a persistent globals dict, instrumented with `sys.settrace` so every line emits a structured event). **Tools live as files** under `tools_dir`; the agent discovers them with `os.listdir`/`glob`/`grep` and imports what it needs — the system prompt enumerates the protocol, not the toolset, so a 10,000-file workspace costs the same prompt tokens as a 10-file one. Small inline tools registered via the `@tool` decorator are also injected into the sandbox globals for convenience. **Sub-agents** are special: at registration time, coda inspects the Pydantic return type and generates an Anthropic tool-use schema; at call-time, the inner Claude call is *forced* to produce that schema and coda deserialises into the typed object. The signature flows into the outer agent's system prompt so its generated code can navigate `verdict.field` structurally. A `HookRegistry` lets external tools (debuggers, audit, model routers) subscribe to typed events: `code_emitted`, `line_executed`, `tool_imported`, `tool_called`, `subagent_called`, `variable_assigned`, `execution_error`, `turn_complete`. The runtime is ~500 lines with two external dependencies (`anthropic`, `pydantic`).

## Why open source

Every agent framework I look at locks you in. `coda` is a library, not a framework: ~500 lines, one external dependency, no microservices, no orchestration layer, no opinion about your storage. The hook protocol is documented and stable. Want to swap Anthropic for OpenAI, route per-tool to different models, plug in your own sandbox? There's nothing fighting you.

## Prior art

- Wang et al. — *Executable Code Actions Elicit Better LLM Agents* (ICML 2024) — [paper](https://arxiv.org/abs/2402.01030)
- [`xingyaoww/code-act`](https://github.com/xingyaoww/code-act) — reference implementation from the paper authors
- LlamaIndex `CodeActAgent`
- Microsoft Hyperlight CodeAct
- **MirageAI EIP** (Epitech, 2024–2026) — where this started for me

## License

MIT.
