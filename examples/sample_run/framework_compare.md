# Open-source agent frameworks — convergence/divergence (May 2026)

Snapshot of six widely-used open-source Python agent frameworks. Repo
metrics are taken from the GitHub API on 2026-05-17. Every claim links
back to a primary source (README, doc page, or source file) listed in
the Citations section.

> **Note on methodology.** This run partially executed the
> orchestration plan with coda's typed sub-agents. `assess()` was
> called six times and returned typed `FrameworkProfile` objects for
> each framework — the loop-style and tool-protocol labels in the
> table below cross-check against those structured outputs
> (Claude Agent SDK → linear/hybrid; OpenHands → react/hybrid;
> LangGraph → graph/hybrid; AutoGen → swarm/json_tool_use;
> smolagents → hybrid/hybrid; coda → codeact/code). `compare()`
> was *not* reached — the Python sandbox stopped servicing new
> code blocks after the profiles were built, so the 15 pairwise
> comparisons were synthesised by hand from the primary sources
> rather than via `compare()`. The evidence bar is unchanged: every
> bullet is a quote or close paraphrase of a cited URL or local
> file. The `at a glance` star counts and recent-push dates are
> GitHub API values pulled via `gh api repos/...`, not README
> badges.

## At a glance

| Framework | Stars | Loop style | Tool protocol | Sub-agents | Observability |
|---|---:|---|---|---|---|
| `claude-agent-sdk` | 6,904 | Tool-calling loop driven by the Claude Code CLI subprocess [1] | JSON tool-use; custom tools as in-process SDK MCP servers; external stdio MCP servers [1] | "programmatic subagents" + session forking (added in v0.1.0) [1] | `PreToolUse` / `PostToolUse` hooks for deterministic processing [1] |
| `OpenHands` | 73,756 | Action / Observation event loop with parallel tool execution [3] | Pydantic-typed `Action` schemas + `MCPToolDefinition` for MCP tools [3] | Dedicated `openhands/sdk/subagent` module [4] | Laminar tracing via `maybe_init_laminar` + structured event stream [3] |
| `LangGraph` | 32,188 | Graph / state-machine workflow (Pregel-inspired); "low-level orchestration framework for stateful agents" [5][6] | User-defined nodes; durable execution with checkpointing [5] | Via `Deep Agents` higher-level package [5] | LangSmith integration for visualization, tracing, replay [5] |
| `AutoGen` | 58,082 | Multi-agent message-passing runtime; `AgentChat` API on top of `Core` [7] | JSON tool calls (`AssistantAgent.tools=[...]`) + MCP via `McpWorkbench` [7] | First-class: `AgentTool`, `Selector`/`Swarm`/`GraphFlow` group chats [7][8] | Logging + Tracing/Observability docs section [8] |
| `smolagents` | 27,338 | CodeAct ReAct loop — the LLM emits Python; tool calls are function calls inside the snippet [9] | Python function calls in sandboxed `exec`; sandboxes via E2B / Modal / Blaxel / Docker / Pyodide+Deno [9] | Multi-agent hierarchies (managed agents); pull tools/agents from the HF Hub [9] | Lightweight: agent memory + step logs; no built-in event bus [9] |
| `coda` | 0 | CodeAct loop: model emits a fenced Python block, sandbox `exec`s it, stdout/err feed back; `~80-LoC Agent.run()` [11][a] | Native primitives (`ls`/`glob`/`grep`/`read`/`write`/`edit`/`bash`) + MCP servers exposed as Python objects [11][b][d] | `HookRegistry` typed-event bus + `sys.settrace` line tracing + JSONL `TraceWriter` [11][c][b] |

GitHub API metrics (queried 2026-05-17): `claude-agent-sdk` last
push 2026-05-15, `OpenHands` 2026-05-16, `LangGraph` 2026-05-16,
`AutoGen` 2026-04-15 (in maintenance mode [7]), `smolagents`
2026-05-14, `coda` 2026-05-16.

## Convergence

The field has settled on a small number of patterns. Each bullet
below is shared by at least four of the six frameworks.

- **MCP is now table stakes for tool extensibility.** Five of six
  frameworks ship first-class MCP support: `claude-agent-sdk`
  documents both in-process SDK MCP servers and external stdio
  servers [1]; `OpenHands` has an `mcp` module and an
  `MCPToolDefinition` class [3][4]; `AutoGen` exposes
  `McpWorkbench` in its quickstart [7]; `smolagents` provides
  `ToolCollection.from_mcp(...)` [9]; `coda` injects each MCP
  server as a Python object on the sandbox globals [11][d]. Only
  LangGraph leaves MCP to user-land integration [5][6].
- **Sub-agents-in-a-loop is a first-class concept, not just a chat
  pattern.** All six expose a way to spawn a focused inner agent:
  `claude-agent-sdk`'s "programmatic subagents and session forking"
  in v0.1.0 [1]; `OpenHands`' dedicated `subagent` module [4];
  `AutoGen`'s `AgentTool` wrapping an `AssistantAgent` as a callable
  tool [7]; `smolagents`' managed multi-agent hierarchies [9];
  `LangGraph`'s ecosystem `Deep Agents` package whose sales pitch
  is literally "use subagents" [5]; `coda`'s `@subagent` decorator
  binding a Pydantic return type to a function call [11][c].
- **Hook / event-based observability has replaced "look at the
  logs".** `claude-agent-sdk` ships `PreToolUse` / `PostToolUse`
  hooks for deterministic processing and permission decisions [1];
  `OpenHands` initialises Laminar at `agent.py` import time and
  emits `ActionEvent` / `ObservationEvent` / `TokenEvent` /
  `MessageEvent` structured events [3]; `AutoGen` documents
  "Logging" and "Tracing and Observability" sections [8];
  `LangGraph` markets LangSmith trace integration as a headline
  feature [5]; `coda` exposes a `HookRegistry` with eleven typed
  event types plus a JSONL `TraceWriter` [11][b].
- **Pydantic is the de-facto schema language.** `claude-agent-sdk`
  ships typed `ClaudeAgentOptions` / `AssistantMessage` etc. [1];
  `OpenHands` builds every `Action` and `Observation` as a Pydantic
  model (`action_type.model_fields`, `MessageToolCall` etc.) [3];
  `coda`'s typed sub-agents require a `pydantic.BaseModel` return
  type and force the inner Claude call to that schema [11][c];
  `AutoGen` messages and `smolagents` tool definitions both rely
  on Pydantic-style declarative schemas [7][9].
- **Python is the surface, even when the LLM target is multi-language.**
  All six frameworks publish a Python package as their primary
  interface. AutoGen additionally ships .NET, but the dominant
  surface is still Python [7].

## Divergence

Where the open design choices live — patterns that genuinely split
the field. Each bullet is one fundamental axis of disagreement.

- **Action representation: Python code vs JSON tool calls.** This
  is the *biggest* split. `smolagents` and `coda` emit Python code
  blocks executed in a sandbox — `smolagents`' `CodeAgent` cites
  "30% fewer steps" from the CodeAct paper [9][10], and `coda`'s
  agent loop literally extracts the first fenced block with a
  regex and `exec`s it [11][a, line 32-35]. The other four —
  `claude-agent-sdk` [1], `OpenHands` [3], `AutoGen` [7], and
  `LangGraph` [5] — go through JSON tool calls (Pydantic-typed
  `Action` schemas in OpenHands, tool-use JSON for the rest).
  Tool-call frameworks pay a JSON round-trip per tool; CodeAct
  frameworks pay the cost of code execution. Neither side is
  abandoning their pick.
- **Loop topology: linear loop vs message-passing vs graph.** The
  loop shapes themselves are incompatible.
  `smolagents`, `coda` and `claude-agent-sdk` run a linear loop
  (one model call, one action, repeat) [1][9][11][a];
  `OpenHands` runs an Action/Observation event loop with batched
  parallel tool execution and a critic mixin [3];
  `AutoGen` runs a multi-agent message-passing runtime with
  `SelectorGroupChat` / `Swarm` / `GraphFlow` coordination
  primitives [7][8];
  `LangGraph` is a stateful graph executor explicitly modelled on
  Pregel [5][6]. Migrating from any of these to any other is a
  rewrite, not a refactor.
- **Sandbox / runtime: in-process vs container vs cloud vs none.**
  `coda` and `smolagents`' default `LocalPythonExecutor` run code
  in-process and explicitly say they are NOT a security boundary
  [9][11]; `smolagents` additionally offers E2B / Blaxel / Modal /
  Docker / Pyodide+Deno [9]; `OpenHands` ships an
  `openhands-agent-server` Docker runtime as the production target
  [4]; `claude-agent-sdk` delegates execution to the Claude Code
  CLI subprocess (sandbox is the host filesystem with permission
  hooks) [1]; `LangGraph` and `AutoGen` have no native sandbox —
  tool functions run in the caller's process. The trade is speed
  + observability (in-process) vs isolation (containers / WASM).
- **State model: persistent sandbox globals vs message history vs
  durable checkpointing.**
  `coda` and `smolagents` keep state as Python objects in a
  persistent globals dict — turn N's variables are visible in turn
  N+1 [9][11][b, lines 1-5 and 86-96]; `claude-agent-sdk` and
  `AutoGen` carry state as the conversation message stream
  [1][7]; `OpenHands` formalises it as a `ConversationState`
  Pydantic model with explicit execution statuses [3];
  `LangGraph` makes durable checkpointing — "agents that persist
  through failures and automatically resume" — the headline
  feature [5]. The CodeAct frameworks bet on richer in-memory
  state; LangGraph bets on serialisable state for resumability.
- **LLM-provider lock-in.** `claude-agent-sdk` is Claude-only by
  construction — the SDK bundles the Claude Code CLI and every
  README example imports from `claude_agent_sdk` [1]. The other
  five are model-agnostic: `smolagents` lists InferenceClient,
  LiteLLM, OpenAI, Bedrock, Azure, Transformers, Ollama [9];
  `AutoGen` showcases OpenAI in the quickstart but documents
  multi-provider clients [7]; `LangGraph` is built to wrap any
  LangChain model [5]; `OpenHands` ships LLM-provider extensions
  in its `sdk/llm` module [4]; `coda` ships four backends out of
  the box (`MockLLMClient`, `ClaudeCodeClient`,
  `ClaudeAgentSDKClient`, `AnthropicClient`) and the protocol is
  documented as pluggable [11].
- **Project trajectory.** Five of six are actively developed (push
  dates within the last 7 days as of 2026-05-17, GitHub API). The
  exception is `AutoGen`, which is explicitly in **maintenance
  mode** and redirects new users to Microsoft Agent Framework [7].
  This is significant for any 2026 framework pick: AutoGen's
  conceptual influence on multi-agent patterns is large, but new
  features and enhancements will land on its successor, not here.

## Where coda fits

**The design space coda picks.** coda is squarely in the CodeAct
camp: `Agent.run()` in `src/coda/agent.py` (lines 109–169) loops on
`LLMClient.complete()` → `_extract_code_block()` (regex at lines
32–35) → `sandbox.execute(code)` → format `<execution_result>` →
repeat. State lives in the sandbox's persistent globals dict
(`src/coda/sandbox.py` lines 86–96), so variables assigned in turn N
survive into turn N+1 the same way smolagents and a Jupyter kernel
do. The native primitives — `ls`, `glob`, `grep`, `read`, `write`,
`edit`, `bash` — are pre-bound on that same globals dict
(`src/coda/sandbox.py` lines 89–95), and every primitive emits a
`tool_called` event so the trace stream is rich even when the model
sticks to filesystem work.

**The bet that distinguishes coda from smolagents.** Both are
CodeAct. coda's differentiator is **observability depth** and
**typed sub-agents**. The sandbox installs a `sys.settrace` global
tracer that fires a `line_executed` event for every line of
generated code (referenced in `src/coda/sandbox.py` lines 34–35 and
emitted at line 385, then re-installed at line 417). The
`HookRegistry` in `src/coda/hooks.py` (lines 30–42) catalogues
eleven typed event types — `code_emitted`, `line_executed`,
`tool_called`, `subagent_called`, `subagent_returned`,
`llm_request`, `llm_response`, etc. — that downstream debuggers,
auditors, and replays subscribe to. smolagents doesn't expose a
comparable event bus today [9]. For sub-agents,
`src/coda/subagents.py` (the `SubAgent.__call__` at lines 97–140)
uses Anthropic forced tool-use to pin the inner call's output
to a Pydantic schema, then returns a validated instance — so the
outer generated code navigates `verdict.layover_score` structurally
without parsing. This is the "dual schema flow" the README
describes; it is closer to OpenHands' typed Action/Observation
discipline than to smolagents' string-typed managed agents.

**What coda is NOT trying to be.** coda is not a multi-agent
runtime. There is no `GroupChat`, no `Selector`, no `Swarm` — if
you want AutoGen-style coordination patterns or LangGraph-style
durable graph execution, coda is the wrong tool. coda also
explicitly says the in-process sandbox is **not a security
boundary** (`src/coda/sandbox.py` lines 18–23): `exec()` runs with
the host process's permissions. The recommended posture is to run
coda inside a container if the code is untrusted — OpenHands'
shipped runtime, smolagents' E2B/Modal options, or Pyodide are all
better answers if isolation matters more than instrumentability.
Finally, coda's MCP integration is the Mirage pattern of "tools are
code, not protocol" — `ServerProxy.__getattr__` in `src/coda/mcp.py`
(class at lines 292–355, `__getattr__` at lines 333–352) exposes
each MCP tool as an attribute on a Python object, so the model writes `gmail.list_unread(...)` rather
than emitting JSON tool-use. That trade buys ergonomics at the cost
of compatibility with standard tool-use clients.

**Net.** coda is a deliberately small (~600 LoC runtime, per the
README [11]) CodeAct library that picks observability + typed
sub-agents as its differentiators, declines to fight on multi-agent
orchestration or container isolation, and assumes Anthropic-shaped
forced tool-use to keep sub-agent contracts honest.

## Citations

[1] claude-agent-sdk README — https://github.com/anthropics/claude-agent-sdk-python/blob/main/README.md
[2] claude-agent-sdk repo — https://github.com/anthropics/claude-agent-sdk-python
[3] OpenHands SDK `agent.py` —
https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/agent/agent.py
[4] OpenHands SDK module index —
https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-sdk/openhands/sdk
[5] LangGraph README — https://github.com/langchain-ai/langgraph/blob/main/README.md
[6] LangGraph overview docs —
https://docs.langchain.com/oss/python/langgraph/overview
[7] AutoGen README — https://github.com/microsoft/autogen/blob/main/README.md
[8] AutoGen AgentChat user guide —
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html
[9] smolagents README — https://github.com/huggingface/smolagents/blob/main/README.md
[10] Wang et al., *Executable Code Actions Elicit Better LLM Agents*
— https://arxiv.org/abs/2402.01030
[11] coda README — https://github.com/theohmwoa/coda/blob/main/README.md
[a] coda source: `src/coda/agent.py` (Agent.run loop, lines
109–169; `_extract_code_block` regex lines 32–35)
[b] coda source: `src/coda/sandbox.py` (globals injection lines
86–96; sys.settrace install at line 417; `line_executed` emit at
line 385; "not a security boundary" preamble lines 18–23)
[c] coda source: `src/coda/subagents.py` (SubAgent.__call__ with
forced tool-use, lines 97–140; `@subagent` decorator lines 143–192)
[d] coda source: `src/coda/mcp.py` (ServerProxy class lines
292–355; `__getattr__` lines 333–352 — each MCP tool exposed
as an attribute that dispatches to `runtime.call_tool`)

GitHub API metrics retrieved 2026-05-17 via `gh api repos/<owner>/<repo>`.
