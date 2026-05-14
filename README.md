# coda — minimal CodeAct runtime with first-class observability

> Python agents that **write code instead of calling tools**, with a hookable runtime so debuggers, replay tools, and model routers can plug straight in.

## Origin

This started at the end of a two-year engineering project — **MirageAI** (Epitech EIP, 2024–2026). Our team built a production CodeAct backend: an LLM workspace where the model generated `workflow.py` files calling typed `interfaces/` instead of emitting tool-call JSON. We did this because we kept hitting the wall every tool-calling agent hits:

- **Loops and branches were impossible without context bloat.** Iterating over 200 items with a reasoning step per item meant 200 round-trips through the main agent's context.
- **Composing tool outputs into a single decision was painful.** Every intermediate value had to round-trip through JSON-serialisable tool results.
- **Workflows that needed sub-agents inside loops were impossible.** No way to call a model with fresh, isolated context from inside a loop body.

Writing Python — with the LLM authoring code that ran in a persistent sandbox, called typed interfaces, and could spawn sub-agent calls *inside loops* — solved all three.

Two years later, the industry has converged on this pattern. The original CodeAct paper (Wang et al., ICML 2024) named it. Microsoft Hyperlight productised it. LlamaIndex shipped `CodeActAgent`. Manus, OpenHands, and Anthropic's internal patterns all run on it.

**What's still missing is a clean, minimal, framework-agnostic runtime.** Every existing implementation is either academic ([`xingyaoww/code-act`](https://github.com/xingyaoww/code-act)), framework-coupled (LlamaIndex), or heavy (Hyperlight requires a WASM sandbox). And none of them emit the kind of structured hooks you'd want for a real debugger.

`coda` is the clean version. The clever bits from Mirage, extracted, plus what I think we got wrong, rebuilt.

## What's clever (kept from Mirage)

- **Persistent execution context** — variables defined in turn N are visible in turn N+1. The LLM can build up state across multiple code blocks within one conversation.
- **Tool functions as native Python** — tools are `@tool`-decorated Python functions injected into the sandbox globals. The model writes `result = search_flights("CDG", "JFK")` — no JSON, no schemas, no bridges.
- **Typed sub-agents as tools** — a sub-agent isn't just "another tool that returns a string". It's a `@subagent`-decorated Python function with a Pydantic return type. The inner Claude call is *forced* to produce that shape (via Anthropic's tool-use schema); the outer agent sees the typed signature in its system prompt and can navigate the return value structurally — `if verdict.is_redeye and verdict.layover_score < 4: ...`. No parsing, no try/except, no "did the model return what I asked for?" guesswork. This is the single biggest power feature carried over from Mirage.
- **Sub-agent calls inside loops** — because a sub-agent call is just a Python function, the LLM can write `for item in items: r = evaluate(item)`, and each inner Claude call runs with fresh, isolated context. No context bloat from the loop.
- **Approval gates as Python decorators** — certain tools pause and ask for human confirmation before running. That's just a wrapper, not a special protocol.

## What I think we got wrong (rebuilt)

- **Subprocess-based execution.** Mirage ran each generated workflow in a fresh Python subprocess via Claude Code. Clean isolation but expensive, and persistent state across turns was awkward. `coda` uses an in-process sandbox with `sys.settrace` for line-by-line instrumentation and an explicit globals dict for state — orders of magnitude faster and instrumentable at the bytecode level.
- **Print-style logs instead of structured events.** Mirage emitted human-readable strings. `coda` emits structured events through a `HookRegistry` — every line execution, every tool call, every variable assignment is a typed event that downstream tools (debuggers, audit, replay) subscribe to.
- **One-shot whole-workflow generation.** Mirage's `WorkflowGenerator` generated a complete `workflow.py` upfront. `coda` is iterative — the agent writes a block, sees the output, writes the next block, observing intermediate state. Better for exploration and error recovery.
- **Closed runtime.** Mirage was a service. `coda` is a library — `pip install coda`, import, run. No microservices required.

## Status

Week 1 of a 4-week solo build:

- [x] Project scaffold
- [ ] Agent loop + sandbox + tool registry + hook registry
- [ ] First end-to-end demo (week 1 target)
- [ ] Trace writer + replay CLI (week 2)
- [ ] `codetrace` web UI on top — time-traveling debugger (week 3)
- [ ] Docs + launch (week 4)

## Target API (week 1)

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
    # No body — coda generates the prompt from signature + docstring + return schema,
    # forces the inner Claude call to produce a FlightVerdict via tool-use schema.

agent = coda.Agent(
    model="claude-sonnet-4-6",
    tools=[search_flights, evaluate_flight],
)

result = agent.run("Find me the best non-redeye flight CDG→JFK under €600 next month.")
print(result)
```

The agent might emit code like:

```python
options = search_flights("CDG", "JFK", max_price=600)
keepers = []
for f in options:
    v = evaluate_flight(f)             # ← typed sub-agent call, fresh context
    if not v.is_redeye and v.layover_score < 4 and v.confidence == "high":
        keepers.append((f, v.reasoning))
best = min(keepers, key=lambda x: x[0]["price"])
print(best)
```

Notice three things:
1. **`evaluate_flight` inside a `for` loop** — each iteration is a fresh Claude call with isolated context. With tool-calling, those 20 iterations would bloat the outer agent's context past 100K tokens.
2. **`v.is_redeye`, `v.layover_score`, `v.confidence`** — the outer agent can write code that navigates the *typed* return value because the schema flows into its system prompt. Mirage's design move; coda's first-class feature.
3. **No JSON parsing.** The inner Claude call is forced (via Anthropic's tool-use schema enforcement) to return a `FlightVerdict`-shaped object. coda deserialises into the Pydantic model. If the model can't produce the schema, coda retries with the error — the outer agent never sees a malformed payload.

## Architecture (one paragraph)

The agent loop alternates between an Anthropic call (model emits text, stopping at a code-block boundary) and a sandbox execution (in-process `exec` against a persistent globals dict, instrumented with `sys.settrace` so every line emits a structured event). Tools and sub-agents are registered via decorator and injected into the sandbox globals as ordinary callables. **Sub-agents** are special: at registration time, coda inspects the Pydantic return type and generates an Anthropic tool-use schema; at call-time, the inner Claude call is *forced* to produce that schema and coda deserialises into the typed object. The signature flows into the outer agent's system prompt so its generated code can navigate `verdict.field` structurally. A `HookRegistry` lets external tools (debuggers, audit, model routers) subscribe to typed events: `code_emitted`, `line_executed`, `tool_called`, `subagent_called`, `variable_assigned`, `execution_error`, `turn_complete`. The runtime is ~500 lines with two external dependencies (`anthropic`, `pydantic`).

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
