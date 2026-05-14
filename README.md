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
- **Sub-agent calls as tools** — `claude_judge(...)` can be a tool, which means an LLM call inside a `for` loop runs with fresh, isolated context. No context bloat from the inner loop.
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
from anthropic import Anthropic

@coda.tool
def add(a: int, b: int) -> int:
    return a + b

@coda.tool
def claude_judge(text: str, question: str) -> str:
    """A sub-agent call — fresh context per invocation."""
    resp = Anthropic().messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": f"{text}\n\nQuestion: {question}\nAnswer with yes/no/maybe."}],
    )
    return resp.content[0].text

agent = coda.Agent(
    model="claude-sonnet-4-6",
    tools=[add, claude_judge],
)

result = agent.run("Sum the first 50 even numbers, then judge whether the result is interesting.")
print(result)
```

The agent might emit:

```python
total = sum(2 * i for i in range(1, 51))
verdict = claude_judge(str(total), "Is this an interesting number?")
print(total, verdict)
```

Notice the `claude_judge` call **inside** a context where Python has already done the arithmetic. The sub-agent reasons about a small concrete value in fresh context, then its verdict comes back as a normal Python variable. That's the CodeAct pattern in one line — and it's the thing tool-calling agents fundamentally can't do.

## Architecture (one paragraph)

The agent loop alternates between an Anthropic call (model emits text, stopping at a code-block boundary) and a sandbox execution (in-process `exec` against a persistent globals dict, instrumented with `sys.settrace` so every line emits a structured event). Tools are registered via decorator and injected into the sandbox globals as ordinary callables. A `HookRegistry` lets external tools (debuggers, audit, model routers) subscribe to typed events: `code_emitted`, `line_executed`, `tool_called`, `variable_assigned`, `execution_error`, `turn_complete`. The runtime is ~500 lines with one external dependency (`anthropic`).

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
