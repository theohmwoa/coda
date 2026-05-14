# Internals — design principles for coda

> What we learned from studying the leaked Claude Code source and the public reverse-engineering work, distilled into design principles for coda.

The Claude Code source-map leak (March 2026) plus the community analysis that followed exposed roughly 1,900 TypeScript files / 512K+ LoC / ~40 built-in tools — and revealed that **only ~1.6% of the codebase is "AI decision logic"**. The other 98.4% is infrastructure: permission gates, context management, tool routing, recovery logic, cache management, security gates. **A great agent isn't `model + tools`; it's an engineered system around those two pieces.**

This document records what to copy in spirit, what to do differently, and what to ignore.

## Principle 1 — Simple agent loop, rich infrastructure

The agent loop in Claude Code is essentially a `while` loop. Almost everything interesting lives *around* it.

**coda follows this.** The `Agent.run()` method should be ~20 lines. The interesting code is:

- Tool primitives and their descriptions
- Permission / approval gates
- Prompt assembly
- Sandbox + observability
- Sub-agent dispatch with typed I/O
- Context / cache management

Resist the temptation to put logic in the loop. Put it in the tools and the prompt assembler.

## Principle 2 — Modular, conditional, cache-aware prompt assembler

Claude Code's system prompt is **not a single string**. It's ~20 sections, dynamically assembled per session, each conditional on environment flags or config:

```
1. Intro                                    (always)
2. System Rules                             (always)
3. Doing Tasks                              (omittable via custom style)
4. Executing Actions with Care              (always — safety)
5. Using Your Tools                         (always, conditional variations)
6. Tone and Style                           (always)
7. Output Efficiency                        (always)
8. Cache Boundary Marker                    (conditional — global_cache_scope)
9. Session Guidance subsections             (~6 conditional subsections)
10. Memory Prompt                           (memory_configured)
11. Environment Info                        (conditional variations)
12. Language                                (language_set)
13. Output Style                            (output_style)
14. MCP Server Instructions                 (MCP servers connected)
15. Scratchpad Instructions                 (scratchpad_enabled)
16. Numeric Length Anchors                  (Anthropic-employees-only)
17. Token Budget                            (token_budget feature)
18. Git Status Snapshot                     (repo + git_instructions)
19. Append System Prompt                    (user-customized appendix)
```

**The order matters for cache.** Sections marked `DANGEROUS_uncachedSystemPromptSection()` in the leaked source are explicitly tagged as cache-breaking. There's a separate `promptCacheBreakDetection.ts` with **14 vectors** for detecting accidental cache breaks. "Sticky latches" prevent mode toggles from invalidating the cache.

**coda's approach:** build a `PromptAssembler` from day one. Each section is a small function. Each section declares whether it's cache-stable or cache-breaking. The assembler emits the prompt in deterministic order with cache-stable sections first. Section components:

```python
class PromptSection:
    name: str
    cache_stable: bool
    condition: Callable[[Context], bool]
    render: Callable[[Context], str]
```

We start with maybe 8 sections (intro, rules, tools, tone, env, interfaces-discovery, subagents-signatures, append). Add more as we discover need.

## Principle 3 — Tool descriptions are 14-17K tokens — the description IS the behavior spec

The biggest single chunk of every Claude Code request isn't the system prompt (~2.5K tokens). It's the **tool descriptions** — ~50 tools at 14-17K total tokens. **Bash alone is 1,558 tokens** because its description includes detailed instructions for git commits, PR creation, pre-commit hooks. That's why Claude Code writes good commit messages without being asked: the behavior is in the tool description, not the model.

**coda's implication:** don't write minimal docstrings for tool primitives. The description is where you encode behavior. Our `bash` tool's docstring should include git conventions, safety notes, when to prefer `read`/`write`/`edit` over shell. Our `read` tool's docstring should specify the line-numbered format we want for output.

The cost concern is real (every request pays for the tool description tokens), but cache mitigates it — descriptions are stable, so they hit the prompt cache after the first call.

## Principle 4 — Native primitives preferred over `bash`, for observability

The leaked prompt literally instructs: *"Do NOT use Bash to run commands when a relevant dedicated tool is provided."* The reason is observability: every `Read` tool call can be logged, latency-measured, surfaced in the UI. A `cat file.txt` via Bash is opaque.

**This is coda's exact thesis.** Our sandbox primitives — `ls`, `glob`, `grep`, `read`, `write`, `edit` — each emit structured events through `HookRegistry`. The agent SHOULD use them. The system prompt should say so.

We keep `bash` as a fallback for things the typed primitives don't cover. But the prompt steers strongly toward native.

## Principle 5 — Sub-agents have specialized prompts and reduced tool inventories

The leak revealed dedicated sub-agent prompts:

- **Plan** subagent — 715 tokens, specialized for planning
- **Explore** subagent — 575 tokens, specialized for code-base exploration
- Plus security-review, onboarding, task management

Each sub-agent has:
- Its own system prompt (not just the parent's prompt + a task description)
- A reduced tool inventory (Plan can't `write`, Explore is read-only)
- A focused output format

**coda's `@subagent` decorator should generate this.** The decorated function's signature + docstring + return type → a specialized system prompt for that sub-agent. Sub-agent inherits a *subset* of parent tools by default (or an explicit subset declared by the decorator).

```python
@coda.subagent(
    model="claude-haiku-4-5",
    tools=[read, grep],          # read-only subset, like Explore
)
def evaluate_flight(flight: dict) -> FlightVerdict:
    """Evaluate one flight..."""
```

## Principle 6 — Project context (`CLAUDE.md`) goes in conversation, not system prompt

Claude Code reads `CLAUDE.md` and injects it as a **conversation attachment**, not as a system prompt section. This is deliberate:

- System prompt stays cacheable across projects
- Project-specific changes only blow per-conversation cache (which is much shorter-lived)
- The model treats project context as user-provided, not authoritative

**coda follows this.** A `CODA.md` or `coda.toml` file in the workspace gets attached to the first user message as project context. System prompt stays project-agnostic and globally cacheable.

## Principle 7 — Per-tool security gates, not centralized validation

The leaked `bashSecurity.ts` has **23 numbered security checks** specific to the bash tool:

- 18 blocked Zsh builtins
- Unicode zero-width space injection defense
- IFS null-byte injection guards

Note: these gates live *inside the bash tool's implementation*, not in a centralized validator. Each tool owns its own safety.

**coda follows this for its bash tool.** Lighter scope than Anthropic's enterprise version, but the pattern is: each tool implementation has its own guard list. No central `validate(call)` function that has to know about every tool.

## Principle 8 — Coordination via prompt directives, not state machines

The leaked `coordinatorMode.ts` orchestrates multi-agent flows entirely through prompt instructions:

> *"Do not rubber-stamp weak work."*
> *"You must understand findings before directing follow-up work."*

There's no Python state machine deciding when the coordinator interrupts the worker. The behavior is encoded in language.

**coda's implication:** if we add multi-agent coordination (future), do it the same way. Don't build orchestration machinery; encode the protocol in prompt.

## What we explicitly DON'T copy

- **Anti-distillation tokens** (`anti_distillation: ['fake_tools']`). Anthropic injects decoy tool definitions to poison training data scraping. We don't need this; it would actively confuse our users.
- **KAIROS** (the unshipped autonomous-daemon mode referenced 150+ times in the source). Too speculative; the design hasn't even shipped from Anthropic.
- **The leaked system prompt verbatim.** Legally and ethically we should write our own prose, informed by the patterns but not the strings.
- **`undercover_mode`** — a flag in the leaked source for anonymizing Anthropic-employee usage. Not applicable.

## References

- [Piebald-AI/claude-code-system-prompts](https://github.com/Piebald-AI/claude-code-system-prompts) — the most up-to-date reference (110+ markdown files per release, including tool descriptions and sub-agent prompts)
- [alex000kim — The Claude Code Source Leak](https://alex000kim.com/posts/2026-03-31-claude-code-source-leak/) — fake tools, frustration regexes, undercover mode, KAIROS
- [dbreunig — How Claude Code Builds a System Prompt](https://www.dbreunig.com/2026/04/04/how-claude-code-builds-a-system-prompt.html) — section-by-section breakdown of dynamic assembly
- [ComeOnOliver/claude-code-analysis](https://github.com/ComeOnOliver/claude-code-analysis) — module boundaries, tool inventories, state management
- [alejandrobalderas/claude-code-from-source](https://github.com/alejandrobalderas/claude-code-from-source) — architecture & design patterns from source maps
- [VILA-Lab/Dive-into-Claude-Code](https://github.com/VILA-Lab/Dive-into-Claude-Code) — systematic analysis paper
- [Kir Shatrov — Reverse engineering Claude Code](https://kirshatrov.com/posts/claude-code-internals) — alternative angle on the same artefacts

## Translation table — patterns → coda components

| Claude Code pattern | coda component | Status |
|---|---|---|
| Dynamic prompt assembler with ~20 sections | `coda.prompt.Assembler` | TODO |
| Rich tool descriptions ≥ 500 tokens each | docstrings on `ls`/`glob`/`grep`/`read`/`write`/`edit`/`bash` | TODO |
| Native tools preferred over bash | system prompt directive + steered prompts | TODO |
| Specialized sub-agent prompts | `@subagent` decorator generates per-subagent prompt | TODO |
| Project context as conversation attachment | `CODA.md` reader → injected into first user message | TODO |
| Per-tool security gates | guards live inside each tool's implementation | TODO |
| Cache-aware section ordering | `cache_stable: bool` on each prompt section | TODO |
| Simple agent loop (~20 lines) | `Agent.run()` stays minimal | TODO |
