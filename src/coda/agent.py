"""Agent — the main CodeAct loop.

The loop is deliberately small (INTERNALS.md Principle 1: simple loop, rich
infrastructure). Each turn:

1. Call LLMClient.complete with the conversation so far
2. Extract the fenced Python code block from the assistant's text
3. If there's no code block → we're done, return the text
4. exec the code in the Sandbox
5. Append the result (stdout/stderr/error) as a user message
6. Loop, up to `max_turns`

Everything else — prompt assembly, tool discovery, observability, sub-agent
dispatch — lives outside this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .context import (
    ContextPolicy,
    build_context_reminder,
    build_print_habit_reminder,
    render_ctx_delta,
)
from .hooks import Event, HookRegistry
from .llm import LLMClient, Message
from .mcp import MCPRuntime, MCPServer, ServerProxy, register_for_atexit
from .prompt import Assembler, build_system_prompt, default_assembler
from .sandbox import ExecutionResult, Sandbox
from .skills import Skill, inject_skills, load_skills, make_save_skill
from .subagents import SubAgent
from .tools import Tool


CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# A line containing only a "# --- next block ---" comment (any number
# of dashes, optional whitespace) is treated as a soft fence boundary
# WITHIN a fenced block. The model is taught to emit multiple ```python
# fences per turn, but in practice it sometimes wraps an entire reply
# in one fence and uses these comment separators instead. Without
# splitting on them, any SyntaxError inside such a mega-block voids
# every segment — observed in framework_compare_020702, diagnosed by
# coda's own self_debug demo (runs/self_debug_diagnosis_20260517_114052.md).
BLOCK_SEPARATOR_RE = re.compile(
    r"^[ \t]*#[ \t]*-{2,}[ \t]*next[ \t]+block[ \t]*-{2,}[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class RunResult:
    text: str
    turns: int
    messages: list[Message] = field(default_factory=list)
    executions: list[ExecutionResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class Agent:
    def __init__(
        self,
        *,
        model: str,
        llm: LLMClient,
        sandbox: Sandbox | None = None,
        hooks: HookRegistry | None = None,
        tools: list[Tool] | None = None,
        subagents: list[SubAgent] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        tools_dir: str | None = None,
        skills_dir: str | Path | None = None,
        system_prompt: str | None = None,
        system_prompt_append: str | None = None,
        prompt_assembler: Assembler | None = None,
        max_turns: int = 25,
        max_tokens: int = 4096,
        context_policy: ContextPolicy | None = None,
    ) -> None:
        self.model = model
        self.llm = llm
        self.hooks = hooks or HookRegistry()
        self.sandbox = sandbox or Sandbox(hooks=self.hooks)
        self.tools = list(tools or [])
        self.subagents = list(subagents or [])
        self.tools_dir = tools_dir
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.context_policy = context_policy

        for sa in self.subagents:
            sa.bind(self.llm, self.hooks)
            self.sandbox.inject(sa.name, sa)
        for t in self.tools:
            self.sandbox.inject(t.name, t)

        # Schemas used by the per-turn linter. Built once from the live
        # sub-agents so it doesn't matter whether skills_dir is set.
        self._lint_schemas: dict[str, Any] = {
            sa.name: sa.return_type for sa in self.subagents
        }

        self.mcp_runtime: MCPRuntime | None = None
        self.mcp_proxies: dict[str, ServerProxy] = {}
        if mcp_servers:
            self.mcp_runtime = MCPRuntime()
            self.mcp_runtime.start(list(mcp_servers))
            register_for_atexit(self.mcp_runtime)
            self.mcp_proxies = self.mcp_runtime.proxies()
            for py_name, proxy in self.mcp_proxies.items():
                self.sandbox.inject(py_name, proxy)

        self.skills_dir: Path | None = (
            Path(skills_dir).resolve() if skills_dir else None
        )
        self.skills: dict[str, Skill] = {}
        if self.skills_dir is not None:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            self.skills = load_skills(self.skills_dir)
            inject_skills(self.sandbox.globals, self.skills)
            self.sandbox.inject(
                "save_skill",
                make_save_skill(self.skills_dir, schemas=self._lint_schemas or None),
            )

        # Expose `lint` as a sandbox primitive whenever there are
        # sub-agents, regardless of whether skills_dir is set. Lets the
        # agent run an extra check at any point in a workflow.
        if self._lint_schemas:
            from .lint import lint_dicts as _lint_dicts
            _schemas = self._lint_schemas
            def _lint(code: str) -> list[dict]:
                """Lint Python code against the live sub-agent schemas.

                Catches comparisons of `x.field == "value"` or
                `x["field"] == "value"` where `field` is a Literal-typed
                field on some sub-agent's return type and `"value"` isn't
                in the allowed set (dead-branch bugs). The Agent already
                runs this automatically on every emitted code block and
                surfaces findings in the execution result; this primitive
                is for explicit one-off checks.
                """
                return _lint_dicts(code, schemas=_schemas)
            self.sandbox.inject("lint", _lint)

        self.prompt_assembler = prompt_assembler or default_assembler()
        if system_prompt is not None:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = build_system_prompt(
                tools=self.tools,
                subagents=self.subagents,
                tools_dir=self.tools_dir,
                extra=system_prompt_append,
                assembler=self.prompt_assembler,
                mcp_proxies=self.mcp_proxies,
                skills=self.skills,
                skills_dir=self.skills_dir,
            )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def run(self, user_prompt: str) -> RunResult:
        messages: list[Message] = [Message(role="user", content=user_prompt)]
        executions: list[ExecutionResult] = []
        in_tokens = 0
        out_tokens = 0
        # Turns since the last context-policy reminder fired. Initialized
        # high so the first over-threshold turn fires without waiting for
        # the cooldown to accumulate. After firing, reset to 0 and counted
        # back up to cooldown_turns before another reminder is allowed.
        turns_since_reminder = 10**9
        # Same idea for the print-habit reminder — fired when this turn's
        # stdout volume crosses the policy's large_print_chars threshold.
        turns_since_print_warn = 10**9

        for turn in range(1, self.max_turns + 1):
            # Stamp this turn on subsequent ctx.* assignments so the model
            # can see "this was set in turn 3" in renders and reminders.
            self.sandbox.ctx._ctx_set_turn(turn)
            self.hooks.emit(
                Event(type="llm_request", payload={"turn": turn, "messages": len(messages)})
            )
            resp = self.llm.complete(
                system=self._system_prompt,
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
            )
            in_tokens += resp.input_tokens
            out_tokens += resp.output_tokens
            self.hooks.emit(
                Event(
                    type="llm_response",
                    payload={
                        "turn": turn,
                        "stop_reason": resp.stop_reason,
                        "text_len": len(resp.text),
                    },
                )
            )
            messages.append(Message(role="assistant", content=resp.text))

            blocks = _extract_code_blocks(resp.text)
            if not blocks:
                self.hooks.emit(
                    Event(type="turn_complete", payload={"turn": turn, "done": True})
                )
                return RunResult(
                    text=resp.text,
                    turns=turn,
                    messages=messages,
                    executions=executions,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )

            # Snapshot ctx entries before execution so we can render the
            # delta to the agent (it uses this in place of print() to see
            # what landed in ctx).
            ctx_before = self.sandbox.ctx._entries()

            # Execute every fenced code block in this turn, one at a time,
            # against the shared persistent sandbox globals. Backends that
            # wrap Claude Code's own loop tend to emit several blocks per
            # response, and we'd silently lose all but the first if we only
            # ran one. We deliberately exec block-by-block (not concatenated)
            # so a SyntaxError in block N doesn't void blocks 1..N-1 — that
            # bit us in framework_compare when one block had stray markdown
            # em-dashes and erased all the prior assess() sub-agent results.
            block_results: list[ExecutionResult] = []
            lint_findings: list[dict] = []
            for i, block in enumerate(blocks, start=1):
                # Lint BEFORE / alongside execution so warnings come back
                # in the same turn that produced the code. The agent is
                # told to fix lint findings in the next turn before
                # continuing — that's the whole point of this surface:
                # catch dead-code bugs the moment they're written, not
                # after the buggy code is saved into a skill.
                if self._lint_schemas:
                    from .lint import lint_dicts
                    for f in lint_dicts(block, schemas=self._lint_schemas):
                        if len(blocks) > 1:
                            f["block"] = i
                        lint_findings.append(f)
                r = self.sandbox.execute(block)
                block_results.append(r)
                executions.append(r)
                if r.error is not None:
                    # Stop on first failure — let the agent see the error
                    # in feedback and decide what to do next turn. Subsequent
                    # blocks would likely depend on this one's state.
                    break
            feedback = _format_block_results(
                block_results,
                total_blocks=len(blocks),
                lint_findings=lint_findings,
            )

            # Render the ctx delta (added / changed / removed entries this
            # turn) right after the execution result. This is the agent's
            # primary signal in place of print() — when the agent assigns
            # `ctx.X = value`, the value shows up here next turn.
            ctx_delta = render_ctx_delta(self.sandbox.ctx, before=ctx_before)
            if ctx_delta:
                feedback = f"{feedback}\n{ctx_delta}"

            # Context-policy: if input_tokens crossed the threshold this
            # turn (and cooldown has elapsed), append a one-shot
            # system-reminder asking the agent to manage its ctx for the
            # next turn. The reminder lists the largest live entries
            # (with their source expressions), and any recently-dropped
            # entries with requery hints so the agent can bring them back
            # without losing the call that produced them.
            if self.context_policy is not None:
                # Resolve the concrete budget for THIS model — auto when the
                # policy didn't pin a number. This is what lets a default
                # ContextPolicy() do the right thing on Opus 4.7 (1M window)
                # vs Sonnet (200K) without the caller having to think about it.
                budget_concrete = self.context_policy.resolve_budget(self.model)
                fire = self.context_policy.should_remind(
                    input_tokens=resp.input_tokens,
                    turns_since_last=turns_since_reminder,
                    model=self.model,
                )
                if fire:
                    reminder = build_context_reminder(
                        self.sandbox.ctx,
                        input_tokens=resp.input_tokens,
                        budget_tokens=budget_concrete,
                        max_largest_entries=self.context_policy.max_largest_entries,
                        include_dropped=self.context_policy.include_dropped,
                    )
                    feedback = f"{feedback}\n\n{reminder}"
                    turns_since_reminder = 0
                    self.hooks.emit(
                        Event(
                            type="context_reminder_emitted",
                            payload={
                                "turn": turn,
                                "input_tokens": resp.input_tokens,
                                "budget_tokens": budget_concrete,
                                "ctx_entries": len(self.sandbox.ctx),
                                "ctx_dropped": len(self.sandbox.ctx._dropped()),
                            },
                        )
                    )
                else:
                    turns_since_reminder += 1

                # Print-habit reminder: two modes, both fire at most once per
                # turn (force takes precedence over size-based when both
                # would apply, since force is the stronger steer).
                #
                # FORCE mode: any non-trivial `print(...)` in the emitted
                # code fires the reminder, regardless of stdout size.
                # Triggered by inspecting the source, not the output —
                # so even small data prints are caught.
                #
                # SIZE mode: total stdout chars across blocks crossed the
                # `large_print_chars` threshold. Triggered by output.
                stdout_total = sum(len(r.stdout) for r in block_results)
                data_print_line: str | None = None
                data_print_block_i: int | None = None
                for i, block_src in enumerate(blocks, start=1):
                    line = _first_loud_print_line(block_src)
                    if line is not None:
                        data_print_line = line
                        data_print_block_i = i
                        break

                fired_print_warn = False
                if self.context_policy.should_force_ctx(
                    has_data_print=(data_print_line is not None),
                    turns_since_last=turns_since_print_warn,
                ):
                    print_reminder = build_print_habit_reminder(
                        stdout_chars=stdout_total,
                        block_index=data_print_block_i if len(block_results) > 1 else None,
                        code_snippet=data_print_line,
                        force=True,
                    )
                    feedback = f"{feedback}\n\n{print_reminder}"
                    fired_print_warn = True
                    self.hooks.emit(
                        Event(
                            type="print_habit_warning_emitted",
                            payload={
                                "turn": turn,
                                "mode": "force",
                                "stdout_chars": stdout_total,
                                "snippet": data_print_line,
                            },
                        )
                    )
                elif self.context_policy.should_warn_print(
                    stdout_chars=stdout_total,
                    turns_since_last=turns_since_print_warn,
                ):
                    loud_block_i, _ = max(
                        enumerate(block_results, start=1),
                        key=lambda ix_r: len(ix_r[1].stdout),
                    )
                    snippet = _first_loud_print_line(
                        blocks[loud_block_i - 1] if loud_block_i - 1 < len(blocks) else None
                    )
                    print_reminder = build_print_habit_reminder(
                        stdout_chars=stdout_total,
                        block_index=loud_block_i if len(block_results) > 1 else None,
                        code_snippet=snippet,
                        force=False,
                    )
                    feedback = f"{feedback}\n\n{print_reminder}"
                    fired_print_warn = True
                    self.hooks.emit(
                        Event(
                            type="print_habit_warning_emitted",
                            payload={
                                "turn": turn,
                                "mode": "size",
                                "stdout_chars": stdout_total,
                                "snippet": snippet,
                            },
                        )
                    )

                if fired_print_warn:
                    turns_since_print_warn = 0
                else:
                    turns_since_print_warn += 1

            messages.append(Message(role="user", content=feedback))
            self.hooks.emit(
                Event(
                    type="turn_complete",
                    payload={"turn": turn, "done": False, "blocks": len(blocks)},
                )
            )

        final_text = messages[-1].content if messages else ""
        return RunResult(
            text=final_text,
            turns=self.max_turns,
            messages=messages,
            executions=executions,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )

    def close(self) -> None:
        """Shut down owned MCP servers. Idempotent."""
        if self.mcp_runtime is not None:
            self.mcp_runtime.stop()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _first_loud_print_line(block: str | None) -> str | None:
    """Return the first `print(...)` line in `block` whose argument is NOT
    a string literal — that's the most likely culprit for a big stdout.

    String-literal prints (`print("ok")`) are almost certainly small; we
    skip them so the reminder's "Offending line" snippet points at the
    actual data dump, not a status message printed alongside it.
    """
    if not block:
        return None
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith("print("):
            continue
        # Heuristic: a print whose argument is just `"..."` or `'...'` —
        # i.e. opens immediately on a quote, with no expression inside —
        # is a tiny status line. Skip it.
        inner = s[len("print("):].lstrip()
        if inner.startswith(("'", '"')) and not any(c in inner for c in "+,({["):
            continue
        return s
    return None


def _extract_code_blocks(text: str) -> list[str]:
    """Return every Python code block in `text`, in order. Empty list if none.

    A fenced block whose body contains `# --- next block ---` separator
    comments is split on those markers so each segment runs independently.
    Without this, a SyntaxError inside any one segment would void every
    other segment of the same fence (see framework_compare_020702 trace).
    """
    out: list[str] = []
    for m in CODE_BLOCK_RE.finditer(text):
        body = m.group(1)
        for segment in BLOCK_SEPARATOR_RE.split(body):
            s = segment.strip("\n").rstrip()
            if s.strip():
                out.append(s)
    return out


def _extract_code_block(text: str) -> str | None:
    """Return the first Python code block in `text`, or None if none.

    Kept for backward compatibility with callers that only wanted one block.
    New code should use `_extract_code_blocks` (plural).
    """
    blocks = _extract_code_blocks(text)
    return blocks[0] if blocks else None


def _format_execution(result: ExecutionResult) -> str:
    parts: list[str] = ["<execution_result>"]
    if result.stdout:
        parts.append(f"<stdout>\n{result.stdout.rstrip()}\n</stdout>")
    if result.stderr:
        parts.append(f"<stderr>\n{result.stderr.rstrip()}\n</stderr>")
    if result.error:
        parts.append(f"<error>{result.error}</error>")
        if result.error_traceback:
            parts.append(f"<traceback>\n{result.error_traceback.rstrip()}\n</traceback>")
    if not result.stdout and not result.stderr and not result.error:
        parts.append("<stdout>(empty)</stdout>")
    parts.append("</execution_result>")
    return "\n".join(parts)


def _format_block_results(
    results: list[ExecutionResult],
    total_blocks: int,
    lint_findings: list[dict] | None = None,
) -> str:
    """Format per-block results for the next user message.

    Reports how many blocks ran successfully out of how many were emitted
    so the model can tell when execution stopped early on an error.
    Appends a ⚠️ lint section if `lint_findings` is non-empty so the
    agent sees code-quality warnings inline with execution output.
    """
    if not results:
        body = "<execution_result><stdout>(no blocks)</stdout></execution_result>"
    elif total_blocks == 1:
        body = _format_execution(results[0])
    else:
        parts: list[str] = []
        succeeded = sum(1 for r in results if r.error is None)
        parts.append(f"<execution_summary blocks_run='{len(results)}' "
                     f"blocks_emitted='{total_blocks}' "
                     f"succeeded='{succeeded}'/>")
        for i, r in enumerate(results, start=1):
            parts.append(f"<block n='{i}'>")
            sub = _format_execution(r)
            sub = sub.removeprefix("<execution_result>").removesuffix("</execution_result>").strip()
            parts.append(sub)
            parts.append("</block>")
        if len(results) < total_blocks:
            parts.append(
                f"<note>Stopped at block {len(results)} due to error; "
                f"{total_blocks - len(results)} subsequent block(s) NOT executed. "
                f"Next turn, address the failure before re-emitting them.</note>"
            )
        body = "\n".join(parts)

    if lint_findings:
        lint_lines = ["", "⚠️ Lint warnings (likely real bugs — fix in your next code block before continuing):"]
        for f in lint_findings:
            block_prefix = f"block {f['block']}, " if "block" in f else ""
            lint_lines.append(f"  - {block_prefix}{f.get('message', f)}")
        body = body + "\n" + "\n".join(lint_lines)

    return body
