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

        for sa in self.subagents:
            sa.bind(self.llm, self.hooks)
            self.sandbox.inject(sa.name, sa)
        for t in self.tools:
            self.sandbox.inject(t.name, t)

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
            # The agent itself can crystallize new skills mid-run.
            self.sandbox.inject("save_skill", make_save_skill(self.skills_dir))

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

        for turn in range(1, self.max_turns + 1):
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

            # Execute every fenced code block in this turn, one at a time,
            # against the shared persistent sandbox globals. Backends that
            # wrap Claude Code's own loop tend to emit several blocks per
            # response, and we'd silently lose all but the first if we only
            # ran one. We deliberately exec block-by-block (not concatenated)
            # so a SyntaxError in block N doesn't void blocks 1..N-1 — that
            # bit us in framework_compare when one block had stray markdown
            # em-dashes and erased all the prior assess() sub-agent results.
            block_results: list[ExecutionResult] = []
            for i, block in enumerate(blocks, start=1):
                r = self.sandbox.execute(block)
                block_results.append(r)
                executions.append(r)
                if r.error is not None:
                    # Stop on first failure — let the agent see the error
                    # in feedback and decide what to do next turn. Subsequent
                    # blocks would likely depend on this one's state.
                    break
            feedback = _format_block_results(block_results, total_blocks=len(blocks))
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


def _extract_code_blocks(text: str) -> list[str]:
    """Return every Python code block in `text`, in order. Empty list if none."""
    return [m.group(1).rstrip() for m in CODE_BLOCK_RE.finditer(text)]


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


def _format_block_results(results: list[ExecutionResult], total_blocks: int) -> str:
    """Format per-block results for the next user message.

    Reports how many blocks ran successfully out of how many were emitted
    so the model can tell when execution stopped early on an error.
    """
    if not results:
        return "<execution_result><stdout>(no blocks)</stdout></execution_result>"
    if total_blocks == 1:
        return _format_execution(results[0])
    parts: list[str] = []
    succeeded = sum(1 for r in results if r.error is None)
    parts.append(f"<execution_summary blocks_run='{len(results)}' "
                 f"blocks_emitted='{total_blocks}' "
                 f"succeeded='{succeeded}'/>")
    for i, r in enumerate(results, start=1):
        parts.append(f"<block n='{i}'>")
        body = _format_execution(r)
        # Strip the outer <execution_result> wrapper for compactness
        body = body.removeprefix("<execution_result>").removesuffix("</execution_result>").strip()
        parts.append(body)
        parts.append("</block>")
    if len(results) < total_blocks:
        parts.append(
            f"<note>Stopped at block {len(results)} due to error; "
            f"{total_blocks - len(results)} subsequent block(s) NOT executed. "
            f"Next turn, address the failure before re-emitting them.</note>"
        )
    return "\n".join(parts)
