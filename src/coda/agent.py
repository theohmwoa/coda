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
from typing import Any

from .hooks import Event, HookRegistry
from .llm import LLMClient, Message
from .mcp import MCPRuntime, MCPServer, ServerProxy, register_for_atexit
from .prompt import Assembler, build_system_prompt, default_assembler
from .sandbox import ExecutionResult, Sandbox
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

            code = _extract_code_block(resp.text)
            if code is None:
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

            result = self.sandbox.execute(code)
            executions.append(result)
            feedback = _format_execution(result)
            messages.append(Message(role="user", content=feedback))
            self.hooks.emit(
                Event(type="turn_complete", payload={"turn": turn, "done": False})
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


def _extract_code_block(text: str) -> str | None:
    """Return the first Python code block in `text`, or None if none.

    We deliberately take only ONE block per turn: turns stay small, errors
    stay localized, and the model can react to each result before continuing.
    """
    m = CODE_BLOCK_RE.search(text)
    if m is None:
        return None
    return m.group(1).rstrip()


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
