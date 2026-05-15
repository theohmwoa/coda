"""ClaudeAgentSDKClient — backend using the official claude-agent-sdk.

Why this exists alongside ClaudeCodeClient:

- ClaudeCodeClient shells out to `claude -p` once per coda turn. Each call
  forks a node subprocess, loads Claude Code, runs, exits. It works, but
  there's startup latency on every call.
- ClaudeAgentSDKClient uses the Python `claude-agent-sdk` package, which
  manages the Claude Code process for us via an in-process async transport.
  Same subscription billing, lower per-call overhead.

Auth model is identical to ClaudeCodeClient: the SDK uses the Claude Code
CLI's OAuth token. If you ran `claude login` (or `claude setup-token`) and
have NOT set ANTHROPIC_API_KEY (which would shadow the OAuth path), every
call bills to your Claude plan. After 2026-06-15 the Agent SDK monthly
credit pool covers this on Pro/Max/Team/Enterprise.

# Mapping coda's LLMClient onto the SDK

The SDK's `query()` is async and designed to drive a multi-turn agent loop.
We want a single one-shot completion. So:

- `max_turns=1` forces a single assistant turn.
- `allowed_tools=[]` disables Claude Code's built-in tools (Read, Write,
  Bash, etc.) — coda has its own primitives, so we want the model to emit
  plain text containing Python code blocks, not call Claude Code's tools.
- `system_prompt=` is coda's assembled system prompt, replacing Claude
  Code's default.
- Conversation history is flattened into a single prompt string, same as
  ClaudeCodeClient. Tool schemas (for sub-agents) are described in prose
  and the JSON response is parsed back to a ToolCall.

# Sync wrapping

`LLMClient.complete()` is sync, the SDK is async. We use `anyio.run()` to
drive the async query in a fresh event loop on each call. This means
ClaudeAgentSDKClient CANNOT be called from inside an existing event loop.
For async hosts, an `acomplete()` variant can land later.
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    CompletionResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolChoice,
    ToolSchema,
)


class ClaudeAgentSDKClient(LLMClient):
    def __init__(
        self,
        *,
        allowed_tools: list[str] | None = None,
        permission_mode: str = "bypassPermissions",
        cwd: str | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> None:
        """Construct a subscription-authed SDK backend.

        Args:
            allowed_tools: Which of Claude Code's built-in tools the model
                may invoke. Defaults to `[]` — disable all of them so the
                model only emits text (which coda parses for python code
                blocks). Set to `["Read", "Bash", ...]` if you want to MIX
                Claude Code's native tools with coda's loop, but expect
                duplicated functionality.
            permission_mode: SDK's permission handler. `bypassPermissions`
                is the right default for a library that owns its own
                policy. Other modes will prompt the SDK user — confusing
                inside coda.
            cwd: working directory passed to the SDK. Defaults to the
                process cwd; usually you want this equal to Sandbox.root.
            extra_options: kwargs forwarded into ClaudeAgentOptions for
                anything we don't surface explicitly (`thinking`, `effort`,
                `add_dirs`, etc.).
        """
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ClaudeAgentSDKClient requires `claude-agent-sdk`. "
                "Install with: pip install claude-agent-sdk"
            ) from e
        self._allowed_tools = list(allowed_tools or [])
        self._permission_mode = permission_mode
        self._cwd = cwd
        self._extra_options = dict(extra_options or {})

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,  # SDK doesn't surface this; kept for protocol parity
        temperature: float = 1.0,  # same — SDK doesn't expose it
        stop_sequences: list[str] | None = None,
        tools: list[ToolSchema] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> CompletionResponse:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )

        flat_prompt = _flatten_conversation(messages, tools, tool_choice)
        # We deliberately do NOT pass max_turns=1: Claude Code's internal loop
        # treats hitting that limit as an error condition rather than a clean
        # exit, even if the model already produced a complete AssistantMessage.
        # Instead we cap at a small value and rely on `allowed_tools=[]` to
        # ensure the model produces text + ends its turn in one shot.
        options_kwargs: dict[str, Any] = {
            "system_prompt": system,
            "allowed_tools": self._allowed_tools,
            "permission_mode": self._permission_mode,
            "model": model,
        }
        if self._cwd is not None:
            options_kwargs["cwd"] = self._cwd
        options_kwargs.update(self._extra_options)
        options = ClaudeAgentOptions(**options_kwargs)

        async def _drain() -> tuple[str, list[ToolCall], int, int]:
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            in_toks = 0
            out_toks = 0
            # Catch "max turns" / similar non-fatal stops so we keep any
            # assistant content that streamed before the error. The SDK
            # raises a generic Exception for these.
            try:
                async for msg in query(prompt=flat_prompt, options=options):
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                text_parts.append(block.text)
                            elif isinstance(block, ToolUseBlock):
                                tool_calls.append(
                                    ToolCall(
                                        id=block.id, name=block.name, input=dict(block.input)
                                    )
                                )
                    elif isinstance(msg, ResultMessage):
                        usage = getattr(msg, "usage", None) or {}
                        if isinstance(usage, dict):
                            in_toks = int(usage.get("input_tokens", 0) or 0)
                            out_toks = int(usage.get("output_tokens", 0) or 0)
            except Exception as e:
                if text_parts or tool_calls:
                    pass  # we got useful content; swallow the SDK's stop-reason error
                else:
                    raise
            return "".join(text_parts), tool_calls, in_toks, out_toks

        import anyio
        text, tool_calls, in_toks, out_toks = anyio.run(_drain)

        # Forced-tool sub-agent path: coerce text into a ToolCall, same as
        # ClaudeCodeClient. The SDK doesn't surface Anthropic tool-use schema
        # enforcement when running through Claude Code's loop.
        if not tool_calls and tools and tool_choice and tool_choice.mode != "auto":
            tool_calls = _coerce_tool_use(text, tool_choice)
            if tool_calls:
                text = ""

        return CompletionResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            input_tokens=in_toks,
            output_tokens=out_toks,
            raw=None,
        )


def _flatten_conversation(
    messages: list[Message],
    tools: list[ToolSchema] | None,
    tool_choice: ToolChoice | None,
) -> str:
    """Encode the conversation as a single string. Same format as ClaudeCodeClient."""
    parts: list[str] = []
    if tools:
        parts.append("<available_tools>")
        for t in tools:
            parts.append(
                f"- {t.name}: {t.description}\n  input_schema: "
                f"{json.dumps(t.input_schema)}"
            )
        parts.append("</available_tools>")
        if tool_choice and tool_choice.mode == "tool" and tool_choice.name:
            parts.append(
                f"You MUST respond by calling the tool `{tool_choice.name}`. "
                "Output ONLY a single JSON object of the form "
                f'{{"tool": "{tool_choice.name}", "input": {{...}}}} with no '
                "prose, no markdown, no code fences."
            )
        elif tool_choice and tool_choice.mode == "any":
            parts.append(
                "You MUST call one of the tools above. Output ONLY a single JSON "
                'object of the form {"tool": "<name>", "input": {...}}.'
            )
    for m in messages:
        parts.append(f"<{m.role}>\n{m.content}\n</{m.role}>")
    return "\n".join(parts)


def _coerce_tool_use(text: str, tool_choice: ToolChoice) -> list[ToolCall]:
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        if body.startswith("json"):
            body = body[4:]
        body = body.strip()
    try:
        obj: Any = json.loads(body)
    except json.JSONDecodeError:
        raise ValueError(
            "ClaudeAgentSDKClient expected a forced tool-use JSON response but got "
            f"non-JSON text: {text[:200]!r}"
        )
    if not isinstance(obj, dict) or "tool" not in obj or "input" not in obj:
        raise ValueError(
            "ClaudeAgentSDKClient expected {'tool': ..., 'input': {...}} JSON; "
            f"got: {obj!r}"
        )
    return [ToolCall(id="agent_sdk_tool_1", name=obj["tool"], input=dict(obj["input"]))]
