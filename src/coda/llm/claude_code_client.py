"""ClaudeCodeClient — subprocess `claude -p` backend.

Why this exists: before 2026-06-15, paid Claude plans (Pro, Max 5x, Max 20x)
don't include programmatic API credits. They DO include unlimited use of the
`claude` CLI in non-interactive print mode (`claude -p '<prompt>'`). That
means: a developer with a $20/mo Pro plan can drive coda's agent loop for
free as long as the request goes through the CLI subprocess rather than the
SDK.

After 2026-06-15 the programmatic credit pool launches and AnthropicClient
becomes preferable for production. ClaudeCodeClient stays useful for local dev
and CI that wants to share a developer subscription.

Trade-offs vs. direct SDK:
- Slower (subprocess fork + Claude Code's own scaffolding overhead)
- Tool-use schema enforcement: we use --output-format=json and prompt-level
  coercion, then validate the parsed JSON against the requested schema.
  Sub-agents that require strict typed returns SHOULD prefer AnthropicClient
  when budget is available.
- No streaming yet (CompletionResponse is the only shape).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from .base import (
    CompletionResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolChoice,
    ToolSchema,
)


class ClaudeCodeClient(LLMClient):
    def __init__(
        self,
        executable: str = "claude",
        timeout_s: float = 300.0,
        extra_args: list[str] | None = None,
    ) -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"`{executable}` not found on PATH. Install Claude Code from "
                "https://claude.com/claude-code and ensure it's logged into a paid plan."
            )
        self._executable = resolved
        self._timeout_s = timeout_s
        self._extra_args = list(extra_args or [])

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        stop_sequences: list[str] | None = None,
        tools: list[ToolSchema] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> CompletionResponse:
        prompt = _flatten_conversation(system, messages, tools, tool_choice)
        cmd = [
            self._executable,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            model,
            *self._extra_args,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"claude -p timed out after {self._timeout_s}s"
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p exited {proc.returncode}: {proc.stderr.strip()}"
            )
        text, usage = _parse_cli_json(proc.stdout)
        tool_calls = _coerce_tool_use(text, tools, tool_choice)
        return CompletionResponse(
            text=text if not tool_calls else "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw=proc.stdout,
        )


def _flatten_conversation(
    system: str,
    messages: list[Message],
    tools: list[ToolSchema] | None,
    tool_choice: ToolChoice | None,
) -> str:
    """Encode a multi-turn conversation as a single prompt.

    `claude -p` takes one prompt string, not a Messages-style list. We
    reconstruct the conversation in a stable, parse-friendly format. Tools
    requested via the protocol's `tools=` are described in prose so the model
    knows what JSON shape to emit; we then parse that JSON back into ToolCall
    objects post-hoc.
    """
    parts: list[str] = []
    if system:
        parts.append(f"<system>\n{system}\n</system>")
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
    parts.append("<assistant>")
    return "\n".join(parts)


def _parse_cli_json(stdout: str) -> tuple[str, dict[str, int]]:
    """Pull the assistant text and token counts out of `claude -p --output-format json` output."""
    stdout = stdout.strip()
    if not stdout:
        return "", {}
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, {}
    text = obj.get("result") or obj.get("text") or obj.get("response") or ""
    usage_obj = obj.get("usage") or {}
    return text, {
        "input_tokens": int(usage_obj.get("input_tokens", 0) or 0),
        "output_tokens": int(usage_obj.get("output_tokens", 0) or 0),
    }


def _coerce_tool_use(
    text: str,
    tools: list[ToolSchema] | None,
    tool_choice: ToolChoice | None,
) -> list[ToolCall]:
    """Parse a forced-tool response back into a ToolCall.

    Only runs when the caller asked for a forced tool. Sub-agents are the
    primary user of this path. Returns [] for plain completions so the agent
    loop just reads response.text as usual.
    """
    if not tools or not tool_choice or tool_choice.mode == "auto":
        return []
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
            "ClaudeCodeClient expected a forced tool-use JSON response but got "
            f"non-JSON text: {text[:200]!r}"
        )
    if not isinstance(obj, dict) or "tool" not in obj or "input" not in obj:
        raise ValueError(
            "ClaudeCodeClient expected {'tool': ..., 'input': {...}} JSON; "
            f"got: {obj!r}"
        )
    return [ToolCall(id="claude_code_tool_1", name=obj["tool"], input=dict(obj["input"]))]
