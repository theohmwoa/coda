"""AnthropicClient — direct Anthropic Messages API.

Production backend. Billed per-token at API rates. The user pays for prompt
cache hits separately; this client surfaces cache token counts on the response
so coda can log prompt-assembly efficiency.

We import the `anthropic` package lazily so that `import coda` doesn't require
it for users on the mock or subprocess backends.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from .base import (
    CompletionResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolChoice,
    ToolSchema,
)


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicClient requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )

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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        if tool_choice is not None:
            kwargs["tool_choice"] = _encode_tool_choice(tool_choice)

        resp = self._client.messages.create(**kwargs)
        return _decode_response(resp)


def _encode_tool_choice(tc: ToolChoice) -> dict[str, Any]:
    if tc.mode == "tool":
        if not tc.name:
            raise ValueError("ToolChoice(mode='tool') requires a `name`.")
        return {"type": "tool", "name": tc.name}
    return {"type": tc.mode}


def _decode_response(resp: Any) -> CompletionResponse:
    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(resp, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_chunks.append(block.text)
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, input=dict(block.input))
            )
    usage = getattr(resp, "usage", None)
    # The Anthropic API splits `input_tokens` (only NEW, uncached) from
    # `cache_read_input_tokens` and `cache_creation_input_tokens`. Context
    # budget checks must sum all three — the model SEES every cached token,
    # cache just reduces $/latency. Sub the missing fields default to 0,
    # which is correct for non-cached responses.
    input_total = 0
    if usage is not None:
        input_total = (
            (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        )
    return CompletionResponse(
        text="".join(text_chunks),
        tool_calls=tool_calls,
        stop_reason=getattr(resp, "stop_reason", "end_turn") or "end_turn",
        input_tokens=input_total,
        output_tokens=(getattr(usage, "output_tokens", 0) or 0) if usage else 0,
        raw=resp,
    )
