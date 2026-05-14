"""MockLLMClient — canned-response backend for tests and offline dev.

Two modes:

- `MockLLMClient(responses=[...])` — return responses in order. Raises if the
  agent loop calls `complete()` more times than there are scripted responses.
- `MockLLMClient(responder=fn)` — call `fn(call_context)` for each request and
  use what it returns. Useful when the response depends on the conversation
  state (e.g. simulating a sub-agent that should fail twice then succeed).

The point is to exercise the agent loop, prompt assembly, and sandbox without
spending money or needing network access. Tests using this backend can assert
exactly what the loop sent and what the model "saw".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .base import (
    CompletionResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolChoice,
    ToolSchema,
)


@dataclass
class MockCall:
    """Snapshot of one call to MockLLMClient. Inspected by tests."""

    system: str
    messages: list[Message]
    model: str
    max_tokens: int
    temperature: float
    stop_sequences: list[str] | None
    tools: list[ToolSchema] | None
    tool_choice: ToolChoice | None


Responder = Callable[[MockCall], CompletionResponse]


class MockLLMClient(LLMClient):
    def __init__(
        self,
        responses: list[CompletionResponse | str] | None = None,
        responder: Responder | None = None,
    ) -> None:
        if responses is None and responder is None:
            raise ValueError("MockLLMClient needs either `responses=` or `responder=`.")
        if responses is not None and responder is not None:
            raise ValueError("Pass `responses=` OR `responder=`, not both.")
        self._responses: list[CompletionResponse] = [
            r if isinstance(r, CompletionResponse) else CompletionResponse(text=r)
            for r in (responses or [])
        ]
        self._responder = responder
        self.calls: list[MockCall] = []

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
        call = MockCall(
            system=system,
            messages=list(messages),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
        )
        self.calls.append(call)
        if self._responder is not None:
            return self._responder(call)
        if not self._responses:
            raise RuntimeError(
                f"MockLLMClient exhausted: agent made {len(self.calls)} call(s) "
                "but no more scripted responses are queued."
            )
        return self._responses.pop(0)


def tool_call_response(
    name: str,
    args: dict,
    call_id: str = "mock_tool_1",
    text: str = "",
) -> CompletionResponse:
    """Convenience constructor for a tool_use response.

    Tests for typed sub-agents need a response that carries a `tool_calls`
    block matching the expected schema. This is verbose to spell out inline.
    """
    return CompletionResponse(
        text=text,
        tool_calls=[ToolCall(id=call_id, name=name, input=args)],
        stop_reason="tool_use",
    )
