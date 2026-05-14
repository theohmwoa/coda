"""LLMClient protocol — the pluggable backend boundary for coda.

coda's agent loop does not depend on Anthropic. It depends on this protocol.
Three concrete backends ship with coda:

- MockLLMClient        — canned responses for tests and offline dev
- ClaudeCodeClient     — subprocess `claude -p ...`, billed to a paid plan
- AnthropicClient      — direct Anthropic SDK, billed at API rates

The shape is deliberately close to Anthropic's Messages API because that's the
target. Other providers (OpenAI, OpenAI-compatible local servers) can be
adapted by translating in their concrete clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


Role = Literal["user", "assistant"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


@dataclass
class Message:
    """One turn in a conversation. Content is plain text for now.

    Tool results from sub-agent calls are encoded as user messages whose text
    body is the serialized tool result. This keeps the wire-format minimal;
    structured content blocks can be added later without breaking the protocol.
    """

    role: Role
    content: str


@dataclass
class ToolSchema:
    """A tool the model is allowed to call.

    Used for sub-agent schema enforcement (the inner Claude call that has to
    return a Pydantic-shaped object). NOT used for native coda primitives
    (`ls`, `read`, etc.) — those are exposed to the model as Python functions
    in the sandbox, not as JSON tool calls.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A tool-use block returned by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolChoice:
    """How aggressively to force a tool call.

    - mode="auto": the model decides
    - mode="any":  the model must call SOME tool
    - mode="tool": the model must call this specific tool (use `name`)
    """

    mode: Literal["auto", "any", "tool"] = "auto"
    name: str | None = None


@dataclass
class CompletionResponse:
    """What a backend returns from `complete()`.

    `text` is the assistant's natural-language output (often containing the
    Python code blocks coda will execute). `tool_calls` is non-empty when the
    model invoked a typed sub-agent or tool. `raw` carries the provider's
    original response object for callers that need provider-specific fields
    (token counts, cache hits, etc.).
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


class LLMClient(Protocol):
    """The single seam coda's agent loop talks through.

    Implementations MUST be deterministic with respect to their inputs when
    `temperature` is 0 (so tests using MockLLMClient stay stable) and SHOULD
    surface token counts when the provider returns them.
    """

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
        """Run one completion.

        Sub-agents pass `tools=[...]` plus `tool_choice=ToolChoice(mode="tool",
        name=...)` to force a structured return. The main agent loop usually
        passes neither and just reads `response.text`.
        """
        ...
