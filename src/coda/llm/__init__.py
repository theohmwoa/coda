"""Pluggable LLM backends for coda.

The agent loop never imports a specific provider. It accepts any object
satisfying the LLMClient protocol. Three concrete backends ship with coda:

    MockLLMClient        — canned responses for tests / offline dev
    ClaudeCodeClient     — subprocess `claude -p`, free with a paid plan pre-2026-06-15
    AnthropicClient      — direct Anthropic SDK, billed at API rates

The AnthropicClient and ClaudeCodeClient are imported lazily so coda has zero
hard runtime dependency on the `anthropic` package or the `claude` CLI for
users sticking to MockLLMClient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    CompletionResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolChoice,
    ToolSchema,
)
from .mock import MockCall, MockLLMClient, tool_call_response

if TYPE_CHECKING:
    from .agent_sdk_client import ClaudeAgentSDKClient
    from .anthropic_client import AnthropicClient
    from .claude_code_client import ClaudeCodeClient


def __getattr__(name: str):
    if name == "AnthropicClient":
        from .anthropic_client import AnthropicClient as _AC
        return _AC
    if name == "ClaudeCodeClient":
        from .claude_code_client import ClaudeCodeClient as _CCC
        return _CCC
    if name == "ClaudeAgentSDKClient":
        from .agent_sdk_client import ClaudeAgentSDKClient as _CAS
        return _CAS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLMClient",
    "Message",
    "CompletionResponse",
    "ToolCall",
    "ToolChoice",
    "ToolSchema",
    "MockLLMClient",
    "MockCall",
    "tool_call_response",
    "AnthropicClient",
    "ClaudeCodeClient",
    "ClaudeAgentSDKClient",
]
