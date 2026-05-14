"""coda — a CodeAct coding harness with typed sub-agents and line-level observability."""

from .llm import (
    AnthropicClient,
    ClaudeCodeClient,
    CompletionResponse,
    LLMClient,
    Message,
    MockLLMClient,
    ToolCall,
    ToolChoice,
    ToolSchema,
)

__version__ = "0.0.1"

__all__ = [
    "LLMClient",
    "Message",
    "CompletionResponse",
    "ToolCall",
    "ToolChoice",
    "ToolSchema",
    "MockLLMClient",
    "ClaudeCodeClient",
    "AnthropicClient",
]
