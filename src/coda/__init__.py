"""coda — a CodeAct coding harness with typed sub-agents and line-level observability."""

from .agent import Agent, RunResult
from .hooks import Event, EventType, HookRegistry
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
from .prompt import build_system_prompt
from .sandbox import ExecutionResult, Sandbox
from .subagents import SubAgent, subagent
from .tools import Tool, tool

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "RunResult",
    "Sandbox",
    "ExecutionResult",
    "HookRegistry",
    "Event",
    "EventType",
    "tool",
    "Tool",
    "subagent",
    "SubAgent",
    "build_system_prompt",
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
