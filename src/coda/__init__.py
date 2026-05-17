"""coda — a CodeAct coding harness with typed sub-agents and line-level observability."""

from .agent import Agent, RunResult
from .hooks import Event, EventType, HookRegistry
from .llm import (
    AnthropicClient,
    ClaudeAgentSDKClient,
    ClaudeCodeClient,
    CompletionResponse,
    LLMClient,
    Message,
    MockLLMClient,
    ToolCall,
    ToolChoice,
    ToolSchema,
)
from .mcp import McpToolError, MCPRuntime, MCPServer, ServerProxy
from .prompt import Assembler, Section, build_system_prompt, default_assembler
from .replay import replay
from .sandbox import ExecutionResult, Sandbox
from .skills import Skill, load_skills, save_skill
from .subagents import SubAgent, subagent
from .tools import Tool, tool
from .trace import TraceWriter, iter_trace, read_trace

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
    "default_assembler",
    "Assembler",
    "Section",
    "MCPServer",
    "MCPRuntime",
    "ServerProxy",
    "McpToolError",
    "Skill",
    "load_skills",
    "save_skill",
    "TraceWriter",
    "read_trace",
    "iter_trace",
    "replay",
    "LLMClient",
    "Message",
    "CompletionResponse",
    "ToolCall",
    "ToolChoice",
    "ToolSchema",
    "MockLLMClient",
    "ClaudeCodeClient",
    "ClaudeAgentSDKClient",
    "AnthropicClient",
]
