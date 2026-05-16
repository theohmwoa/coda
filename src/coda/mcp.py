"""MCP integration — expose Model Context Protocol servers as native Python objects.

Every tool every popular MCP server exposes (gmail, slack, github, stripe,
filesystem, ...) becomes a callable Python method inside the coda sandbox.
The model writes:

    unread = gmail.list_unread(label="INBOX", max_results=50)
    for email in unread:
        v = is_urgent(email)               # typed sub-agent
        if v.score >= 8:
            slack.post_message(channel="#triage", text=email["subject"])

instead of emitting tool-use JSON. Same MirageAI move we kept in coda —
tools are code, not protocol.

# Async-to-sync bridge

MCP is async (stdio sessions, JSON-RPC, awaitable everything). Coda's
sandbox is sync. We run a background daemon thread with its own asyncio
event loop, hold the MCP `ClientSession` contexts open inside an
`AsyncExitStack` on that loop, and dispatch tool calls into it via
`asyncio.run_coroutine_threadsafe(...).result()`. The sync caller blocks
until the MCP server replies.

# Lifecycle

`MCPRuntime.start(servers)` boots the event-loop thread and connects to
every server in parallel. `MCPRuntime.stop()` closes the contexts and
joins the thread. `Agent` owns one runtime and registers `stop()` with
`atexit` so subprocess MCP servers don't leak if the user forgets to
clean up.

# Result decoding

MCP returns `CallToolResult` with a list of content blocks. Coda's
decoder unwraps:

- isError → raise McpToolError(content)
- single TextContent that parses as JSON → return parsed value
- single TextContent that doesn't → return the string
- anything else → return [{type, ...}] list of content blocks

The "try JSON first" rule matters: most modern MCP servers return
structured payloads inside a TextContent for backward compat, and we
want the model to see dicts/lists not strings to parse.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import re
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


__all__ = [
    "MCPServer",
    "MCPRuntime",
    "ServerProxy",
    "McpToolError",
    "McpStartupError",
]


@dataclass
class MCPServer:
    """How to launch / connect to one MCP server."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        args: Iterable[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> "MCPServer":
        """Convenience constructor for an stdio-transport server."""
        return cls(
            name=name,
            command=command,
            args=list(args or []),
            env=dict(env or {}),
            cwd=cwd,
        )


class McpToolError(RuntimeError):
    """Raised when an MCP tool call returns isError=True."""


class McpStartupError(RuntimeError):
    """Raised when an MCP server fails to connect or initialize."""


_INVALID_PY_IDENT = re.compile(r"[^0-9a-zA-Z_]")


def _to_py_ident(name: str) -> str:
    """Sanitize an MCP tool/server name into a Python identifier."""
    s = _INVALID_PY_IDENT.sub("_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "_"


class MCPRuntime:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, list[Any]] = {}
        self._stopped = False

    # --- thread / event loop lifecycle ---------------------------------------

    def _start_loop(self) -> None:
        ready = threading.Event()

        def runner() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=runner, name="coda-mcp-runtime", daemon=True
        )
        self._thread.start()
        if not ready.wait(timeout=5):
            raise McpStartupError("MCP event loop thread failed to start within 5s")

    def _run(self, coro) -> Any:
        assert self._loop is not None, "MCPRuntime not started"
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    # --- public API ----------------------------------------------------------

    def start(self, servers: list[MCPServer]) -> None:
        if not servers:
            return
        try:
            import mcp  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MCP integration requires the `mcp` package. "
                "Install with: pip install 'coda[mcp]' or pip install mcp"
            ) from e
        self._start_loop()

        async def _setup() -> None:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            self._stack = AsyncExitStack()
            for cfg in servers:
                if cfg.command is None:
                    raise McpStartupError(
                        f"MCPServer {cfg.name!r}: only stdio is supported in v0 — "
                        "set `command=` via MCPServer.stdio(...)."
                    )
                params = StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args,
                    env={**cfg.env} if cfg.env else None,
                    cwd=cfg.cwd,
                )
                try:
                    read, write = await self._stack.enter_async_context(
                        stdio_client(params)
                    )
                    session = await self._stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    listed = await session.list_tools()
                except Exception as e:
                    raise McpStartupError(
                        f"MCPServer {cfg.name!r} failed to start: {e}"
                    ) from e
                self._sessions[cfg.name] = session
                self._tools[cfg.name] = list(listed.tools)

        self._run(_setup())

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._stack is not None and self._loop is not None:
            try:
                self._run(self._stack.aclose())
            except Exception:
                pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def call_tool(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        session = self._sessions.get(server)
        if session is None:
            raise KeyError(f"No MCP server registered as {server!r}")

        async def _go():
            return await session.call_tool(tool, args)

        result = self._run(_go())
        return _decode_result(result)

    def proxies(self) -> dict[str, "ServerProxy"]:
        """Build a {python_name: proxy} map of every connected server."""
        out: dict[str, ServerProxy] = {}
        for name, tools in self._tools.items():
            py_name = _to_py_ident(name)
            out[py_name] = ServerProxy(runtime=self, server_name=name, tools=tools)
        return out

    def list_servers(self) -> list[str]:
        return list(self._tools.keys())

    def list_tools(self, server: str) -> list[Any]:
        return list(self._tools.get(server, []))


def _decode_result(result: Any) -> Any:
    """Unwrap CallToolResult → Python-native value."""
    if getattr(result, "isError", False):
        raise McpToolError(_render_content(result))
    content = getattr(result, "content", []) or []
    if len(content) == 1 and hasattr(content[0], "text"):
        text = content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
    return [_content_block_to_dict(b) for b in content]


def _content_block_to_dict(b: Any) -> dict[str, Any]:
    if hasattr(b, "model_dump"):
        return b.model_dump()
    out: dict[str, Any] = {"type": getattr(b, "type", "unknown")}
    for attr in ("text", "data", "mimeType", "uri"):
        if hasattr(b, attr):
            out[attr] = getattr(b, attr)
    return out


def _render_content(result: Any) -> str:
    content = getattr(result, "content", []) or []
    parts = []
    for b in content:
        if hasattr(b, "text"):
            parts.append(b.text)
        else:
            parts.append(repr(b))
    return " ".join(parts) or "(no content)"


@dataclass
class _ToolMeta:
    name: str
    description: str
    input_schema: dict[str, Any]


def _tool_meta(t: Any) -> _ToolMeta:
    return _ToolMeta(
        name=getattr(t, "name", "?"),
        description=(getattr(t, "description", "") or "").strip(),
        input_schema=dict(getattr(t, "inputSchema", {}) or {}),
    )


class ServerProxy:
    """One MCP server, exposed as a Python object.

    Each MCP tool is reachable as an attribute:

        gmail.list_unread(label="INBOX", max_results=50)
        github.create_issue(repo="me/x", title="bug", body="...")

    Tool names are sanitized to be valid Python identifiers (`-` → `_`).
    Calls are synchronous from the caller's perspective; under the hood
    they're dispatched to the MCP runtime's background event loop.

    Reserved attributes (`list_tools`, `describe`, `_name`) shadow tool
    names if any conflict — rare, but it can happen.
    """

    def __init__(
        self,
        runtime: MCPRuntime,
        server_name: str,
        tools: list[Any],
    ) -> None:
        self._runtime = runtime
        self._name = server_name
        self._tools: dict[str, _ToolMeta] = {}
        for raw in tools:
            meta = _tool_meta(raw)
            py = _to_py_ident(meta.name)
            self._tools[py] = meta

    def list_tools(self) -> list[str]:
        """Return the Python-identifier names of every tool on this server."""
        return list(self._tools.keys())

    def describe(self, tool: str) -> dict[str, Any]:
        """Return {description, input_schema} for one tool, or raise KeyError."""
        meta = self._tools.get(tool)
        if meta is None:
            raise KeyError(f"{self._name!r} has no tool {tool!r}")
        return {"description": meta.description, "input_schema": meta.input_schema}

    def __getattr__(self, attr: str) -> Callable[..., Any]:
        if attr.startswith("_"):
            raise AttributeError(attr)
        meta = self._tools.get(attr)
        if meta is None:
            raise AttributeError(
                f"MCP server {self._name!r} has no tool {attr!r}. "
                f"Available: {list(self._tools.keys())}"
            )
        runtime = self._runtime
        server_name = self._name
        original_name = meta.name

        def caller(**kwargs):
            return runtime.call_tool(server_name, original_name, kwargs)

        caller.__name__ = attr
        caller.__qualname__ = f"{self._name}.{attr}"
        caller.__doc__ = meta.description
        return caller

    def __repr__(self) -> str:
        return f"<MCP server {self._name!r} with {len(self._tools)} tool(s)>"


_RUNTIMES_FOR_CLEANUP: list[MCPRuntime] = []


def _cleanup_all() -> None:
    for r in _RUNTIMES_FOR_CLEANUP:
        try:
            r.stop()
        except Exception:
            pass


atexit.register(_cleanup_all)


def register_for_atexit(runtime: MCPRuntime) -> None:
    _RUNTIMES_FOR_CLEANUP.append(runtime)
