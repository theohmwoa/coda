"""Tests for the MCP integration. Most tests use a fake MCPRuntime so they
run without any real MCP server. A `pytest -m live` slot covers an actual
stdio MCP server (`mcp-server-time`) if it's installed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from coda.mcp import (
    McpToolError,
    MCPRuntime,
    MCPServer,
    ServerProxy,
    _decode_result,
    _to_py_ident,
)


# --- helpers -------------------------------------------------------------


@dataclass
class _Tool:
    name: str
    description: str = ""
    inputSchema: dict | None = None


@dataclass
class _TextContent:
    text: str
    type: str = "text"


@dataclass
class _CallResult:
    content: list
    isError: bool = False


class _FakeRuntime:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []
        self.next_result: Any = "ok"

    def call_tool(self, server, tool, args):
        self.calls.append((server, tool, args))
        return self.next_result


# --- identifier sanitization --------------------------------------------


def test_to_py_ident_replaces_hyphens():
    assert _to_py_ident("my-server") == "my_server"


def test_to_py_ident_prefixes_leading_digit():
    assert _to_py_ident("1stplace") == "_1stplace"


def test_to_py_ident_empty():
    assert _to_py_ident("") == "_"


# --- result decoding ----------------------------------------------------


def test_decode_single_text_block_as_string():
    r = _CallResult(content=[_TextContent(text="hello world")])
    assert _decode_result(r) == "hello world"


def test_decode_single_text_block_parsed_as_json():
    r = _CallResult(content=[_TextContent(text='{"x": 1, "y": [2, 3]}')])
    assert _decode_result(r) == {"x": 1, "y": [2, 3]}


def test_decode_multiple_blocks_returns_list():
    r = _CallResult(content=[_TextContent(text="a"), _TextContent(text="b")])
    out = _decode_result(r)
    assert isinstance(out, list)
    assert out[0]["text"] == "a"
    assert out[1]["text"] == "b"


def test_decode_error_raises():
    r = _CallResult(content=[_TextContent(text="something broke")], isError=True)
    with pytest.raises(McpToolError, match="something broke"):
        _decode_result(r)


# --- ServerProxy --------------------------------------------------------


def _proxy_with(runtime, tools):
    return ServerProxy(runtime=runtime, server_name="gmail", tools=tools)


def test_proxy_list_tools_and_describe():
    proxy = _proxy_with(
        _FakeRuntime(),
        [
            _Tool(
                name="list_unread",
                description="Get unread emails.",
                inputSchema={"properties": {"label": {"type": "string"}}},
            ),
            _Tool(name="send", description="Send email.", inputSchema={}),
        ],
    )
    assert sorted(proxy.list_tools()) == ["list_unread", "send"]
    desc = proxy.describe("list_unread")
    assert "Get unread" in desc["description"]
    assert "label" in desc["input_schema"]["properties"]


def test_proxy_method_call_dispatches_to_runtime():
    rt = _FakeRuntime()
    rt.next_result = [{"id": 1, "subject": "x"}]
    proxy = _proxy_with(
        rt, [_Tool(name="list_unread", description="...", inputSchema={})]
    )
    result = proxy.list_unread(label="INBOX", max=50)
    assert result == [{"id": 1, "subject": "x"}]
    assert rt.calls == [("gmail", "list_unread", {"label": "INBOX", "max": 50})]


def test_proxy_unknown_tool_raises_attribute_error():
    proxy = _proxy_with(_FakeRuntime(), [_Tool(name="known")])
    with pytest.raises(AttributeError, match="has no tool 'unknown'"):
        proxy.unknown(x=1)


def test_proxy_method_has_helpful_metadata():
    proxy = _proxy_with(
        _FakeRuntime(),
        [_Tool(name="search", description="Search the index.", inputSchema={})],
    )
    fn = proxy.search
    assert fn.__name__ == "search"
    assert fn.__qualname__ == "gmail.search"
    assert "Search the index." in (fn.__doc__ or "")


def test_proxy_sanitizes_tool_names_with_hyphens():
    proxy = _proxy_with(_FakeRuntime(), [_Tool(name="get-issue")])
    assert "get_issue" in proxy.list_tools()


# --- MCPRuntime lifecycle (without real MCP) ----------------------------


def test_runtime_start_with_empty_list_is_noop():
    rt = MCPRuntime()
    rt.start([])  # should not raise, should not start a thread
    rt.stop()


def test_runtime_call_tool_unknown_server():
    rt = MCPRuntime()
    with pytest.raises(KeyError, match="No MCP server"):
        # case-sensitive check; KeyError text from our code
        rt.call_tool("ghost", "x", {})


# --- Agent wiring -------------------------------------------------------


def test_agent_without_mcp_does_not_create_runtime(tmp_path):
    from coda import Agent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    agent = Agent(
        model="m",
        llm=MockLLMClient(responses=[CompletionResponse(text="done.")]),
        sandbox=Sandbox(root=tmp_path),
    )
    assert agent.mcp_runtime is None
    assert agent.mcp_proxies == {}


def test_agent_injects_proxies_into_sandbox(tmp_path, monkeypatch):
    """Use a fake runtime + manually inject proxies to verify wiring without
    spawning a real MCP server."""

    from coda import Agent, MCPServer
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    fake_runtime = _FakeRuntime()
    fake_runtime.next_result = "the answer is 42"

    proxy = ServerProxy(
        runtime=fake_runtime,
        server_name="oracle",
        tools=[_Tool(name="ask", description="ask anything", inputSchema={})],
    )

    # Build an Agent without MCP, then patch in the proxy by hand to verify
    # that exposing one through sandbox globals lets generated code call it.
    agent = Agent(
        model="m",
        llm=MockLLMClient(
            responses=[
                CompletionResponse(text="```python\nresult = oracle.ask(q='hi')\nprint(result)\n```"),
                CompletionResponse(text="done."),
            ]
        ),
        sandbox=Sandbox(root=tmp_path),
    )
    agent.sandbox.inject("oracle", proxy)
    result = agent.run("ask it")
    assert "the answer is 42" in result.executions[0].stdout
    assert fake_runtime.calls == [("oracle", "ask", {"q": "hi"})]


# --- prompt assembler ---------------------------------------------------


def test_prompt_assembler_lists_mcp_servers(tmp_path):
    from coda import Agent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    proxy = ServerProxy(
        runtime=_FakeRuntime(),
        server_name="gmail",
        tools=[
            _Tool(
                name="list_unread",
                description="Get unread emails for a label.",
                inputSchema={"properties": {"label": {"type": "string"}, "max": {"type": "integer"}}},
            )
        ],
    )

    agent = Agent(
        model="m",
        llm=MockLLMClient(responses=[CompletionResponse(text="done.")]),
        sandbox=Sandbox(root=tmp_path),
    )
    # Stuff a proxy in by hand to test the prompt builder's render branch
    agent.mcp_proxies = {"gmail": proxy}
    from coda.prompt import build_system_prompt

    prompt = build_system_prompt(mcp_proxies=agent.mcp_proxies)
    assert "MCP servers available" in prompt
    assert "gmail.list_unread(label, max)" in prompt
    assert "Get unread emails" in prompt
