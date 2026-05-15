"""Tests for ClaudeAgentSDKClient.

Uses monkeypatching to fake the SDK's async iterator. No real subscription
or Claude Code install needed; the live smoke test is in examples/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

from coda.llm import Message, ToolChoice, ToolSchema
from coda.llm.agent_sdk_client import ClaudeAgentSDKClient, _flatten_conversation


@dataclass
class _Text:
    text: str
    type: str = "text"


@dataclass
class _ToolUse:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _Assistant:
    content: list[Any] = field(default_factory=list)


@dataclass
class _Result:
    usage: dict


async def _async_iter(items):
    for x in items:
        yield x


def _patch_sdk(monkeypatch, messages, capture: dict):
    """Replace query() and the message classes the client isinstance-checks."""
    from coda.llm import agent_sdk_client as mod

    # Make isinstance checks see our fakes
    monkeypatch.setattr(
        claude_agent_sdk, "AssistantMessage", _Assistant, raising=False
    )
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _Result, raising=False)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _Text, raising=False)
    monkeypatch.setattr(claude_agent_sdk, "ToolUseBlock", _ToolUse, raising=False)

    def fake_query(*, prompt, options, transport=None):
        capture["prompt"] = prompt
        capture["options"] = options
        return _async_iter(messages)

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query, raising=False)


def test_plain_text_completion(monkeypatch):
    capture: dict = {}
    _patch_sdk(
        monkeypatch,
        [
            _Assistant(content=[_Text(text="hello world")]),
            _Result(usage={"input_tokens": 42, "output_tokens": 9}),
        ],
        capture,
    )
    client = ClaudeAgentSDKClient()
    resp = client.complete(
        system="sys",
        messages=[Message("user", "hi")],
        model="claude-sonnet-4-6",
    )
    assert resp.text == "hello world"
    assert resp.tool_calls == []
    assert resp.input_tokens == 42
    assert resp.output_tokens == 9
    opts = capture["options"]
    assert opts.system_prompt == "sys"
    assert opts.model == "claude-sonnet-4-6"
    assert opts.allowed_tools == []


def test_forced_tool_use_coerced_from_text(monkeypatch):
    capture: dict = {}
    payload = '{"tool": "evaluate", "input": {"id": 7}}'
    _patch_sdk(
        monkeypatch,
        [_Assistant(content=[_Text(text=payload)])],
        capture,
    )
    client = ClaudeAgentSDKClient()
    resp = client.complete(
        system="",
        messages=[Message("user", "go")],
        model="m",
        tools=[ToolSchema("evaluate", "eval", {"type": "object"})],
        tool_choice=ToolChoice(mode="tool", name="evaluate"),
    )
    assert resp.stop_reason == "tool_use"
    assert resp.text == ""
    assert resp.tool_calls[0].name == "evaluate"
    assert resp.tool_calls[0].input == {"id": 7}


def test_native_tool_use_block_decoded(monkeypatch):
    """If the SDK returns a real ToolUseBlock (allowed_tools path), use it directly."""
    capture: dict = {}
    _patch_sdk(
        monkeypatch,
        [_Assistant(content=[_ToolUse(id="t1", name="Read", input={"path": "x"})])],
        capture,
    )
    client = ClaudeAgentSDKClient(allowed_tools=["Read"])
    resp = client.complete(
        system="",
        messages=[Message("user", "read x")],
        model="m",
    )
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "Read"
    assert resp.tool_calls[0].input == {"path": "x"}


def test_flatten_includes_tools_and_history():
    prompt = _flatten_conversation(
        messages=[
            Message("user", "hi"),
            Message("assistant", "hello"),
            Message("user", "go"),
        ],
        tools=[ToolSchema("t", "d", {"type": "object"})],
        tool_choice=ToolChoice(mode="tool", name="t"),
    )
    assert "<available_tools>" in prompt
    assert "<user>\nhi\n</user>" in prompt
    assert "<assistant>\nhello\n</assistant>" in prompt
    assert "MUST respond by calling the tool `t`" in prompt


def test_missing_package_raises_helpful_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(ImportError, match="claude-agent-sdk"):
        ClaudeAgentSDKClient()
