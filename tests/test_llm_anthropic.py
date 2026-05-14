"""Tests for AnthropicClient using an injected fake SDK client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coda.llm import Message, ToolChoice, ToolSchema
from coda.llm.anthropic_client import AnthropicClient


@dataclass
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict | None = None


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Resp:
    content: list[_Block]
    stop_reason: str
    usage: _Usage


class _FakeMessages:
    def __init__(self):
        self.last_kwargs: dict[str, Any] | None = None
        self.next_response: _Resp | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.next_response


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _make() -> tuple[AnthropicClient, _FakeClient]:
    fake = _FakeClient()
    client = AnthropicClient(client=fake)
    return client, fake


def test_plain_text_round_trip():
    client, fake = _make()
    fake.messages.next_response = _Resp(
        content=[_Block(type="text", text="hello")],
        stop_reason="end_turn",
        usage=_Usage(input_tokens=10, output_tokens=5),
    )
    resp = client.complete(
        system="sys",
        messages=[Message("user", "hi")],
        model="claude-test",
    )
    assert resp.text == "hello"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5
    sent = fake.messages.last_kwargs
    assert sent["model"] == "claude-test"
    assert sent["system"] == "sys"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_tool_use_block_decoded():
    client, fake = _make()
    fake.messages.next_response = _Resp(
        content=[
            _Block(type="text", text="calling tool"),
            _Block(
                type="tool_use",
                id="t1",
                name="evaluate",
                input={"flight": {"id": 7}},
            ),
        ],
        stop_reason="tool_use",
        usage=_Usage(input_tokens=20, output_tokens=8),
    )
    resp = client.complete(
        system="",
        messages=[Message("user", "go")],
        model="m",
        tools=[ToolSchema("evaluate", "eval", {"type": "object"})],
        tool_choice=ToolChoice(mode="tool", name="evaluate"),
    )
    assert resp.text == "calling tool"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "evaluate"
    assert resp.tool_calls[0].input == {"flight": {"id": 7}}
    assert resp.stop_reason == "tool_use"
    sent = fake.messages.last_kwargs
    assert sent["tool_choice"] == {"type": "tool", "name": "evaluate"}
    assert sent["tools"] == [
        {"name": "evaluate", "description": "eval", "input_schema": {"type": "object"}}
    ]


def test_tool_choice_modes():
    client, fake = _make()
    fake.messages.next_response = _Resp(
        content=[_Block(type="text", text="")],
        stop_reason="end_turn",
        usage=_Usage(input_tokens=0, output_tokens=0),
    )
    client.complete(
        system="",
        messages=[Message("user", "x")],
        model="m",
        tools=[ToolSchema("t", "d", {})],
        tool_choice=ToolChoice(mode="any"),
    )
    assert fake.messages.last_kwargs["tool_choice"] == {"type": "any"}
    client.complete(
        system="",
        messages=[Message("user", "x")],
        model="m",
        tools=[ToolSchema("t", "d", {})],
        tool_choice=ToolChoice(mode="auto"),
    )
    assert fake.messages.last_kwargs["tool_choice"] == {"type": "auto"}
