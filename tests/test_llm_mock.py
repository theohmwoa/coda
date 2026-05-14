"""Tests for MockLLMClient — the canned-response backend."""

from __future__ import annotations

import pytest

from coda.llm import (
    CompletionResponse,
    Message,
    MockLLMClient,
    ToolChoice,
    ToolSchema,
    tool_call_response,
)


def test_returns_canned_text_in_order():
    client = MockLLMClient(responses=["first", "second"])
    a = client.complete(system="s", messages=[Message("user", "u1")], model="m")
    b = client.complete(system="s", messages=[Message("user", "u2")], model="m")
    assert a.text == "first"
    assert b.text == "second"
    assert len(client.calls) == 2


def test_records_call_snapshot():
    client = MockLLMClient(responses=["ok"])
    tools = [ToolSchema(name="t", description="d", input_schema={"type": "object"})]
    client.complete(
        system="sys",
        messages=[Message("user", "hi")],
        model="claude-test",
        max_tokens=128,
        temperature=0.7,
        tools=tools,
        tool_choice=ToolChoice(mode="tool", name="t"),
    )
    call = client.calls[0]
    assert call.system == "sys"
    assert call.model == "claude-test"
    assert call.max_tokens == 128
    assert call.temperature == 0.7
    assert call.tools == tools
    assert call.tool_choice and call.tool_choice.name == "t"


def test_exhaustion_raises():
    client = MockLLMClient(responses=["only"])
    client.complete(system="", messages=[], model="m")
    with pytest.raises(RuntimeError, match="exhausted"):
        client.complete(system="", messages=[], model="m")


def test_responder_mode_sees_state():
    seen: list[int] = []

    def responder(call):
        seen.append(len(call.messages))
        return CompletionResponse(text=f"reply-{len(call.messages)}")

    client = MockLLMClient(responder=responder)
    r1 = client.complete(system="", messages=[Message("user", "a")], model="m")
    r2 = client.complete(
        system="",
        messages=[Message("user", "a"), Message("assistant", "b"), Message("user", "c")],
        model="m",
    )
    assert r1.text == "reply-1"
    assert r2.text == "reply-3"
    assert seen == [1, 3]


def test_tool_call_response_helper():
    resp = tool_call_response("evaluate", {"flight": {"id": 42}})
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "evaluate"
    assert resp.tool_calls[0].input == {"flight": {"id": 42}}


def test_rejects_ambiguous_construction():
    with pytest.raises(ValueError):
        MockLLMClient()
    with pytest.raises(ValueError):
        MockLLMClient(responses=["x"], responder=lambda c: CompletionResponse(text=""))
