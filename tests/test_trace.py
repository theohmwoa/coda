"""Tests for the JSONL TraceWriter and reader."""

from __future__ import annotations

import io
import json

import pytest
from pydantic import BaseModel

from coda.hooks import Event, HookRegistry
from coda.trace import TraceWriter, iter_trace, read_trace


def test_writes_jsonl_for_every_event(tmp_path):
    path = tmp_path / "run.jsonl"
    hooks = HookRegistry()
    with TraceWriter(path).attach(hooks):
        hooks.emit(Event(type="code_emitted", payload={"code": "x = 1"}))
        hooks.emit(Event(type="line_executed", payload={"lineno": 1}))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "code_emitted"
    assert first["payload"] == {"code": "x = 1"}
    assert "ts" in first


def test_filters_by_type(tmp_path):
    path = tmp_path / "run.jsonl"
    hooks = HookRegistry()
    with TraceWriter(path, types={"tool_called"}).attach(hooks):
        hooks.emit(Event(type="line_executed", payload={}))
        hooks.emit(Event(type="tool_called", payload={"tool": "ls"}))
        hooks.emit(Event(type="line_executed", payload={}))
    rows = read_trace(path)
    assert len(rows) == 1
    assert rows[0]["type"] == "tool_called"


def test_stream_mode_does_not_close_external_buffer():
    buf = io.StringIO()
    hooks = HookRegistry()
    writer = TraceWriter(stream=buf).attach(hooks)
    hooks.emit(Event(type="code_emitted", payload={"n": 1}))
    writer.close()
    # buf is still readable — only owned streams are closed
    assert "code_emitted" in buf.getvalue()
    assert not buf.closed


def test_rejects_both_path_and_stream():
    with pytest.raises(ValueError, match="exactly one"):
        TraceWriter(path="x.jsonl", stream=io.StringIO())


def test_rejects_neither():
    with pytest.raises(ValueError, match="exactly one"):
        TraceWriter()


def test_serializes_pydantic_models(tmp_path):
    class Foo(BaseModel):
        x: int
        y: str

    path = tmp_path / "run.jsonl"
    hooks = HookRegistry()
    with TraceWriter(path).attach(hooks):
        hooks.emit(Event(type="subagent_returned", payload={"result": Foo(x=1, y="hi")}))
    row = read_trace(path)[0]
    assert row["payload"]["result"] == {"x": 1, "y": "hi"}


def test_iter_trace_is_streaming(tmp_path):
    path = tmp_path / "run.jsonl"
    hooks = HookRegistry()
    with TraceWriter(path).attach(hooks):
        for i in range(5):
            hooks.emit(Event(type="line_executed", payload={"lineno": i}))
    seen = [r["payload"]["lineno"] for r in iter_trace(path)]
    assert seen == [0, 1, 2, 3, 4]


def test_events_written_counter(tmp_path):
    path = tmp_path / "run.jsonl"
    hooks = HookRegistry()
    writer = TraceWriter(path).attach(hooks)
    hooks.emit(Event(type="code_emitted", payload={}))
    hooks.emit(Event(type="code_executed", payload={}))
    writer.close()
    assert writer.events_written == 2


def test_attaches_to_multiple_registries(tmp_path):
    path = tmp_path / "run.jsonl"
    hooks_a = HookRegistry()
    hooks_b = HookRegistry()
    with TraceWriter(path).attach(hooks_a).attach(hooks_b):
        hooks_a.emit(Event(type="code_emitted", payload={"src": "a"}))
        hooks_b.emit(Event(type="code_emitted", payload={"src": "b"}))
    rows = read_trace(path)
    assert [r["payload"]["src"] for r in rows] == ["a", "b"]


def test_end_to_end_agent_trace(tmp_path):
    from coda.agent import Agent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    llm = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nls('.')\n```"),
            CompletionResponse(text="done."),
        ]
    )
    trace_path = tmp_path / "trace.jsonl"
    with TraceWriter(trace_path).attach(hooks):
        Agent(model="m", llm=llm, sandbox=sandbox, hooks=hooks).run("explore")
    rows = read_trace(trace_path)
    types = {r["type"] for r in rows}
    for must in ("llm_request", "llm_response", "code_emitted",
                 "tool_called", "code_executed", "turn_complete"):
        assert must in types
