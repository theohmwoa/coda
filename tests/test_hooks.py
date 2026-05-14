from __future__ import annotations

from coda.hooks import Event, HookRegistry


def test_emit_dispatches_to_subscriber():
    seen: list[Event] = []
    reg = HookRegistry()
    reg.on("tool_called", lambda e: seen.append(e))
    reg.emit(Event(type="tool_called", payload={"tool": "ls"}))
    reg.emit(Event(type="line_executed", payload={}))
    assert len(seen) == 1
    assert seen[0].payload["tool"] == "ls"


def test_wildcard_handler_sees_everything():
    seen: list[str] = []
    reg = HookRegistry()
    reg.on("*", lambda e: seen.append(e.type))
    reg.emit(Event(type="tool_called", payload={}))
    reg.emit(Event(type="code_emitted", payload={}))
    assert seen == ["tool_called", "code_emitted"]


def test_capture_records_in_order():
    reg = HookRegistry()
    captured = reg.capture()
    reg.emit(Event(type="code_emitted", payload={"n": 1}))
    reg.emit(Event(type="code_executed", payload={"n": 2}))
    reg.stop_capture()
    reg.emit(Event(type="line_executed", payload={"n": 3}))
    assert [e.type for e in captured] == ["code_emitted", "code_executed"]


def test_multiple_handlers_run_in_order():
    seen: list[int] = []
    reg = HookRegistry()
    reg.on("tool_called", lambda e: seen.append(1))
    reg.on("tool_called", lambda e: seen.append(2))
    reg.emit(Event(type="tool_called", payload={}))
    assert seen == [1, 2]
