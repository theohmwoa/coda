"""Tests for ClaudeCodeClient — subprocess `claude -p` backend.

We monkeypatch subprocess.run and shutil.which to keep tests offline and
independent of whether the user has Claude Code installed.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import pytest

from coda.llm import Message, ToolChoice, ToolSchema
from coda.llm.claude_code_client import ClaudeCodeClient, _flatten_conversation


@dataclass
class _FakeProc:
    stdout: str
    stderr: str = ""
    returncode: int = 0


@pytest.fixture(autouse=True)
def fake_claude(monkeypatch):
    """Stub `which('claude')` so the class instantiates without the binary."""
    monkeypatch.setattr(
        "coda.llm.claude_code_client.shutil.which",
        lambda name: f"/fake/bin/{name}",
    )


def test_plain_text_completion(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(
            stdout=json.dumps(
                {
                    "result": "hello from claude",
                    "usage": {"input_tokens": 42, "output_tokens": 9},
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ClaudeCodeClient()
    resp = client.complete(
        system="sys",
        messages=[Message("user", "ping")],
        model="claude-sonnet-4-6",
    )
    assert resp.text == "hello from claude"
    assert resp.input_tokens == 42
    assert resp.output_tokens == 9
    assert resp.stop_reason == "end_turn"
    assert "--model" in captured["cmd"]
    assert "claude-sonnet-4-6" in captured["cmd"]


def test_forced_tool_use_parsed(monkeypatch):
    payload = json.dumps(
        {"tool": "evaluate", "input": {"flight": {"id": 7}, "strict": True}}
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: _FakeProc(stdout=json.dumps({"result": payload})),
    )
    client = ClaudeCodeClient()
    resp = client.complete(
        system="",
        messages=[Message("user", "do it")],
        model="m",
        tools=[ToolSchema("evaluate", "eval one flight", {"type": "object"})],
        tool_choice=ToolChoice(mode="tool", name="evaluate"),
    )
    assert resp.stop_reason == "tool_use"
    assert resp.text == ""
    assert resp.tool_calls[0].name == "evaluate"
    assert resp.tool_calls[0].input == {"flight": {"id": 7}, "strict": True}


def test_forced_tool_use_with_markdown_fence(monkeypatch):
    fenced = "```json\n" + json.dumps({"tool": "t", "input": {"a": 1}}) + "\n```"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: _FakeProc(stdout=json.dumps({"result": fenced})),
    )
    client = ClaudeCodeClient()
    resp = client.complete(
        system="",
        messages=[Message("user", "go")],
        model="m",
        tools=[ToolSchema("t", "d", {})],
        tool_choice=ToolChoice(mode="tool", name="t"),
    )
    assert resp.tool_calls[0].input == {"a": 1}


def test_non_zero_exit_raises(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: _FakeProc(stdout="", stderr="boom", returncode=1),
    )
    client = ClaudeCodeClient()
    with pytest.raises(RuntimeError, match="claude -p exited 1"):
        client.complete(system="", messages=[Message("user", "x")], model="m")


def test_timeout_raises(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ClaudeCodeClient(timeout_s=1)
    with pytest.raises(TimeoutError):
        client.complete(system="", messages=[Message("user", "x")], model="m")


def test_flatten_includes_tool_schemas():
    prompt = _flatten_conversation(
        system="be helpful",
        messages=[Message("user", "rate this flight")],
        tools=[ToolSchema("evaluate", "rates a flight", {"type": "object"})],
        tool_choice=ToolChoice(mode="tool", name="evaluate"),
    )
    assert "<system>" in prompt
    assert "be helpful" in prompt
    assert "<available_tools>" in prompt
    assert "evaluate" in prompt
    assert "rates a flight" in prompt
    assert "MUST respond by calling the tool `evaluate`" in prompt
    assert prompt.endswith("<assistant>")


def test_missing_binary_raises_at_construction(monkeypatch):
    monkeypatch.setattr(
        "coda.llm.claude_code_client.shutil.which",
        lambda name: None,
    )
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        ClaudeCodeClient()
