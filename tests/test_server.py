"""Smoke tests for the live UI server.

These use FastAPI's TestClient and a MockLLMClient — no Opus, no MCPs.
We're not testing the agent here, we're testing the WebSocket plumbing:
events emitted from a HookRegistry while a (mock) Agent runs make it
out through the WebSocket as JSON frames in the right order, with the
right shape, and the run terminates with a `done` frame.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from coda.llm import CompletionResponse, MockLLMClient
from coda.server import ServerConfig, build_agent, create_app, event_to_json
from coda.hooks import Event


# --- pure functions ----------------------------------------------------


def test_event_to_json_round_trip():
    e = Event(type="tool_called", payload={"tool": "ls", "args": {"path": "."}})
    out = event_to_json(e)
    assert out["type"] == "tool_called"
    assert out["payload"]["tool"] == "ls"
    assert "ts" in out


def test_event_to_json_pydantic_fields_serialize():
    from pydantic import BaseModel

    class V(BaseModel):
        x: int

    e = Event(type="subagent_returned", payload={"result": V(x=42)})
    out = event_to_json(e)
    assert out["payload"]["result"]["x"] == 42


# --- /health -----------------------------------------------------------


def test_health_endpoint(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "foo.py").write_text("def foo():\n    return 1\n")
    app = create_app(ServerConfig(skills_dir=skills))
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "foo" in body["skills"]


# --- /ws/run end-to-end with mock LLM ----------------------------------


def test_ws_run_streams_events_and_terminates(tmp_path, monkeypatch):
    """The WS sends `ready`, then a stream of `event` frames mirroring
    the HookRegistry, then `final` + `done`. Order matters."""
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nx = 1\nprint(x)\n```"),
            CompletionResponse(text="done — x is 1."),
        ]
    )

    # Patch ClaudeAgentSDKClient out of build_agent's path: we patch the
    # function that constructs the agent so we can substitute a mock LLM.
    from coda import server as srv_mod

    real_build = srv_mod.build_agent

    def fake_build(config, hooks):
        from coda.agent import Agent
        from coda.sandbox import Sandbox

        workspace = tmp_path
        sandbox = Sandbox(root=workspace, hooks=hooks)
        agent = Agent(
            model="mock",
            llm=mock,
            sandbox=sandbox,
            hooks=hooks,
            max_turns=5,
        )
        return agent, workspace

    monkeypatch.setattr(srv_mod, "build_agent", fake_build)

    app = create_app(ServerConfig())
    c = TestClient(app)

    frames: list[dict] = []
    with c.websocket_connect("/ws/run") as ws:
        ws.send_json({"prompt": "trivial test"})
        while True:
            msg = ws.receive_json()
            frames.append(msg)
            if msg.get("kind") == "done":
                break

    kinds = [f["kind"] for f in frames]
    assert kinds[0] == "ready"
    assert kinds[-1] == "done"
    assert "final" in kinds
    # Should have streamed at least: llm_request, llm_response,
    # code_emitted, code_executed, turn_complete (x2)
    event_types = [f.get("type") for f in frames if f.get("kind") == "event"]
    assert "llm_request" in event_types
    assert "code_emitted" in event_types
    assert "turn_complete" in event_types

    # The `done` frame carries usage stats
    done = frames[-1]
    assert "elapsed_s" in done
    assert done["turns"] >= 1

    # The `final` frame carries the agent's last assistant text
    finals = [f for f in frames if f.get("kind") == "final"]
    assert "x is 1" in finals[0]["text"]


def test_ws_rejects_missing_prompt(tmp_path):
    app = create_app(ServerConfig())
    c = TestClient(app)
    with c.websocket_connect("/ws/run") as ws:
        ws.send_json({"not_a_prompt": "oops"})
        err = ws.receive_json()
        assert err["kind"] == "error"
        assert "prompt" in err["error"]
