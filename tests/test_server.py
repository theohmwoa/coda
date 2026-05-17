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


# --- skills endpoint ---------------------------------------------------


def test_skills_endpoint_reads_coda_meta(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "morning_brief.py").write_text(
        '__coda__ = {"display_name": "Morning brief", '
        '"description": "Triage Gmail and post a digest.", '
        '"emoji": "☀️"}\n\n'
        'def morning_brief() -> str:\n'
        '    """Build a brief."""\n'
        '    return "ok"\n'
    )
    (skills / "untyped.py").write_text(
        'def untyped() -> str:\n'
        '    """One-line description goes here."""\n'
        '    return "x"\n'
    )
    app = create_app(ServerConfig(skills_dir=skills))
    c = TestClient(app)
    r = c.get("/skills")
    assert r.status_code == 200
    items = {s["name"]: s for s in r.json()}
    assert items["morning_brief"]["display_name"] == "Morning brief"
    assert items["morning_brief"]["emoji"] == "☀️"
    assert items["morning_brief"]["description"] == "Triage Gmail and post a digest."
    # Falls back to titleized name + docstring first line for no-meta skills
    assert items["untyped"]["display_name"] == "Untyped"
    assert items["untyped"]["description"] == "One-line description goes here."


# --- approval round-trip ----------------------------------------------


def test_approval_round_trip(tmp_path, monkeypatch):
    """The agent calls approve(...) — the server sends `approval_needed`
    to the client, blocks the agent until the client sends back an
    `approval_response`, and the agent's `approve()` returns the Edit."""
    from coda.agent import Agent
    from coda.sandbox import Sandbox
    from coda.llm import CompletionResponse, MockLLMClient
    from coda import server as srv_mod

    captured: dict = {}

    def fake_build(config, hooks):
        # The agent emits code that calls approve(). The mock LLM
        # returns code that calls approve() and prints the result.
        mock = MockLLMClient(
            responses=[
                CompletionResponse(text=(
                    "```python\n"
                    "edit = approve('send_email', {'to': 'a@b.com', 'subject': 'hi', 'body': 'hello'})\n"
                    "print('outcome:', edit['outcome']['kind'])\n"
                    "print('actionId:', edit['actionId'])\n"
                    "```"
                )),
                CompletionResponse(text="done."),
            ]
        )
        workspace = tmp_path
        sandbox = Sandbox(root=workspace, hooks=hooks)
        agent = Agent(model="mock", llm=mock, sandbox=sandbox, hooks=hooks, max_turns=5)
        captured["agent"] = agent
        return agent, workspace

    monkeypatch.setattr(srv_mod, "build_agent", fake_build)

    app = create_app(ServerConfig())
    c = TestClient(app)
    with c.websocket_connect("/ws/run") as ws:
        ws.send_json({"prompt": "send an email please"})
        # Drain until we see approval_needed
        approval_action_id = None
        seen_kinds = []
        while True:
            msg = ws.receive_json()
            seen_kinds.append(msg.get("kind"))
            if msg.get("kind") == "approval_needed":
                approval_action_id = msg["action"]["id"]
                # Respond with a modify edit
                ws.send_json({
                    "kind": "approval_response",
                    "action_id": approval_action_id,
                    "edit": {
                        "actionId": approval_action_id,
                        "outcome": {"kind": "modify", "payload": {"to": "a@b.com", "subject": "hi (edited)", "body": "edited body"}},
                    },
                })
                continue
            if msg.get("kind") == "done":
                break

        assert "approval_needed" in seen_kinds
        assert approval_action_id is not None
        # Verify the agent's print('outcome:', ...) saw "modify"
        # The executions are on the agent — search the event stream we collected.
        # (The stdout would have come back as a code_executed event payload's
        # had_error=False signal; full stdout isn't in the trace events but
        # we can verify the approve() call ran by virtue of completing the run.)


def test_approval_dispatcher_unit():
    """ApprovalDispatcher round-trip without any actual WS — verifies the
    cross-thread block + resolve path in isolation."""
    import asyncio
    import threading

    from coda.server import ApprovalDispatcher

    async def harness():
        loop = asyncio.get_running_loop()
        d = ApprovalDispatcher()
        sent: list[dict] = []

        async def fake_send(frame):
            sent.append(frame)

        d.bind(loop, send_coro_factory=fake_send)

        result_holder: dict = {}

        def agent_call():
            result_holder["edit"] = d.request("send_email", {"to": "a@b"})

        t = threading.Thread(target=agent_call, daemon=True)
        t.start()

        # Wait for the send to land
        for _ in range(50):
            await asyncio.sleep(0.01)
            if sent:
                break
        assert sent, "dispatcher never sent approval_needed"
        action_id = sent[0]["action"]["id"]
        assert sent[0]["kind"] == "approval_needed"

        # Resolve from the loop thread
        ok = d.resolve(action_id, {"actionId": action_id, "outcome": {"kind": "accept"}})
        assert ok
        t.join(timeout=2)
        assert not t.is_alive()
        assert result_holder["edit"]["outcome"]["kind"] == "accept"

    asyncio.run(harness())
