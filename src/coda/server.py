"""coda serve — WebSocket-backed live UI for a real coda Agent.

What this is: a FastAPI app that, on `python -m coda serve`, opens a
browser-clickable URL. Inside, you type a prompt. The agent runs a
real coda loop — Gmail MCP, the typed `triage_email` sub-agent, the
Discord webhook tool, the skills library — and every event from the
HookRegistry streams live to the page over a WebSocket: code being
emitted, lines as they execute, tool calls with their args + result
summaries, sub-agent calls with their typed Pydantic returns, lint
warnings, the final response.

It's the UI for coda's actual differentiator: the trace stream is
detailed enough that watching it feels like watching the agent think.

# Architecture

- FastAPI app with one main websocket endpoint `/ws/run`.
- Static SPA served from `/` (the index.html in this package).
- Agent runs in a background thread. The HookRegistry's `*` handler
  fires from that thread; we schedule each event onto the
  websocket's event loop with `asyncio.run_coroutine_threadsafe`,
  pump it into an `asyncio.Queue`, and the WS coroutine drains and
  sends. Same producer/consumer pattern as the on-call demo's
  TraceWriter, just the sink is a WebSocket instead of a file.

# Lifecycle

For v1, one agent per WS connection. New connection = fresh Agent
+ MCPRuntime + sandbox. Closing the WS shuts down the runtime
cleanly via the existing `Agent.close()`. Persistent multi-turn
session is a v2 problem.
"""

# NOTE: deliberately NOT `from __future__ import annotations` here.
# FastAPI uses runtime type-introspection on route handlers to decide
# how to inject parameters (path / query / body / WebSocket). With
# PEP 563 string annotations, `websocket: WebSocket` becomes the string
# "WebSocket" and FastAPI falls through to query-param resolution,
# rejecting every WS connection with a Pydantic "Field required" error.
# Concrete annotations only in this module.

import asyncio
import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .agent import Agent
from .hooks import Event, HookRegistry
from .llm import ClaudeAgentSDKClient
from .sandbox import Sandbox


log = logging.getLogger("coda.server")


# --- payload normalization ----------------------------------------------


def _jsonable(o: Any) -> Any:
    """Best-effort: convert anything we might find in an Event.payload to JSON.

    The trace JSONL writer already does this for files; we replicate
    here for the live wire so the WS frame is always valid JSON. Pydantic
    models become dicts, sets become lists, unknowns become repr().
    """
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            pass
    if isinstance(o, (set, frozenset)):
        return [_jsonable(v) for v in o]
    try:
        return str(o)
    except Exception:
        return repr(o)


def event_to_json(ev: Event) -> dict[str, Any]:
    return {
        "ts": ev.timestamp.isoformat(timespec="milliseconds"),
        "type": ev.type,
        "payload": _jsonable(ev.payload),
    }


# --- agent factory ------------------------------------------------------


@dataclass
class ServerConfig:
    """What the server uses to construct an Agent per WS session.

    Hardcoded for v1 to the Gmail + Discord + skills setup we built up
    over the morning_brief demos. A `~/.coda/config.toml` will be the
    natural v2 next step.
    """

    model: str = "claude-opus-4-7"
    workspace_root: Optional[Path] = None  # if None, mktemp per session
    skills_dir: Optional[Path] = None
    sub_agents: list[Any] = field(default_factory=list)
    inline_tools: list[Any] = field(default_factory=list)
    mcp_servers: list[Any] = field(default_factory=list)


def build_agent(config: ServerConfig, hooks: HookRegistry) -> tuple[Agent, Path]:
    import tempfile

    workspace = (
        config.workspace_root
        if config.workspace_root is not None
        else Path(tempfile.mkdtemp(prefix="coda-serve-")).resolve()
    )
    sandbox = Sandbox(root=workspace, hooks=hooks, bash_timeout_s=120)
    agent = Agent(
        model=config.model,
        llm=ClaudeAgentSDKClient(cwd=str(workspace)),
        sandbox=sandbox,
        hooks=hooks,
        subagents=config.sub_agents,
        tools=config.inline_tools,
        mcp_servers=config.mcp_servers,
        skills_dir=config.skills_dir,
        max_turns=25,
        max_tokens=8192,
    )
    return agent, workspace


# --- WebSocket bridge ----------------------------------------------------


async def _stream_run(
    websocket,
    config: ServerConfig,
    user_prompt: str,
) -> None:
    """Run one Agent against `user_prompt`, streaming every event back.

    Send shapes:
      - {"kind": "event", ...event_to_json}       # every HookRegistry event
      - {"kind": "ready"}                          # agent constructed, run starting
      - {"kind": "final", "text": "..."}           # the agent's final assistant text
      - {"kind": "error", "error": "..."}          # construction or run blew up
      - {"kind": "done", "elapsed_s": float, ...}  # tail; safe for client to close
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    hooks = HookRegistry()

    def on_event(ev: Event) -> None:
        # Hop from the agent thread to the WS thread.
        try:
            payload = {"kind": "event", **event_to_json(ev)}
            asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
        except Exception:
            # Never let a bad payload kill the agent — log and drop.
            log.exception("event serialization failed")

    hooks.on("*", on_event)

    agent: Agent | None = None
    workspace: Path | None = None
    try:
        agent, workspace = build_agent(config, hooks)
    except Exception as e:
        await websocket.send_json({"kind": "error", "error": f"agent construction failed: {e}"})
        return

    await websocket.send_json(
        {
            "kind": "ready",
            "workspace": str(workspace),
            "model": config.model,
            "skills": list(agent.skills.keys()),
            "mcp_servers": list(agent.mcp_proxies.keys()),
            "sub_agents": [sa.name for sa in agent.subagents],
        }
    )

    result_holder: dict[str, Any] = {}
    t0 = time.monotonic()

    def run_agent() -> None:
        try:
            r = agent.run(user_prompt)
            result_holder["final_text"] = r.text
            result_holder["turns"] = r.turns
            result_holder["tokens_in"] = r.input_tokens
            result_holder["tokens_out"] = r.output_tokens
            result_holder["executions"] = len(r.executions)
        except Exception as e:
            result_holder["error"] = repr(e)
            result_holder["traceback"] = traceback.format_exc()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=run_agent, daemon=True, name="coda-agent-run")
    thread.start()

    # Drain events as they arrive, sentinel = run finished.
    while True:
        item = await queue.get()
        if item is None:
            break
        try:
            await websocket.send_json(item)
        except Exception:
            # Client closed mid-stream — let the agent finish anyway
            # (the thread will exit when the run completes) and bail.
            log.info("websocket closed during stream; abandoning sends")
            return

    elapsed = time.monotonic() - t0
    if "error" in result_holder:
        await websocket.send_json(
            {
                "kind": "error",
                "error": result_holder["error"],
                "traceback": result_holder.get("traceback"),
            }
        )
    else:
        await websocket.send_json(
            {"kind": "final", "text": result_holder.get("final_text", "")}
        )
    await websocket.send_json(
        {
            "kind": "done",
            "elapsed_s": elapsed,
            "turns": result_holder.get("turns", 0),
            "tokens_in": result_holder.get("tokens_in", 0),
            "tokens_out": result_holder.get("tokens_out", 0),
        }
    )

    try:
        agent.close()
    except Exception:
        log.exception("agent.close() failed")


# --- app factory --------------------------------------------------------


def create_app(config: ServerConfig | None = None):
    """Build the FastAPI app. Importable for tests; called by `python -m coda serve`."""
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="coda")
    cfg = config or ServerConfig()
    app.state.config = cfg

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index():
        index_html = static_dir / "index.html"
        if index_html.is_file():
            return FileResponse(str(index_html))
        return JSONResponse(
            {"error": "index.html not found", "expected_at": str(index_html)},
            status_code=500,
        )

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": cfg.model,
            "skills": (
                [p.stem for p in cfg.skills_dir.glob("*.py") if not p.name.startswith("_")]
                if cfg.skills_dir is not None and cfg.skills_dir.is_dir()
                else []
            ),
        }

    @app.websocket("/ws/run")
    async def ws_run(websocket: WebSocket):
        await websocket.accept()
        try:
            first = await websocket.receive_json()
        except Exception as e:
            await websocket.close(code=1003, reason=f"bad initial message: {e}")
            return

        prompt = first.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            await websocket.send_json({"kind": "error", "error": "missing 'prompt' field"})
            await websocket.close()
            return

        try:
            await _stream_run(websocket, cfg, prompt)
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    return app


# --- entry point --------------------------------------------------------


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    config: ServerConfig | None = None,
) -> None:
    """Boot the coda live UI on `host:port`. Blocks until killed."""
    import uvicorn

    app = create_app(config)
    log.info("coda live UI starting on http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
