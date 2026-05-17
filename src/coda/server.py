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
import inspect
import json
import logging
import threading
import time
import traceback
import uuid
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


def _titleize(snake: str) -> str:
    """Convert `morning_brief_to_discord` → `Morning brief to discord`.

    Intentionally not Title Case for every word — first word capitalized,
    rest lower. Matches how Notion / Linear titles read.
    """
    parts = snake.split("_")
    if not parts:
        return snake
    return parts[0].capitalize() + (" " + " ".join(parts[1:]) if len(parts) > 1 else "")


def _read_skill_meta(file_path: Path) -> dict[str, Any]:
    """Pull a `__coda__ = {...}` module-level dict out of a skill file.

    Convention: a skill's display metadata lives in a module-level dict
    literal. Same pattern as `__all__` / `__version__` — pure Python,
    picked up by IDEs, no separate config file.

        # skills/morning_brief.py
        __coda__ = {
            "display_name": "Morning brief",
            "description": "Triage recent Gmail and post a digest.",
            "emoji": "☀️",
        }

    We parse with `ast.literal_eval` against the dict-literal RHS so we
    never execute skill code just to read its metadata.
    """
    import ast

    try:
        src = file_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__coda__":
                    try:
                        v = ast.literal_eval(node.value)
                        if isinstance(v, dict):
                            return {str(k): val for k, val in v.items()}
                    except (ValueError, SyntaxError):
                        return {}
    return {}


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


APPROVE_DOC = """\
## Asking the user before destructive actions

A function `approve(action_type: str, payload: dict) -> dict` is in
your sandbox globals. Use it BEFORE any irreversible side-effect:
send an email, post to Slack, run a destructive shell command, file
a ticket, charge a card.

`approve()` BLOCKS until the user reviews your draft in the UI, then
returns the (possibly edited) payload — you don't need to unwrap any
envelope. If the user rejects, `approve()` raises `PermissionError`.

Canonical usage:

```python
draft = {
    "to": "alice@example.com",
    "subject": "Re: Q3 — your draft",
    "body": "Hey Alice,\\n\\nQuick note...",
}
try:
    final = approve("send_email", draft)   # blocks; returns dict
except PermissionError as e:
    print(f"User declined: {e}")
else:
    gmail.send_email(**final)              # always pass the returned dict
```

Three contracts to remember:
1. The return value is the dict to ACTUALLY use — it may be your
   original draft (user clicked Send as-is) or an edited version
   (user changed fields before sending). Always pass it forward
   with `**final` or by keys; don't re-use the input `draft`.
2. A rejection raises `PermissionError`. Catch it, stop the workflow,
   and tell the user you stopped.
3. Call `approve()` ONCE per destructive action. Don't loop on it —
   the user already saw and chose.

Supported `action_type` values today: `"send_email"` (renders an
email-compose modal). Unknown types fall back to a generic JSON
preview; future types will be `"shell_command"`, `"sql_query"`,
`"slack_message"`, etc.
"""


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
        # IMPORTANT: tells the agent the approve() primitive exists. Without
        # this section the agent has no idea it can ask for confirmation
        # and will just call gmail.send_email() directly.
        system_prompt_append=APPROVE_DOC,
        max_turns=25,
        max_tokens=8192,
    )
    return agent, workspace


# --- approval dispatcher -------------------------------------------------


class ApprovalDispatcher:
    """Cross-thread bridge: the agent's `approve()` call (sync, on the
    agent thread) sends a frame to the UI and blocks until the UI sends
    back an `Edit`. The WebSocket coroutine resolves pending approvals
    via `resolve()` from the event loop.

    Used by the live-UI server. NOT injected into a sandbox that runs
    headless — `approve()` is a UI-only primitive.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send: Optional[Callable[[dict], Any]] = None
        self._pending: dict[str, tuple[threading.Event, list]] = {}

    def bind(
        self,
        loop: asyncio.AbstractEventLoop,
        send_coro_factory: Callable[[dict], Any],
    ) -> None:
        """Bind to a live event loop + a function that, given a frame dict,
        returns the coroutine that sends it over the WS."""
        self._loop = loop
        self._send = send_coro_factory

    def request(self, action_type: str, payload: Any, timeout: float = 600.0) -> dict:
        """Called from agent thread. Send `approval_needed`, block, return Edit."""
        if self._loop is None or self._send is None:
            raise RuntimeError(
                "approve() requires a UI session. Run via `python -m coda serve`."
            )
        action_id = str(uuid.uuid4())
        action = {"type": action_type, "id": action_id, "payload": _jsonable(payload)}
        event = threading.Event()
        slot: list = [None]
        self._pending[action_id] = (event, slot)

        # Schedule the send on the WS loop. Don't wait for it to complete —
        # we just need the message to flush; the UI may take its time.
        coro = self._send({"kind": "approval_needed", "action": action})
        asyncio.run_coroutine_threadsafe(coro, self._loop)

        if not event.wait(timeout=timeout):
            self._pending.pop(action_id, None)
            raise TimeoutError(f"approve() timed out after {timeout}s waiting for UI")
        return slot[0]

    def resolve(self, action_id: str, edit: dict) -> bool:
        """Called from the WS coroutine when an `approval_response` arrives."""
        slot_pair = self._pending.pop(action_id, None)
        if slot_pair is None:
            return False
        event, slot = slot_pair
        slot[0] = edit
        event.set()
        return True


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

    # Persist every UI run as a JSONL trace under runs/. Cheap (~2 KB
    # per request typically) and makes "what did the agent actually do?"
    # answerable without re-running. Reuse the existing TraceWriter so
    # the file is identical to what python -m coda replay expects.
    import datetime as _dt
    from .trace import TraceWriter
    _runs_dir = Path(__file__).resolve().parent.parent.parent / "runs"
    _runs_dir.mkdir(exist_ok=True)
    _stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _trace_path = _runs_dir / f"server_{_stamp}.jsonl"
    _tracer = TraceWriter(_trace_path).attach(hooks)

    def on_event(ev: Event) -> None:
        # Hop from the agent thread to the WS thread.
        try:
            payload = {"kind": "event", **event_to_json(ev)}
            asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
        except Exception:
            # Never let a bad payload kill the agent — log and drop.
            log.exception("event serialization failed")

    hooks.on("*", on_event)

    # Approval dispatcher: lets the agent's emitted code call
    # `approve("send_email", {...})`, see the UI mount an EmailModal,
    # and receive the edited payload back as a Python dict.
    dispatcher = ApprovalDispatcher()
    dispatcher.bind(loop, send_coro_factory=websocket.send_json)

    agent: Agent | None = None
    workspace: Path | None = None
    try:
        agent, workspace = build_agent(config, hooks)
    except Exception as e:
        await websocket.send_json({"kind": "error", "error": f"agent construction failed: {e}"})
        return

    # Inject approve() into the sandbox AFTER build_agent. The function
    # returns the (possibly edited) payload directly — the agent never
    # has to unwrap an Edit envelope. Reject becomes an exception so
    # control flow stays linear.
    def _approve(action_type: str, payload: Any) -> Any:
        """Pause the agent until the user reviews the draft in the UI.

        Args:
            action_type: a stable identifier the UI maps to a component
                (e.g. "send_email" → EmailCompose modal).
            payload: the agent's draft. For "send_email", a dict with
                `to`, `subject`, `body`, optionally `cc` / `bcc`.

        Returns:
            The payload to actually use:
              - the original `payload` if the user clicked Accept,
              - the user's edited dict if they clicked Send-after-editing.

        Raises:
            PermissionError: if the user rejected the action. Carry the
                user's stated reason in the exception message; agent code
                can `try: ... except PermissionError as e: ...` to handle.
        """
        edit = dispatcher.request(action_type, payload)
        outcome = edit.get("outcome", {}) if isinstance(edit, dict) else {}
        kind = outcome.get("kind")
        if kind == "reject":
            reason = outcome.get("reason") or "no reason given"
            raise PermissionError(f"User rejected {action_type}: {reason}")
        if kind == "modify":
            return outcome.get("payload", payload)
        # "accept" (or anything unknown — treat as accept of original)
        return payload

    agent.sandbox.inject("approve", _approve)

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

    # Concurrently receive approval responses from the client while we
    # drain event frames out. The receive task lives until the run ends.
    async def receive_loop():
        while True:
            try:
                msg = await websocket.receive_json()
            except Exception:
                return  # client closed
            kind = msg.get("kind")
            if kind == "approval_response":
                action_id = msg.get("action_id")
                edit = msg.get("edit")
                if action_id and edit is not None:
                    dispatcher.resolve(action_id, edit)

    recv_task = asyncio.create_task(receive_loop())

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
            recv_task.cancel()
            return

    recv_task.cancel()

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
    try:
        _tracer.close()
    except Exception:
        log.exception("tracer.close() failed")


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
    # The built React SPA lives under static/app/. Mount its assets at
    # /assets/* so the absolute-base build resolves them at the URL the
    # browser asks for. Also expose the whole tree at /app/ for direct
    # access / future routing.
    app_dir = static_dir / "app"
    if app_dir.is_dir():
        assets_dir = app_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        app.mount("/app", StaticFiles(directory=str(app_dir), html=True), name="app")

    @app.get("/")
    def index():
        # Prefer the built React app if available; fall back to the
        # debugger view otherwise.
        app_html = static_dir / "app" / "index.html"
        if app_html.is_file():
            return FileResponse(str(app_html))
        index_html = static_dir / "index.html"
        if index_html.is_file():
            return FileResponse(str(index_html))
        return JSONResponse(
            {"error": "no UI built", "hint": "run `cd app && npm run build` first"},
            status_code=500,
        )

    @app.get("/debug")
    def debug_index():
        """The original event-stream debugger UI (vanilla JS, no React)."""
        index_html = static_dir / "index.html"
        if index_html.is_file():
            return FileResponse(str(index_html))
        return JSONResponse({"error": "debug index.html not found"}, status_code=404)

    @app.get("/skills")
    def list_skills():
        """Enumerate saved skills with display metadata for the UI's skill cards."""
        out: list[dict] = []
        if cfg.skills_dir is None or not cfg.skills_dir.is_dir():
            return out
        from .skills import load_skills

        skills = load_skills(cfg.skills_dir)
        for s in skills.values():
            doc = s.docstring or ""
            first_line = doc.splitlines()[0] if doc else ""
            meta = _read_skill_meta(s.file_path)
            out.append(
                {
                    "name": s.name,
                    "display_name": meta.get("display_name") or _titleize(s.name),
                    "description": meta.get("description") or first_line,
                    "emoji": meta.get("emoji"),
                    "signature": s.signature,
                }
            )
        return out

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
