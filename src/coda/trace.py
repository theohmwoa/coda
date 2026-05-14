"""TraceWriter — JSONL writer for coda run events. Reader / replay helpers.

A coda run with TraceWriter attached produces one JSON object per line:

    {"ts": "2026-05-14T12:34:56.789", "type": "code_emitted", "payload": {...}}
    {"ts": "...",                    "type": "line_executed", "payload": {"lineno": 3}}
    {"ts": "...",                    "type": "tool_called",   "payload": {"tool": "read", ...}}
    ...

This is the wire format the future codetrace debugger UI will consume.
JSONL was chosen over Protobuf / sqlite / pickle because it streams cleanly,
greps cleanly, and is trivial to ingest from any language. The format can
always be migrated; the only stable promise is "one Event per line".

Usage:

    from coda import Agent, HookRegistry, TraceWriter

    hooks = HookRegistry()
    with TraceWriter("run.jsonl").attach(hooks):
        agent = Agent(model="...", llm=..., hooks=hooks, sandbox=Sandbox(hooks=hooks))
        agent.run("...")

    # later, in another process / language:
    events = read_trace("run.jsonl")

Context-manager exit closes the file (when opened by path) and detaches
the handler from the registry.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from .hooks import Event, EventType, HookRegistry


def _default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    try:
        return str(obj)
    except Exception:
        return repr(obj)


class TraceWriter:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        stream: TextIO | None = None,
        types: Iterable[EventType] | None = None,
    ) -> None:
        """Open a writer.

        Args:
            path: file path to append to; the file is opened in write mode.
            stream: an already-open writable text stream (mutually exclusive
                with `path`). Useful for streaming to stdout or a buffer in
                tests.
            types: if set, only events whose type is in this set are written.
                Default: write everything.
        """
        if (path is None) == (stream is None):
            raise ValueError("TraceWriter needs exactly one of `path=` or `stream=`.")
        if path is not None:
            self._stream = open(path, "w", encoding="utf-8")
            self._owns_stream = True
        else:
            assert stream is not None
            self._stream = stream
            self._owns_stream = False
        self._types = set(types) if types is not None else None
        self._registries: list[HookRegistry] = []
        self._closed = False
        self.events_written = 0

    def attach(self, hooks: HookRegistry) -> "TraceWriter":
        """Subscribe to all events on `hooks`. Returns self for chaining."""
        hooks.on("*", self._on_event)
        self._registries.append(hooks)
        return self

    def _on_event(self, event: Event) -> None:
        if self._closed:
            return
        if self._types is not None and event.type not in self._types:
            return
        line = json.dumps(
            {
                "ts": event.timestamp.isoformat(),
                "type": event.type,
                "payload": event.payload,
            },
            default=_default,
            ensure_ascii=False,
        )
        self._stream.write(line + "\n")
        self._stream.flush()
        self.events_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_stream:
            self._stream.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load every event from a JSONL trace file into a list."""
    return list(iter_trace(path))


def iter_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield events from a JSONL trace file one at a time.

    Use this for large traces where loading the whole list would be wasteful.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield json.loads(line)
