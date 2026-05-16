"""Replay a JSONL trace file as a colored, chronological timeline.

The trace files we read here are produced by `coda.TraceWriter` (one
JSON object per line, in emit order). We deliberately parse them as
raw dicts rather than going through `Event.from_dict`, for two reasons:

  - Older traces in `examples/sample_run/` use a leaner shape
    (`ts` / `type` / `payload`) and include event types from prior
    iterations of the agent loop. A replay tool that bails on
    unrecognised types is useless for historical debugging.
  - Replay is read-only; we don't need typed Events, just a sensible
    one-liner per record.

Output format (one event per line) is intentionally terse and
grep-friendly:

    HH:MM:SS.fff  type_name           details ...

Colors are applied per event type when `color=True`. Set `color=False`
(or pass `--no-color` on the CLI) for plain output suitable for pipes
and tests.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO


# ---------------------------------------------------------------------------
# ANSI palette. Kept in one place so re-theming is a one-file change.

_RESET = "\033[0m"
_DIM = "\033[2m"

_COLORS: dict[str, str] = {
    # Lifecycle
    "run_started":        "\033[1;32m",  # bold green
    "run_finished":       "\033[1;32m",
    "turn_started":       "\033[36m",    # cyan
    "turn_finished":      "\033[36m",
    "turn_complete":      "\033[36m",
    # Model I/O
    "prompt_built":       "\033[35m",    # magenta
    "model_requested":    "\033[34m",    # blue
    "llm_request":        "\033[34m",
    "model_responded":    "\033[94m",    # bright blue
    "llm_response":       "\033[94m",
    # CodeAct
    "code_emitted":       "\033[33m",    # yellow
    "code_parsed":        "\033[33m",
    "code_executed":      "\033[33m",
    "execution_finished": "\033[33m",
    "line_executed":      "\033[2;37m",  # dim grey
    # Tools
    "tool_called":        "\033[35m",    # magenta
    "tool_returned":      "\033[95m",    # bright magenta
    # Errors
    "error_raised":       "\033[1;31m",  # bold red
}


# ---------------------------------------------------------------------------
# Parsing helpers.


def _iter_raw(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSONL record as a plain dict. Skips blank lines."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _format_timestamp(ev: dict[str, Any]) -> str:
    """Return HH:MM:SS.fff for the event, accepting either shape."""
    if "ts" in ev:
        ts = str(ev["ts"])
        if "T" in ts:
            return ts.split("T", 1)[1][:12]
        return ts[:12]
    if "timestamp" in ev:
        try:
            t = float(ev["timestamp"])
        except (TypeError, ValueError):
            return str(ev["timestamp"])[:12]
        return _dt.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:12]
    return " " * 12


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_dict(d: dict[str, Any], n: int = 80) -> str:
    parts: list[str] = []
    for k, v in d.items():
        parts.append(f"{k}={_truncate(str(v), 30)}")
    return _truncate(", ".join(parts), n)


def _format_summary(ev: dict[str, Any]) -> str:
    """One-line summary for an event payload. Type-aware where useful."""
    t = ev.get("type", "?")
    p = ev.get("payload", {}) or {}

    if t == "code_emitted":
        code = p.get("code", "")
        first = code.splitlines()[0] if code else ""
        n = len(code.splitlines())
        return f"[{n}L] {_truncate(first, 80)}"

    if t == "line_executed":
        return f"L{p.get('lineno', '?')}"

    if t in ("code_executed", "execution_finished"):
        had_err = p.get("had_error")
        flag = "ERR" if had_err else "ok"
        return (
            f"lines={p.get('lines', '?')} {flag} "
            f"stdout={p.get('stdout_len', '?')}"
        )

    if t == "tool_called":
        tool = p.get("tool", "?")
        args = p.get("args", {}) if isinstance(p.get("args"), dict) else {}
        result = p.get("result", {}) if isinstance(p.get("result"), dict) else {}
        rc = result.get("returncode")
        if tool == "bash":
            cmd = args.get("cmd", "")
            tail = f" rc={rc}" if rc is not None else ""
            return f"bash{tail} :: {_truncate(cmd, 80)}"
        return f"{tool}({_format_dict(args, 60)})"

    if t == "tool_returned":
        return f"{p.get('tool', '?')}"

    if t in ("llm_request", "model_requested"):
        return f"turn={p.get('turn', '?')} messages={p.get('messages', '?')}"

    if t in ("llm_response", "model_responded"):
        return (
            f"turn={p.get('turn', '?')} stop={p.get('stop_reason', '?')} "
            f"text={p.get('text_len', '?')}"
        )

    if t in ("turn_complete", "turn_finished"):
        return f"turn={p.get('turn', '?')} done={p.get('done', '?')}"

    if t == "turn_started":
        return f"turn={p.get('turn', '?')}"

    if t == "run_started":
        task = p.get("task", "")
        return _truncate(str(task), 80) if task else ""

    if t == "run_finished":
        return f"reason={p.get('reason', '?')}"

    if t == "error_raised":
        err = p.get("error", "?")
        msg = p.get("message", "")
        loc = f" L{p['lineno']}" if "lineno" in p else ""
        return f"{err}{loc}: {_truncate(str(msg), 80)}"

    # Unknown type — show payload as best we can.
    return _format_dict(p, 80)


def _render(ev: dict[str, Any], color: bool) -> str:
    """Render a single event as a line (without trailing newline)."""
    t = ev.get("type", "?")
    ts = _format_timestamp(ev)
    summary = _format_summary(ev)
    # 20-char type column keeps the eye-line straight without truncating
    # the longest type names we know about ("execution_finished").
    type_col = f"{t:<20}"
    if color:
        c = _COLORS.get(t, "")
        return f"{_DIM}{ts}{_RESET}  {c}{type_col}{_RESET} {summary}"
    return f"{ts}  {type_col} {summary}"


# ---------------------------------------------------------------------------
# Public API.


def replay(
    path: str | Path,
    filter_types: set[str] | None = None,
    color: bool = True,
    stream: TextIO | None = None,
) -> int:
    """Print a chronological timeline of a JSONL trace.

    Parameters
    ----------
    path:
        Path to a JSONL trace file (as produced by `coda.TraceWriter`).
    filter_types:
        If given, only events whose `type` is in this set are printed.
        `None` means print every event.
    color:
        Whether to emit ANSI color escapes. Pass `False` for plain text
        suitable for pipes or files.
    stream:
        Where to write output. Defaults to `sys.stdout`.

    Returns
    -------
    int
        The number of events that were actually printed (i.e. survived
        the filter).

    Raises
    ------
    FileNotFoundError
        If `path` does not point to an existing file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"trace file not found: {p}")

    out = stream if stream is not None else sys.stdout
    printed = 0
    for ev in _iter_raw(p):
        if filter_types is not None and ev.get("type", "") not in filter_types:
            continue
        out.write(_render(ev, color=color))
        out.write("\n")
        printed += 1
    return printed


def count_by_type(
    path: str | Path,
    filter_types: set[str] | None = None,
) -> dict[str, int]:
    """Return a {type: count} map for the trace, honouring `filter_types`.

    Used by the CLI to print its trailing summary without forcing the
    caller of `replay()` to pay for a second pass.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"trace file not found: {p}")
    counts: dict[str, int] = {}
    for ev in _iter_raw(p):
        t = ev.get("type", "")
        if filter_types is not None and t not in filter_types:
            continue
        counts[t] = counts.get(t, 0) + 1
    return counts
