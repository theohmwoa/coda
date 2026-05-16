"""Tests for the replay CLI / library.

Replay is a read-only pretty-printer: round-tripping isn't the point,
shape and ordering are. We assert the timeline preserves trace order,
honours the filter, and respects the color flag. The subprocess test
guards the wiring between argparse and the library function.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from coda.replay import replay


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# --- Fixtures ---------------------------------------------------------------


def _write_trace(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _sample_events() -> list[dict]:
    return [
        {"ts": "2026-05-16T10:00:00.000001", "type": "run_started",
         "payload": {"task": "demo"}},
        {"ts": "2026-05-16T10:00:00.100000", "type": "llm_request",
         "payload": {"turn": 1, "messages": 1}},
        {"ts": "2026-05-16T10:00:00.200000", "type": "llm_response",
         "payload": {"turn": 1, "stop_reason": "end_turn", "text_len": 42}},
        {"ts": "2026-05-16T10:00:00.300000", "type": "code_emitted",
         "payload": {"code": "x = 1\nprint(x)"}},
        {"ts": "2026-05-16T10:00:00.400000", "type": "line_executed",
         "payload": {"lineno": 1}},
        {"ts": "2026-05-16T10:00:00.500000", "type": "line_executed",
         "payload": {"lineno": 2}},
        {"ts": "2026-05-16T10:00:00.600000", "type": "run_finished",
         "payload": {"reason": "done"}},
    ]


# --- Library-level behaviour ------------------------------------------------


def test_ordering_preserves_trace_order(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _write_trace(p, _sample_events())

    buf = io.StringIO()
    n = replay(p, color=False, stream=buf)

    assert n == 7
    lines = buf.getvalue().splitlines()
    assert len(lines) == 7

    # The "type" column appears in column 2 of each line; check that the
    # sequence of types in the output exactly matches the input order.
    seen_types = [line.split()[1] for line in lines]
    expected = [ev["type"] for ev in _sample_events()]
    assert seen_types == expected


def test_filter_types_keeps_only_matching(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _write_trace(p, _sample_events())

    buf = io.StringIO()
    n = replay(p, filter_types={"line_executed"}, color=False, stream=buf)

    assert n == 2
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert "line_executed" in line
    # Nothing leaked through.
    assert "code_emitted" not in buf.getvalue()
    assert "run_started" not in buf.getvalue()


def test_color_false_emits_no_ansi_escapes(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _write_trace(p, _sample_events())

    buf = io.StringIO()
    replay(p, color=False, stream=buf)
    out = buf.getvalue()
    assert ANSI_RE.search(out) is None
    assert "\x1b" not in out


def test_color_true_emits_ansi_escapes(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _write_trace(p, _sample_events())

    buf = io.StringIO()
    replay(p, color=True, stream=buf)
    assert ANSI_RE.search(buf.getvalue()) is not None


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        replay(missing)


def test_empty_trace_returns_zero(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    buf = io.StringIO()
    assert replay(p, color=False, stream=buf) == 0
    assert buf.getvalue() == ""


# --- CLI wiring -------------------------------------------------------------


def _repo_root() -> Path:
    # tests/ lives directly under the repo root.
    return Path(__file__).resolve().parent.parent


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    root = _repo_root()
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "coda", *args],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )


def _fixture_line_count() -> int:
    fixture = _repo_root() / "examples" / "sample_run" / "trace.jsonl"
    return sum(
        1
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def test_cli_replay_on_fixture_prints_timeline_and_summary() -> None:
    fixture = _repo_root() / "examples" / "sample_run" / "trace.jsonl"
    expected_events = _fixture_line_count()
    result = _run_cli(["replay", str(fixture), "--no-color"])

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # One line per event, plus one summary line at the end.
    assert len(lines) == expected_events + 1
    summary = lines[-1]
    assert summary.startswith(f"{expected_events} events (")
    # And there must be no ANSI escapes when --no-color is set.
    assert "\x1b" not in result.stdout


def test_cli_replay_filter_narrows_output() -> None:
    fixture = _repo_root() / "examples" / "sample_run" / "trace.jsonl"
    # Pick a type that is actually present in the fixture so the test
    # has something to assert on, but determine the expected count from
    # the file itself to stay robust against fixture rewrites.
    import json
    from collections import Counter

    counts: Counter[str] = Counter()
    for line in fixture.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        counts[json.loads(line).get("type", "")] += 1
    chosen_type, expected = counts.most_common(1)[0]

    result = _run_cli(
        ["replay", str(fixture), "--no-color", "--filter", chosen_type]
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # `expected` event lines + 1 summary line.
    assert len(lines) == expected + 1
    for line in lines[:-1]:
        assert chosen_type in line
    assert lines[-1].startswith(f"{expected} events ({chosen_type}={expected})")


def test_cli_replay_missing_file_returns_error_code(tmp_path: Path) -> None:
    result = _run_cli(["replay", str(tmp_path / "does_not_exist.jsonl")])
    assert result.returncode == 2
    assert "trace file not found" in result.stderr
