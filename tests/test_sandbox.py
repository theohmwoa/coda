from __future__ import annotations

import pytest

from coda.hooks import HookRegistry
from coda.sandbox import Sandbox


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\ngamma\n")
    (tmp_path / "b.py").write_text("def foo():\n    return 1\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("# hello\n# pattern_to_find\n")
    return tmp_path


@pytest.fixture
def sandbox(workspace):
    hooks = HookRegistry()
    return Sandbox(root=workspace, hooks=hooks)


def test_ls_returns_sorted_with_trailing_slash_on_dirs(sandbox):
    entries = sandbox.ls(".")
    assert "a.txt" in entries
    assert "b.py" in entries
    assert "sub/" in entries
    assert entries == sorted(entries)


def test_ls_path_traversal_blocked(sandbox, tmp_path):
    with pytest.raises(PermissionError):
        sandbox.ls("../..")


def test_glob_recursive(sandbox):
    py_files = sandbox.glob("**/*.py")
    assert "b.py" in py_files
    assert "sub/c.py" in py_files


def test_grep_finds_pattern(sandbox):
    hits = sandbox.grep("pattern_to_find", path=".")
    assert len(hits) == 1
    lineno, path, line = hits[0]
    assert lineno == 2
    assert path == "sub/c.py"
    assert "pattern_to_find" in line


def test_grep_literal_mode(sandbox, workspace):
    (workspace / "regex_test.txt").write_text("hello.world\nhello world\n")
    rx_hits = sandbox.grep(r"hello\.world", path="regex_test.txt", regex=True)
    lit_hits = sandbox.grep("hello.world", path="regex_test.txt", regex=False)
    assert len(rx_hits) == 1
    assert len(lit_hits) == 1


def test_read_with_line_numbers(sandbox):
    out = sandbox.read("a.txt")
    assert "1\talpha" in out
    assert "2\tbeta" in out
    assert "3\tgamma" in out


def test_read_raw(sandbox):
    out = sandbox.read("a.txt", with_line_numbers=False)
    assert out == "alpha\nbeta\ngamma\n"


def test_read_size_cap(sandbox, workspace):
    big = workspace / "big.txt"
    big.write_text("x" * 1024)
    sandbox.max_read_bytes = 100
    with pytest.raises(ValueError, match="exceeds max_read_bytes"):
        sandbox.read("big.txt")


def test_write_creates_parent_dirs(sandbox, workspace):
    sandbox.write("deep/nested/file.txt", "content")
    assert (workspace / "deep" / "nested" / "file.txt").read_text() == "content"


def test_edit_replaces_once(sandbox, workspace):
    n = sandbox.edit("a.txt", "beta", "BETA")
    assert n == 1
    assert (workspace / "a.txt").read_text() == "alpha\nBETA\ngamma\n"


def test_edit_ambiguous_fails(sandbox, workspace):
    (workspace / "dup.txt").write_text("x\nx\nx\n")
    with pytest.raises(ValueError, match="appears 3 times"):
        sandbox.edit("dup.txt", "x", "y")


def test_edit_replace_all(sandbox, workspace):
    (workspace / "dup.txt").write_text("x\nx\nx\n")
    n = sandbox.edit("dup.txt", "x", "y", replace_all=True)
    assert n == 3
    assert (workspace / "dup.txt").read_text() == "y\ny\ny\n"


def test_edit_missing_old_fails(sandbox):
    with pytest.raises(ValueError, match="not found"):
        sandbox.edit("a.txt", "ZZZZ", "y")


def test_bash_returns_stdout_and_returncode(sandbox):
    r = sandbox.bash("echo hi && false")
    assert r["stdout"].strip() == "hi"
    assert r["returncode"] == 1
    assert r["timed_out"] is False


def test_tool_events_emitted(sandbox):
    captured = sandbox.hooks.capture()
    sandbox.ls(".")
    sandbox.read("a.txt")
    types = [e.type for e in captured]
    assert "tool_called" in types
    tools_seen = [e.payload["tool"] for e in captured if e.type == "tool_called"]
    assert tools_seen == ["ls", "read"]


def test_execute_persists_globals_across_calls(sandbox):
    r1 = sandbox.execute("x = 41\ny = x + 1")
    assert r1.error is None
    r2 = sandbox.execute("print(y)")
    assert r2.error is None
    assert r2.stdout.strip() == "42"


def test_execute_captures_stdout_and_stderr(sandbox):
    r = sandbox.execute("import sys\nprint('hi')\nprint('err', file=sys.stderr)")
    assert r.stdout.strip() == "hi"
    assert r.stderr.strip() == "err"


def test_execute_records_error_without_raising(sandbox):
    r = sandbox.execute("1 / 0")
    assert r.error is not None
    assert "ZeroDivisionError" in r.error
    assert r.error_traceback is not None


def test_execute_syntax_error_returned(sandbox):
    r = sandbox.execute("def broken(:")
    assert r.error is not None
    assert "SyntaxError" in r.error


def test_execute_line_tracing(sandbox):
    captured = sandbox.hooks.capture()
    sandbox.execute("a = 1\nb = 2\nc = a + b\n")
    line_events = [e for e in captured if e.type == "line_executed"]
    assert len(line_events) == 3


def test_execute_primitives_callable_from_code(sandbox):
    r = sandbox.execute("entries = ls('.')\nprint(len(entries))")
    assert r.error is None
    # 3 entries: a.txt, b.py, sub/
    assert r.stdout.strip() == "3"


def test_execute_does_not_leak_trace_after_exec(sandbox):
    import sys as _sys

    before = _sys.gettrace()
    sandbox.execute("x = 1")
    after = _sys.gettrace()
    assert after is before


def test_inject_makes_value_visible(sandbox):
    sandbox.inject("MY_VAR", 42)
    r = sandbox.execute("print(MY_VAR * 2)")
    assert r.stdout.strip() == "84"
