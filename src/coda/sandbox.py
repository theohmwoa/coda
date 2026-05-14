"""Sandbox — in-process Python execution context with native filesystem primitives.

The sandbox is what makes coda "CodeAct": the model emits Python code, the
sandbox runs it against a persistent globals dict, and the result feeds back
into the next turn. Variables in turn N are visible in turn N+1.

# Why primitives, not raw builtins

The model COULD write `import pathlib; list(pathlib.Path('.').iterdir())`. We
expose `ls('.')` instead, for two reasons (see INTERNALS.md Principle 4):

1. Observability — every primitive call emits a structured `tool_called`
   event. A `cat file.txt` via bash is opaque; a `read('file.txt')` is not.
2. Behavior steering — primitive docstrings encode our conventions (line
   numbering on read, find-and-replace semantics on edit, etc.) so the model
   uses them consistently.

# Security posture

The sandbox is NOT a security boundary. `exec()` runs Python with your
process's permissions. For untrusted code, run coda in a container. The
in-process design buys us speed and observability; isolation is the
container's job.

Per-tool gates we DO enforce:
- Path resolution: ls/glob/grep/read/write/edit reject paths outside `root`
- Read size cap: prevents accidentally loading multi-GB files
- Edit uniqueness: `old` must match exactly once by default
- Bash timeout: subprocess gets a deadline

# Execution model

`execute(code)` compiles `code` against the persistent globals dict and
installs a sys.settrace global tracer. The tracer only attaches per-frame
trace functions to frames whose filename is `<coda>` — so a `line_executed`
event fires for each line of GENERATED code, but not for the implementation
of `ls()` running inside coda itself. Stdout and stderr are captured.
"""

from __future__ import annotations

import builtins
import io
import re
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hooks import Event, HookRegistry


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_traceback: str | None = None
    lines_executed: int = 0
    globals_after: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None


SKIPPED_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", ".coda", "dist", "build"})


class Sandbox:
    def __init__(
        self,
        root: str | Path | None = None,
        hooks: HookRegistry | None = None,
        bash_timeout_s: float = 30.0,
        max_read_bytes: int = 5_000_000,
        extra_globals: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.hooks = hooks or HookRegistry()
        self.bash_timeout_s = bash_timeout_s
        self.max_read_bytes = max_read_bytes
        self.globals: dict[str, Any] = {
            "__builtins__": builtins.__dict__,
            "__name__": "__coda__",
            "ls": self.ls,
            "glob": self.glob,
            "grep": self.grep,
            "read": self.read,
            "write": self.write,
            "edit": self.edit,
            "bash": self.bash,
        }
        if extra_globals:
            self.globals.update(extra_globals)

    def inject(self, name: str, value: Any) -> None:
        """Make `value` available to executed code as the global `name`."""
        self.globals[name] = value

    def _resolve(self, path: str | Path) -> Path:
        target = Path(path)
        p = (target if target.is_absolute() else self.root / target).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise PermissionError(
                f"{path!r} resolves to {p}, which is outside the sandbox root "
                f"{self.root}. Construct Sandbox(root=...) to widen the scope."
            )
        return p

    def _emit_tool(self, name: str, args: dict, result_summary: Any) -> None:
        self.hooks.emit(
            Event(
                type="tool_called",
                payload={"tool": name, "args": args, "result": result_summary},
            )
        )

    # --- primitives -----------------------------------------------------------

    def ls(self, path: str = ".") -> list[str]:
        """List entries in a directory.

        Args:
            path: directory to list, relative to the sandbox root.

        Returns:
            A sorted list of names. Directories end with '/'.

        Use this instead of `os.listdir` or `pathlib.Path.iterdir` — every call
        is observed, logged, and replayable.
        """
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"{path!r} does not exist")
        if not p.is_dir():
            raise NotADirectoryError(f"{path!r} is not a directory")
        names = sorted(
            (entry.name + "/" if entry.is_dir() else entry.name)
            for entry in p.iterdir()
        )
        self._emit_tool("ls", {"path": str(path)}, {"count": len(names)})
        return names

    def glob(self, pattern: str, root: str = ".") -> list[str]:
        """Return paths matching a glob, relative to the sandbox root.

        Args:
            pattern: e.g. `*.py`, `src/**/*.ts`, `**/*test*.py`. Recursive `**`
                is supported.
            root: where to start the glob (default: current sandbox root).

        Returns:
            Sorted list of paths relative to the sandbox root. Empty if nothing
            matches; this is not an error.
        """
        base = self._resolve(root)
        out = sorted(str(m.relative_to(self.root)) for m in base.glob(pattern))
        self._emit_tool(
            "glob", {"pattern": pattern, "root": root}, {"count": len(out)}
        )
        return out

    def grep(
        self,
        pattern: str,
        path: str = ".",
        regex: bool = True,
    ) -> list[tuple[int, str, str]]:
        """Search files for a pattern.

        Args:
            pattern: regex (default) or literal string if `regex=False`.
            path: file or directory. Directories are searched recursively;
                common junk (`.git`, `__pycache__`, `node_modules`, `.venv`,
                `dist`, `build`) is skipped automatically.
            regex: when False, do a fixed-string substring search.

        Returns:
            A list of `(lineno, relative_path, line)` tuples. `lineno` is
            1-indexed. The line is trimmed of its trailing newline.

        Use grep for "is the symbol there" / "where is this referenced"
        questions. Use read() once you know which file to dig into.
        """
        target = self._resolve(path)
        rx = re.compile(pattern) if regex else None
        matches: list[tuple[int, str, str]] = []
        files: list[Path] = (
            [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
        )
        for f in files:
            if any(part in SKIPPED_DIRS for part in f.relative_to(self.root).parts):
                continue
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, start=1):
                        hit = bool(rx.search(line)) if rx else (pattern in line)
                        if hit:
                            matches.append(
                                (i, str(f.relative_to(self.root)), line.rstrip("\n"))
                            )
            except OSError:
                continue
        self._emit_tool(
            "grep",
            {"pattern": pattern, "path": path, "regex": regex},
            {"count": len(matches)},
        )
        return matches

    def read(self, path: str, with_line_numbers: bool = True) -> str:
        """Read a text file.

        Args:
            path: file path, relative to the sandbox root.
            with_line_numbers: when True (default), prepend `cat -n`-style line
                numbers to each line. This matches Claude Code's output format
                and makes it natural for the model to reference specific lines
                in follow-up `edit` calls.

        Returns:
            File contents. Files larger than `max_read_bytes` are rejected —
            use `bash('head -c ...')` for a sample of huge files.

        For binary or non-UTF-8 files, errors are replaced rather than raised.
        """
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"{path!r} does not exist")
        size = p.stat().st_size
        if size > self.max_read_bytes:
            raise ValueError(
                f"{path} is {size} bytes; exceeds max_read_bytes="
                f"{self.max_read_bytes}. Use bash('head -c N {path}') for a sample."
            )
        body = p.read_text(encoding="utf-8", errors="replace")
        if with_line_numbers:
            body = "\n".join(
                f"{i:>6}\t{line}" for i, line in enumerate(body.splitlines(), start=1)
            )
        self._emit_tool("read", {"path": path}, {"bytes": size})
        return body

    def write(self, path: str, content: str) -> int:
        """Write a text file, overwriting any existing contents.

        Args:
            path: file path, relative to the sandbox root.
            content: text to write.

        Returns:
            Number of bytes written.

        Parent directories are created as needed. Prefer `edit()` for changes
        to an existing file — it's safer because it requires the existing text
        you intend to replace, so concurrent or unexpected edits surface as
        errors instead of silent overwrites.
        """
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        n = p.write_text(content, encoding="utf-8")
        self._emit_tool("write", {"path": path}, {"bytes": n})
        return n

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> int:
        """Find and replace text in a file. By default `old` must occur exactly once.

        Args:
            path: file path, relative to the sandbox root.
            old: the exact text to replace. Must be a contiguous slice of the
                file as it currently exists.
            new: replacement text.
            replace_all: when True, replace every occurrence; otherwise the
                call fails if `old` appears more than once.

        Returns:
            Number of replacements performed.

        If a short `old` matches multiple times, include more surrounding
        context until it uniquely identifies the spot you want to edit. The
        model should usually grab 1-3 lines of context above and below the
        target span.
        """
        p = self._resolve(path)
        body = p.read_text(encoding="utf-8")
        count = body.count(old)
        if count == 0:
            raise ValueError(f"`old` text not found in {path}")
        if count > 1 and not replace_all:
            raise ValueError(
                f"`old` text appears {count} times in {path}. Provide more "
                "context to disambiguate, or pass replace_all=True."
            )
        new_body = body.replace(old, new)
        p.write_text(new_body, encoding="utf-8")
        n_done = count if replace_all else 1
        self._emit_tool("edit", {"path": path}, {"replacements": n_done})
        return n_done

    def bash(self, cmd: str, timeout_s: float | None = None) -> dict[str, Any]:
        """Run a shell command in the sandbox root.

        Args:
            cmd: shell command (single-line; use `&&` to chain).
            timeout_s: per-call override of the sandbox's default.

        Returns:
            `{"stdout": str, "stderr": str, "returncode": int, "timed_out": bool}`.

        Prefer the typed primitives (ls / glob / grep / read / write / edit)
        when they apply — they emit structured events that the bash call
        cannot. Use bash for:
        - running tests (`pytest`, `cargo test`, …)
        - git (`git status`, `git diff`, `git log`)
        - package managers (`pip install`, `npm i`)
        - anything else not covered by the typed tools

        Avoid:
        - destructive ops without explicit user confirmation (`rm -rf`,
          `git push --force`)
        - `cat`/`head`/`tail`/`sed`/`awk` to read or edit files — use the
          typed primitives instead
        """
        timeout = timeout_s or self.bash_timeout_s
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            result = {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            result = {
                "stdout": _decode(getattr(e, "stdout", None)),
                "stderr": _decode(getattr(e, "stderr", None)),
                "returncode": -1,
                "timed_out": True,
            }
        self._emit_tool(
            "bash",
            {"cmd": cmd},
            {"returncode": result["returncode"], "timed_out": result["timed_out"]},
        )
        return result

    # --- execution -----------------------------------------------------------

    def execute(self, code: str) -> ExecutionResult:
        """Compile and execute `code` against the persistent globals dict.

        Returns an ExecutionResult with stdout, stderr, error info, and the
        number of generated-code lines executed (tracked via sys.settrace,
        scoped to frames with filename `<coda>` so we don't trace coda itself).
        """
        stdout, stderr = io.StringIO(), io.StringIO()
        lines = [0]
        hooks = self.hooks

        def line_tracer(frame, event, arg):
            if event == "line":
                lines[0] += 1
                hooks.emit(
                    Event(
                        type="line_executed",
                        payload={"lineno": frame.f_lineno},
                    )
                )
            return line_tracer

        def global_tracer(frame, event, arg):
            if event == "call" and frame.f_code.co_filename == "<coda>":
                return line_tracer
            return None

        hooks.emit(Event(type="code_emitted", payload={"code": code}))
        err_repr: str | None = None
        tb_str: str | None = None
        prev_tracer = sys.gettrace()
        try:
            compiled = compile(code, "<coda>", "exec")
        except SyntaxError as e:
            tb_str = traceback.format_exc()
            err_repr = repr(e)
            hooks.emit(Event(type="execution_error", payload={"error": err_repr}))
            return ExecutionResult(
                stdout="",
                stderr="",
                error=err_repr,
                error_traceback=tb_str,
                lines_executed=0,
                globals_after=self.globals,
            )

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                sys.settrace(global_tracer)
                try:
                    exec(compiled, self.globals)
                finally:
                    sys.settrace(prev_tracer)
        except BaseException as e:
            err_repr = repr(e)
            tb_str = traceback.format_exc()
            hooks.emit(Event(type="execution_error", payload={"error": err_repr}))

        result = ExecutionResult(
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            error=err_repr,
            error_traceback=tb_str,
            lines_executed=lines[0],
            globals_after=self.globals,
        )
        hooks.emit(
            Event(
                type="code_executed",
                payload={
                    "lines": result.lines_executed,
                    "had_error": result.error is not None,
                    "stdout_len": len(result.stdout),
                },
            )
        )
        return result


def _decode(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    return str(x)
