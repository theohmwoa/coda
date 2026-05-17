"""Skills — agent-curated reusable code workflows.

The pattern: after a coda agent successfully writes Python that solves a
task, that code is itself an asset. We save it as a `.py` file in a
skills directory; the next run loads it and exposes the top-level
function as a sandbox global. The agent calls `brief_morning()` instead
of re-discovering how to build a morning brief.

This is the Voyager skill library (Wang et al. 2023) and the MirageAI
`interfaces/` pattern, productized into coda's substrate:

- Each skill = one `.py` file with exactly one top-level function whose
  name matches the file stem. The function's docstring + signature is
  its tool description.
- Skills are loaded at `Agent` construction and their callables are
  rebound to the sandbox globals (`types.FunctionType` swap on
  `__globals__`) so references inside their bodies — e.g. `gmail`,
  `slack`, `ls`, `read` — resolve against the live sandbox at call
  time. A skill written when the agent had Gmail wired up only works
  in a future run that also has Gmail wired up. Documented contract.
- `save_skill(code=...)` is exposed to the agent as an inline tool. The
  agent passes a full Python function source; we AST-parse it,
  validate, write the file. Available in the *next* run.

Skill files are also human-editable and human-readable. They're just
Python.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")

# Names skills can't shadow without confusion.
_RESERVED_NAMES = frozenset({
    "save_skill", "ls", "glob", "grep", "read", "write", "edit", "bash",
    "print", "input", "open", "exec", "eval", "compile",
})


@dataclass
class Skill:
    """One loaded skill ready to inject into a sandbox."""

    name: str
    docstring: str
    signature: str
    file_path: Path
    fn: Callable[..., Any]


def load_skills(directory: str | Path) -> dict[str, Skill]:
    """Discover and import every `.py` file in `directory` as a skill.

    Returns a `{name: Skill}` map. Files starting with `_` are skipped
    (so `__init__.py` and any helper modules don't get loaded as
    skills). Files that don't define a top-level function matching the
    file stem are skipped with no error — they may be data fixtures.
    """
    directory = Path(directory).resolve()
    out: dict[str, Skill] = {}
    if not directory.is_dir():
        return out

    # Make the directory's parent importable so `from <dirname> import x`
    # works inside skill bodies if they ever need it. The skills
    # themselves are loaded individually via spec_from_file_location.
    parent = str(directory.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    pkg_name = directory.name
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        skill_name = py_file.stem
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{skill_name}", py_file
        )
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            # A skill that fails to import is silently skipped. The user
            # can fix it manually; the rest of the registry stays usable.
            continue
        fn = getattr(mod, skill_name, None)
        if not callable(fn):
            continue
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        out[skill_name] = Skill(
            name=skill_name,
            docstring=(inspect.getdoc(fn) or "").strip(),
            signature=sig,
            file_path=py_file,
            fn=fn,
        )
    return out


def inject_skills(
    sandbox_globals: dict[str, Any], skills: dict[str, Skill]
) -> None:
    """Make each skill callable inside the sandbox.

    Crucially: rebinds each skill's `__globals__` to point at the
    sandbox's globals via `types.FunctionType`. Without this, calls
    inside the skill body referencing names like `gmail`, `slack`, or
    coda primitives would resolve against the (long-gone) module
    namespace where the skill was first loaded — not against what's
    actually live in the current run's sandbox.
    """
    for name, skill in skills.items():
        rebound = types.FunctionType(
            skill.fn.__code__,
            sandbox_globals,
            skill.fn.__name__,
            skill.fn.__defaults__,
            skill.fn.__closure__,
        )
        rebound.__doc__ = skill.fn.__doc__
        sandbox_globals[name] = rebound


def save_skill(*, code: str, directory: str | Path) -> dict[str, Any]:
    """Save a reusable skill from a full Python function source.

    Args:
        code: complete Python source defining EXACTLY ONE top-level
            function. The function's name becomes the file name. The
            function's docstring is used as the file's module docstring.
        directory: where to write the skill (`Agent.skills_dir`).

    Returns:
        `{"ok": True, "path": ..., "name": ...}` on success.
        `{"ok": False, "error": ...}` on failure (no file written).

    Validation:
    - The code must `ast.parse` without errors.
    - It must contain exactly one top-level function definition.
    - The function name must be a snake_case identifier and NOT one of
      coda's reserved names (the sandbox primitives, `save_skill` itself).
    - The function's body must `compile` cleanly.

    Existing files at the target path are overwritten — letting the
    agent iterate on a skill across runs is intentional.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"SyntaxError: {e}"}

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return {"ok": False, "error": "no top-level function found in code"}
    if len(funcs) > 1:
        names = [f.name for f in funcs]
        return {
            "ok": False,
            "error": f"expected exactly one top-level function, got {names}",
        }

    fn_def = funcs[0]
    name = fn_def.name
    if not _IDENT.match(name):
        return {"ok": False, "error": f"invalid skill name {name!r}; must be snake_case"}
    if name in _RESERVED_NAMES:
        return {
            "ok": False,
            "error": f"skill name {name!r} shadows a coda built-in; pick another",
        }

    try:
        compile(code, f"<skill:{name}>", "exec")
    except SyntaxError as e:
        return {"ok": False, "error": f"SyntaxError in body: {e}"}

    docstring = ast.get_docstring(fn_def) or ""
    module_doc_line = docstring.splitlines()[0] if docstring else f"Skill: {name}."

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{name}.py"

    file_content = f'"""{module_doc_line}"""\n\n{code.rstrip()}\n'
    out_path.write_text(file_content, encoding="utf-8")

    return {"ok": True, "path": str(out_path), "name": name}


def make_save_skill(directory: str | Path) -> Callable[..., dict[str, Any]]:
    """Build a sandbox-injectable `save_skill(code: str) -> dict` bound to a directory."""
    bound_dir = directory

    def _save_skill(*, code: str) -> dict[str, Any]:
        """Save the given Python function source as a reusable skill.

        Pass `code` containing EXACTLY ONE top-level `def` whose name
        becomes the skill's name. The function may reference any names
        present in the sandbox globals at call time (MCP proxies like
        `gmail` or `slack`, sub-agents, native primitives like `read`
        and `bash`). Existing skills with the same name are overwritten.

        Returns `{"ok": True, "path": ..., "name": ...}` or
        `{"ok": False, "error": ...}`. Saved skills are available
        immediately in the NEXT run with the same `skills_dir`.
        """
        return save_skill(code=code, directory=bound_dir)

    _save_skill.__name__ = "save_skill"
    return _save_skill
