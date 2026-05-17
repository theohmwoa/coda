"""Schema-aware AST linting for skills (and any Python coda would exec).

Catches the specific class of bug the morning_brief thin-prompt run
exposed: the agent saved `t["importance"] == "medium"` when the
sub-agent's return type was `Literal["critical", "high", "normal",
"low"]`. The comparison parses, runs without error, and is just dead
code that silently makes a feature not work. Text-grep can't catch it
because there's nothing textually wrong. AST + type info catches it
in 30 lines.

This is `ast-grep` flavored — structural pattern matching — but with a
twist: it knows about the runtime's Pydantic schemas. Every Literal
field in every connected sub-agent's return type goes into a
{field_name: allowed_values} table. We scan the code for comparisons
of those field names against string constants; if a string constant
isn't in the allowed set, we flag it.

# Known limitations (intentional, for v0)

- Field-name matching is positional, not flow-typed. If a skill has a
  local `{"importance": "medium"}` dict unrelated to a sub-agent
  return, we still flag it. False positives are fine; the agent
  decides what to do with warnings.
- Only handles `Literal[...]`. Union/Optional containing Literal is
  not unwrapped yet.
- Only `==` (and `!=` mirror). Doesn't handle `x in (...)`.

These are all addressable with more rules but each adds false-
positive surface; we'd rather start narrow.
"""

from __future__ import annotations

import ast
import typing
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class Finding:
    kind: str           # currently: "literal_mismatch"
    line: int           # 1-indexed
    col: int            # 0-indexed
    field: str          # the field name (e.g., "importance")
    value: str          # the bad string (e.g., "medium")
    allowed: list[str]  # the allowed set
    message: str        # human-readable rendering for the agent

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "line": self.line,
            "col": self.col,
            "field": self.field,
            "value": self.value,
            "allowed": list(self.allowed),
            "message": self.message,
        }


def collect_literal_fields(model_cls: type[BaseModel]) -> dict[str, set[str]]:
    """Return `{field_name: allowed_values}` for every `Literal[...]` field on the model.

    Only string-typed Literal values are returned (the only ones the
    AST matcher considers).
    """
    out: dict[str, set[str]] = {}
    for field_name, field_info in model_cls.model_fields.items():
        ann = field_info.annotation
        # Literal[...] directly
        if typing.get_origin(ann) is typing.Literal:
            vals = {v for v in typing.get_args(ann) if isinstance(v, str)}
            if vals:
                out[field_name] = vals
        # Optional[Literal[...]] etc. — peel one union layer
        elif typing.get_origin(ann) in (typing.Union, type(int | str)):
            for arg in typing.get_args(ann):
                if typing.get_origin(arg) is typing.Literal:
                    vals = {v for v in typing.get_args(arg) if isinstance(v, str)}
                    if vals:
                        out.setdefault(field_name, set()).update(vals)
    return out


def build_schema_table(
    schemas: dict[str, type[BaseModel]],
) -> dict[str, set[str]]:
    """Union the Literal-field tables of every sub-agent's return type.

    If two sub-agents both have a field named `severity` with disjoint
    Literal values, the table is the union — a value is "valid" if any
    sub-agent allows it. Conservative: minimizes false positives.
    """
    out: dict[str, set[str]] = {}
    for _name, model_cls in schemas.items():
        for f, vals in collect_literal_fields(model_cls).items():
            out.setdefault(f, set()).update(vals)
    return out


def _field_name_from_node(node: ast.expr) -> str | None:
    """If `node` is `x.NAME` or `x["NAME"]`, return NAME. Otherwise None."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        # Python 3.9+: node.slice is the index expr directly
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
            return idx.value
    return None


def lint(
    code: str,
    *,
    schemas: dict[str, type[BaseModel]] | None = None,
) -> list[Finding]:
    """Return a list of findings for `code` against the given sub-agent schemas.

    `schemas` is `{sub_agent_name: return_type_class}` — typically built
    from an `Agent`'s active sub-agents. Empty `schemas` means an empty
    finding list (nothing to compare against).
    """
    findings: list[Finding] = []
    if not schemas:
        return findings

    table = build_schema_table(schemas)
    if not table:
        return findings

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Caller should handle SyntaxError separately; lint only runs on
        # parseable code.
        return findings

    class _V(ast.NodeVisitor):
        def visit_Compare(self, node: ast.Compare) -> None:
            for op, comp in zip(node.ops, node.comparators):
                if not isinstance(op, (ast.Eq, ast.NotEq)):
                    continue
                if not (isinstance(comp, ast.Constant) and isinstance(comp.value, str)):
                    continue
                field = _field_name_from_node(node.left)
                if field is None or field not in table:
                    continue
                if comp.value in table[field]:
                    continue
                allowed = sorted(table[field])
                findings.append(
                    Finding(
                        kind="literal_mismatch",
                        line=node.lineno,
                        col=node.col_offset,
                        field=field,
                        value=comp.value,
                        allowed=allowed,
                        message=(
                            f"line {node.lineno}: comparison "
                            f"`{field} {'!=' if isinstance(op, ast.NotEq) else '=='} "
                            f"{comp.value!r}` — `{field}` is a Literal field whose "
                            f"allowed values are {allowed}. This branch is dead code."
                        ),
                    )
                )
            self.generic_visit(node)

    _V().visit(tree)
    return findings


def lint_dicts(
    code: str,
    *,
    schemas: dict[str, type[BaseModel]] | None = None,
) -> list[dict[str, Any]]:
    """JSON-serializable variant for use over the wire / in tool responses."""
    return [f.to_dict() for f in lint(code, schemas=schemas)]
