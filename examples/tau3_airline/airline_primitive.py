"""Expose tau3-bench (sierra-research/tau2-bench) airline tools to a coda sandbox.

This is the τ³ counterpart of `examples/tau_airline/airline_primitive.py`.
The v1 helper iterated `tau_bench.envs.airline.tools.ALL_TOOLS` (class
objects with `.invoke(data=..., **kwargs)` classmethods). τ³ replaces
that with a `ToolKitBase` class whose tools are *bound methods* on an
`AirlineTools(db)` instance, with `self`/`db` already captured. We iterate
`env.tools.tools` (a dict of `{name: bound_method}`) and wrap each into
the same `airline.*` SimpleNamespace shape the agent already knows from
the v1 bridge.

Other shifts from v1 that matter to us:
- Tool results are Pydantic `BaseModel` instances (Reservation, User, …)
  rather than JSON strings. We `.model_dump()` them so ctx_delta renders
  as plain dicts; primitives pass through unchanged.
- Errors raise `ValueError` instead of returning "Error: ..." strings.
  We catch and convert so the agent's branching logic from v1 still works.
- Whether a tool mutates state is queried via
  `env.tools.tool_mutates_state(name)` instead of inferred from name.
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from pydantic import BaseModel

_DEFAULT_TAU2_BENCH_PATH = os.path.expanduser("~/code/tau2-bench")
TAU2_BENCH_PATH = os.environ.get("TAU2_BENCH_PATH", _DEFAULT_TAU2_BENCH_PATH)
_TAU2_SRC = os.path.join(TAU2_BENCH_PATH, "src")
if _TAU2_SRC not in sys.path:
    sys.path.insert(0, _TAU2_SRC)

from tau2.data_model.tasks import Task  # noqa: E402
from tau2.domains.airline.environment import (  # noqa: E402
    get_environment,
    get_tasks,
)


@dataclass
class RecordedAction:
    name: str
    kwargs: dict[str, Any]
    result_preview: str  # first ~200 chars of the raw tool result


def _result_to_python(raw: Any) -> Any:
    """Convert tool return values into something ctx/print-friendly.

    Pydantic models → dicts. Lists of models → lists of dicts. Primitives
    pass through. This way `ctx.user = airline.get_user_details(...)` shows
    a real dict in the ctx_delta, the same shape the v1 bridge produced
    via `json.loads(raw)`.
    """
    if isinstance(raw, BaseModel):
        return raw.model_dump()
    if isinstance(raw, list):
        return [_result_to_python(x) for x in raw]
    if isinstance(raw, dict):
        return {k: _result_to_python(v) for k, v in raw.items()}
    return raw


def build_airline() -> tuple[
    SimpleNamespace, list[RecordedAction], Any, list[bool], str
]:
    """Build the airline namespace backed by a fresh tau3 environment.

    Returns:
        (airline, actions, db, terminate, policy)
        - airline: SimpleNamespace whose attributes are the 14 callable
          tools. Inject as `sandbox.inject("airline", airline)`.
        - actions: live list; every call appends a RecordedAction.
        - db: the live FlightDB (Pydantic) the tools mutate. Compared to
          a fresh `get_environment()` snapshot to see what changed.
        - terminate: [bool] — set to [True] when transfer_to_human_agents
          fires.
        - policy: the tau3 policy.md as a string, ready to drop into the
          prompt preamble.
    """
    env = get_environment()
    tools = env.tools  # AirlineTools instance bound to a fresh FlightDB
    db = tools.db
    policy = env.policy

    actions: list[RecordedAction] = []
    terminate: list[bool] = [False]

    ns_attrs: dict[str, Callable] = {}
    for name, bound_method in tools.tools.items():
        # `bound_method` already has `self` bound — inspect.signature gives
        # just the user-facing params.
        sig = inspect.signature(bound_method)
        param_order = list(sig.parameters.keys())

        def _make(method=bound_method, n=name, order=param_order):
            def call(*args, **kwargs):
                # Translate positional args to kwargs by schema order so
                # `airline.get_user_details("...")` works the same as
                # `airline.get_user_details(user_id="...")`.
                if args:
                    for i, val in enumerate(args):
                        if i >= len(order):
                            raise TypeError(
                                f"{n}() takes at most {len(order)} positional "
                                f"arg(s), got {len(args)}"
                            )
                        pname = order[i]
                        if pname in kwargs:
                            raise TypeError(
                                f"{n}() got multiple values for argument {pname!r}"
                            )
                        kwargs[pname] = val
                try:
                    raw = method(**kwargs)
                except Exception as e:
                    # v1's tools returned "Error: ..." strings rather than
                    # raising. Preserve that contract for the agent's
                    # downstream branching.
                    err = f"Error: {e}"
                    actions.append(
                        RecordedAction(name=n, kwargs=dict(kwargs), result_preview=err[:200])
                    )
                    return err

                preview = (raw if isinstance(raw, str) else str(raw))[:200]
                actions.append(
                    RecordedAction(name=n, kwargs=dict(kwargs), result_preview=preview)
                )
                if n == "transfer_to_human_agents":
                    terminate[0] = True
                return _result_to_python(raw)

            call.__name__ = n
            # Tool description comes from the method's docstring directly in
            # τ³ — no need for an extra schema lookup.
            call.__doc__ = bound_method.__doc__ or ""
            return call

        ns_attrs[name] = _make()

    airline = SimpleNamespace(**ns_attrs)
    return airline, actions, db, terminate, policy


def load_tasks() -> list[Task]:
    """Load the full τ³ airline task list (50 tasks)."""
    return get_tasks(task_split_name=None)


def task_instruction_text(task: Task) -> str:
    """Render a τ³ task's user-scenario into the single-string format
    the v1 bridge passed as `task.instruction`.

    The customer simulator only needs the situational context: who they
    are, why they're calling, what they want, and any behavioral hints.
    This concatenates the structured fields in the order they read most
    naturally.
    """
    inst = task.user_scenario.instructions
    if isinstance(inst, str):
        return inst
    parts: list[str] = []
    if inst.known_info:
        parts.append(inst.known_info.strip())
    if inst.reason_for_call:
        parts.append(inst.reason_for_call.strip())
    if inst.task_instructions:
        parts.append(inst.task_instructions.strip())
    if inst.unknown_info:
        parts.append(f"(unknown to you: {inst.unknown_info.strip()})")
    return "\n\n".join(parts)
