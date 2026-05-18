"""Expose tau-bench airline tools to a coda sandbox under an `airline.*` namespace.

No manual signature duplication. We iterate tau-bench's `ALL_TOOLS`, pull
each tool's name and parameter description out of `get_info()`, and wrap
it dynamically into a callable with `data` already bound. The set of
callables lives on a SimpleNamespace exposed as `airline` in the sandbox:

    user = airline.get_user_details(user_id="mia_li_3668")
    flights = airline.search_direct_flight(origin="JFK", destination="SEA",
                                           date="2024-05-20")

Why a namespace and not flat names: tau-bench's tool set includes
`calculate`, `think`, `transfer_to_human_agents` — generic words that
read better as `airline.calculate(...)` than as bare `calculate(...)`
in the agent's code. It also mirrors how MCP servers appear in coda
(`gmail.list_unread`, `slack.post_message`), which is the agent's
existing mental model.

Every invocation is also recorded into an `actions` list (one entry per
call, with the kwargs) so we can compare against the task's gold actions
for a coarse pass/fail signal. `transfer_to_human_agents` flips a
terminate flag so the runner knows the agent ended the run that way.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

_DEFAULT_TAU_BENCH_PATH = os.path.expanduser("~/code/tau-bench")
TAU_BENCH_PATH = os.environ.get("TAU_BENCH_PATH", _DEFAULT_TAU_BENCH_PATH)
if TAU_BENCH_PATH not in sys.path:
    sys.path.insert(0, TAU_BENCH_PATH)

from tau_bench.envs.airline.data import load_data  # noqa: E402
from tau_bench.envs.airline.tools import ALL_TOOLS  # noqa: E402
from tau_bench.envs.airline.wiki import WIKI  # noqa: E402


@dataclass
class RecordedAction:
    name: str
    kwargs: dict[str, Any]
    result_preview: str  # first ~200 chars of the raw tool result


def _parse_result(raw: Any) -> Any:
    """JSON-decode if possible, otherwise pass through.

    tau-bench tools return str — JSON when there's structured data, a
    plain "Error: ..." string on failure, or a plain confirmation string
    on send_certificate. Parsing with fallback gives the agent dicts
    where dicts are appropriate and leaves errors/confirmations as
    diagnostic strings it can branch on.
    """
    if not isinstance(raw, str):
        return raw
    if not raw or raw.startswith("Error:"):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def build_airline(*, fresh_data: bool = True) -> tuple[SimpleNamespace, list[RecordedAction], dict[str, Any], list[bool]]:
    """Build the `airline` namespace, backed by a fresh tau-bench data dict.

    Returns:
        (airline, actions, data, terminate)
        - airline: SimpleNamespace whose attributes are the 14 callable
          tools. Inject as `sandbox.inject("airline", airline)`.
        - actions: live list; every call appends a RecordedAction.
        - data: the live database the tools mutate; compare to a fresh
          load_data() snapshot to see what state changed.
        - terminate: [bool] — set to [True] when transfer_to_human_agents
          is called, so the runner can detect end-of-conversation.
    """
    data = load_data() if fresh_data else {}
    actions: list[RecordedAction] = []
    terminate: list[bool] = [False]

    ns_attrs: dict[str, Callable] = {}
    for tool_cls in ALL_TOOLS:
        name = tool_cls.get_info()["function"]["name"]

        def _make(cls=tool_cls, n=name):
            # Resolve the parameter order ONCE per tool, from tau-bench's
            # own schema. Used to map positional args (which a Python agent
            # may naturally use, e.g. `airline.get_user_details("mia_li_...")`)
            # onto the right keyword.
            info = cls.get_info()["function"]
            params = info.get("parameters", {}).get("properties", {})
            param_order = list(params.keys())

            def call(*args, **kwargs):
                # Translate any positional args into kwargs by schema order
                # so `airline.get_user_details("mia_li_...")` works the
                # same as `airline.get_user_details(user_id="mia_li_...")`.
                if args:
                    for i, val in enumerate(args):
                        if i >= len(param_order):
                            raise TypeError(
                                f"{n}() takes at most {len(param_order)} "
                                f"positional arg(s), got {len(args)}"
                            )
                        pname = param_order[i]
                        if pname in kwargs:
                            raise TypeError(
                                f"{n}() got multiple values for argument {pname!r}"
                            )
                        kwargs[pname] = val
                raw = cls.invoke(data=data, **kwargs)
                preview = (raw if isinstance(raw, str) else str(raw))[:200]
                actions.append(
                    RecordedAction(name=n, kwargs=dict(kwargs), result_preview=preview)
                )
                if n == "transfer_to_human_agents":
                    terminate[0] = True
                return _parse_result(raw)

            # Copy tau-bench's own description into the wrapper's __doc__
            # so the agent sees the same prose tau-bench wrote, not a
            # second-hand restatement.
            param_lines = [
                f"    {pname} ({pinfo.get('type', '?')}): {pinfo.get('description', '')}"
                for pname, pinfo in params.items()
            ]
            call.__name__ = n
            call.__doc__ = (
                info.get("description", "")
                + ("\n\nArgs:\n" + "\n".join(param_lines) if param_lines else "")
            )
            return call

        ns_attrs[name] = _make()

    airline = SimpleNamespace(**ns_attrs)
    return airline, actions, data, terminate


def get_wiki() -> str:
    """Return tau-bench's airline policy document (~6KB)."""
    return WIKI
