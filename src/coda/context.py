"""Ctx — the agent-authored context namespace.

# What this is

CodeAct decouples two things JSON-tool-calling agents fuse together: what the
agent can compute over (sandbox globals) and what the agent thinks about (the
LLM's context window). The `ctx` namespace makes that separation explicit and
agent-controlled.

The agent writes:

    bookings_today = bookings.find(date=today)        # sandbox-only, 500 records
    high_value = [b for b in bookings_today if b.fare > 1000]
    ctx.high_value_count = len(high_value)            # visible to LLM next turn
    ctx.top_5 = high_value[:5]                        # visible to LLM next turn
    del ctx.policy_doc                                # drop from context; requery hint kept

Only `ctx.*` state is serialized into the next user message. Plain globals
persist in the sandbox across turns but don't enter the LLM's context. This
keeps the context window small even when the agent is working over large
in-memory datasets.

# Requery hints

When `ctx.X = EXPR` runs, the source expression `EXPR` is captured from the
currently-executing source line. When `del ctx.X` runs, the entry moves to a
`_dropped` buffer that keeps the name + source_expr so the agent can be told,
later: "you had `policy_doc` (produced by `policies.get(\"refund\")`) — requery
it any time." The requery hint is just Python code the agent can re-run; the
sandbox doesn't try to be clever about what's idempotent.

# How source capture works

`Ctx.__setattr__` walks the call stack to find the user-code frame (the one
whose filename is `<coda>`, which is what `Sandbox.execute` compiles to). It
reads `frame.f_lineno`, looks up that line in the source code that's
currently executing (provided by the Sandbox via a callable), and parses out
the RHS of the assignment with `ast`. If the line can't be parsed cleanly
(multi-line assignment, augmented assignment, etc.) the entry still works —
just without a requery hint.

# Why a custom class, not SimpleNamespace

We need three things SimpleNamespace can't give us: assignment interception
(for source capture), deletion interception (for the dropped buffer), and a
clean way to keep our bookkeeping off the public attribute surface (so
`dir(ctx)` shows only what the agent put there).
"""

from __future__ import annotations

import ast
import json
import pprint
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


__all__ = [
    "Ctx",
    "CtxEntry",
    "render_ctx",
    "render_value",
    "render_value_pretty",
    "render_ctx_delta",
    "estimate_tokens",
    "ctx_token_estimate",
    "ContextPolicy",
    "build_context_reminder",
    "build_print_habit_reminder",
    "MODEL_CONTEXT_WINDOWS",
    "resolve_budget_for_model",
]


# Rough character-to-token ratio used by the inline estimator. The Anthropic
# tokenizer averages ~3.5-4.0 chars per token for English prose and slightly
# fewer for JSON-like content; 4 is a good budget-side estimate (it under-
# counts a bit, which is what we want for a "do we need to remind the agent"
# threshold check — false positives are cheaper than missed overflows).
#
# For exact counts at decision time, callers can swap in a real tokenizer.
# This module deliberately stays dependency-free for the threshold path.
_CHARS_PER_TOKEN = 4.0

# Max chars to inline per ctx entry's value repr. Past this point the value
# is truncated with a `…(N more chars)` marker. The agent can always inspect
# the full value via Python (it lives in the sandbox); the marker tells it
# "this was big, you may want to summarize or drop me." Sized generously so
# small business records, short docs, and modest lists pass through whole.
DEFAULT_MAX_VALUE_CHARS = 2000

# Per-entry cap when rendering ctx_delta values for the agent. Larger than
# DEFAULT_MAX_VALUE_CHARS because this is a one-shot inspection (only fires
# the turn the entry is set/changed, not on subsequent turns), and the
# agent needs enough of the value to verify its own work. The persistent
# render_ctx output (used in threshold reminders) keeps the smaller cap
# since it can fire repeatedly.
DEFAULT_DELTA_VALUE_CHARS = 6000


# Context windows in tokens for known Claude models. Used to auto-derive a
# sensible `budget_tokens` on ContextPolicy when the caller doesn't pin it
# explicitly. Numbers reflect the documented full-context capacity; you can
# of course pin a smaller budget for testing or for cost reasons.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    # older lines, kept for compatibility
    "claude-opus-4-5": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
}

# What to return when the model name doesn't prefix-match any known entry.
# 200K is a safe, conservative floor — modern Claude models all support
# at least that much. If the agent runs against an unknown future model
# the policy will under-estimate the available room, never over-estimate.
DEFAULT_BUDGET_FALLBACK = 200_000


def resolve_budget_for_model(model: str) -> int:
    """Look up the context window for `model`, prefix-matching MODEL_CONTEXT_WINDOWS.

    Prefix match so `claude-opus-4-7[1m]` (with the [1m] suffix Anthropic
    sometimes appends) resolves to the same window as `claude-opus-4-7`.
    Falls back to DEFAULT_BUDGET_FALLBACK on no match — better to underrun
    than overrun the model's real capacity.
    """
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(key):
            return window
    return DEFAULT_BUDGET_FALLBACK


# Filename Sandbox.execute() compiles user code to. We walk the stack
# looking for a frame with this filename to find "the user-code frame"
# regardless of how deeply nested the assignment is inside helper calls.
_SANDBOX_FILENAME = "<coda>"

# Private attribute names that bypass the tracking machinery. Anything
# starting with `_ctx_` is treated as Ctx-internal state — stored via
# `object.__setattr__` and never enters the tracked entries dict.
_PRIVATE_PREFIX = "_ctx_"


SourceProvider = Callable[[], list[str] | None]
"""A callable that returns the source lines of the code currently executing
in the sandbox, or None if no code is executing right now. Sandbox installs
one of these when it starts an execute(); Ctx queries it on each assignment
to read the source line by lineno."""


@dataclass
class CtxEntry:
    """One value the agent has chosen to keep in context.

    Fields:
        name: attribute name on `ctx` (e.g. `"high_value_count"`).
        value: the actual Python value.
        source_expr: the RHS of the assignment as written in source code,
            e.g. `"len(high_value)"` for `ctx.high_value_count = len(high_value)`.
            None if the source couldn't be parsed (rare; multi-line
            assignments, eval'd code, etc.).
        set_at_turn: which agent turn this entry was set in. Filled in by
            the Agent loop, not by Ctx itself. None when running outside
            an agent loop (e.g. in unit tests).
    """

    name: str
    value: Any
    source_expr: str | None = None
    set_at_turn: int | None = None


class Ctx:
    """The agent's writable context namespace.

    Use attribute syntax: `ctx.foo = 42`, `del ctx.foo`, `ctx.foo` to read.

    Bookkeeping (entries, dropped buffer, source provider) is stored on
    private `_ctx_*` attributes via `object.__setattr__` so it never appears
    in the tracked entries dict.
    """

    def __init__(self, source_provider: SourceProvider | None = None) -> None:
        # Use object.__setattr__ to bypass our own __setattr__ for setup.
        object.__setattr__(self, "_ctx_entries", {})
        object.__setattr__(self, "_ctx_dropped", {})
        object.__setattr__(self, "_ctx_source_provider", source_provider)
        # Current agent turn, stamped onto each entry as it's set. Owned by
        # the Agent loop; defaults to None for tests using Ctx standalone.
        object.__setattr__(self, "_ctx_current_turn", None)
        # Hook callback: called as `hook(event_type, payload_dict)` on every
        # assignment and deletion. Set by Sandbox so events flow through
        # HookRegistry. None means no observability (still works fine).
        object.__setattr__(self, "_ctx_hook", None)

    # --- internal helpers ----------------------------------------------------

    def _ctx_set_source_provider(self, provider: SourceProvider | None) -> None:
        """Install a source-lookup callable. Sandbox calls this once at boot."""
        object.__setattr__(self, "_ctx_source_provider", provider)

    def _ctx_set_hook(self, hook: Callable[[str, dict], None] | None) -> None:
        """Install an event hook. Called as hook(event_type, payload)."""
        object.__setattr__(self, "_ctx_hook", hook)

    def _ctx_set_turn(self, turn: int | None) -> None:
        """Stamp subsequent assignments with `turn`. The Agent calls this."""
        object.__setattr__(self, "_ctx_current_turn", turn)

    def _ctx_emit(self, event_type: str, payload: dict) -> None:
        # _ctx_hook is set via object.__setattr__ in __init__, so a direct
        # instance-attribute read picks it up without recursing through
        # our own __getattr__ (which fires only when normal lookup fails).
        hook = object.__getattribute__(self, "_ctx_hook")
        if hook is not None:
            hook(event_type, payload)

    def _ctx_lookup_source_expr(self, name: str) -> str | None:
        """Try to parse `ctx.NAME = EXPR` out of the source line that's running.

        Returns the unparsed RHS expression if we can confidently extract it.
        Returns None if we can't (multi-line assignment, augmented assignment,
        unparseable line, no source provider, etc.). Always errs on the side
        of None — a missing requery hint is better than a wrong one.
        """
        provider = self._ctx_source_provider
        if provider is None:
            return None
        try:
            source_lines = provider()
        except Exception:
            return None
        if not source_lines:
            return None

        # Walk the call stack to find the user-code frame. We start from our
        # caller (__setattr__) and look outward. sys._getframe is faster than
        # inspect.currentframe; the frame at depth 1 is __setattr__'s caller.
        # For nested cases (e.g. `ctx.x = helper()` where helper does the
        # assignment internally) the assignment frame may be deeper, but the
        # user-code frame is identified by filename == <coda>.
        frame = sys._getframe(1)
        lineno: int | None = None
        while frame is not None:
            if frame.f_code.co_filename == _SANDBOX_FILENAME:
                lineno = frame.f_lineno
                break
            frame = frame.f_back
        if lineno is None:
            return None
        if lineno < 1 or lineno > len(source_lines):
            return None

        source_line = source_lines[lineno - 1]
        return _parse_assignment_rhs(source_line, name)

    # --- entry / dropped accessors -------------------------------------------

    def _entries(self) -> dict[str, CtxEntry]:
        """Return a snapshot of the live entries (name -> CtxEntry).

        Returns a copy so callers can't mutate our internal dict.
        """
        return dict(self._ctx_entries)

    def _dropped(self) -> dict[str, CtxEntry]:
        """Return a snapshot of recently-dropped entries (name -> CtxEntry).

        Dropped entries keep their `source_expr` so the agent can be reminded
        how to bring them back. The buffer is unbounded by default; threshold
        policy code is responsible for evicting old dropped entries when it
        decides they're no longer useful.
        """
        return dict(self._ctx_dropped)

    def _clear_dropped(self, names: list[str] | None = None) -> None:
        """Forget specific dropped entries, or all of them if `names` is None."""
        if names is None:
            self._ctx_dropped.clear()
        else:
            for n in names:
                self._ctx_dropped.pop(n, None)

    def _names(self) -> list[str]:
        """Live entry names, in insertion order (dict preserves it)."""
        return list(self._ctx_entries.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self._ctx_entries)

    def __len__(self) -> int:
        return len(self._ctx_entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._ctx_entries

    def __dir__(self) -> list[str]:
        # `dir(ctx)` should show what the agent put there, not our internals.
        return sorted(self._ctx_entries.keys())

    def __repr__(self) -> str:
        names = ", ".join(self._ctx_entries.keys())
        return f"<Ctx [{names}]>"

    # --- the three attribute hooks -------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith(_PRIVATE_PREFIX):
            # Internal state: store directly, do not track.
            object.__setattr__(self, name, value)
            return
        if name.startswith("_") or name in {"value"}:
            # Defensive: refuse names that could shadow our hooks/methods.
            # The agent has the entire alphabet; we can have leading
            # underscores to ourselves.
            raise AttributeError(
                f"ctx attribute {name!r} starts with '_' (reserved for Ctx internals)"
            )
        source_expr = self._ctx_lookup_source_expr(name)
        entry = CtxEntry(
            name=name,
            value=value,
            source_expr=source_expr,
            set_at_turn=self._ctx_current_turn,
        )
        self._ctx_entries[name] = entry
        # If this name was previously dropped, remove the dropped record.
        self._ctx_dropped.pop(name, None)
        self._ctx_emit(
            "ctx_assigned",
            {"name": name, "source_expr": source_expr, "turn": entry.set_at_turn},
        )

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails. Our entries
        # aren't on the instance dict (they live in _ctx_entries), so every
        # `ctx.foo` for a tracked entry routes here.
        if name.startswith(_PRIVATE_PREFIX):
            # Shouldn't happen via attribute lookup, but be defensive.
            raise AttributeError(name)
        entries = object.__getattribute__(self, "_ctx_entries")
        if name in entries:
            return entries[name].value
        raise AttributeError(f"ctx has no attribute {name!r}")

    def __delattr__(self, name: str) -> None:
        if name.startswith(_PRIVATE_PREFIX):
            object.__delattr__(self, name)
            return
        if name not in self._ctx_entries:
            raise AttributeError(f"ctx has no attribute {name!r}")
        entry = self._ctx_entries.pop(name)
        # Park the entry in the dropped buffer — value is gone, but we keep
        # name + source_expr so the agent can be told what to requery.
        self._ctx_dropped[name] = CtxEntry(
            name=entry.name,
            value=None,  # don't hold a reference; this is a tombstone
            source_expr=entry.source_expr,
            set_at_turn=entry.set_at_turn,
        )
        self._ctx_emit(
            "ctx_dropped",
            {"name": name, "source_expr": entry.source_expr},
        )


# --- serialization for LLM context ------------------------------------------


def render_value(value: Any, max_chars: int = DEFAULT_MAX_VALUE_CHARS) -> str:
    """Render `value` as a string the LLM can read, truncated if too long.

    Uses `repr()` by default. Pydantic models get `.model_dump()` first so
    their representation shows fields, not a `<Customer id=...>` object header
    the model can't introspect. Strings pass through quoted (via repr) so the
    model can see the boundaries — important when a value happens to contain
    XML-looking content that would confuse the surrounding wrapper.

    Long values get `…(N more chars)` rather than silent truncation. We want
    the model to know its view is partial so it can decide whether to
    summarize, slice, or drop.
    """
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            rendered = repr(value.model_dump())
        except Exception:
            rendered = repr(value)
    else:
        try:
            rendered = repr(value)
        except Exception as e:
            rendered = f"<unrepresentable: {type(value).__name__}: {e}>"
    if len(rendered) <= max_chars:
        return rendered
    head = rendered[:max_chars]
    omitted = len(rendered) - max_chars
    return f"{head}…({omitted} more chars)"


def render_ctx(
    ctx: "Ctx",
    *,
    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS,
    include_dropped: bool = False,
) -> str:
    """Serialize the ctx state for inclusion in the next user message.

    Format:

        <ctx>
        <entry name="NAME" set_at_turn="N" tokens="EST"><value>REPR</value></entry>
        ...
        </ctx>

    Returns an empty string when ctx has no live entries (and no dropped
    entries are being requested). The Agent code is responsible for deciding
    whether to call this at all and where the output gets placed in the
    next message.

    The `set_at_turn` attribute helps the model reason about freshness ("this
    was set 6 turns ago, may be stale"). The per-entry `tokens` estimate lets
    the model see at a glance which entries are spending the budget without
    needing to count for itself.
    """
    live = ctx._entries()
    if not live and not (include_dropped and ctx._dropped()):
        return ""
    parts: list[str] = ["<ctx>"]
    for name, entry in live.items():
        rendered = render_value(entry.value, max_chars=max_value_chars)
        tokens = estimate_tokens(rendered)
        attrs = [f'name="{name}"', f'tokens="{tokens}"']
        if entry.set_at_turn is not None:
            attrs.append(f'set_at_turn="{entry.set_at_turn}"')
        if entry.source_expr is not None:
            # Source expr is shown so the model can recognize what produced
            # the value without scrolling back to earlier turns. Escape the
            # only characters that would confuse the XML-ish wrapper.
            escaped = entry.source_expr.replace('"', "&quot;")
            attrs.append(f'source_expr="{escaped}"')
        parts.append(f"<entry {' '.join(attrs)}><value>{rendered}</value></entry>")
    if include_dropped:
        dropped = ctx._dropped()
        if dropped:
            parts.append("<dropped>")
            for name, entry in dropped.items():
                hint = entry.source_expr or "(no source captured)"
                escaped = hint.replace('"', "&quot;")
                parts.append(f'<entry name="{name}" requery="{escaped}"/>')
            parts.append("</dropped>")
    parts.append("</ctx>")
    return "\n".join(parts)


def render_value_pretty(value: Any, max_chars: int = DEFAULT_DELTA_VALUE_CHARS) -> str:
    """Render `value` in a print-like, multi-line form for verification.

    Prefers `json.dumps(..., indent=2)` for JSON-compatible values (dicts,
    lists, primitives) because the indentation makes nested structure
    scannable. Falls back to `pprint.pformat` for everything else (sets,
    tuples, custom objects). Pydantic models get `.model_dump()` first.

    Long output is truncated with a `…(N more chars)` marker, like the
    compact `render_value` — the agent should know its view is partial
    so it can fetch the full value via plain globals if needed.
    """
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            value = value.model_dump()
        except Exception:
            pass
    try:
        rendered = json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            rendered = pprint.pformat(value, width=80, sort_dicts=False)
        except Exception as e:
            rendered = f"<unrepresentable: {type(value).__name__}: {e}>"
    if len(rendered) <= max_chars:
        return rendered
    head = rendered[:max_chars]
    omitted = len(rendered) - max_chars
    return f"{head}\n…({omitted} more chars)"


def render_ctx_delta(
    ctx: "Ctx",
    *,
    before: dict[str, Any],
    max_value_chars: int = DEFAULT_DELTA_VALUE_CHARS,
) -> str:
    """Render the changes to ctx since `before`, formatted to read like print output.

    The agent uses this in place of print(). After every code execution,
    the runner snapshots `ctx._entries()` before, runs the code, and calls
    this with the before-snapshot. The returned block shows every entry
    that was added, changed, or removed — values pretty-printed across
    multiple lines so the agent can verify structure (catching e.g.
    `set(prices)` accidentally labeled as `cabin`, which a one-line repr
    would obscure).

    Format:

        <ctx_delta>
        # added: ctx.NAME
        ctx.NAME = <pretty-printed value>

        # changed: ctx.NAME
        ctx.NAME = <pretty-printed new value>

        # removed: ctx.NAME (was: <pretty-printed last value>)
        </ctx_delta>

    The block fires once per assignment — when the same entry stays
    unchanged on a later turn, it does NOT reappear. So the per-turn
    token cost is bounded by what actually changed this turn, not by
    the size of ctx.

    Returns empty string when nothing changed.
    """
    after = ctx._entries()
    added: list[str] = []
    changed: list[str] = []
    removed: list[str] = []
    for name, entry in after.items():
        prev = before.get(name)
        if prev is None:
            added.append(name)
        else:
            # Use source_expr + id() of value as a coarse "did this change"
            # check. Mutating an existing value in place (without
            # reassigning) won't be flagged here — that's intentional,
            # since no source_expr fired.
            if prev.source_expr != entry.source_expr or id(prev.value) != id(entry.value):
                changed.append(name)
    for name in before:
        if name not in after:
            removed.append(name)

    if not (added or changed or removed):
        return ""

    parts: list[str] = ["<ctx_delta>"]
    for name in added:
        rendered = render_value_pretty(after[name].value, max_chars=max_value_chars)
        parts.append(f"# added: ctx.{name}")
        parts.append(f"ctx.{name} = {rendered}")
        parts.append("")
    for name in changed:
        rendered = render_value_pretty(after[name].value, max_chars=max_value_chars)
        parts.append(f"# changed: ctx.{name}")
        parts.append(f"ctx.{name} = {rendered}")
        parts.append("")
    for name in removed:
        prev = before[name]
        if prev.value is not None:
            rendered = render_value_pretty(prev.value, max_chars=max_value_chars)
            parts.append(f"# removed: ctx.{name} (was)")
            parts.append(f"ctx.{name} = {rendered}")
            parts.append("")
        else:
            parts.append(f"# removed: ctx.{name}")
            parts.append("")
    parts.append("</ctx_delta>")
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: chars / 4. See _CHARS_PER_TOKEN for rationale.

    Off by 10-20% on either side for most English/code content. Fine for
    threshold decisions; not appropriate for billing or hard cutoffs.
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def ctx_token_estimate(
    ctx: "Ctx",
    *,
    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS,
    include_dropped: bool = False,
) -> int:
    """Estimated token cost of the rendered ctx block.

    This is what the threshold policy compares against its budget. Computing
    the full rendered string and counting chars is the simplest faithful
    answer (it accounts for the XML wrapper and per-entry attribute overhead);
    we don't try to optimize because ctx is small by design.
    """
    return estimate_tokens(render_ctx(ctx, max_value_chars=max_value_chars, include_dropped=include_dropped))


# --- threshold policy + reminder formatter ----------------------------------


@dataclass
class ContextPolicy:
    """Configurable policy for when to remind the agent about context size.

    The Agent loop consults `should_remind()` after every turn using the
    `input_tokens` count the LLM backend just reported (which is the exact
    size of what we sent — more reliable than estimating). When True, it
    asks the policy to format a reminder via `build_context_reminder`, and
    injects that reminder as a one-shot system-reminder block in the next
    user message.

    Defaults are tuned for a typical 200K-context model and a 75% trigger
    point — plenty of room left to do useful work after the agent drops
    or summarizes a few entries. Smaller-window models should pass a
    smaller `budget_tokens` (or just leave the default and hit the trigger
    earlier in absolute terms; the fraction is what matters for behavior).

    `cooldown_turns` debounces the threshold reminder so the agent isn't
    pinged every single turn if it's slow to react. With cooldown=2, after
    a reminder fires we suppress the next 2 turns regardless of size; the
    third turn can re-trigger. Set to 0 to remind every turn.

    `large_print_chars` enables a SECOND, finer-grained reminder: if any
    block prints more than this many characters to stdout in one turn,
    the harness appends a one-shot system-reminder steering the agent
    toward `ctx`. This catches "print(full_record)" bloat *before* it
    accumulates into a threshold cross. Set to 0 to disable.

    `large_print_cooldown_turns` is the same idea: don't ping the agent
    on every consecutive large-print turn.

    `force_ctx_on_print` is an aggressive variant: fires for EVERY
    non-trivial print (any `print(...)` whose argument isn't a pure
    string literal), regardless of stdout size, with stronger
    error-shaped wording. Use this when you suspect the model is
    defaulting to print() instead of ctx and you want to push it.
    Pure status prints (`print("done")`) are still allowed because
    they don't carry data.
    """

    budget_tokens: int | None = None     # None → resolve from model at Agent init
    threshold_fraction: float = 0.85     # raised from 0.75: at this point ctx
                                          # management is mostly handled by the
                                          # print-habit reminder; threshold is a
                                          # safety net, not a primary surface
    cooldown_turns: int = 2
    max_largest_entries: int = 5
    include_dropped: bool = True
    large_print_chars: int = 4_000        # ~1K tokens of stdout in one turn
    large_print_cooldown_turns: int = 1
    force_ctx_on_print: bool = False

    def resolve_budget(self, model: str) -> int:
        """Return the concrete budget for this run: explicit value, else from model."""
        if self.budget_tokens is not None:
            return self.budget_tokens
        return resolve_budget_for_model(model)

    @property
    def threshold_tokens(self) -> int:
        """Threshold against the explicit budget. Raises if budget is auto;
        callers in that case should use `threshold_for(model)` instead."""
        if self.budget_tokens is None:
            raise ValueError(
                "ContextPolicy.budget_tokens is None (auto). Use "
                "policy.threshold_for(model) or policy.resolve_budget(model) "
                "in code that needs the concrete number."
            )
        return int(self.budget_tokens * self.threshold_fraction)

    def threshold_for(self, model: str) -> int:
        return int(self.resolve_budget(model) * self.threshold_fraction)

    def should_remind(
        self,
        *,
        input_tokens: int,
        turns_since_last: int,
        model: str | None = None,
    ) -> bool:
        """Decide whether to fire the threshold reminder this turn.

        `model` is required when `budget_tokens` is None (auto). Callers
        that pin budget_tokens explicitly can omit it for backwards
        compatibility with existing tests.
        """
        if self.budget_tokens is None:
            if model is None:
                raise ValueError(
                    "should_remind: model must be passed when budget_tokens is auto"
                )
            threshold = self.threshold_for(model)
        else:
            threshold = self.threshold_tokens
        if input_tokens < threshold:
            return False
        if turns_since_last < self.cooldown_turns:
            return False
        return True

    def should_warn_print(self, *, stdout_chars: int, turns_since_last: int) -> bool:
        if self.large_print_chars <= 0:
            return False
        if stdout_chars < self.large_print_chars:
            return False
        if turns_since_last < self.large_print_cooldown_turns:
            return False
        return True

    def should_force_ctx(self, *, has_data_print: bool, turns_since_last: int) -> bool:
        if not self.force_ctx_on_print:
            return False
        if not has_data_print:
            return False
        if turns_since_last < self.large_print_cooldown_turns:
            return False
        return True


def build_context_reminder(
    ctx: "Ctx",
    *,
    input_tokens: int,
    budget_tokens: int,
    max_largest_entries: int = 5,
    include_dropped: bool = True,
) -> str:
    """Format the system-reminder string for an over-threshold turn.

    The reminder is structured for the agent to act on with Python:

        <system-reminder>
        Context at 152,000 / 200,000 tokens (76%). Largest in ctx:
          - bookings_today (4,100 tokens) — set turn 3, source: bookings.find(...)
          - policy_doc (2,500 tokens) — set turn 1, source: policies.get('refund')
        Recently dropped (requery any time):
          - customer_history — requery: customers.get('ABC123').history()
        Drop or summarize entries that aren't needed for the next step,
        or proceed if everything here is still relevant.
        </system-reminder>

    The reminder is deliberately a single contiguous string with no
    framework-specific tags inside — the only structural marker is the
    outer `<system-reminder>` block, which Claude is trained to treat as
    sandboxed instruction.
    """
    pct = int(round((input_tokens / max(budget_tokens, 1)) * 100))
    lines = [
        "<system-reminder>",
        f"Context at {input_tokens:,} / {budget_tokens:,} tokens ({pct}%).",
    ]

    # Rank live entries by their rendered-token cost.
    live = ctx._entries()
    if live:
        sized: list[tuple[str, int, "CtxEntry"]] = []
        for name, entry in live.items():
            rendered = render_value(entry.value)
            sized.append((name, estimate_tokens(rendered), entry))
        sized.sort(key=lambda t: t[1], reverse=True)
        top = sized[:max_largest_entries]
        if top:
            lines.append("Largest in ctx:")
            for name, tok, entry in top:
                bits = [f"  - {name} (~{tok:,} tokens)"]
                if entry.set_at_turn is not None:
                    bits.append(f"set turn {entry.set_at_turn}")
                if entry.source_expr is not None:
                    bits.append(f"source: {entry.source_expr}")
                lines.append(" — ".join(bits))

    # Dropped entries with requery hints. This is the load-bearing
    # affordance — without it, the agent doesn't know what's recoverable.
    if include_dropped:
        dropped = ctx._dropped()
        if dropped:
            lines.append("Recently dropped (requery any time):")
            for name, entry in dropped.items():
                hint = entry.source_expr or "(source not captured; rebuild manually)"
                lines.append(f"  - {name} — requery: {hint}")

    lines.append(
        "Drop or summarize entries that aren't needed for the next step "
        "(use `del ctx.NAME` or reassign), or proceed if everything here "
        "is still relevant."
    )
    lines.append("</system-reminder>")
    return "\n".join(lines)


def build_print_habit_reminder(
    *,
    stdout_chars: int,
    block_index: int | None = None,
    code_snippet: str | None = None,
    force: bool = False,
) -> str:
    """Format the system-reminder shown after a big-stdout turn.

    Two modes, controlled by `force`:

    - **Normal** (`force=False`): a soft nudge tied to size. Suggests
      slicing, plain globals, or ctx as alternatives. Used when a block
      crossed `ContextPolicy.large_print_chars` but the policy isn't
      configured to force ctx usage.

    - **Force** (`force=True`): an error-shaped reminder used when the
      policy has `force_ctx_on_print=True`. Stronger language ("DO NOT",
      "violation"), no soft alternatives — the model is told plainly to
      switch to ctx for any print of fetched data.

    Args:
        stdout_chars: total stdout chars in this turn (across all blocks).
            Ignored when `force=True` (the reminder is triggered by the
            presence of a data-print, not by its size).
        block_index: which block triggered, if multi-block.
        code_snippet: a short excerpt of the offending `print(...)` line.
        force: switch to force-mode wording.
    """
    lines: list[str] = ["<system-reminder>"]
    if force:
        lines.append(
            "❌ **CTX POLICY VIOLATION** — this run is configured with "
            "`force_ctx_on_print=True`. Calls like `print(fetched_data)` "
            "are not allowed; data printed to stdout enters your "
            "conversation history and is paid for on every subsequent "
            "turn."
        )
    else:
        approx_tokens = max(1, int(stdout_chars / 4))
        lines.append(
            f"You printed {stdout_chars:,} characters (~{approx_tokens:,} tokens) "
            f"to stdout this turn"
            + (f" in block {block_index}" if block_index is not None else "")
            + ". This now lives in your conversation history and you'll pay for "
            "it on every subsequent turn."
        )
    if code_snippet:
        snippet = code_snippet.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"Offending line: `{snippet}`")
    if force:
        lines.append(
            "Fix next turn: replace the print with one of:"
        )
        lines.append(
            "  - `ctx.NAME = value` — visible across turns, surfaced when "
            "context fills up. Use this when you want the LLM to see a "
            "summary."
        )
        lines.append(
            "  - plain assignment `NAME = value` — lives in the sandbox, "
            "NOT in your context. Use this for full records you'll "
            "compute over but don't need to read."
        )
        lines.append(
            "Status prints like `print(\"done\")` are fine (no data); "
            "data prints like `print(user)` or `print(reservation)` are "
            "the problem."
        )
    else:
        lines.append("Prefer one of these patterns next turn:")
        lines.append(
            "  - Print only the slice you actually need: "
            "`print(len(x))`, `print(x['key'])`, etc."
        )
        lines.append(
            "  - Keep the full result in a plain global (sandbox-only, "
            "not in your context): `data = primitive.fetch(...)` (no print)."
        )
        lines.append(
            "  - Stash a small summary in ctx for cross-turn use: "
            "`ctx.summary = {...}` (no print needed; ctx is surfaced when "
            "useful)."
        )
    lines.append("</system-reminder>")
    return "\n".join(lines)


# --- source-line parsing -----------------------------------------------------


def _parse_assignment_rhs(source_line: str, name: str) -> str | None:
    """If `source_line` is exactly `ctx.NAME = EXPR`, return EXPR (as written).

    Falls back to None for:
      - augmented assignments (`ctx.x += 1`)
      - tuple/multi-target assignments (`ctx.x, ctx.y = 1, 2`)
      - assignments via subscript (`ctx['x'] = 1`) — not our path anyway
      - lines we can't ast.parse (continuation lines from a multi-line stmt)

    Why ast and not regex: nested brackets, strings containing `=`, lambdas,
    list-comprehensions all break naive splitting. ast handles them. The
    overhead is ~tens of microseconds per assignment, which is fine because
    assignments are not on any hot path the agent cares about.
    """
    stripped = source_line.strip()
    if not stripped:
        return None
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError:
        # Likely a continuation line from a multi-line stmt. Give up.
        return None
    if len(tree.body) != 1:
        return None
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Assign):
        return None
    if len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Attribute):
        return None
    if target.attr != name:
        return None
    # We don't strictly check that the target's base is the literal name
    # `ctx` — the agent could rebind it. The attr-name match is enough
    # signal that this is the assignment we're looking at.
    try:
        return ast.unparse(stmt.value)
    except Exception:
        return None
