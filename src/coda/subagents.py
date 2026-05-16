"""@subagent decorator — typed sub-agents callable from inside the sandbox.

A sub-agent is a function-shaped LLM call with a Pydantic return type. The
outer agent's generated Python invokes it like any other call:

    verdict = evaluate_flight(flight={"id": 7, "redeye": True})
    if verdict.layover_score >= 7 and verdict.confidence == "high":
        ...

Under the hood, `evaluate_flight(...)` is a fresh `LLMClient.complete()` call
with `tool_choice` set to FORCE the model to return JSON matching the
Pydantic schema. The result is parsed back into a Pydantic instance. This is
the "dual schema flow" referenced in the README:

- Inner Claude call is schema-forced → guaranteed-shape output
- Outer agent's prompt is augmented with the function signature → its
  generated code navigates `verdict.score` structurally instead of guessing

Decorator form:

    @coda.subagent(model="claude-haiku-4-5", tools=[read, grep])
    def evaluate_flight(flight: dict) -> FlightVerdict:
        '''Evaluate one flight on redeye + layover dimensions. Be strict.'''
"""

from __future__ import annotations

import inspect
import json
import sys
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .hooks import Event, HookRegistry
from .llm import LLMClient, Message, ToolChoice, ToolSchema


@dataclass
class SubAgent:
    name: str
    fn: Callable[..., Any]
    return_type: type[BaseModel]
    docstring: str
    signature: inspect.Signature
    model: str
    max_tokens: int = 2048
    temperature: float = 1.0
    _llm: LLMClient | None = field(default=None, repr=False)
    _hooks: HookRegistry | None = field(default=None, repr=False)

    def bind(self, llm: LLMClient, hooks: HookRegistry | None = None) -> None:
        """Attach the LLMClient (and optional HookRegistry) the sub-agent should use.

        The outer Agent calls this at construction so SubAgent instances can be
        defined module-level (before any Agent exists) and rebound per-Agent.
        """
        self._llm = llm
        self._hooks = hooks

    def tool_schema(self) -> ToolSchema:
        """Anthropic-shaped tool schema derived from the Pydantic return type."""
        return ToolSchema(
            name=f"return_{self.name}",
            description=(
                f"Return the result of `{self.name}` as a structured "
                f"{self.return_type.__name__} object. "
                f"Provide every required field."
            ),
            input_schema=self.return_type.model_json_schema(),
        )

    def _system_prompt(self) -> str:
        sig_str = f"def {self.name}{self.signature} -> {self.return_type.__name__}"
        schema_str = json.dumps(self.return_type.model_json_schema(), indent=2)
        return (
            f"You are a focused sub-agent that implements one function:\n\n"
            f"    {sig_str}\n\n"
            f"Docstring:\n{self.docstring}\n\n"
            f"You MUST return your answer by calling the tool "
            f"`return_{self.name}` with arguments matching this JSON schema:\n"
            f"{schema_str}\n\n"
            f"Be decisive. Do not ask clarifying questions."
        )

    def _user_prompt(self, args: tuple, kwargs: dict) -> str:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return "Inputs:\n" + json.dumps(
            {k: _jsonable(v) for k, v in bound.arguments.items()},
            indent=2,
            default=str,
        )

    def __call__(self, *args, **kwargs) -> BaseModel:
        if self._llm is None:
            raise RuntimeError(
                f"SubAgent {self.name!r} is unbound. Pass it to an Agent via "
                "subagents=[...] so it gets an LLMClient before being called."
            )
        schema = self.tool_schema()
        if self._hooks is not None:
            self._hooks.emit(
                Event(
                    type="subagent_called",
                    payload={"name": self.name, "args": args, "kwargs": kwargs},
                )
            )
        resp = self._llm.complete(
            system=self._system_prompt(),
            messages=[Message(role="user", content=self._user_prompt(args, kwargs))],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=[schema],
            tool_choice=ToolChoice(mode="tool", name=schema.name),
        )
        if not resp.tool_calls:
            raise RuntimeError(
                f"SubAgent {self.name!r} expected a forced tool-use response "
                f"but got plain text: {resp.text[:200]!r}"
            )
        raw = resp.tool_calls[0].input
        try:
            result = self.return_type.model_validate(raw)
        except ValidationError:
            # Backends that fake forced tool-use via prompted JSON
            # (ClaudeAgentSDKClient, ClaudeCodeClient) sometimes return
            # nested lists/dicts as JSON-encoded STRINGS rather than
            # native objects. Coerce them once and retry — this
            # eliminates a large class of spurious validation failures
            # that otherwise burn through retries (and rate limits).
            try:
                result = self.return_type.model_validate(_coerce_json_strings(raw))
            except ValidationError as e:
                raise ValueError(
                    f"SubAgent {self.name!r} returned an object that failed "
                    f"{self.return_type.__name__} validation: {e}"
                )
        if self._hooks is not None:
            self._hooks.emit(
                Event(
                    type="subagent_returned",
                    payload={"name": self.name, "result": raw},
                )
            )
        return result


def subagent(
    *,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 2048,
    temperature: float = 1.0,
):
    """Decorate a function as a typed sub-agent.

    The decorated function's return type annotation MUST be a Pydantic
    BaseModel. The function body is unused — coda generates the
    implementation from the signature, docstring, and return type.
    """

    def decorator(fn: Callable[..., Any]) -> SubAgent:
        sig = inspect.signature(fn)
        # Resolve forward-ref annotations (PEP 563 / `from __future__ import annotations`).
        # Pull in the caller's local scope so return types defined inside a
        # function body (common in tests) can be resolved.
        localns: dict[str, Any] | None = None
        try:
            localns = sys._getframe(1).f_locals
        except (ValueError, AttributeError):
            pass
        try:
            hints = typing.get_type_hints(fn, localns=localns)
        except Exception:
            hints = {}
        ret = hints.get("return", sig.return_annotation)
        if ret is inspect.Signature.empty:
            raise TypeError(
                f"@subagent {fn.__name__!r} requires a return-type annotation "
                "(a Pydantic BaseModel subclass)."
            )
        if not (isinstance(ret, type) and issubclass(ret, BaseModel)):
            raise TypeError(
                f"@subagent {fn.__name__!r} return type must subclass "
                f"pydantic.BaseModel; got {ret!r}."
            )
        return SubAgent(
            name=fn.__name__,
            fn=fn,
            return_type=ret,
            docstring=(inspect.getdoc(fn) or "").strip(),
            signature=sig,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return decorator


def _jsonable(v: Any) -> Any:
    if isinstance(v, BaseModel):
        return v.model_dump()
    if isinstance(v, (dict, list, str, int, float, bool)) or v is None:
        return v
    return repr(v)


def _coerce_json_strings(obj: Any) -> Any:
    """Recursively try to JSON-parse string values that look like JSON.

    Used as a Pydantic-validation safety net. Some LLM backends (the ones
    that fake forced tool-use via prompted JSON instead of Anthropic's
    native tool-use schema enforcement) occasionally return nested
    lists/dicts as JSON-encoded strings — e.g. `agreements: "[\"a\",\"b\"]"`
    instead of `agreements: ["a", "b"]`. Without this, every such response
    triggers a ValidationError and the agent retries (often hitting the
    same model and the same mistake), wasting time and rate budget.

    Conservative: only parses strings that already start with `[` or `{`,
    silently keeps the original on parse failure.
    """
    if isinstance(obj, dict):
        return {k: _coerce_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_json_strings(x) for x in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if len(stripped) >= 2 and stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return obj
            return _coerce_json_strings(parsed)
    return obj
