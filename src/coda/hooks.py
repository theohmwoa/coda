"""HookRegistry — typed event bus for everything coda does.

Every interesting moment in the agent loop emits a structured Event:

    code_emitted        a model turn produced code to run
    code_executed       that code finished
    line_executed       one line inside that code ran (from sys.settrace)
    tool_called         a sandbox primitive (ls/glob/grep/read/write/edit/bash)
                        was invoked from user code
    variable_assigned   a global was assigned in user code
    execution_error     exec raised
    subagent_called     a typed sub-agent was dispatched
    subagent_returned   the sub-agent returned (success or schema fail)
    llm_request         about to call LLMClient.complete
    llm_response        got the completion back
    turn_complete       one full agent turn finished

Subscribers register a handler per event type (or '*' for all). Handlers
run in registration order, synchronously, in the calling thread. If your
handler is slow, do the work on a queue — the agent loop blocks on you.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

EventType = Literal[
    "code_emitted",
    "code_executed",
    "line_executed",
    "tool_called",
    "variable_assigned",
    "execution_error",
    "subagent_called",
    "subagent_returned",
    "llm_request",
    "llm_response",
    "turn_complete",
]


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


Handler = Callable[[Event], None]


class HookRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._captured: list[Event] | None = None

    def on(self, event_type: EventType | str, handler: Handler) -> None:
        """Register `handler` to fire on `event_type`. Use '*' to catch all."""
        self._handlers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        """Dispatch `event` to subscribers. Exceptions in handlers propagate."""
        if self._captured is not None:
            self._captured.append(event)
        for h in self._handlers.get(event.type, ()):
            h(event)
        for h in self._handlers.get("*", ()):
            h(event)

    def capture(self) -> list[Event]:
        """Start recording every event into a list. Returns the list (live).

        Useful for tests and for the trace writer that will land in week 2.
        Calling capture() a second time replaces the recording buffer.
        """
        self._captured = []
        return self._captured

    def stop_capture(self) -> None:
        self._captured = None
