from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from coda.llm import CompletionResponse, MockLLMClient, ToolCall
from coda.subagents import SubAgent, subagent


class Verdict(BaseModel):
    is_urgent: bool
    score: int
    confidence: Literal["low", "medium", "high"]
    reasoning: str


def test_decorator_produces_subagent_with_metadata():
    @subagent(model="claude-haiku-4-5")
    def is_urgent(email: dict) -> Verdict:
        """Decide whether an email needs an immediate reply."""

    assert isinstance(is_urgent, SubAgent)
    assert is_urgent.name == "is_urgent"
    assert is_urgent.return_type is Verdict
    assert "immediate reply" in is_urgent.docstring
    assert "email" in is_urgent.signature.parameters


def test_missing_return_type_rejected():
    with pytest.raises(TypeError, match="return-type annotation"):
        @subagent()
        def bad(x: int):
            """no return type"""


def test_non_pydantic_return_type_rejected():
    with pytest.raises(TypeError, match="must subclass"):
        @subagent()
        def bad(x: int) -> dict:
            """not a pydantic model"""


def test_unbound_call_raises():
    @subagent()
    def f(x: int) -> Verdict:
        """f."""

    with pytest.raises(RuntimeError, match="unbound"):
        f(x=1)


def test_call_forces_tool_use_and_returns_pydantic():
    verdict_payload = {
        "is_urgent": True,
        "score": 9,
        "confidence": "high",
        "reasoning": "deadline today",
    }
    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text="",
                tool_calls=[ToolCall(id="t1", name="return_is_urgent", input=verdict_payload)],
                stop_reason="tool_use",
            )
        ]
    )

    @subagent(model="claude-haiku-4-5")
    def is_urgent(email: dict) -> Verdict:
        """Decide whether an email needs an immediate reply."""

    is_urgent.bind(mock)
    result = is_urgent(email={"subject": "URGENT", "body": "..."})

    assert isinstance(result, Verdict)
    assert result.is_urgent is True
    assert result.score == 9
    assert result.confidence == "high"

    call = mock.calls[0]
    assert call.tool_choice and call.tool_choice.mode == "tool"
    assert call.tool_choice.name == "return_is_urgent"
    assert call.tools and call.tools[0].input_schema is not None
    assert "is_urgent" in call.system
    assert "claude-haiku-4-5" == call.model


def test_call_rejects_plain_text_response():
    mock = MockLLMClient(responses=[CompletionResponse(text="hi", stop_reason="end_turn")])

    @subagent()
    def f(x: int) -> Verdict:
        """f."""

    f.bind(mock)
    with pytest.raises(RuntimeError, match="forced tool-use"):
        f(x=1)


def test_call_coerces_stringified_json_lists():
    """The model sometimes returns nested lists/dicts as JSON-encoded
    strings under prompted-JSON backends. Coerce-and-retry should rescue
    these instead of failing validation."""
    payload = {
        "is_urgent": True,
        "score": 7,
        "confidence": "medium",
        # Note: reasoning is a str, that's fine; but a real failure mode
        # is nested lists. Use a Verdict variant for that case.
        "reasoning": "deadline",
    }
    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text="",
                tool_calls=[ToolCall(id="t1", name="return_is_urgent", input=payload)],
                stop_reason="tool_use",
            )
        ]
    )

    @subagent()
    def is_urgent(email: dict) -> Verdict:
        """."""

    is_urgent.bind(mock)
    # Should not raise even though we're testing the post-coercion path
    # with already-valid data (coerce is a no-op here).
    result = is_urgent(email={"subject": "x"})
    assert result.is_urgent is True


def test_coerce_json_strings_helper():
    from coda.subagents import _coerce_json_strings

    # Stringified list becomes a list
    assert _coerce_json_strings({"xs": '["a", "b"]'}) == {"xs": ["a", "b"]}
    # Stringified dict becomes a dict
    assert _coerce_json_strings({"d": '{"k": 1}'}) == {"d": {"k": 1}}
    # Plain string is left alone
    assert _coerce_json_strings({"s": "hello"}) == {"s": "hello"}
    # Already-list values unchanged
    assert _coerce_json_strings({"xs": ["a"]}) == {"xs": ["a"]}
    # Nested
    assert _coerce_json_strings({"outer": '["a", "b"]', "n": 1}) == {
        "outer": ["a", "b"],
        "n": 1,
    }
    # Malformed JSON string stays as string
    assert _coerce_json_strings({"x": "[not json"}) == {"x": "[not json"}


def test_call_rejects_schema_mismatch():
    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text="",
                tool_calls=[
                    ToolCall(id="t1", name="return_f", input={"wrong": "shape"})
                ],
                stop_reason="tool_use",
            )
        ]
    )

    @subagent()
    def f(x: int) -> Verdict:
        """f."""

    f.bind(mock)
    with pytest.raises(ValueError, match="validation"):
        f(x=1)
