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
