from __future__ import annotations

from coda.tools import Tool, tool


def test_tool_decorator_wraps_function():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert "Add two numbers." in add.description
    assert add(2, 3) == 5
    assert "a" in add.signature and "int" in add.signature


def test_tool_idempotent():
    @tool
    def f():
        """f."""

    again = tool(f)
    assert again is f
