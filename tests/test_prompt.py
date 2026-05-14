"""Tests for the modular system-prompt Assembler (Principle 2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from coda.prompt import (
    Assembler,
    Section,
    build_system_prompt,
    default_assembler,
)
from coda.subagents import subagent
from coda.tools import tool


# --- raw Assembler --------------------------------------------------------


def test_render_concatenates_sections():
    a = Assembler()
    a.add(Section("a", render=lambda ctx: "AAA"))
    a.add(Section("b", render=lambda ctx: "BBB"))
    out = a.render()
    assert "AAA" in out and "BBB" in out
    assert out.index("AAA") < out.index("BBB")


def test_cache_stable_renders_before_breaking():
    a = Assembler()
    a.add(Section("breaking_1", render=lambda c: "BREAK", cache_stable=False))
    a.add(Section("stable_1", render=lambda c: "STABLE", cache_stable=True))
    out = a.render()
    assert out.index("STABLE") < out.index("BREAK")


def test_condition_omits_section():
    a = Assembler()
    a.add(
        Section(
            "maybe",
            render=lambda c: "INCLUDED",
            condition=lambda c: c.get("on", False),
        )
    )
    assert "INCLUDED" not in a.render({})
    assert "INCLUDED" in a.render({"on": True})


def test_empty_render_dropped():
    a = Assembler()
    a.add(Section("empty", render=lambda c: ""))
    a.add(Section("solid", render=lambda c: "X"))
    assert a.render() == "X"


def test_remove_section():
    a = Assembler()
    a.add(Section("keep", render=lambda c: "K"))
    a.add(Section("drop", render=lambda c: "D"))
    a.remove("drop")
    out = a.render()
    assert "K" in out
    assert "D" not in out


def test_replace_overrides_existing():
    a = Assembler()
    a.add(Section("intro", render=lambda c: "v1"))
    a.replace(Section("intro", render=lambda c: "v2"))
    assert "v2" in a.render()
    assert "v1" not in a.render()


def test_replace_appends_if_absent():
    a = Assembler()
    a.replace(Section("new", render=lambda c: "NEW"))
    assert "NEW" in a.render()


def test_preview_describes_each_section():
    a = Assembler()
    a.add(Section("on", render=lambda c: "hello"))
    a.add(
        Section(
            "off",
            render=lambda c: "hidden",
            condition=lambda c: False,
            cache_stable=False,
        )
    )
    rows = a.preview()
    assert rows[0] == {
        "name": "on",
        "cache_stable": True,
        "included": True,
        "chars": 5,
    }
    assert rows[1]["included"] is False
    assert rows[1]["chars"] == 0
    assert rows[1]["cache_stable"] is False


# --- default_assembler() -------------------------------------------------


def test_default_assembler_has_core_sections():
    a = default_assembler()
    names = [row["name"] for row in a.preview()]
    for must_have in ("intro", "primitives", "output_format"):
        assert must_have in names


def test_default_renders_minimum_when_context_is_empty():
    out = build_system_prompt()
    assert "ls(path='.')" in out
    assert "Output format" in out
    assert "Inline tools" not in out
    assert "sub-agents" not in out
    assert "Filesystem-as-toolspace" not in out


def test_default_renders_filesystem_section_when_tools_dir_set():
    out = build_system_prompt(tools_dir="interfaces")
    assert "Filesystem-as-toolspace" in out
    assert "from interfaces" in out


def test_default_renders_tools_and_subagents_when_provided():
    @tool
    def search(q: str) -> list:
        """Look things up."""
        return []

    class V(BaseModel):
        ok: bool
        score: int
        tier: Literal["a", "b"]

    @subagent()
    def check(item: dict) -> V:
        """Validate one item."""

    out = build_system_prompt(tools=[search], subagents=[check])
    assert "search" in out and "Look things up" in out
    assert "check" in out and "Validate one item" in out
    assert "ok" in out and "score" in out and "tier" in out


def test_default_cache_stable_ordering_matches_principle_2():
    """Cache-stable sections (intro, primitives, output_format, filesystem)
    must all render before cache-breaking ones (tools, subagents, extra).

    This is the property that makes coda's prompt cache-friendly even when
    the per-agent tool inventory changes."""

    @tool
    def t(x: int) -> int:
        """t."""
        return x

    class V(BaseModel):
        ok: bool

    @subagent()
    def s(x: dict) -> V:
        """s."""

    out = build_system_prompt(
        tools_dir="interfaces",
        tools=[t],
        subagents=[s],
        extra="custom appendix",
    )
    stable_anchor = out.index("Native primitives")
    fs_anchor = out.index("Filesystem-as-toolspace")
    breaking_tools = out.index("Inline tools available")
    breaking_subs = out.index("Typed sub-agents available")
    extra_anchor = out.index("custom appendix")
    assert stable_anchor < breaking_tools
    assert fs_anchor < breaking_tools
    assert breaking_tools < breaking_subs < extra_anchor


def test_custom_assembler_passthrough():
    a = default_assembler()
    a.replace(Section("intro", render=lambda c: "CUSTOM INTRO"))
    out = build_system_prompt(assembler=a)
    assert "CUSTOM INTRO" in out
    assert "You are coda" not in out


def test_agent_accepts_custom_assembler(tmp_path):
    from coda import Agent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    a = default_assembler()
    a.add(Section("marker", render=lambda c: "## CUSTOM SECTION FOR TEST"))
    agent = Agent(
        model="m",
        llm=MockLLMClient(responses=[CompletionResponse(text="done.")]),
        sandbox=Sandbox(root=tmp_path),
        prompt_assembler=a,
    )
    assert "## CUSTOM SECTION FOR TEST" in agent.system_prompt
