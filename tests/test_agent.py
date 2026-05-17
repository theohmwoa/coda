"""End-to-end tests for the coda agent loop with MockLLMClient."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from coda.agent import Agent, _extract_code_block, _extract_code_blocks, _format_execution
from coda.hooks import HookRegistry
from coda.llm import CompletionResponse, MockLLMClient, ToolCall
from coda.sandbox import ExecutionResult, Sandbox
from coda.subagents import subagent
from coda.tools import tool


def test_extract_code_block_python_fence():
    text = "Let me try:\n```python\nx = 1\nprint(x)\n```\nDone."
    assert _extract_code_block(text) == "x = 1\nprint(x)"


def test_extract_code_block_py_fence():
    text = "```py\nls('.')\n```"
    assert _extract_code_block(text) == "ls('.')"


def test_extract_code_block_unspecified_language():
    text = "```\nprint('hi')\n```"
    assert _extract_code_block(text) == "print('hi')"


def test_extract_code_block_none():
    text = "All done — no code needed."
    assert _extract_code_block(text) is None


def test_format_execution_renders_streams():
    r = ExecutionResult(stdout="hello\n", stderr="warn\n")
    out = _format_execution(r)
    assert "<stdout>" in out
    assert "hello" in out
    assert "<stderr>" in out
    assert "warn" in out


def test_format_execution_renders_error():
    r = ExecutionResult(error="ZeroDivisionError(...)", error_traceback="tb")
    out = _format_execution(r)
    assert "<error>ZeroDivisionError" in out
    assert "tb" in out


def test_loop_terminates_when_no_code_block(tmp_path):
    mock = MockLLMClient(
        responses=[CompletionResponse(text="Nothing to do here.")]
    )
    agent = Agent(model="claude-test", llm=mock, sandbox=Sandbox(root=tmp_path))
    result = agent.run("hi")
    assert result.turns == 1
    assert result.text == "Nothing to do here."
    assert result.executions == []


def test_loop_runs_one_code_block_and_finishes(tmp_path):
    (tmp_path / "x.txt").write_text("content")
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="Listing:\n```python\nprint(ls('.'))\n```"),
            CompletionResponse(text="Done — file is `x.txt`."),
        ]
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(model="claude-test", llm=mock, sandbox=sandbox)
    result = agent.run("list the files")
    assert result.turns == 2
    assert result.text == "Done — file is `x.txt`."
    assert len(result.executions) == 1
    assert "x.txt" in result.executions[0].stdout


def test_multi_block_response_runs_all_blocks(tmp_path):
    """Backends that wrap an internal agent loop emit several fenced blocks
    per response; coda must run every one of them, not just the first."""
    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text="""I'll do this in two parts.

```python
x = 7
print('part one done')
```

Now the second part:

```python
y = x * 6
print(f'part two: {y}')
```
"""
            ),
            CompletionResponse(text="all done."),
        ]
    )
    agent = Agent(model="m", llm=mock, sandbox=Sandbox(root=tmp_path))
    result = agent.run("two parts")
    assert result.turns == 2
    # Both blocks must have written stdout
    assert "part one done" in result.executions[0].stdout
    assert "part two: 42" in result.executions[1].stdout


def test_internal_next_block_separator_splits_a_fence():
    """A fenced block containing `# --- next block ---` separator comments
    must split into multiple segments — otherwise a SyntaxError in one
    sub-segment voids them all (framework_compare_020702 failure mode,
    diagnosed by the self_debug demo)."""
    text = """```python
print('a')
# --- next block ---
print('b')
# --- next block ---
print('c')
```"""
    blocks = _extract_code_blocks(text)
    assert len(blocks) == 3
    assert "print('a')" in blocks[0]
    assert "print('b')" in blocks[1]
    assert "print('c')" in blocks[2]


def test_internal_next_block_separator_is_case_and_dash_tolerant():
    text = """```python
x = 1
#--- NEXT BLOCK ---
y = 2
#  ---- next block -----
z = 3
```"""
    blocks = _extract_code_blocks(text)
    assert len(blocks) == 3


def test_internal_separator_only_inside_fence_is_a_split(tmp_path):
    """End-to-end: a fenced block with an internal SyntaxError after a
    separator must NOT void the segment before it."""
    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text="""```python
write('survived.txt', 'hello')
print('block 1 ok')
# --- next block ---
def broken(:
    pass
# --- next block ---
write('never_ran.txt', 'should not exist')
```"""
            ),
            CompletionResponse(text="done."),
        ]
    )
    agent = Agent(model="m", llm=mock, sandbox=Sandbox(root=tmp_path))
    result = agent.run("mixed")
    assert (tmp_path / "survived.txt").exists()
    assert not (tmp_path / "never_ran.txt").exists()
    feedback = result.messages[2].content
    assert "SyntaxError" in feedback


def test_syntax_error_in_one_block_does_not_void_prior_blocks(tmp_path):
    """The framework_compare bug: one syntax-erroring block must not erase
    the side effects (file writes, prints, state) of earlier successful
    blocks in the same turn."""
    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text="""```python
write('survived.txt', 'hello')
print('block 1 ok')
```

```python
def broken(:
    pass
```

```python
write('never_ran.txt', 'should not exist')
```
"""
            ),
            CompletionResponse(text="done."),
        ]
    )
    agent = Agent(model="m", llm=mock, sandbox=Sandbox(root=tmp_path))
    result = agent.run("mixed")
    # Block 1 produced its side effect (the file)
    assert (tmp_path / "survived.txt").exists()
    assert (tmp_path / "survived.txt").read_text() == "hello"
    # Block 3 never ran
    assert not (tmp_path / "never_ran.txt").exists()
    # Block 2 surfaced as an error
    # messages: [0]=user prompt, [1]=assistant response, [2]=execution feedback
    feedback = result.messages[2].content
    assert "SyntaxError" in feedback
    assert "blocks_run='2'" in feedback
    assert "blocks_emitted='3'" in feedback


def test_persistent_state_across_turns(tmp_path):
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nx = 21\n```"),
            CompletionResponse(text="```python\nprint(x * 2)\n```"),
            CompletionResponse(text="Result is 42."),
        ]
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(model="m", llm=mock, sandbox=sandbox)
    result = agent.run("compute 21*2")
    assert result.turns == 3
    assert result.executions[1].stdout.strip() == "42"


def test_execution_error_fed_back_to_model(tmp_path):
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\n1/0\n```"),
            CompletionResponse(text="Saw the divide-by-zero — stopping."),
        ]
    )
    agent = Agent(model="m", llm=mock, sandbox=Sandbox(root=tmp_path))
    result = agent.run("divide by zero")
    assert result.turns == 2
    last_user_msg = result.messages[-2].content
    assert "ZeroDivisionError" in last_user_msg


def test_inline_tool_callable_from_sandbox(tmp_path):
    @tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nprint(double(21))\n```"),
            CompletionResponse(text="Got 42."),
        ]
    )
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=Sandbox(root=tmp_path),
        tools=[double],
    )
    result = agent.run("use the tool")
    assert result.executions[0].stdout.strip() == "42"


def test_max_turns_safeguard(tmp_path):
    mock = MockLLMClient(
        responder=lambda call: CompletionResponse(text="```python\nx = 1\n```")
    )
    agent = Agent(model="m", llm=mock, sandbox=Sandbox(root=tmp_path), max_turns=3)
    result = agent.run("infinite loop")
    assert result.turns == 3
    assert len(result.executions) == 3


def test_subagent_callable_from_sandbox(tmp_path):
    class Verdict(BaseModel):
        score: int
        confidence: Literal["low", "medium", "high"]

    @subagent()
    def rate(item: dict) -> Verdict:
        """Rate one item."""

    main_responses = [
        CompletionResponse(
            text="```python\nv = rate(item={'name': 'x'})\nprint(v.score, v.confidence)\n```"
        ),
        CompletionResponse(text="Saw score 9 — done."),
    ]
    subagent_response = CompletionResponse(
        text="",
        tool_calls=[
            ToolCall(id="t1", name="return_rate", input={"score": 9, "confidence": "high"})
        ],
        stop_reason="tool_use",
    )

    queue = list(main_responses)
    sub_queue = [subagent_response]

    def responder(call):
        if call.tools and call.tool_choice and call.tool_choice.name == "return_rate":
            return sub_queue.pop(0)
        return queue.pop(0)

    mock = MockLLMClient(responder=responder)
    agent = Agent(model="m", llm=mock, sandbox=Sandbox(root=tmp_path), subagents=[rate])
    result = agent.run("rate the item")
    assert result.turns == 2
    assert "9 high" in result.executions[0].stdout


def test_hooks_observe_a_full_run(tmp_path):
    hooks = HookRegistry()
    captured = hooks.capture()
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nx = 1\n```"),
            CompletionResponse(text="done."),
        ]
    )
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=Sandbox(root=tmp_path, hooks=hooks),
        hooks=hooks,
    )
    agent.run("trivial")
    types = [e.type for e in captured]
    assert "llm_request" in types
    assert "llm_response" in types
    assert "code_emitted" in types
    assert "code_executed" in types
    assert "turn_complete" in types


def test_system_prompt_lists_inline_tools_and_subagents(tmp_path):
    @tool
    def search(q: str) -> list:
        """Search for things."""
        return []

    class Verdict(BaseModel):
        ok: bool

    @subagent()
    def check(item: dict) -> Verdict:
        """Check one item."""

    mock = MockLLMClient(responses=[CompletionResponse(text="done.")])
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=Sandbox(root=tmp_path),
        tools=[search],
        subagents=[check],
    )
    prompt = agent.system_prompt
    assert "ls(" in prompt and "grep(" in prompt
    assert "search" in prompt
    assert "check" in prompt
    assert "Pydantic" in prompt or "typed" in prompt.lower()
