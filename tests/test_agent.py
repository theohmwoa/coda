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


# --- ctx + ContextPolicy integration -----------------------------------------


def _responder_factory(scripted):
    """Build a MockLLMClient responder that yields scripted texts in order,
    but stamps each response with a deterministic input_tokens so we can
    drive the threshold policy from the test.

    scripted: list of (text, input_tokens) tuples
    """
    iterator = iter(scripted)

    def responder(call):
        text, input_tokens = next(iterator)
        return CompletionResponse(
            text=text, input_tokens=input_tokens, output_tokens=10
        )

    return responder


def test_agent_stamps_turn_number_on_ctx_assignments(tmp_path):
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nctx.x = 1\n```", input_tokens=100),
            CompletionResponse(text="```python\nctx.y = 2\n```", input_tokens=110),
            CompletionResponse(text="Done."),
        ]
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(model="m", llm=mock, sandbox=sandbox)
    agent.run("set things")
    entries = sandbox.ctx._entries()
    assert entries["x"].set_at_turn == 1
    assert entries["y"].set_at_turn == 2


def test_no_reminder_when_policy_unset(tmp_path):
    # Without a context_policy, no reminder is ever appended, regardless
    # of how many input tokens the mock reports.
    mock = MockLLMClient(
        responder=_responder_factory(
            [
                ("```python\nctx.x = 'x' * 5000\n```", 500_000),  # absurdly large
                ("Done.", 500_000),
            ]
        )
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(model="m", llm=mock, sandbox=sandbox)
    agent.run("go")
    # Walk the messages: no system-reminder should have been added.
    for msg in mock.calls[-1].messages:
        assert "<system-reminder>" not in msg.content


def test_reminder_fires_when_threshold_crossed(tmp_path):
    from coda.context import ContextPolicy

    # First turn: agent assigns something to ctx and overshoots threshold.
    # Second turn (where we observe the reminder): the conversation it
    # sees should now contain the system-reminder appended after the
    # execution result of turn 1.
    mock = MockLLMClient(
        responder=_responder_factory(
            [
                (
                    "```python\n"
                    "def fetch_policy():\n"
                    "    return 'x' * 100\n"
                    "ctx.policy = fetch_policy()\n"
                    "```",
                    9_000,  # just over the 75% of 10K threshold
                ),
                ("All set.", 9_500),
            ]
        )
    )
    sandbox = Sandbox(root=tmp_path)
    policy = ContextPolicy(budget_tokens=10_000, cooldown_turns=0)
    agent = Agent(model="m", llm=mock, sandbox=sandbox, context_policy=policy)
    agent.run("go")

    # The second LLM call should have seen the reminder appended to the
    # user-feedback message from turn 1.
    second_call = mock.calls[1]
    user_messages = [m.content for m in second_call.messages if m.role == "user"]
    last_user = user_messages[-1]
    assert "<system-reminder>" in last_user
    # Source expression of the assignment must appear as a requery /
    # source hint somewhere in the reminder.
    assert "fetch_policy()" in last_user


def test_reminder_lists_largest_ctx_entries(tmp_path):
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                (
                    "```python\n"
                    "ctx.tiny = 1\n"
                    "ctx.huge = 'x' * 2000\n"
                    "```",
                    9_000,
                ),
                ("Done.", 9_500),
            ]
        )
    )
    sandbox = Sandbox(root=tmp_path)
    policy = ContextPolicy(budget_tokens=10_000, cooldown_turns=0)
    agent = Agent(model="m", llm=mock, sandbox=sandbox, context_policy=policy)
    agent.run("go")

    last_user = mock.calls[1].messages[-1].content
    assert "<system-reminder>" in last_user
    # Scope the ordering check to the reminder section — the ctx_delta
    # above it lists entries in insertion order, which is not what this
    # test is about.
    reminder = last_user.split("<system-reminder>", 1)[1]
    # Both names should be listed, and "huge" should come first (by size).
    assert reminder.find("huge") < reminder.find("tiny")


def test_reminder_includes_dropped_with_requery_hint(tmp_path):
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                (
                    "```python\n"
                    "def history(): return 'x' * 200\n"
                    "ctx.h = history()\n"
                    "del ctx.h\n"
                    "ctx.something_else = 'y' * 2000\n"
                    "```",
                    9_000,
                ),
                ("Done.", 9_500),
            ]
        )
    )
    sandbox = Sandbox(root=tmp_path)
    policy = ContextPolicy(budget_tokens=10_000, cooldown_turns=0)
    agent = Agent(model="m", llm=mock, sandbox=sandbox, context_policy=policy)
    agent.run("go")

    last_user = mock.calls[1].messages[-1].content
    assert "Recently dropped" in last_user
    assert "requery: history()" in last_user


def test_reminder_cooldown_suppresses_back_to_back(tmp_path):
    from coda.context import ContextPolicy

    # Three turns all over threshold. With cooldown=2, reminder fires
    # on turn 1, suppressed on turn 2, then we stop (no more code).
    mock = MockLLMClient(
        responder=_responder_factory(
            [
                ("```python\nctx.a = 'x' * 500\n```", 9_000),
                ("```python\nctx.b = 'y' * 500\n```", 9_500),
                ("Done.", 10_000),
            ]
        )
    )
    # Share hooks between sandbox and agent so we can observe events from
    # both in one place. context_reminder_emitted is fired by the Agent
    # via its own hook registry, so without sharing we'd only see the
    # sandbox-side events.
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(budget_tokens=10_000, cooldown_turns=2)
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )

    reminder_events: list[int] = []
    hooks.on(
        "context_reminder_emitted",
        lambda e: reminder_events.append(e.payload["turn"]),
    )
    agent.run("go")

    # Only the first over-threshold turn should have fired the reminder.
    assert reminder_events == [1]


def test_reminder_emits_hook_event(tmp_path):
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                ("```python\nctx.x = 'y' * 1000\n```", 9_000),
                ("Done.", 9_500),
            ]
        )
    )
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(budget_tokens=10_000, cooldown_turns=0)
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )

    captured = hooks.capture()
    agent.run("go")
    reminder_events = [e for e in captured if e.type == "context_reminder_emitted"]
    assert len(reminder_events) == 1
    payload = reminder_events[0].payload
    assert payload["turn"] == 1
    assert payload["input_tokens"] == 9_000
    assert payload["budget_tokens"] == 10_000
    assert payload["ctx_entries"] >= 1


# --- print-habit reminder integration ---------------------------------------


def test_print_habit_reminder_fires_on_big_stdout(tmp_path):
    from coda.context import ContextPolicy

    # Agent prints a big blob; print-habit threshold set low.
    mock = MockLLMClient(
        responder=_responder_factory(
            [
                (
                    "```python\n"
                    "data = 'x' * 5000\n"
                    "print(data)\n"
                    "```",
                    100,  # tiny input_tokens — threshold reminder won't fire
                ),
                ("Done.", 5_300),
            ]
        )
    )
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(
        budget_tokens=1_000_000,  # huge — so threshold reminder DOES NOT fire
        large_print_chars=2_000,
        large_print_cooldown_turns=0,
    )
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )
    captured = hooks.capture()
    agent.run("go")

    print_events = [e for e in captured if e.type == "print_habit_warning_emitted"]
    threshold_events = [e for e in captured if e.type == "context_reminder_emitted"]
    assert len(print_events) == 1
    assert len(threshold_events) == 0  # not over threshold; only print-habit fired
    payload = print_events[0].payload
    assert payload["turn"] == 1
    assert payload["stdout_chars"] >= 5_000


def test_print_habit_reminder_includes_snippet_in_feedback(tmp_path):
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                (
                    "```python\n"
                    "user = {'id': 'abc', 'data': 'x' * 4000}\n"
                    "print(user)\n"
                    "```",
                    100,
                ),
                ("Done.", 5_000),
            ]
        )
    )
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(
        budget_tokens=1_000_000,
        large_print_chars=1_000,
        large_print_cooldown_turns=0,
    )
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )
    agent.run("go")

    # The second LLM call should see the print-habit reminder appended
    # to the user-feedback message.
    last_user = mock.calls[1].messages[-1].content
    assert "<system-reminder>" in last_user
    assert "Offending line" in last_user
    assert "print(user)" in last_user


def test_print_habit_disabled_when_chars_zero(tmp_path):
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                ("```python\nprint('x' * 100_000)\n```", 100),
                ("Done.", 100_000),
            ]
        )
    )
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(budget_tokens=1_000_000, large_print_chars=0)
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )
    captured = hooks.capture()
    agent.run("go")
    assert [e for e in captured if e.type == "print_habit_warning_emitted"] == []


def test_print_habit_cooldown_suppresses(tmp_path):
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                ("```python\nprint('x' * 5000)\n```", 100),
                ("```python\nprint('y' * 5000)\n```", 200),
                ("Done.", 5_000),
            ]
        )
    )
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(
        budget_tokens=1_000_000,
        large_print_chars=2_000,
        large_print_cooldown_turns=2,
    )
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )
    captured = hooks.capture()
    agent.run("go")

    fires = [e.payload["turn"] for e in captured if e.type == "print_habit_warning_emitted"]
    # Only turn 1 fires; turn 2 suppressed by cooldown.
    assert fires == [1]


def test_force_ctx_fires_on_tiny_data_print(tmp_path):
    """Force mode fires regardless of stdout size, as long as a data print exists."""
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                (
                    "```python\n"
                    "user = {'id': 'abc'}\n"
                    "print(user)\n"  # tiny stdout but it IS a data print
                    "```",
                    100,
                ),
                ("Done.", 200),
            ]
        )
    )
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(
        budget_tokens=1_000_000,
        large_print_chars=10_000_000,   # size-based won't fire
        force_ctx_on_print=True,
        large_print_cooldown_turns=0,
    )
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )
    captured = hooks.capture()
    agent.run("go")

    fires = [e for e in captured if e.type == "print_habit_warning_emitted"]
    assert len(fires) == 1
    assert fires[0].payload["mode"] == "force"
    # And the feedback message includes the force-mode wording
    last_user = mock.calls[1].messages[-1].content
    assert "VIOLATION" in last_user
    assert "print(user)" in last_user


def test_force_ctx_ignores_pure_status_prints(tmp_path):
    """`print('done')` is not a data print and should NOT fire force mode."""
    from coda.context import ContextPolicy

    mock = MockLLMClient(
        responder=_responder_factory(
            [
                (
                    "```python\n"
                    "print('done')\n"
                    "print(\"all good\")\n"
                    "```",
                    100,
                ),
                ("Done.", 200),
            ]
        )
    )
    hooks = HookRegistry()
    sandbox = Sandbox(root=tmp_path, hooks=hooks)
    policy = ContextPolicy(
        budget_tokens=1_000_000,
        force_ctx_on_print=True,
        large_print_cooldown_turns=0,
    )
    agent = Agent(
        model="m", llm=mock, sandbox=sandbox, hooks=hooks, context_policy=policy
    )
    captured = hooks.capture()
    agent.run("go")
    assert [e for e in captured if e.type == "print_habit_warning_emitted"] == []


# --- ctx_delta in feedback --------------------------------------------------


def test_ctx_delta_added_to_feedback_on_assignment(tmp_path):
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nctx.user_tier = 'platinum'\n```"),
            CompletionResponse(text="Done."),
        ]
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(model="m", llm=mock, sandbox=sandbox)
    agent.run("go")
    # The second LLM call should see the ctx_delta in the feedback message
    last_user = mock.calls[1].messages[-1].content
    assert "<ctx_delta>" in last_user
    assert "# added: ctx.user_tier" in last_user
    assert 'ctx.user_tier = "platinum"' in last_user


def test_ctx_delta_omitted_when_no_change(tmp_path):
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nx = 1 + 1\n```"),  # plain global, no ctx
            CompletionResponse(text="Done."),
        ]
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(model="m", llm=mock, sandbox=sandbox)
    agent.run("go")
    last_user = mock.calls[1].messages[-1].content
    assert "<ctx_delta>" not in last_user


def test_ctx_delta_shows_drop(tmp_path):
    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nctx.x = 42\n```"),
            CompletionResponse(text="```python\ndel ctx.x\n```"),
            CompletionResponse(text="Done."),
        ]
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(model="m", llm=mock, sandbox=sandbox)
    agent.run("go")
    # Second user message (after turn 2) shows the drop in the delta
    last_user = mock.calls[2].messages[-1].content
    assert "<ctx_delta>" in last_user
    assert "# removed: ctx.x" in last_user
