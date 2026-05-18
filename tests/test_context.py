from __future__ import annotations

import pytest

from coda.context import (
    ContextPolicy,
    Ctx,
    _parse_assignment_rhs,
    build_context_reminder,
    build_print_habit_reminder,
    ctx_token_estimate,
    estimate_tokens,
    render_ctx,
    render_ctx_delta,
    render_value,
)


# --- basic attribute mechanics -----------------------------------------------


def test_set_and_get_attribute():
    ctx = Ctx()
    ctx.x = 42
    assert ctx.x == 42


def test_get_missing_attribute_raises():
    ctx = Ctx()
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = ctx.missing


def test_delete_moves_to_dropped():
    ctx = Ctx()
    ctx.x = 42
    del ctx.x
    assert "x" not in ctx._entries()
    assert "x" in ctx._dropped()


def test_delete_missing_raises():
    ctx = Ctx()
    with pytest.raises(AttributeError, match="no attribute 'x'"):
        del ctx.x


def test_reassign_clears_dropped_record():
    ctx = Ctx()
    ctx.x = 1
    del ctx.x
    assert "x" in ctx._dropped()
    ctx.x = 2
    assert "x" not in ctx._dropped()
    assert "x" in ctx._entries()
    assert ctx.x == 2


def test_entries_returns_snapshot_not_live_dict():
    ctx = Ctx()
    ctx.x = 1
    snap = ctx._entries()
    snap["x"] = "tampered"
    # internal state unaffected
    assert ctx.x == 1


def test_dropped_returns_snapshot_not_live_dict():
    ctx = Ctx()
    ctx.x = 1
    del ctx.x
    snap = ctx._dropped()
    snap.clear()
    assert "x" in ctx._dropped()


def test_clear_dropped_all():
    ctx = Ctx()
    ctx.x = 1
    ctx.y = 2
    del ctx.x
    del ctx.y
    assert len(ctx._dropped()) == 2
    ctx._clear_dropped()
    assert ctx._dropped() == {}


def test_clear_dropped_selective():
    ctx = Ctx()
    ctx.x = 1
    ctx.y = 2
    del ctx.x
    del ctx.y
    ctx._clear_dropped(["x"])
    assert "x" not in ctx._dropped()
    assert "y" in ctx._dropped()


# --- iteration / dunder methods ---------------------------------------------


def test_iter_yields_names_in_insertion_order():
    ctx = Ctx()
    ctx.first = 1
    ctx.second = 2
    ctx.third = 3
    assert list(ctx) == ["first", "second", "third"]


def test_len_counts_live_entries_only():
    ctx = Ctx()
    ctx.a = 1
    ctx.b = 2
    assert len(ctx) == 2
    del ctx.a
    assert len(ctx) == 1


def test_contains_checks_live_entries():
    ctx = Ctx()
    ctx.x = 1
    assert "x" in ctx
    assert "y" not in ctx
    del ctx.x
    assert "x" not in ctx


def test_dir_shows_only_user_entries():
    ctx = Ctx()
    ctx.foo = 1
    ctx.bar = 2
    names = dir(ctx)
    assert names == ["bar", "foo"]
    # internal names not exposed
    assert "_ctx_entries" not in names


def test_repr_lists_entries():
    ctx = Ctx()
    ctx.a = 1
    ctx.b = 2
    assert repr(ctx) == "<Ctx [a, b]>"


# --- reserved / private attribute names -------------------------------------


def test_underscore_attr_rejected():
    ctx = Ctx()
    with pytest.raises(AttributeError, match="reserved for Ctx internals"):
        ctx._sneaky = 1


def test_ctx_private_attrs_bypass_tracking():
    ctx = Ctx()
    # Internal API uses this. Verify it works and stays untracked.
    ctx._ctx_set_turn(5)
    assert "_ctx_current_turn" not in ctx._entries()
    assert len(ctx) == 0


# --- turn stamping ----------------------------------------------------------


def test_set_at_turn_recorded_when_turn_set():
    ctx = Ctx()
    ctx._ctx_set_turn(3)
    ctx.x = 1
    entry = ctx._entries()["x"]
    assert entry.set_at_turn == 3


def test_set_at_turn_none_by_default():
    ctx = Ctx()
    ctx.x = 1
    entry = ctx._entries()["x"]
    assert entry.set_at_turn is None


def test_turn_can_be_updated_between_assignments():
    ctx = Ctx()
    ctx._ctx_set_turn(1)
    ctx.a = "first"
    ctx._ctx_set_turn(2)
    ctx.b = "second"
    assert ctx._entries()["a"].set_at_turn == 1
    assert ctx._entries()["b"].set_at_turn == 2


# --- hook emission ----------------------------------------------------------


def test_hook_fires_on_assignment():
    events = []
    ctx = Ctx()
    ctx._ctx_set_hook(lambda t, p: events.append((t, p)))
    ctx.x = 42
    assert len(events) == 1
    t, payload = events[0]
    assert t == "ctx_assigned"
    assert payload["name"] == "x"
    assert payload["source_expr"] is None  # no provider installed
    assert payload["turn"] is None


def test_hook_fires_on_deletion():
    events = []
    ctx = Ctx()
    ctx.x = 1
    ctx._ctx_set_hook(lambda t, p: events.append((t, p)))
    del ctx.x
    assert any(e[0] == "ctx_dropped" and e[1]["name"] == "x" for e in events)


def test_hook_none_means_silent():
    ctx = Ctx()
    # No hook installed; should not raise.
    ctx.x = 1
    del ctx.x


# --- source-expression parsing (unit, no exec) ------------------------------


@pytest.mark.parametrize(
    "line,name,expected",
    [
        ("ctx.x = 42", "x", "42"),
        ("ctx.x = foo()", "x", "foo()"),
        ("ctx.high_value = [b for b in bookings if b.fare > 1000]", "high_value",
         "[b for b in bookings if b.fare > 1000]"),
        ("ctx.policy_doc = policies.get('refund')", "policy_doc", "policies.get('refund')"),
        ("    ctx.x = 1  ", "x", "1"),  # leading/trailing whitespace
        ("ctx.x = 1 + 2 * 3", "x", "1 + 2 * 3"),
        ("ctx.x = {'a': 1}", "x", "{'a': 1}"),
    ],
)
def test_parse_assignment_rhs_extracts_expression(line, name, expected):
    result = _parse_assignment_rhs(line, name)
    # ast.unparse may normalize whitespace/parens but the meaning is preserved.
    assert result == expected


@pytest.mark.parametrize(
    "line,name",
    [
        ("ctx.x += 1", "x"),  # augmented assignment
        ("ctx.x = ctx.y = 1", "x"),  # multi-target
        ("x = ctx.x = 1", "x"),  # mixed
        ("ctx['x'] = 1", "x"),  # subscript, not attribute
        ("not_an_assignment(ctx.x)", "x"),
        ("", "x"),  # empty
        ("    ", "x"),  # whitespace only
        ("def f(): pass", "x"),  # unrelated stmt
        ("ctx.y = 1", "x"),  # name mismatch
    ],
)
def test_parse_assignment_rhs_returns_none_when_unparseable(line, name):
    assert _parse_assignment_rhs(line, name) is None


def test_parse_assignment_rhs_handles_continuation_line():
    # If we only see the second line of a multi-line stmt, return None
    # rather than guessing.
    assert _parse_assignment_rhs("    b in bookings", "high_value") is None


# --- source-expression capture (with provider, no real sandbox) -------------


def test_source_capture_returns_none_without_provider():
    ctx = Ctx()  # no provider
    ctx.x = 42
    assert ctx._entries()["x"].source_expr is None


def test_source_capture_handles_provider_returning_none():
    ctx = Ctx(source_provider=lambda: None)
    ctx.x = 42
    assert ctx._entries()["x"].source_expr is None


def test_source_capture_handles_provider_raising():
    def bad_provider():
        raise RuntimeError("boom")

    ctx = Ctx(source_provider=bad_provider)
    # Must not propagate the provider error.
    ctx.x = 42
    assert ctx._entries()["x"].source_expr is None


def test_source_capture_needs_coda_frame():
    # Without a frame whose filename is "<coda>", we can't know which line
    # is currently executing, so source_expr stays None.
    ctx = Ctx(source_provider=lambda: ["ctx.x = 42"])
    ctx.x = 42  # this call originates from this test file, not <coda>
    assert ctx._entries()["x"].source_expr is None


# --- source-expression capture VIA actual exec in <coda> frame --------------


def _exec_with_ctx(ctx: Ctx, code: str) -> None:
    """Compile-and-exec helper that matches Sandbox.execute's filename.

    The provider returns the exact source lines we compiled, so source
    capture has something to look up.
    """
    lines = code.splitlines()
    ctx._ctx_set_source_provider(lambda: lines)
    compiled = compile(code, "<coda>", "exec")
    exec(compiled, {"ctx": ctx})


def test_source_capture_simple_literal():
    ctx = Ctx()
    _exec_with_ctx(ctx, "ctx.x = 42\n")
    assert ctx._entries()["x"].source_expr == "42"


def test_source_capture_function_call():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "def f(): return 99\n"
        "ctx.result = f()\n",
    )
    assert ctx._entries()["result"].source_expr == "f()"


def test_source_capture_complex_expression():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "data = [1, 2, 3, 4, 5]\n"
        "ctx.high = [x for x in data if x > 2]\n",
    )
    assert "[x for x in data if x > 2]" == ctx._entries()["high"].source_expr


def test_source_capture_across_multiple_assignments():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "ctx.first = 1\n"
        "ctx.second = 'hello'\n"
        "ctx.third = [1, 2, 3]\n",
    )
    entries = ctx._entries()
    assert entries["first"].source_expr == "1"
    assert entries["second"].source_expr == "'hello'"
    assert entries["third"].source_expr == "[1, 2, 3]"


def test_dropped_entry_keeps_source_expr_for_requery():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "def fetch(): return {'doc': True}\n"
        "ctx.policy = fetch()\n",
    )
    assert ctx._entries()["policy"].source_expr == "fetch()"
    del ctx.policy
    dropped = ctx._dropped()["policy"]
    assert dropped.source_expr == "fetch()"
    assert dropped.value is None  # tombstone, no live reference


# --- mutation through values doesn't change source_expr ---------------------


def test_mutating_value_after_set_keeps_original_source_expr():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "ctx.items = [1, 2, 3]\n"
        "ctx.items.append(4)\n",
    )
    # The .append() isn't an assignment, so source_expr stays the original RHS.
    entry = ctx._entries()["items"]
    assert entry.value == [1, 2, 3, 4]
    assert entry.source_expr == "[1, 2, 3]"


# --- value rendering --------------------------------------------------------


def test_render_value_int():
    assert render_value(42) == "42"


def test_render_value_string_is_quoted():
    # repr-style — preserves boundary visibility
    assert render_value("hello") == "'hello'"


def test_render_value_dict():
    assert render_value({"a": 1, "b": 2}) == "{'a': 1, 'b': 2}"


def test_render_value_truncates_long_repr():
    big = "x" * 5000
    out = render_value(big, max_chars=100)
    assert len(out) < 200  # head + marker
    assert "more chars" in out
    assert out.startswith("'xxxx")


def test_render_value_handles_pydantic_models():
    from pydantic import BaseModel

    class Customer(BaseModel):
        id: str
        tier: str

    out = render_value(Customer(id="abc", tier="gold"))
    assert "abc" in out
    assert "gold" in out


def test_render_value_handles_unrepresentable_object():
    class Bad:
        def __repr__(self):
            raise RuntimeError("nope")

    out = render_value(Bad())
    assert "unrepresentable" in out


# --- ctx rendering ----------------------------------------------------------


def test_render_ctx_empty_returns_empty_string():
    ctx = Ctx()
    assert render_ctx(ctx) == ""


def test_render_ctx_single_entry():
    ctx = Ctx()
    ctx.x = 42
    out = render_ctx(ctx)
    assert out.startswith("<ctx>")
    assert out.endswith("</ctx>")
    assert 'name="x"' in out
    assert "<value>42</value>" in out


def test_render_ctx_includes_token_estimate_per_entry():
    ctx = Ctx()
    ctx.short = 1
    ctx.long = "x" * 400  # ~100 tokens at chars/4
    out = render_ctx(ctx)
    assert 'tokens="1"' in out  # short entry should be minimal
    # long entry token count should be > 50
    import re

    matches = re.findall(r'name="long" tokens="(\d+)"', out)
    assert matches
    assert int(matches[0]) >= 50


def test_render_ctx_includes_set_at_turn_when_present():
    ctx = Ctx()
    ctx._ctx_set_turn(7)
    ctx.x = 1
    out = render_ctx(ctx)
    assert 'set_at_turn="7"' in out


def test_render_ctx_omits_set_at_turn_when_none():
    ctx = Ctx()
    ctx.x = 1
    out = render_ctx(ctx)
    assert "set_at_turn" not in out


def test_render_ctx_includes_source_expr_when_captured():
    ctx = Ctx()
    _exec_with_ctx(ctx, "ctx.x = 1 + 2\n")
    out = render_ctx(ctx)
    assert 'source_expr="1 + 2"' in out


def test_render_ctx_escapes_double_quotes_in_source_expr():
    ctx = Ctx()
    _exec_with_ctx(ctx, "ctx.x = \"hi\"\n")
    out = render_ctx(ctx)
    # ast.unparse uses single quotes by default; but if RHS literally
    # contained a double quote, we'd need to escape. Force the case here:
    _exec_with_ctx(ctx, 'ctx.y = {"key": "val"}\n')
    out = render_ctx(ctx)
    # Inside the source_expr attribute we should never see an unescaped "
    # that would prematurely close the attribute.
    import re

    for attr in re.findall(r'source_expr="([^"]*)"', out):
        assert '"' not in attr or "&quot;" in attr


def test_render_ctx_with_dropped_entries():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "def get_pol(): return 'mock'\n"
        "ctx.pol = get_pol()\n",
    )
    del ctx.pol
    out = render_ctx(ctx, include_dropped=True)
    assert "<dropped>" in out
    assert 'name="pol"' in out
    assert 'requery="get_pol()"' in out


def test_render_ctx_skip_dropped_by_default():
    ctx = Ctx()
    ctx.x = 1
    del ctx.x
    # Live entries are now empty; dropped has "x". Without include_dropped,
    # we should get empty string (nothing to show).
    assert render_ctx(ctx) == ""


def test_render_ctx_dropped_section_only_when_dropped_exists():
    ctx = Ctx()
    ctx.x = 1
    out = render_ctx(ctx, include_dropped=True)
    assert "<dropped>" not in out


def test_render_ctx_truncates_huge_values():
    ctx = Ctx()
    ctx.big = "x" * 100_000
    out = render_ctx(ctx, max_value_chars=500)
    # Rendered output should respect the cap (plus wrapper overhead)
    assert "more chars" in out
    assert len(out) < 2000  # nowhere near the 100K input


def test_render_ctx_preserves_insertion_order():
    ctx = Ctx()
    ctx.alpha = 1
    ctx.beta = 2
    ctx.gamma = 3
    out = render_ctx(ctx)
    assert out.find("alpha") < out.find("beta") < out.find("gamma")


# --- token estimation -------------------------------------------------------


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short_string_is_at_least_one():
    assert estimate_tokens("hi") == 1


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("hello")
    long = estimate_tokens("hello" * 100)
    assert long > short * 50


def test_ctx_token_estimate_empty():
    ctx = Ctx()
    assert ctx_token_estimate(ctx) == 0


def test_ctx_token_estimate_includes_wrapper_overhead():
    ctx = Ctx()
    ctx.x = 1
    # Should be at least a few tokens for the <ctx><entry .../></ctx> shape.
    est = ctx_token_estimate(ctx)
    assert est >= 5


def test_ctx_token_estimate_grows_with_entries():
    ctx = Ctx()
    one = 0
    ctx.first = "x" * 200
    one = ctx_token_estimate(ctx)
    ctx.second = "y" * 200
    two = ctx_token_estimate(ctx)
    # Second entry roughly doubles the cost
    assert two > one * 1.5


# --- ContextPolicy ----------------------------------------------------------


def test_policy_default_threshold_at_75pct():
    p = ContextPolicy(budget_tokens=10_000)
    assert p.threshold_tokens == 7_500


def test_policy_below_threshold_no_remind():
    p = ContextPolicy(budget_tokens=10_000, cooldown_turns=0)
    assert p.should_remind(input_tokens=5_000, turns_since_last=999) is False


def test_policy_at_threshold_reminds():
    p = ContextPolicy(budget_tokens=10_000, cooldown_turns=0)
    assert p.should_remind(input_tokens=7_500, turns_since_last=999) is True


def test_policy_cooldown_suppresses_recent_reminders():
    p = ContextPolicy(budget_tokens=10_000, cooldown_turns=2)
    # Just reminded last turn — suppressed even though over threshold
    assert p.should_remind(input_tokens=9_000, turns_since_last=0) is False
    assert p.should_remind(input_tokens=9_000, turns_since_last=1) is False
    assert p.should_remind(input_tokens=9_000, turns_since_last=2) is True


def test_policy_custom_threshold_fraction():
    p = ContextPolicy(budget_tokens=10_000, threshold_fraction=0.5, cooldown_turns=0)
    assert p.threshold_tokens == 5_000
    assert p.should_remind(input_tokens=4_999, turns_since_last=999) is False
    assert p.should_remind(input_tokens=5_000, turns_since_last=999) is True


# --- build_context_reminder -------------------------------------------------


def test_reminder_shows_utilization():
    ctx = Ctx()
    ctx.x = 1
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    assert "<system-reminder>" in out
    assert "</system-reminder>" in out
    assert "15,000" in out
    assert "20,000" in out
    assert "75%" in out


def test_reminder_lists_largest_entries_descending():
    ctx = Ctx()
    ctx.tiny = "a"
    ctx.huge = "x" * 4000
    ctx.medium = "y" * 800
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    # "huge" should appear before "medium" should appear before "tiny"
    assert out.find("huge") < out.find("medium")
    assert out.find("medium") < out.find("tiny")


def test_reminder_respects_max_largest_entries():
    ctx = Ctx()
    for i in range(10):
        setattr(ctx, f"entry_{i}", "x" * 100)
    out = build_context_reminder(
        ctx, input_tokens=15_000, budget_tokens=20_000, max_largest_entries=3
    )
    # Only 3 entry bullets should appear
    entry_lines = [line for line in out.splitlines() if line.startswith("  - entry_")]
    assert len(entry_lines) == 3


def test_reminder_includes_source_expression():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "def fetch(): return 'huge_doc' * 500\n"
        "ctx.doc = fetch()\n",
    )
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    assert "source: fetch()" in out


def test_reminder_includes_turn_when_present():
    ctx = Ctx()
    ctx._ctx_set_turn(4)
    ctx.x = "x" * 800
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    assert "set turn 4" in out


def test_reminder_includes_dropped_with_requery_hint():
    ctx = Ctx()
    _exec_with_ctx(
        ctx,
        "def get_history(): return 'mock'\n"
        "ctx.history = get_history()\n",
    )
    del ctx.history
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    assert "Recently dropped" in out
    assert "history" in out
    assert "requery: get_history()" in out


def test_reminder_handles_no_source_for_dropped():
    ctx = Ctx()
    ctx.mystery = 1  # no exec, so no source_expr
    del ctx.mystery
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    assert "source not captured" in out


def test_reminder_can_exclude_dropped():
    ctx = Ctx()
    ctx.x = 1
    del ctx.x
    ctx.y = 2
    out = build_context_reminder(
        ctx, input_tokens=15_000, budget_tokens=20_000, include_dropped=False
    )
    assert "Recently dropped" not in out


def test_reminder_works_with_empty_ctx():
    ctx = Ctx()
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    # Still gives the budget readout and the closing instruction
    assert "75%" in out
    assert "<system-reminder>" in out


def test_reminder_actionable_instruction_present():
    ctx = Ctx()
    ctx.x = "x" * 1000
    out = build_context_reminder(ctx, input_tokens=15_000, budget_tokens=20_000)
    assert "del ctx" in out  # tells the agent how to act


# --- ContextPolicy.should_warn_print ----------------------------------------


def test_should_warn_print_disabled_when_threshold_zero():
    p = ContextPolicy(large_print_chars=0)
    assert p.should_warn_print(stdout_chars=10_000_000, turns_since_last=999) is False


def test_should_warn_print_below_threshold():
    p = ContextPolicy(large_print_chars=4_000, large_print_cooldown_turns=0)
    assert p.should_warn_print(stdout_chars=3_999, turns_since_last=999) is False


def test_should_warn_print_at_threshold():
    p = ContextPolicy(large_print_chars=4_000, large_print_cooldown_turns=0)
    assert p.should_warn_print(stdout_chars=4_000, turns_since_last=999) is True


def test_should_warn_print_cooldown():
    p = ContextPolicy(large_print_chars=4_000, large_print_cooldown_turns=1)
    assert p.should_warn_print(stdout_chars=5_000, turns_since_last=0) is False
    assert p.should_warn_print(stdout_chars=5_000, turns_since_last=1) is True


# --- build_print_habit_reminder ---------------------------------------------


def test_print_habit_reminder_shows_size():
    out = build_print_habit_reminder(stdout_chars=8_000)
    assert "<system-reminder>" in out
    assert "8,000" in out
    assert "tokens" in out


def test_print_habit_reminder_includes_block_index_for_multiblock():
    out = build_print_habit_reminder(stdout_chars=5_000, block_index=3)
    assert "block 3" in out


def test_print_habit_reminder_skips_block_index_when_single_block():
    out = build_print_habit_reminder(stdout_chars=5_000, block_index=None)
    assert "block " not in out


def test_print_habit_reminder_includes_snippet():
    out = build_print_habit_reminder(
        stdout_chars=5_000,
        code_snippet="print(airline.get_user_details(user_id='abc'))",
    )
    assert "Offending line" in out
    assert "print(airline.get_user_details" in out


def test_print_habit_reminder_truncates_long_snippets():
    long = "print(" + "x" * 500 + ")"
    out = build_print_habit_reminder(stdout_chars=5_000, code_snippet=long)
    assert "..." in out
    # No raw line of 500+ x's
    assert "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in out.split("Offending line")[1] or len(out) < 1000


def test_print_habit_reminder_suggests_alternatives():
    out = build_print_habit_reminder(stdout_chars=5_000)
    # The three alternative patterns the prompt also documents.
    assert "ctx" in out
    assert "print(" in out


def test_print_habit_reminder_force_mode_has_stronger_wording():
    out = build_print_habit_reminder(
        stdout_chars=200, force=True, code_snippet="print(user)"
    )
    # Force mode uses error-shaped wording.
    assert "VIOLATION" in out or "DO NOT" in out or "❌" in out
    assert "ctx" in out
    assert "print(user)" in out


def test_should_force_ctx_off_by_default():
    p = ContextPolicy()
    assert p.should_force_ctx(has_data_print=True, turns_since_last=999) is False


def test_should_force_ctx_when_enabled():
    p = ContextPolicy(force_ctx_on_print=True, large_print_cooldown_turns=0)
    assert p.should_force_ctx(has_data_print=True, turns_since_last=999) is True
    assert p.should_force_ctx(has_data_print=False, turns_since_last=999) is False


def test_should_force_ctx_respects_cooldown():
    p = ContextPolicy(force_ctx_on_print=True, large_print_cooldown_turns=2)
    assert p.should_force_ctx(has_data_print=True, turns_since_last=0) is False
    assert p.should_force_ctx(has_data_print=True, turns_since_last=2) is True


# --- render_ctx_delta -------------------------------------------------------


def test_ctx_delta_empty_when_no_change():
    ctx = Ctx()
    ctx.x = 1
    before = ctx._entries()
    assert render_ctx_delta(ctx, before=before) == ""


def test_ctx_delta_added_entry():
    ctx = Ctx()
    before = ctx._entries()
    ctx.user_tier = "platinum"
    out = render_ctx_delta(ctx, before=before)
    assert "<ctx_delta>" in out
    assert "# added: ctx.user_tier" in out
    assert 'ctx.user_tier = "platinum"' in out


def test_ctx_delta_removed_entry():
    ctx = Ctx()
    ctx.transient = "bye"
    before = ctx._entries()
    del ctx.transient
    out = render_ctx_delta(ctx, before=before)
    # Tombstone has value=None so the "(was)" header still appears but no
    # value is rendered for it.
    assert "# removed: ctx.transient" in out


def test_ctx_delta_changed_entry():
    ctx = Ctx()
    _exec_with_ctx(ctx, "ctx.x = 1\n")
    before = ctx._entries()
    _exec_with_ctx(ctx, "ctx.x = 2\n")
    out = render_ctx_delta(ctx, before=before)
    # Reassignment changes source_expr and id() — should appear as changed
    assert "# changed: ctx.x" in out
    assert "ctx.x = 2" in out


def test_ctx_delta_multiple_changes():
    ctx = Ctx()
    ctx.keep = "static"
    ctx.drop_me = "old"
    before = ctx._entries()
    ctx.added = 99
    del ctx.drop_me
    out = render_ctx_delta(ctx, before=before)
    assert "# added: ctx.added" in out
    assert "ctx.added = 99" in out
    assert "# removed: ctx.drop_me" in out
    # keep was not touched; should NOT appear in delta
    assert "keep" not in out


def test_ctx_delta_truncates_huge_values():
    ctx = Ctx()
    before = ctx._entries()
    ctx.big = "x" * 100_000
    out = render_ctx_delta(ctx, before=before, max_value_chars=200)
    # Truncation marker present, full blob not
    assert "more chars" in out
    assert len(out) < 1_500


def test_ctx_delta_pretty_prints_nested_dict():
    """The whole point of the pretty rendering: nested structure should be
    visible across multiple indented lines, not crushed onto one line."""
    ctx = Ctx()
    before = ctx._entries()
    ctx.user = {
        "id": "abc",
        "membership": "gold",
        "reservations": ["JG7FMM", "LQ940Q"],
    }
    out = render_ctx_delta(ctx, before=before)
    # Multi-line dict rendering with indentation
    assert '"id": "abc"' in out
    assert '"membership": "gold"' in out
    # Some indentation on inner lines
    assert "\n  " in out


def test_ctx_delta_falls_back_to_pprint_for_non_json_values():
    """Sets aren't JSON-serializable; ensure we fall back gracefully."""
    ctx = Ctx()
    before = ctx._entries()
    ctx.distinct_prices = {100, 200, 300}
    out = render_ctx_delta(ctx, before=before)
    # Just confirm we got something readable, not a crash
    assert "ctx.distinct_prices" in out
    # Set repr appears (order may vary)
    assert "100" in out and "200" in out and "300" in out
