"""Tests for the schema-aware AST linter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from coda.lint import build_schema_table, collect_literal_fields, lint, lint_dicts


class Triage(BaseModel):
    importance: Literal["critical", "high", "normal", "low"]
    category: Literal["bug", "feature", "docs"]
    needs_reply: bool
    reasoning: str


class Other(BaseModel):
    severity: Literal["minor", "moderate", "severe"]


# --- schema introspection ---


def test_collect_literal_fields_finds_string_literals():
    fields = collect_literal_fields(Triage)
    assert fields["importance"] == {"critical", "high", "normal", "low"}
    assert fields["category"] == {"bug", "feature", "docs"}
    # needs_reply (bool) and reasoning (str) aren't Literal — excluded
    assert "needs_reply" not in fields
    assert "reasoning" not in fields


def test_build_schema_table_unions_across_models():
    t = build_schema_table({"triage": Triage, "other": Other})
    assert t["importance"] == {"critical", "high", "normal", "low"}
    assert t["severity"] == {"minor", "moderate", "severe"}


def test_build_schema_table_empty_when_no_schemas():
    assert build_schema_table({}) == {}


# --- lint: positive cases (should flag) ---


def test_attribute_compare_to_bad_literal_is_flagged():
    code = """
def f(t):
    if t.importance == "medium":
        return True
    return False
"""
    findings = lint(code, schemas={"triage": Triage})
    assert len(findings) == 1
    assert findings[0].field == "importance"
    assert findings[0].value == "medium"
    assert findings[0].allowed == ["critical", "high", "low", "normal"]


def test_subscript_compare_to_bad_literal_is_flagged():
    code = """
def f(d):
    return [x for x in d if x["importance"] == "medium"]
"""
    findings = lint(code, schemas={"triage": Triage})
    assert len(findings) == 1
    assert findings[0].field == "importance"


def test_not_eq_also_flagged():
    code = '''def f(t):
    return t.importance != "spam"
'''
    findings = lint(code, schemas={"triage": Triage})
    assert len(findings) == 1
    assert "!=" in findings[0].message


def test_multiple_findings_in_one_function():
    code = """
def f(t):
    if t.importance == "medium" and t.category == "wontfix":
        pass
"""
    findings = lint(code, schemas={"triage": Triage})
    assert len(findings) == 2
    fields = {f.field for f in findings}
    assert fields == {"importance", "category"}


# --- lint: negative cases (should NOT flag) ---


def test_valid_literal_value_is_not_flagged():
    code = """
def f(t):
    return t.importance == "high"
"""
    assert lint(code, schemas={"triage": Triage}) == []


def test_unknown_field_name_is_not_flagged():
    code = """
def f(t):
    return t.unknown_field == "medium"
"""
    assert lint(code, schemas={"triage": Triage}) == []


def test_no_schemas_returns_empty():
    code = '''def f(t):
    return t.importance == "medium"
'''
    assert lint(code, schemas={}) == []
    assert lint(code) == []


def test_unparseable_code_returns_empty():
    assert lint("def broken(: pass", schemas={"triage": Triage}) == []


def test_non_string_constant_comparison_ignored():
    code = """
def f(t):
    return t.importance == 42
"""
    assert lint(code, schemas={"triage": Triage}) == []


def test_dict_with_literal_field_name_still_flagged_known_false_positive():
    """Documented limitation: we can't tell from AST alone that a dict
    is a local construction unrelated to a sub-agent. This is fine —
    warnings are heuristic and the agent decides."""
    code = """
def f():
    d = {"importance": "medium"}
    return d["importance"] == "medium"
"""
    findings = lint(code, schemas={"triage": Triage})
    # Yes, false positive — and we accept it.
    assert len(findings) == 1


# --- json variant ---


def test_lint_dicts_returns_serializable():
    code = '''def f(t):
    return t.importance == "medium"
'''
    rows = lint_dicts(code, schemas={"triage": Triage})
    assert isinstance(rows, list)
    assert rows[0]["field"] == "importance"
    assert rows[0]["value"] == "medium"
    assert isinstance(rows[0]["allowed"], list)


# --- end-to-end via save_skill ---


def test_save_skill_attaches_warnings_when_skill_has_bad_literal(tmp_path):
    from coda.skills import make_save_skill

    save = make_save_skill(tmp_path, schemas={"triage": Triage})
    code = '''def bad_filter(items: list) -> list:
    """Filter for medium-importance items."""
    return [i for i in items if i["importance"] == "medium"]
'''
    result = save(code=code)
    assert result["ok"] is True
    assert "warnings" in result
    assert len(result["warnings"]) == 1
    assert result["warnings"][0]["field"] == "importance"
    assert result["warnings"][0]["value"] == "medium"


def test_save_skill_clean_code_has_no_warnings_key(tmp_path):
    from coda.skills import make_save_skill

    save = make_save_skill(tmp_path, schemas={"triage": Triage})
    code = '''def good_filter(items: list) -> list:
    """Filter for high-importance items."""
    return [i for i in items if i["importance"] == "high"]
'''
    result = save(code=code)
    assert result["ok"] is True
    assert "warnings" not in result


def test_save_skill_without_schemas_skips_lint(tmp_path):
    from coda.skills import make_save_skill

    save = make_save_skill(tmp_path)  # no schemas
    code = '''def f(t):
    return t.importance == "medium"
'''
    result = save(code=code)
    assert result["ok"] is True
    assert "warnings" not in result


# --- agent integration ---


def test_agent_run_surfaces_lint_in_execution_feedback(tmp_path):
    """The lint runs PER TURN — not just at save_skill — so the agent
    sees `⚠️ Lint warnings` in its execution result and can fix in the
    next code block, before the bad code ever ends up in a skill."""
    from typing import Literal
    from pydantic import BaseModel

    from coda import Agent, subagent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    class Verdict(BaseModel):
        importance: Literal["critical", "high", "normal", "low"]

    @subagent()
    def triage_email(email: dict) -> Verdict:
        """."""

    # Agent emits code with a bad literal comparison. Lint should
    # detect it and surface in the feedback message.
    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text='''```python
items = [{"importance": "high"}, {"importance": "normal"}]
medium = [i for i in items if i["importance"] == "medium"]
print("count:", len(medium))
```'''
            ),
            CompletionResponse(text="done — I see the warning."),
        ]
    )
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=Sandbox(root=tmp_path),
        subagents=[triage_email],
    )
    result = agent.run("write code with a dead branch")
    assert result.turns == 2
    # The user-feedback message back to the model must contain the
    # lint warning section. messages = [user_prompt, assistant, feedback, assistant]
    feedback = result.messages[2].content
    assert "Lint warnings" in feedback
    assert "importance" in feedback
    assert "medium" in feedback
    assert "critical" in feedback  # the allowed-values list is shown


def test_agent_run_no_lint_warnings_when_clean(tmp_path):
    from typing import Literal
    from pydantic import BaseModel

    from coda import Agent, subagent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    class Verdict(BaseModel):
        importance: Literal["critical", "high", "normal", "low"]

    @subagent()
    def triage_email(email: dict) -> Verdict:
        """."""

    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text='```python\nitems = [{"importance": "high"}]\nhigh = [i for i in items if i["importance"] == "high"]\nprint(len(high))\n```'
            ),
            CompletionResponse(text="done."),
        ]
    )
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=Sandbox(root=tmp_path),
        subagents=[triage_email],
    )
    result = agent.run("clean code")
    feedback = result.messages[2].content
    assert "Lint warnings" not in feedback


def test_lint_runs_without_skills_dir(tmp_path):
    """Lint surfaces in execution feedback even when no skills_dir is
    configured — it's a sub-agent-schema check, not a skill check."""
    from typing import Literal
    from pydantic import BaseModel

    from coda import Agent, subagent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    class V(BaseModel):
        kind: Literal["a", "b"]

    @subagent()
    def cls(x: dict) -> V:
        """."""

    mock = MockLLMClient(
        responses=[
            CompletionResponse(text='```python\nx = {"kind": "z"}\nprint(x["kind"] == "z")\n```'),
            CompletionResponse(text="done."),
        ]
    )
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=Sandbox(root=tmp_path),
        subagents=[cls],
        # NO skills_dir!
    )
    result = agent.run("emit code with a dead branch")
    feedback = result.messages[2].content
    assert "Lint warnings" in feedback


def test_agent_exposes_lint_as_primitive(tmp_path):
    from typing import Literal
    from pydantic import BaseModel

    from coda import Agent, subagent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    class V(BaseModel):
        kind: Literal["a", "b", "c"]

    @subagent()
    def cls(x: dict) -> V:
        """."""

    mock = MockLLMClient(
        responses=[
            CompletionResponse(
                text='''```python
findings = lint('def f(t):\\n    return t.kind == "z"\\n')
print("count:", len(findings))
print("first kind:", findings[0]["kind"] if findings else "none")
```'''
            ),
            CompletionResponse(text="done."),
        ]
    )
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=Sandbox(root=tmp_path),
        subagents=[cls],
        skills_dir=tmp_path / "skills",
    )
    result = agent.run("lint a snippet")
    out = result.executions[0].stdout
    assert "count: 1" in out
    assert "first kind: literal_mismatch" in out
