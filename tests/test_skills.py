"""Tests for the skills library — agent-curated reusable workflows."""

from __future__ import annotations

from pathlib import Path

from coda.skills import (
    Skill,
    inject_skills,
    load_skills,
    make_save_skill,
    save_skill,
)


# --- save_skill ----------------------------------------------------------


def test_save_skill_writes_well_formed_file(tmp_path: Path):
    code = '''def brief_morning(vips: list[str]) -> str:
    """Build a morning brief from VIP messages."""
    return ", ".join(vips)
'''
    res = save_skill(code=code, directory=tmp_path)
    assert res["ok"] is True
    assert res["name"] == "brief_morning"
    saved = Path(res["path"])
    assert saved.exists()
    body = saved.read_text()
    assert body.startswith('"""Build a morning brief from VIP messages."""')
    assert "def brief_morning(vips: list[str]) -> str:" in body


def test_save_skill_rejects_syntax_error(tmp_path: Path):
    res = save_skill(code="def broken(: pass", directory=tmp_path)
    assert res["ok"] is False
    assert "SyntaxError" in res["error"]
    assert list(tmp_path.iterdir()) == []  # no file written


def test_save_skill_rejects_zero_or_multiple_functions(tmp_path: Path):
    assert save_skill(code="x = 1", directory=tmp_path)["ok"] is False
    res = save_skill(
        code="def one(): pass\ndef two(): pass\n", directory=tmp_path
    )
    assert res["ok"] is False
    assert "expected exactly one" in res["error"]


def test_save_skill_rejects_invalid_name(tmp_path: Path):
    res = save_skill(code="def BadName(): pass", directory=tmp_path)
    assert res["ok"] is False
    assert "snake_case" in res["error"]


def test_save_skill_rejects_reserved_names(tmp_path: Path):
    for reserved in ("save_skill", "read", "bash"):
        res = save_skill(code=f"def {reserved}(): pass", directory=tmp_path)
        assert res["ok"] is False
        assert "built-in" in res["error"]


def test_save_skill_overwrites_existing(tmp_path: Path):
    save_skill(code='def f(): return 1', directory=tmp_path)
    save_skill(code='def f(): return 2', directory=tmp_path)
    body = (tmp_path / "f.py").read_text()
    assert "return 2" in body
    assert "return 1" not in body


# --- load_skills ---------------------------------------------------------


def test_load_skills_returns_callables(tmp_path: Path):
    save_skill(code='def double(x: int) -> int:\n    """Double a number."""\n    return x * 2\n', directory=tmp_path)
    skills = load_skills(tmp_path)
    assert "double" in skills
    s = skills["double"]
    assert isinstance(s, Skill)
    assert s.docstring == "Double a number."
    assert "x: int" in s.signature
    assert s.fn(5) == 10


def test_load_skills_skips_underscore_files(tmp_path: Path):
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "_private.py").write_text("def _private(): pass")
    save_skill(code="def real(): return 42", directory=tmp_path)
    skills = load_skills(tmp_path)
    assert list(skills.keys()) == ["real"]


def test_load_skills_skips_broken_files(tmp_path: Path):
    (tmp_path / "broken.py").write_text("this is not valid python(")
    save_skill(code="def working(): return 1", directory=tmp_path)
    skills = load_skills(tmp_path)
    assert "working" in skills
    assert "broken" not in skills


def test_load_skills_missing_dir_returns_empty(tmp_path: Path):
    assert load_skills(tmp_path / "does_not_exist") == {}


# --- inject_skills -------------------------------------------------------


def test_inject_skills_rebinds_globals(tmp_path: Path):
    # A skill that calls a name that doesn't exist in its module — only
    # in the sandbox globals we're about to inject it into.
    save_skill(
        code='def call_helper(x: int) -> int:\n    """Use a sandbox helper."""\n    return helper(x) + 1\n',
        directory=tmp_path,
    )
    skills = load_skills(tmp_path)

    sandbox_globals: dict = {"helper": lambda x: x * 100}
    inject_skills(sandbox_globals, skills)

    assert "call_helper" in sandbox_globals
    # The skill body references `helper` — which lives ONLY in sandbox_globals,
    # not in the skill's source file. Rebinding __globals__ must make this work.
    assert sandbox_globals["call_helper"](3) == 301


# --- make_save_skill -----------------------------------------------------


def test_make_save_skill_binds_directory(tmp_path: Path):
    fn = make_save_skill(tmp_path)
    res = fn(code='def hi(): return "hi"')
    assert res["ok"] is True
    assert Path(res["path"]).parent == tmp_path.resolve()


# --- Agent integration --------------------------------------------------


def test_agent_loads_and_injects_skills(tmp_path: Path):
    from coda import Agent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    save_skill(
        code='def shout(s: str) -> str:\n    """Uppercase echo."""\n    return s.upper()\n',
        directory=skills_dir,
    )

    mock = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nprint(shout('hello'))\n```"),
            CompletionResponse(text="done."),
        ]
    )
    sandbox = Sandbox(root=tmp_path)
    agent = Agent(
        model="m",
        llm=mock,
        sandbox=sandbox,
        skills_dir=skills_dir,
    )
    assert "shout" in agent.skills
    result = agent.run("use shout")
    assert "HELLO" in result.executions[0].stdout


def test_agent_save_skill_then_next_run_picks_it_up(tmp_path: Path):
    from coda import Agent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    skills_dir = tmp_path / "skills"

    # Run 1 — agent saves a skill
    mock1 = MockLLMClient(
        responses=[
            CompletionResponse(
                text='```python\nresult = save_skill(code="""\ndef triple(x: int) -> int:\n    \\"\\"\\"Triple a number.\\"\\"\\"\n    return x * 3\n""")\nprint(result)\n```'
            ),
            CompletionResponse(text="done."),
        ]
    )
    Agent(
        model="m",
        llm=mock1,
        sandbox=Sandbox(root=tmp_path / "ws1"),
        skills_dir=skills_dir,
    ).run("save triple")

    saved_files = list(skills_dir.glob("*.py"))
    assert any(f.name == "triple.py" for f in saved_files)

    # Run 2 — fresh agent should load + use the skill
    mock2 = MockLLMClient(
        responses=[
            CompletionResponse(text="```python\nprint(triple(7))\n```"),
            CompletionResponse(text="21."),
        ]
    )
    agent2 = Agent(
        model="m",
        llm=mock2,
        sandbox=Sandbox(root=tmp_path / "ws2"),
        skills_dir=skills_dir,
    )
    assert "triple" in agent2.skills
    result = agent2.run("use the saved skill")
    assert "21" in result.executions[0].stdout


def test_prompt_mentions_save_skill(tmp_path: Path):
    from coda import Agent
    from coda.llm import CompletionResponse, MockLLMClient
    from coda.sandbox import Sandbox

    agent = Agent(
        model="m",
        llm=MockLLMClient(responses=[CompletionResponse(text="done.")]),
        sandbox=Sandbox(root=tmp_path),
        skills_dir=tmp_path / "skills",
    )
    p = agent.system_prompt
    assert "save_skill" in p
    assert "skills" in p.lower()
    # The prompt MUST tell the agent it can save skills
    assert "crystallize" in p.lower() or "save_skill(code" in p
