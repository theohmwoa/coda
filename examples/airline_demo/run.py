"""Real-LLM airline IRROPS demo — exercises ctx + ContextPolicy.

Runs an Agent against Anthropic via the Agent SDK on a multi-step task that
naturally produces enough lookups to overflow a small context budget. The
budget is set deliberately small (12K tokens against a 200K-window model) so
the threshold reminder fires after a few turns and we can see how the model
responds to it.

Usage:
    python -m examples.airline_demo.run

What it captures:
    - Full JSONL trace at examples/airline_demo/.runs/run_<timestamp>.jsonl
    - Markdown report at examples/airline_demo/.runs/run_<timestamp>.md
      summarizing turns, ctx evolution, reminder firings, and final answer.

The mock airline data lives in examples/airline_demo/mock_data.py. The
primitives the agent calls live in examples/airline_demo/primitives.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Make `from airline_demo...` imports work whether run from the repo root
# (python -m examples.airline_demo.run) or directly.
_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from coda import Agent, ContextPolicy, HookRegistry, Sandbox, TraceWriter
from coda.llm import ClaudeAgentSDKClient

from airline_demo.primitives import Bookings, Flights, Passengers, Policies, Tickets


# --- task ---


TASK = """\
Passenger Mira Chen called about flight UA1234 on 2026-05-19 being cancelled.
Her booking is B-77001.

Your job is to produce a complete handling brief for the agent who will call
her back. The brief must include:

1. Who she is (tier, year joined, YTD spend) and a one-paragraph summary
   of her support history that calls out anything operationally relevant.
2. The exact cause of the cancellation and the operational situation
   (including whether her downstream connection on AA9876 is also affected
   and how).
3. The tier-specific compensation she is entitled to under the involuntary
   cancellation policy — pulled from the policy document, with the
   section number cited.
4. The compensation she is entitled to under the connecting-itinerary
   provisions of the policy, again with the section number.
5. A draft script the agent can read aloud, in plain language, that
   summarizes the situation and offers her the options the policy entitles
   her to. Six sentences or fewer.

Constraints:
- All facts must come from the data primitives (passengers, bookings,
  flights, policies, tickets); do not invent numbers.
- This is a small context budget (12,000 tokens). Use the ctx namespace
  thoughtfully — keep large records in plain globals and expose only
  what you need to reason about in ctx. The harness will warn you with a
  system-reminder if you get close to the budget; respond by dropping
  ctx entries you no longer need or summarizing them.

Primitives available in your sandbox globals (all are typed Python objects):
  passengers.get(id)
  passengers.find(email=, name_contains=)
  bookings.get(id)
  bookings.find(passenger_id=, flight_id=, date=, status=)
  flights.get(flight_id, date)
  flights.find(date=, operator=, status=, origin=, destination=)
  tickets.find(passenger_id=, category=, since=)
  policies.list_available()
  policies.lookup(name)
  policies.get_section(name, section_number)

Produce the brief as a single final assistant message (no code block) when
you're done.
"""


# --- runner ---


@dataclass
class RunSummary:
    timestamp: str
    turns: int
    input_tokens: int
    output_tokens: int
    ctx_entries_final: int
    ctx_dropped_final: int
    reminder_fires: int
    final_text: str
    trace_path: Path
    report_path: Path


def run_demo(*, model: str = "claude-sonnet-4-6", budget_tokens: int = 12_000) -> RunSummary:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = _THIS_DIR / ".runs"
    runs_dir.mkdir(exist_ok=True)
    trace_path = runs_dir / f"run_{timestamp}.jsonl"
    report_path = runs_dir / f"run_{timestamp}.md"

    hooks = HookRegistry()
    sandbox = Sandbox(hooks=hooks)
    # Inject airline primitives as named objects — same shape as MCP servers
    # would appear from the agent's perspective.
    sandbox.inject("passengers", Passengers())
    sandbox.inject("bookings", Bookings())
    sandbox.inject("flights", Flights())
    sandbox.inject("policies", Policies())
    sandbox.inject("tickets", Tickets())

    policy = ContextPolicy(
        budget_tokens=budget_tokens,
        threshold_fraction=0.70,
        cooldown_turns=1,
    )

    llm = ClaudeAgentSDKClient(
        permission_mode="bypassPermissions",
        cwd=str(_THIS_DIR),
    )

    agent = Agent(
        model=model,
        llm=llm,
        sandbox=sandbox,
        hooks=hooks,
        context_policy=policy,
        max_turns=12,
    )

    reminder_fires: list[dict] = []
    hooks.on(
        "context_reminder_emitted",
        lambda e: reminder_fires.append(dict(e.payload)),
    )

    with TraceWriter(trace_path).attach(hooks):
        result = agent.run(TASK)

    summary = RunSummary(
        timestamp=timestamp,
        turns=result.turns,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        ctx_entries_final=len(sandbox.ctx),
        ctx_dropped_final=len(sandbox.ctx._dropped()),
        reminder_fires=len(reminder_fires),
        final_text=result.text,
        trace_path=trace_path,
        report_path=report_path,
    )

    _write_report(summary, sandbox, reminder_fires)
    return summary


def _write_report(summary: RunSummary, sandbox: Sandbox, reminder_fires: list[dict]) -> None:
    lines: list[str] = [
        f"# Airline demo run — {summary.timestamp}",
        "",
        "## Top-line numbers",
        f"- Turns: **{summary.turns}**",
        f"- Input tokens (sum across turns): **{summary.input_tokens:,}**",
        f"- Output tokens (sum across turns): **{summary.output_tokens:,}**",
        f"- ctx live entries at end: **{summary.ctx_entries_final}**",
        f"- ctx dropped entries at end: **{summary.ctx_dropped_final}**",
        f"- Threshold reminders fired: **{summary.reminder_fires}**",
        "",
    ]
    if reminder_fires:
        lines.append("## Reminders fired")
        for r in reminder_fires:
            lines.append(
                f"- turn {r['turn']}: input_tokens={r['input_tokens']:,} / "
                f"budget={r['budget_tokens']:,}, "
                f"ctx_entries={r['ctx_entries']}, dropped={r['ctx_dropped']}"
            )
        lines.append("")

    lines.append("## Final ctx state")
    if sandbox.ctx._entries():
        for name, entry in sandbox.ctx._entries().items():
            preview = repr(entry.value)
            if len(preview) > 200:
                preview = preview[:200] + "…"
            src = entry.source_expr or "(no source)"
            lines.append(f"- `{name}` (turn {entry.set_at_turn}) — source: `{src}`")
            lines.append(f"  - value preview: `{preview}`")
    else:
        lines.append("(empty)")
    lines.append("")

    if sandbox.ctx._dropped():
        lines.append("## Dropped (requery hints)")
        for name, entry in sandbox.ctx._dropped().items():
            lines.append(f"- `{name}` — requery: `{entry.source_expr or '(no source)'}`")
        lines.append("")

    lines.append("## Final assistant message")
    lines.append("")
    lines.append("```")
    lines.append(summary.final_text)
    lines.append("```")
    lines.append("")
    lines.append(f"_Full JSONL trace: `{summary.trace_path.name}`_")
    summary.report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summary = run_demo()
    print(f"\nDone. Turns: {summary.turns}. Reminders: {summary.reminder_fires}.")
    print(f"Report: {summary.report_path}")
    print(f"Trace:  {summary.trace_path}")
