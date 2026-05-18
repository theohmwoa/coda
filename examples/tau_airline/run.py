"""Run a tau-bench airline task against coda + Agent SDK + Opus 4.7.

Usage:
    python -m examples.tau_airline.run                  # default: task 2
    python -m examples.tau_airline.run --task 4
    python -m examples.tau_airline.run --task 2 --budget 80000

What this is and isn't:

This is NOT a faithful tau-bench scoring run. The proper tau-bench harness
includes a simulated user (an LLM playing the customer) that the agent must
talk back-and-forth with, including obtaining explicit `yes` confirmation
before any database write. We don't run that here — we feed the task's
instruction in as the initial prompt and let the agent operate.

That means we trade two things for simplicity:
  - We can't measure tau-bench's pass@1 reward, which depends on the
    simulated-user transcript matching gold actions.
  - The agent may refuse to mutate state because it can't get user
    confirmation; the prompt explicitly tells it that the user has
    pre-confirmed any action consistent with the stated preferences.

What we DO measure here:
  - tau-bench-shaped data and tools, real sizes (~5MB of JSON across
    flights/reservations/users; 6KB policy wiki).
  - The action sequence the agent emitted (compared coarsely against gold
    via name + kwargs subset match).
  - ctx behavior: assignments, drops, threshold-reminder fires.
  - Token counts (now correctly summed across uncached + cache_read +
    cache_creation, after the bug we found in run #1).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from coda import Agent, ContextPolicy, HookRegistry, Sandbox, TraceWriter
from coda.llm import ClaudeAgentSDKClient

from tau_airline.airline_primitive import RecordedAction, build_airline, get_wiki  # noqa: F401


# Make tau-bench importable here too (for TASKS).
import os
TAU_BENCH_PATH = os.environ.get("TAU_BENCH_PATH", os.path.expanduser("~/code/tau-bench"))
if TAU_BENCH_PATH not in sys.path:
    sys.path.insert(0, TAU_BENCH_PATH)
from tau_bench.envs.airline.tasks_test import TASKS  # noqa: E402


PROMPT_PREAMBLE = """\
You are an airline customer service agent. You have a single Python object
in your sandbox called `airline` that exposes the booking system. Call its
methods directly from your Python code blocks; do NOT emit JSON tool calls.

The airline policy document is given to you below in this prompt. Cite
section names when making policy decisions.

IMPORTANT for this run: the user has pre-confirmed any action that is
consistent with their stated preferences and conforms to airline policy.
Do NOT ask the user for confirmation again before executing writes — just
execute them. (Real-world deployment would gate writes on an explicit
yes; we're bypassing that here so we can test the harness behavior on
a fixed task spec.)

`airline` methods:
  Read-only:
    airline.get_user_details(user_id)
    airline.get_reservation_details(reservation_id)
    airline.list_all_airports()
    airline.search_direct_flight(origin, destination, date)   # date: YYYY-MM-DD
    airline.search_onestop_flight(origin, destination, date)
    airline.calculate(expression)   # eval an arithmetic expression
    airline.think(thought)          # no-op scratch

  Writes (database-mutating):
    airline.book_reservation(user_id, origin, destination, flight_type, cabin,
        flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
    airline.cancel_reservation(reservation_id)
    airline.update_reservation_flights(reservation_id, cabin, flights, payment_id)
    airline.update_reservation_baggages(reservation_id, total_baggages, nonfree_baggages, payment_id)
    airline.update_reservation_passengers(reservation_id, passengers)
    airline.send_certificate(user_id, amount)
    airline.transfer_to_human_agents(summary)  # terminates the run

Each call returns a parsed Python value (dict / list / int / string) — not
JSON-encoded text. Errors come back as strings beginning with "Error:".

The wiki you must follow:
---
{wiki}
---

Your task is below. Operate on this single specification; do not ask
clarifying questions. When done, end with a prose summary (no code block).
"""


def run_one(task_index: int, *, budget_tokens: int, model: str, force_ctx: bool = False) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = _THIS_DIR / ".runs"
    runs_dir.mkdir(exist_ok=True)
    trace_path = runs_dir / f"task{task_index}_{timestamp}.jsonl"
    report_path = runs_dir / f"task{task_index}_{timestamp}.md"

    task = TASKS[task_index]
    airline, actions, data, terminate = build_airline()

    hooks = HookRegistry()
    sandbox = Sandbox(hooks=hooks)
    sandbox.inject("airline", airline)

    policy = ContextPolicy(
        budget_tokens=budget_tokens,
        threshold_fraction=0.70,
        cooldown_turns=1,
        force_ctx_on_print=force_ctx,
        large_print_cooldown_turns=0 if force_ctx else 1,
    )

    llm = ClaudeAgentSDKClient(
        permission_mode="bypassPermissions",
        cwd=str(_THIS_DIR),
    )

    system_append = PROMPT_PREAMBLE.format(wiki=get_wiki())
    agent = Agent(
        model=model,
        llm=llm,
        sandbox=sandbox,
        hooks=hooks,
        context_policy=policy,
        max_turns=20,
        system_prompt_append=system_append,
    )

    reminder_fires: list[dict] = []
    hooks.on(
        "context_reminder_emitted",
        lambda e: reminder_fires.append(dict(e.payload)),
    )

    with TraceWriter(trace_path).attach(hooks):
        result = agent.run(task.instruction)

    # Compare recorded actions to gold actions, ignoring respond-only actions.
    gold = [a for a in task.actions if a.name != "respond"]
    matched, missing = _action_match(actions, gold)

    summary = {
        "task_index": task_index,
        "timestamp": timestamp,
        "model": model,
        "budget_tokens": budget_tokens,
        "turns": result.turns,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "ctx_entries_final": len(sandbox.ctx),
        "ctx_dropped_final": len(sandbox.ctx._dropped()),
        "reminder_fires": len(reminder_fires),
        "actions_recorded": len(actions),
        "gold_actions": len(gold),
        "actions_matched": matched,
        "actions_missing": missing,
        "terminated_by_transfer": terminate[0],
        "final_text": result.text,
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "ctx_dropped_hints": {
            name: entry.source_expr
            for name, entry in sandbox.ctx._dropped().items()
        },
    }

    _write_report(summary, task, actions, reminder_fires, report_path)
    return summary


def _action_match(recorded: list[RecordedAction], gold) -> tuple[int, list[str]]:
    """Coarse pass: an action is 'matched' if a recorded write with the
    same name + same key argument was emitted. For update_reservation_*,
    the reservation_id is the key argument; for book_reservation, the
    user_id; for cancel/send_certificate, reservation_id/user_id."""
    key_arg_by_tool = {
        "book_reservation": "user_id",
        "cancel_reservation": "reservation_id",
        "update_reservation_flights": "reservation_id",
        "update_reservation_baggages": "reservation_id",
        "update_reservation_passengers": "reservation_id",
        "send_certificate": "user_id",
    }
    rec_by_key: dict[tuple[str, str], int] = {}
    for r in recorded:
        if r.name in key_arg_by_tool:
            k = key_arg_by_tool[r.name]
            kv = str(r.kwargs.get(k, ""))
            rec_by_key[(r.name, kv)] = rec_by_key.get((r.name, kv), 0) + 1
    matched = 0
    missing: list[str] = []
    for g in gold:
        if g.name in key_arg_by_tool:
            k = key_arg_by_tool[g.name]
            kv = str(g.kwargs.get(k, ""))
            if rec_by_key.get((g.name, kv), 0) > 0:
                matched += 1
                rec_by_key[(g.name, kv)] -= 1
            else:
                missing.append(f"{g.name}(...{k}={kv}...)")
    return matched, missing


def _write_report(summary: dict, task, actions: list[RecordedAction], reminders: list[dict], path: Path):
    lines: list[str] = [
        f"# tau-bench airline — task {summary['task_index']} — {summary['timestamp']}",
        "",
        f"**Model:** {summary['model']}  |  **Budget:** {summary['budget_tokens']:,} tokens",
        "",
        "## Task instruction",
        "",
        f"> {task.instruction}",
        "",
        "## Top-line",
        f"- Turns: **{summary['turns']}**",
        f"- Input tokens (sum): **{summary['input_tokens']:,}**",
        f"- Output tokens (sum): **{summary['output_tokens']:,}**",
        f"- Reminders fired: **{summary['reminder_fires']}**",
        f"- ctx live / dropped at end: {summary['ctx_entries_final']} / {summary['ctx_dropped_final']}",
        f"- Actions recorded / gold: **{summary['actions_recorded']} / {summary['gold_actions']}**",
        f"- Actions matched: **{summary['actions_matched']}**",
    ]
    if summary["actions_missing"]:
        lines.append("- Actions missing:")
        for m in summary["actions_missing"]:
            lines.append(f"  - {m}")
    lines.append("")

    if reminders:
        lines.append("## Reminders fired")
        for r in reminders:
            lines.append(
                f"- turn {r['turn']}: input_tokens={r['input_tokens']:,} / "
                f"budget={r['budget_tokens']:,}, "
                f"ctx_entries={r['ctx_entries']}, dropped={r['ctx_dropped']}"
            )
        lines.append("")

    if summary["ctx_dropped_hints"]:
        lines.append("## Dropped ctx entries (requery hints)")
        for name, hint in summary["ctx_dropped_hints"].items():
            lines.append(f"- `{name}` — requery: `{hint or '(no source)'}`")
        lines.append("")

    lines.append("## Actions emitted")
    for i, a in enumerate(actions, start=1):
        kw_preview = ", ".join(f"{k}={v}" for k, v in list(a.kwargs.items())[:3])
        if len(a.kwargs) > 3:
            kw_preview += ", ..."
        lines.append(f"{i}. `{a.name}({kw_preview})` → {a.result_preview[:80]}...")
    lines.append("")

    lines.append("## Final assistant message")
    lines.append("```")
    lines.append(summary["final_text"][:4000])
    lines.append("```")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--budget", type=int, default=80_000)
    ap.add_argument("--model", type=str, default="claude-opus-4-7")
    ap.add_argument(
        "--force-ctx",
        action="store_true",
        help="Force-ctx mode: every non-trivial print() fires an error-shaped reminder pushing the agent toward ctx.",
    )
    args = ap.parse_args()

    summary = run_one(
        args.task,
        budget_tokens=args.budget,
        model=args.model,
        force_ctx=args.force_ctx,
    )
    print(f"\nDone. Turns: {summary['turns']}. Reminders: {summary['reminder_fires']}. "
          f"Actions: {summary['actions_matched']}/{summary['gold_actions']}.")
    print(f"Report: {summary['report_path']}")
    print(f"Trace:  {summary['trace_path']}")


if __name__ == "__main__":
    main()
