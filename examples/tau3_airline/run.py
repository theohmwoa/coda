"""Run a τ³ (tau2-bench) airline task against coda + Agent SDK + Opus 4.7.

Counterpart of `examples/tau_airline/run.py`. Same shape, swapped to τ³'s
data model:
  - Tasks come from `tau3_airline.airline_primitive.load_tasks()` and are
    indexed by position in the airline task file.
  - Gold actions live at `task.evaluation_criteria.actions` (a list of
    `tau2.data_model.tasks.Action`); each action has `.name` and
    `.arguments`.
  - The instruction string handed to the customer simulator is built by
    `task_instruction_text()` from the structured `UserScenario`.
  - Writes are identified via `env.tools.tool_mutates_state(name)` rather
    than hardcoded — same matcher logic, sourced from the τ³ metadata.

Usage:
    python -m examples.tau3_airline.run                  # default: task 0
    python -m examples.tau3_airline.run --task 44
    python -m examples.tau3_airline.run --task 44 --budget 80000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from coda import Agent, ContextPolicy, HookRegistry, Sandbox, TraceWriter
from coda.llm import ClaudeAgentSDKClient

from tau3_airline.airline_primitive import (
    RecordedAction,
    build_airline,
    load_tasks,
    task_instruction_text,
)
from tau3_airline.user_sim import Tau3UserSimulator


PROMPT_PREAMBLE = """\
You are an airline customer service agent. You have a Python object in
your sandbox called `airline` that exposes the booking system, and a
`respond(message: str) -> str` primitive for talking to the customer.
Call them directly from your Python code blocks; do NOT emit JSON tool
calls.

The airline policy document is given to you below in this prompt. Cite
section names when making policy decisions.

Conversation flow:
- The customer's opening message is in the user prompt below.
- Call `respond("...")` to ask the customer a question or confirm an
  action. The function returns the customer's reply as a string.
- Each `respond()` call counts as one customer turn — keep them
  purposeful (one question at a time, gather what you need, then act).
- If the customer's reply contains '###STOP###', '###TRANSFER###', or
  '###OUT-OF-SCOPE###', the conversation is over. Do NOT call `respond`
  again. End your turn with a prose summary (no code block) so the loop
  exits.

`airline` methods:
  Read-only:
    airline.get_user_details(user_id)
    airline.get_reservation_details(reservation_id)
    airline.list_all_airports()
    airline.search_direct_flight(origin, destination, date)   # date: YYYY-MM-DD
    airline.search_onestop_flight(origin, destination, date)
    airline.get_flight_status(flight_number, date)
    airline.calculate(expression)   # eval an arithmetic expression

  Writes (database-mutating; require explicit user 'yes' first):
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

The policy you must follow:
---
{policy}
---
"""


# Writes that touch the DB and therefore matter for our coarse DB-hash
# scoring. Hardcoded by key arg because the matcher needs to know which
# kwarg identifies the row; τ³'s `tool_mutates_state` tells us *what*
# mutates but not the key. (transfer_to_human_agents is omitted: τ³
# decorates it WRITE for accounting, but it doesn't touch the DB.)
_KEY_ARG_BY_TOOL = {
    "book_reservation": "user_id",
    "cancel_reservation": "reservation_id",
    "update_reservation_flights": "reservation_id",
    "update_reservation_baggages": "reservation_id",
    "update_reservation_passengers": "reservation_id",
    "send_certificate": "user_id",
}


def run_one(
    task_index: int,
    *,
    budget_tokens: int,
    model: str,
    force_ctx: bool = False,
    use_user_sim: bool = True,
    user_sim_model: str = "claude-sonnet-4-6",
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = _THIS_DIR / ".runs"
    runs_dir.mkdir(exist_ok=True)
    trace_path = runs_dir / f"task{task_index}_{timestamp}.jsonl"
    report_path = runs_dir / f"task{task_index}_{timestamp}.md"

    tasks = load_tasks()
    task = tasks[task_index]
    airline, actions, db, terminate, policy = build_airline()

    hooks = HookRegistry()
    sandbox = Sandbox(hooks=hooks)
    sandbox.inject("airline", airline)

    instr_text = task_instruction_text(task)
    user_sim: Tau3UserSimulator | None = None
    if use_user_sim:
        user_sim = Tau3UserSimulator(instructions_text=instr_text, model=user_sim_model)
        opening_user_msg = user_sim.reset()

        def respond(message: str) -> str:
            if user_sim is None:
                return "###STOP###"
            return user_sim.step(message)

        sandbox.inject("respond", respond)
    else:
        opening_user_msg = instr_text

    context_policy = ContextPolicy(
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

    system_append = PROMPT_PREAMBLE.format(policy=policy)
    agent = Agent(
        model=model,
        llm=llm,
        sandbox=sandbox,
        hooks=hooks,
        context_policy=context_policy,
        max_turns=20,
        system_prompt_append=system_append,
    )

    reminder_fires: list[dict] = []
    hooks.on(
        "context_reminder_emitted",
        lambda e: reminder_fires.append(dict(e.payload)),
    )

    with TraceWriter(trace_path).attach(hooks):
        result = agent.run(opening_user_msg)

    gold = list(task.evaluation_criteria.actions or [])
    matched, missing, gold_writes = _action_match(actions, gold)
    recorded_writes = sum(1 for a in actions if a.name in _KEY_ARG_BY_TOOL)

    summary = {
        "task_index": task_index,
        "task_id": task.id,
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
        "recorded_writes": recorded_writes,
        "gold_actions": len(gold),
        "gold_writes": gold_writes,
        "actions_matched": matched,
        "actions_missing": missing,
        "terminated_by_transfer": terminate[0],
        "user_sim_calls": user_sim.call_count if user_sim else 0,
        "user_sim_stopped": user_sim.stopped if user_sim else False,
        "final_text": result.text,
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "ctx_dropped_hints": {
            name: entry.source_expr
            for name, entry in sandbox.ctx._dropped().items()
        },
        "nl_assertions": list(task.evaluation_criteria.nl_assertions or []),
    }

    _write_report(summary, task, instr_text, actions, reminder_fires, report_path)
    return summary


def _action_match(recorded: list[RecordedAction], gold) -> tuple[int, list[str], int]:
    """Match recorded WRITES to gold WRITES (same algorithm as v1 bridge).

    Reads/searches/calculate aren't counted on either side — τ³'s DB
    reward is unchanged from τ-bench's, and reads can't change the
    end-state DB hash. We pair on (name, key-arg) where key-arg is
    reservation_id for reservation tools and user_id for book/send.
    Args beyond the key (cabin, flights[]) aren't checked — for the
    canonical DB-hash score, replay against τ³'s evaluator instead.
    """
    rec_by_key: dict[tuple[str, str], int] = {}
    for r in recorded:
        if r.name in _KEY_ARG_BY_TOOL:
            k = _KEY_ARG_BY_TOOL[r.name]
            kv = str(r.kwargs.get(k, ""))
            rec_by_key[(r.name, kv)] = rec_by_key.get((r.name, kv), 0) + 1
    matched = 0
    missing: list[str] = []
    gold_writes = 0
    for g in gold:
        if g.name in _KEY_ARG_BY_TOOL:
            gold_writes += 1
            k = _KEY_ARG_BY_TOOL[g.name]
            kv = str(g.arguments.get(k, ""))
            if rec_by_key.get((g.name, kv), 0) > 0:
                matched += 1
                rec_by_key[(g.name, kv)] -= 1
            else:
                missing.append(f"{g.name}(...{k}={kv}...)")
    return matched, missing, gold_writes


def _write_report(
    summary: dict,
    task,
    instr_text: str,
    actions: list[RecordedAction],
    reminders: list[dict],
    path: Path,
) -> None:
    lines: list[str] = [
        f"# tau3-bench airline — task {summary['task_index']} "
        f"(id={summary['task_id']}) — {summary['timestamp']}",
        "",
        f"**Model:** {summary['model']}  |  **Budget:** {summary['budget_tokens']:,} tokens",
        "",
        "## Task instruction (rendered)",
        "",
    ]
    for line in instr_text.splitlines():
        lines.append(f"> {line}" if line else ">")
    lines.append("")
    lines.append("## Top-line")
    lines.append(f"- Turns: **{summary['turns']}**")
    lines.append(f"- Input tokens (sum): **{summary['input_tokens']:,}**")
    lines.append(f"- Output tokens (sum): **{summary['output_tokens']:,}**")
    lines.append(f"- Reminders fired: **{summary['reminder_fires']}**")
    lines.append(
        f"- User-sim calls: **{summary['user_sim_calls']}** "
        f"(stopped: {summary['user_sim_stopped']})"
    )
    lines.append(
        f"- ctx live / dropped at end: "
        f"{summary['ctx_entries_final']} / {summary['ctx_dropped_final']}"
    )
    lines.append(
        f"- Actions recorded / gold (incl reads): "
        f"**{summary['actions_recorded']} / {summary['gold_actions']}**"
    )
    lines.append(
        f"- Writes matched: **{summary['actions_matched']} / {summary['gold_writes']}**"
    )
    if summary["actions_missing"]:
        lines.append("- Actions missing:")
        for m in summary["actions_missing"]:
            lines.append(f"  - {m}")
    lines.append("")

    if summary["nl_assertions"]:
        lines.append("## NL assertions to check (eyeball)")
        for a in summary["nl_assertions"]:
            lines.append(f"- {a}")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--budget", type=int, default=80_000)
    ap.add_argument("--model", type=str, default="claude-opus-4-7")
    ap.add_argument(
        "--force-ctx",
        action="store_true",
        help="Force-ctx mode: every non-trivial print() fires a reminder.",
    )
    ap.add_argument(
        "--no-user-sim",
        action="store_true",
        help="Disable the user simulator; agent gets the task instruction as the user message.",
    )
    ap.add_argument(
        "--user-sim-model",
        type=str,
        default="claude-sonnet-4-6",
    )
    args = ap.parse_args()

    summary = run_one(
        args.task,
        budget_tokens=args.budget,
        model=args.model,
        force_ctx=args.force_ctx,
        use_user_sim=not args.no_user_sim,
        user_sim_model=args.user_sim_model,
    )
    print(
        f"\nDone. Turns: {summary['turns']}. Reminders: {summary['reminder_fires']}. "
        f"Writes: {summary['actions_matched']}/{summary['gold_writes']}."
    )
    print(f"Report: {summary['report_path']}")
    print(f"Trace:  {summary['trace_path']}")


if __name__ == "__main__":
    main()
