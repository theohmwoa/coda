"""Batch-run several τ³ airline tasks and produce a summary table.

τ³ counterpart of `examples/tau_airline/batch.py`. Same shape, points at
`tau3_airline.run.run_one`.

Default selection mirrors the v1 batch (task 14 / 26 / 33 equivalents):
in τ³ the airline task list is reordered, so the defaults below pick
the τ³ analogues by ID:
  - task 33 in v1 → task 44 in τ³ (sophia_silva_7557, S61CZX is the
    one our issue #85 was about; gold now correctly excludes its
    cancellation).
  - task 26 in v1 (IFOYYZ refusal) is task ID 0 in τ³'s reordered file —
    the canonical 'agent should refuse' test for the airline domain.
  - task 14 is a representative book/refund scenario.

Usage:
    python -m examples.tau3_airline.batch                   # default selection
    python -m examples.tau3_airline.batch --tasks 0,14,44
    python -m examples.tau3_airline.batch --tasks 44 --budget 100000
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

from tau3_airline.run import run_one


DEFAULT_TASKS = [0, 14, 44]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TASKS),
        help="Comma-separated task indices (positions in the τ³ airline tasks file).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Override --tasks: run all 50 base-split tasks (leaderboard denominator).",
    )
    ap.add_argument("--budget", type=int, default=80_000)
    ap.add_argument("--model", type=str, default="claude-opus-4-7")
    ap.add_argument(
        "--no-force-ctx",
        action="store_true",
        help="Disable force-ctx mode (default ON).",
    )
    args = ap.parse_args()
    if args.all:
        # Base split = all 50 tasks in file order (positions 0..49).
        task_indices = list(range(50))
    else:
        task_indices = [int(s) for s in args.tasks.split(",")]
    force_ctx = not args.no_force_ctx

    summaries: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx in task_indices:
        print(f"\n=== running task {idx} ===", flush=True)
        summary = run_one(
            idx,
            budget_tokens=args.budget,
            model=args.model,
            force_ctx=force_ctx,
        )
        summaries.append(summary)
        print(
            f"  task_id={summary['task_id']} "
            f"turns={summary['turns']} "
            f"ctx={summary['ctx_entries_final']}/{summary['ctx_dropped_final']} "
            f"writes={summary['actions_matched']}/{summary['gold_writes']} "
            f"reminders={summary['reminder_fires']}",
            flush=True,
        )

    out = _THIS_DIR / ".runs" / f"batch_{timestamp}.md"
    lines: list[str] = [
        f"# tau3-bench batch — {timestamp}",
        "",
        f"**Model:** {args.model}  |  **Budget:** {args.budget:,}  |  "
        f"**force_ctx:** {force_ctx}",
        "",
        "| Task # | task_id | Turns | In tokens | ctx live | ctx dropped | "
        "Reminders | Writes | Result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    total_matched = 0
    total_gold = 0
    full_passes = 0
    for s in summaries:
        gold_writes = s["gold_writes"]
        recorded_writes = s["recorded_writes"]
        match_str = f"{s['actions_matched']}/{gold_writes}"
        if gold_writes > 0 and s["actions_matched"] == gold_writes:
            result = "✅ full match"
            full_passes += 1
        elif gold_writes == 0 and recorded_writes == 0:
            # Refuse-style tasks: gold expects no writes, agent did none.
            result = "✅ refusal match"
            full_passes += 1
        elif gold_writes == 0 and recorded_writes > 0:
            # Gold says don't mutate; agent mutated. Hard fail.
            result = "❌ over-mutated"
        elif s["actions_matched"] > 0:
            result = "⚠️ partial"
        else:
            result = "❌ no match"
        total_matched += s["actions_matched"]
        total_gold += gold_writes
        lines.append(
            f"| {s['task_index']} | {s['task_id']} | {s['turns']} | "
            f"{s['input_tokens']:,} | "
            f"{s['ctx_entries_final']} | {s['ctx_dropped_final']} | "
            f"{s['reminder_fires']} | {match_str} | {result} |"
        )
    lines.append("")
    lines.append(
        f"**Overall:** {full_passes}/{len(summaries)} full passes, "
        f"{total_matched}/{total_gold} writes matched."
    )
    lines.append("")
    lines.append("## Per-task reports")
    for s in summaries:
        lines.append(
            f"- task {s['task_index']} (id={s['task_id']}): "
            f"`{Path(s['report_path']).name}`"
        )
    out.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== batch summary ===")
    print(f"full passes: {full_passes}/{len(summaries)}")
    print(f"actions matched: {total_matched}/{total_gold}")
    print(f"summary written to: {out}")


if __name__ == "__main__":
    main()
