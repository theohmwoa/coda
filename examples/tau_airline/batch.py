"""Batch-run several tau-bench airline tasks with the run-#6 setup, summarize.

Default config: force-ctx + pretty ctx_delta + Opus 4.7. Each task gets a
fresh sandbox + fresh airline data. Produces a single markdown table
comparing turns, ctx usage, action match, and final-answer correctness.

Usage:
    python -m examples.tau_airline.batch                # default selection
    python -m examples.tau_airline.batch --tasks 14,26,33
    python -m examples.tau_airline.batch --tasks 2,14,26 --budget 100000
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

from tau_airline.run import run_one


DEFAULT_TASKS = [14, 26, 33]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TASKS),
        help="Comma-separated task indices.",
    )
    ap.add_argument("--budget", type=int, default=80_000)
    ap.add_argument("--model", type=str, default="claude-opus-4-7")
    ap.add_argument(
        "--no-force-ctx",
        action="store_true",
        help="Disable force-ctx mode for comparison (default ON).",
    )
    args = ap.parse_args()
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
            f"  turns={summary['turns']} "
            f"ctx={summary['ctx_entries_final']}/{summary['ctx_dropped_final']} "
            f"writes={summary['actions_matched']}/{summary['gold_writes']} "
            f"reminders={summary['reminder_fires']}",
            flush=True,
        )

    # Aggregate report
    out = _THIS_DIR / ".runs" / f"batch_{timestamp}.md"
    lines: list[str] = [
        f"# tau-bench batch — {timestamp}",
        "",
        f"**Model:** {args.model}  |  **Budget:** {args.budget:,}  |  "
        f"**force_ctx:** {force_ctx}",
        "",
        "| Task | Turns | In tokens | ctx live | ctx dropped | Reminders | "
        "Writes | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    total_matched = 0
    total_gold = 0
    full_passes = 0
    for s in summaries:
        gold_writes = s["gold_writes"]
        match_str = f"{s['actions_matched']}/{gold_writes}"
        if gold_writes > 0 and s["actions_matched"] == gold_writes:
            result = "✅ full match"
            full_passes += 1
        elif gold_writes == 0 and s["actions_recorded"] == 0:
            # Some tau-bench tasks have zero gold writes (the right move is
            # refuse / no-mutation). Count those as pass when the agent
            # also did no writes.
            result = "✅ no-write match"
            full_passes += 1
        elif s["actions_matched"] > 0:
            result = "⚠️ partial"
        else:
            result = "❌ no match"
        total_matched += s["actions_matched"]
        total_gold += gold_writes
        lines.append(
            f"| {s['task_index']} | {s['turns']} | {s['input_tokens']:,} | "
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
        lines.append(f"- task {s['task_index']}: `{Path(s['report_path']).name}`")
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== batch summary ===")
    print(f"full passes: {full_passes}/{len(summaries)}")
    print(f"actions matched: {total_matched}/{total_gold}")
    print(f"summary written to: {out}")


if __name__ == "__main__":
    main()
