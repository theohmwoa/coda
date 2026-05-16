"""CLI entrypoint: `python -m coda <subcommand> ...`.

For now the only subcommand is `replay`. We keep the dispatch hand-
rolled with argparse — no click, no typer — to match the existing
"stdlib only" tone of the codebase.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .replay import count_by_type, replay


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coda",
        description="Inspect coda agent traces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser(
        "replay",
        help="Print a chronological timeline of a JSONL trace file.",
    )
    rep.add_argument("path", help="Path to the JSONL trace file.")
    rep.add_argument(
        "--filter",
        default=None,
        metavar="TYPE[,TYPE...]",
        help="Comma-separated event types to keep (default: all).",
    )
    rep.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        default=True,
        help="Disable ANSI color escapes.",
    )
    return parser


def _format_summary(total: int, by_type: dict[str, int]) -> str:
    if not by_type:
        return f"{total} events"
    # Sort by count descending, then by type name for stability.
    ordered = sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = ", ".join(f"{t}={n}" for t, n in ordered)
    return f"{total} events ({parts})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "replay":
        filter_types: set[str] | None
        if args.filter:
            filter_types = {t for t in args.filter.split(",") if t}
        else:
            filter_types = None

        try:
            total = replay(args.path, filter_types=filter_types, color=args.color)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        by_type = count_by_type(args.path, filter_types=filter_types)
        print(_format_summary(total, by_type))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 1  # unreachable; parser.error exits.


if __name__ == "__main__":
    raise SystemExit(main())
