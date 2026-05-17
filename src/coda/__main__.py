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

    srv = sub.add_parser(
        "serve",
        help="Run the live UI on http://localhost:PORT (default 8000).",
    )
    srv.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    srv.add_argument("--port", type=int, default=8000, help="Port to bind.")
    srv.add_argument(
        "--skills-dir",
        default=None,
        help="Directory of saved skills to load (defaults to ./skills).",
    )
    srv.add_argument(
        "--no-gmail",
        action="store_true",
        help="Skip the Gmail MCP server (faster startup, no Gmail tools).",
    )
    srv.add_argument(
        "--no-discord",
        action="store_true",
        help="Skip the Discord webhook tool.",
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

    if args.command == "serve":
        try:
            from .server import ServerConfig, serve
        except ImportError as e:
            print(
                f"error: serve requires fastapi + uvicorn. "
                f"Install with: pip install 'coda[server]'\n  ({e})",
                file=sys.stderr,
            )
            return 2

        # Defaults match the morning_brief setup: Gmail + Discord webhook
        # tool + triage_email sub-agent + skills_dir. Examples are
        # repo-local so this works out of the box from a `coda` checkout.
        import os
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        skills_dir = Path(args.skills_dir).resolve() if args.skills_dir else repo_root / "skills"

        sub_agents: list = []
        inline_tools: list = []
        mcp_servers: list = []

        # Pull the triage_email sub-agent + post_to_discord tool from the
        # bundled example so the live UI mirrors what the user has seen
        # before. Failing to import any of them is non-fatal — UI still
        # works, just with a smaller toolbelt.
        examples_dir = repo_root / "examples"
        if examples_dir.is_dir():
            sys.path.insert(0, str(examples_dir))
            try:
                import morning_brief as _mb  # noqa: E402
                sub_agents.append(_mb.triage_email)
                if not args.no_discord and os.environ.get("DISCORD_WEBHOOK_URL"):
                    inline_tools.append(_mb.post_to_discord)
                if not args.no_gmail and (Path.home() / ".gmail-mcp" / "credentials.json").exists():
                    mcp_servers.append(_mb.gmail_mcp())
            except ImportError as e:
                print(f"warning: couldn't load example tools ({e})", file=sys.stderr)

        config = ServerConfig(
            skills_dir=skills_dir,
            sub_agents=sub_agents,
            inline_tools=inline_tools,
            mcp_servers=mcp_servers,
        )
        print(f"coda serve  →  http://{args.host}:{args.port}", file=sys.stderr)
        print(f"  skills:       {skills_dir} ({len(list(skills_dir.glob('*.py')))} files)" if skills_dir.exists() else f"  skills:       {skills_dir} (will be created)", file=sys.stderr)
        print(f"  sub-agents:   {[sa.name for sa in sub_agents] or '—'}", file=sys.stderr)
        print(f"  inline tools: {[t.name for t in inline_tools] or '—'}", file=sys.stderr)
        print(f"  mcp servers:  {[s.name for s in mcp_servers] or '—'}", file=sys.stderr)
        serve(host=args.host, port=args.port, config=config)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 1  # unreachable; parser.error exits.


if __name__ == "__main__":
    raise SystemExit(main())
