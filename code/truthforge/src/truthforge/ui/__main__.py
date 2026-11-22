"""`python -m truthforge.ui` entry point."""

from __future__ import annotations

import argparse

from .server import serve


def main() -> int:
    ap = argparse.ArgumentParser(description="TruthForge local UI")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--open", action="store_true", help="open a browser window on start"
    )
    ap.add_argument(
        "--preload",
        nargs="*",
        default=[],
        help="detectors to load at startup instead of on first use "
        "(e.g. --preload nli minicheck)",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return serve(
        host=args.host,
        port=args.port,
        open_browser=args.open,
        preload=args.preload,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
