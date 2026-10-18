"""Command line: `python -m rac run configs/<name>.yaml`."""

from __future__ import annotations

import argparse

from rac.eval import runner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rac", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one experiment config")
    run_p.add_argument("config")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(runner.run(args.config))
