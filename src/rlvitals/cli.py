"""The rlvitals command. `rlvitals replay LOG` reads a saved run's log through a watch: each
event on standard error where it starts, then one summary line on standard output."""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rlvitals.replay import read_log, replay
from rlvitals.watch import Watch


def parse_names(pairs: Sequence[str]) -> dict[str, str]:
    """TRAINER=RLVITALS pairs as a mapping from trainer names to rlvitals names."""
    names: dict[str, str] = {}
    for pair in pairs:
        trainer_name, equals, name = pair.rpartition("=")
        if not equals or not trainer_name:
            raise ValueError(f"--name {pair!r}: expected TRAINER=RLVITALS")
        names[trainer_name] = name
    return names


def main(argv: Sequence[str] | None = None) -> int:
    """Exit codes: 0 the log was read (whatever was found), 2 it couldn't be."""
    parser = argparse.ArgumentParser(prog="rlvitals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay_command = commands.add_parser("replay", help="check a saved run's log")
    replay_command.add_argument("log", type=Path, help="a trainer_state.json or a .jsonl file")
    replay_command.add_argument(
        "--name",
        action="append",
        default=[],
        metavar="TRAINER=RLVITALS",
        help="read the trainer's metric TRAINER as rlvitals' RLVITALS (repeatable)",
    )
    replay_command.add_argument("--step-key", default="step", help="a .jsonl log's step key")
    args = parser.parse_args(argv)
    try:
        watch = Watch(parse_names(args.name))
        records = read_log(args.log, args.step_key)
        events = replay(records, watch)
    except (OSError, ValueError, TypeError) as error:
        print(f"rlvitals: {error}", file=sys.stderr)
        return 2
    counts = Counter(event.severity for event in events)
    by_severity = ", ".join(f"{counts[s]} {s}" for s in ("HALT", "SNAPSHOT", "WARN"))
    print(f"{len(records)} steps, {len(events)} events ({by_severity})")
    return 0
