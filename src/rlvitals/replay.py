"""Replaying a finished run: its saved log, read step by step through a Watch, as if live. Two
formats: a Hugging Face trainer's trainer_state.json (TRL's trainers save one with every
checkpoint; its log_history holds each logged step) and JSON Lines, one JSON object a line,
each carrying its step."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from rlvitals.monitor import Event
from rlvitals.watch import Watch

Record = tuple[int, dict[str, Any]]  # a step and everything logged at it


def record(entry: object, step_key: str, where: str) -> Record:
    """One logged entry as a record: its step, a whole number under `step_key`, and itself."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: not a JSON object")
    values = cast(dict[str, Any], entry)
    step = values.get(step_key)
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError(f"{where}: no whole-number {step_key!r}")
    return step, values


def merge(records: Iterable[Record]) -> list[Record]:
    """Records joined by step: a trainer may log twice at one step (its training metrics, then
    an evaluation's), and the monitor sees each step once. Steps must not fall."""
    merged: list[Record] = []
    for step, values in records:
        if merged and step < merged[-1][0]:
            raise ValueError(f"step {step} comes after step {merged[-1][0]}: steps must not fall")
        if merged and step == merged[-1][0]:
            merged[-1][1].update(values)
        else:
            merged.append((step, dict(values)))
    return merged


def trainer_state(path: Path) -> list[Record]:
    """The steps of a trainer_state.json: the entries of its log_history, joined by step."""
    state: object = json.loads(path.read_text())
    history = cast(dict[str, Any], state).get("log_history") if isinstance(state, dict) else None
    if not isinstance(history, list):
        raise ValueError(f"{path}: no log_history, so not a trainer_state.json")
    entries = cast(list[object], history)
    return merge(
        record(entry, "step", f"{path}: log_history[{index}]")
        for index, entry in enumerate(entries)
    )


def json_lines(path: Path, step_key: str = "step") -> list[Record]:
    """The steps of a JSON Lines log, joined by step; blank lines are skipped."""
    records: list[Record] = []
    with path.open() as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            records.append(record(entry, step_key, f"{path}:{number}"))
    return merge(records)


def read_log(path: Path, step_key: str = "step") -> list[Record]:
    """A saved log, by its suffix: .jsonl is JSON Lines, .json a trainer_state.json."""
    if path.suffix == ".jsonl":
        return json_lines(path, step_key)
    if path.suffix == ".json":
        return trainer_state(path)
    raise ValueError(f"{path}: expected a .json (trainer_state.json) or a .jsonl file")


def replay(records: Iterable[Record], watch: Watch) -> list[Event]:
    """Every event the watch finds over a saved run, in step order, as it would have live."""
    events: list[Event] = []
    for step, values in records:
        events.extend(watch.step(step, values))
    return events
