import json
import math
import random
from pathlib import Path

import pytest

from rlvitals.replay import json_lines, merge, read_log, replay, trainer_state
from rlvitals.watch import Watch


def a_run(seed: int) -> list[dict[str, float]]:
    """150 steps of a quiet loss with one jump at step 100."""
    rng = random.Random(seed)
    losses = [1.0 + 0.01 * rng.gauss(0, 1) for _ in range(150)]
    losses[100] = 2.0
    return [{"policy_loss": loss} for loss in losses]


def test_two_entries_at_one_step_are_joined() -> None:
    records = [(10, {"loss": 0.5}), (10, {"eval_loss": 0.4}), (20, {"loss": 0.4})]
    assert merge(records) == [(10, {"loss": 0.5, "eval_loss": 0.4}), (20, {"loss": 0.4})]


def test_a_falling_step_is_refused() -> None:
    with pytest.raises(ValueError, match="step 5 comes after step 10"):
        merge([(10, {}), (5, {})])


def test_a_trainer_state_is_read_by_step_nan_included(tmp_path: Path) -> None:
    path = tmp_path / "trainer_state.json"
    history = [
        {"loss": 0.5, "grad_norm": 1.0, "step": 1},
        {"eval_loss": 0.4, "step": 1},
        {"loss": math.nan, "grad_norm": 1.0, "step": 2},
    ]
    path.write_text(json.dumps({"global_step": 2, "log_history": history}))
    records = trainer_state(path)
    assert [step for step, _ in records] == [1, 2]
    assert records[0][1] == {"loss": 0.5, "grad_norm": 1.0, "eval_loss": 0.4, "step": 1}
    events = replay(records, Watch(hooks=()))
    assert [(e.step, e.detector, e.metric, e.severity) for e in events] == [
        (2, "divergence", "loss", "HALT")
    ]


def test_a_file_without_log_history_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"lr": 1e-5}))
    with pytest.raises(ValueError, match="no log_history"):
        trainer_state(path)


def test_json_lines_take_a_step_key_and_skip_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"_step": 0, "loss": 1.0}\n\n{"_step": 1, "loss": 0.9}\n')
    assert json_lines(path, "_step") == [
        (0, {"_step": 0, "loss": 1.0}),
        (1, {"_step": 1, "loss": 0.9}),
    ]


@pytest.mark.parametrize(
    ("line", "error"),
    [
        ('{"loss": 1.0}', "no whole-number 'step'"),
        ('{"step": 1.5}', "no whole-number"),
        ("[1]", "not a JSON object"),
        ("{", "Expecting"),
    ],
)
def test_a_bad_line_is_refused_with_its_number(tmp_path: Path, line: str, error: str) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"step": 0}\n' + line + "\n")
    with pytest.raises(ValueError, match=f"log.jsonl:2: {error}"):
        json_lines(path)


def test_a_log_is_read_by_its_suffix(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text("")
    with pytest.raises(ValueError, match="expected a .json"):
        read_log(path)


def test_a_replay_finds_what_a_live_watch_found(tmp_path: Path) -> None:
    names = {"policy_loss": "loss"}
    live = Watch(names, hooks=())
    found = [event for step, values in enumerate(a_run(0)) for event in live.step(step, values)]
    path = tmp_path / "log.jsonl"
    path.write_text("".join(json.dumps({"step": s} | v) + "\n" for s, v in enumerate(a_run(0))))
    assert replay(read_log(path), Watch(names, hooks=())) == found
    assert [(e.step, e.detector, e.metric) for e in found] == [(100, "spike", "loss")]
