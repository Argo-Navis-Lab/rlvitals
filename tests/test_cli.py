import json
import random
from pathlib import Path

import pytest

from rlvitals.cli import main, parse_names


def test_names_are_pairs() -> None:
    assert parse_names(["actor/pg_loss=loss", "a=b=kl"]) == {"actor/pg_loss": "loss", "a=b": "kl"}
    with pytest.raises(ValueError, match="expected TRAINER=RLVITALS"):
        parse_names(["loss"])


def test_replay_prints_each_event_and_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = random.Random(0)
    history = [{"step": step, "pg_loss": 1.0 + 0.01 * rng.gauss(0, 1)} for step in range(120)]
    history[100]["pg_loss"] = 2.0
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"log_history": history}))
    assert main(["replay", str(path), "--name", "pg_loss=loss"]) == 0
    out, err = capsys.readouterr()
    assert out == "120 steps, 1 events (0 HALT, 0 SNAPSHOT, 1 WARN)\n"
    assert err.startswith("rlvitals: step 100: WARN, spike in loss (value 2, median ")


def test_a_log_that_cannot_be_read_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["replay", str(tmp_path / "missing.jsonl")]) == 2
    assert "No such file" in capsys.readouterr().err
    assert main(["replay", str(tmp_path / "x.jsonl"), "--name", "pg_loss=los"]) == 2
    assert "'pg_loss' maps onto 'los'" in capsys.readouterr().err
