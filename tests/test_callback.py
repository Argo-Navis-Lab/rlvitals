import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from rlvitals.callback import RLVitalsCallback
from rlvitals.monitor import Event

# What a Trainer passes every callback besides args, state and control.
TRAINER_KWARGS = dict(model=None, processing_class=None, optimizer=None, lr_scheduler=None)


def train(
    callback: RLVitalsCallback,
    logs: Sequence[Mapping[str, object]],
    tmp_path: Path,
    main_process: bool = True,
) -> TrainerControl:
    """Each log at global step 10, 20, ... as a Trainer logging every 10 steps would."""
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    state = TrainerState(is_world_process_zero=main_process)
    control = TrainerControl()
    callback.on_train_begin(args, state, control, **TRAINER_KWARGS)
    for i, log in enumerate(logs, start=1):
        state.global_step = 10 * i
        callback.on_log(args, state, control, logs=dict(log), **TRAINER_KWARGS)
    return control


def trl_log(rng: random.Random, i: int) -> dict[str, object]:
    """A healthy GRPOTrainer log: its names, sound values, and names rlvitals doesn't read."""
    return {
        "loss": 0.01 + 0.001 * rng.gauss(0, 1),
        "grad_norm": 1.0 + 0.05 * rng.gauss(0, 1),
        "learning_rate": 1e-6,
        "reward": 0.5,
        "frac_reward_zero_std": 0.2,
        "entropy": 0.5,
        "completions/mean_length": 100.0,
        "epoch": i / 100,
    }


def test_a_healthy_run_finds_nothing(tmp_path: Path) -> None:
    rng = random.Random(0)
    callback = RLVitalsCallback(hooks=())
    control = train(callback, [trl_log(rng, i) for i in range(200)], tmp_path)
    assert callback.events == []
    assert not control.should_training_stop


def test_trls_flat_groups_are_read_as_flat_groups(tmp_path: Path) -> None:
    rng = random.Random(0)
    logs = [trl_log(rng, i) | {"frac_reward_zero_std": 0.95 if i >= 50 else 0.2} for i in range(80)]
    seen: list[Event] = []
    callback = RLVitalsCallback(hooks=(seen.append,))
    train(callback, logs, tmp_path)
    assert [(e.step, e.detector, e.metric) for e in seen] == [(690, "collapse", "flat_groups")]
    assert seen == callback.events


def test_eval_logs_and_the_summary_are_skipped(tmp_path: Path) -> None:
    callback = RLVitalsCallback(hooks=())
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    state, control = TrainerState(global_step=10), TrainerControl()
    for logs in ({"loss": 1.0}, {"eval_loss": 1.0, "eval_entropy": 0.0}, {"train_runtime": 9.0}):
        callback.on_log(args, state, control, logs=logs, **TRAINER_KWARGS)
    state.global_step = 20
    callback.on_log(args, state, control, logs={"loss": 1.0}, **TRAINER_KWARGS)  # no ValueError


@pytest.mark.parametrize("stop_on_halt", [False, True])
def test_a_nan_loss_halts_and_stops_only_if_asked(tmp_path: Path, stop_on_halt: bool) -> None:
    rng = random.Random(0)
    logs = [trl_log(rng, i) for i in range(10)] + [{"loss": math.nan, "grad_norm": 1.0}]
    callback = RLVitalsCallback(hooks=(), stop_on_halt=stop_on_halt)
    control = train(callback, logs, tmp_path)
    assert [(e.step, e.severity) for e in callback.events] == [(110, "HALT")]
    assert control.should_training_stop == stop_on_halt


@pytest.mark.parametrize("save_on_snapshot", [False, True])
def test_a_growing_grad_norm_saves_only_if_asked(tmp_path: Path, save_on_snapshot: bool) -> None:
    # 100 logs 10 steps apart, log grad_norm up 0.02 a step: double the divergence slope
    logs = [{"loss": 0.01, "grad_norm": math.exp(0.2 * i)} for i in range(100)]
    callback = RLVitalsCallback(hooks=(), save_on_snapshot=save_on_snapshot)
    control = train(callback, logs, tmp_path)
    assert [e.step for e in callback.events if e.severity == "SNAPSHOT"] == [1000]
    assert control.should_save == save_on_snapshot


def test_other_processes_stop_too_but_stay_silent(tmp_path: Path) -> None:
    seen: list[Event] = []
    callback = RLVitalsCallback(hooks=(seen.append,), stop_on_halt=True)
    control = train(callback, [{"grad_norm": math.inf}], tmp_path, main_process=False)
    assert seen == []
    assert [e.severity for e in callback.events] == ["HALT"]
    assert control.should_training_stop
