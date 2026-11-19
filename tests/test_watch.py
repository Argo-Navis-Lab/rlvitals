import math
import random

import pytest

from rlvitals.monitor import Event
from rlvitals.watch import METRICS, Watch, format_event, reading, rename


def test_by_default_rlvitals_names_are_read_and_nothing_else() -> None:
    values = {"loss": 0.5, "grad_norm": 2, "reward": 0.3, "epoch": "one"}
    assert rename(values, reading()) == {"loss": 0.5, "grad_norm": 2.0}


def test_a_trainer_name_replaces_the_rlvitals_name_it_maps_onto() -> None:
    read = reading({"actor/grad_norm": "grad_norm"})
    values = {"actor/grad_norm": 3.0, "grad_norm": 99.0, "loss": 0.5}
    assert rename(values, read) == {"grad_norm": 3.0, "loss": 0.5}
    assert set(read.values()) == set(METRICS)


def test_none_counts_as_absent_and_a_word_is_an_error() -> None:
    assert rename({"loss": None, "kl": 0.1}, reading()) == {"kl": 0.1}
    with pytest.raises(TypeError, match="kl is 'high': not a number"):
        rename({"kl": "high"}, reading())


def test_a_name_rlvitals_does_not_read_is_refused() -> None:
    with pytest.raises(ValueError, match="'grad-norm'"):
        reading({"actor/grad_norm": "grad-norm"})


def test_two_trainer_names_onto_one_rlvitals_name_are_refused() -> None:
    with pytest.raises(ValueError, match="more than one trainer name maps onto 'kl'"):
        reading({"kl_loss": "kl", "kl_ref": "kl"})


def test_an_event_reads_as_one_line() -> None:
    event = Event(
        301, "spike", "kl", "WARN", {"value": 5.0, "median": 1.0, "mad": 0.25, "z": 10.79}
    )
    assert format_event(event) == (
        "rlvitals: step 301: WARN, spike in kl (value 5, median 1, mad 0.25, z 10.79)"
    )


def test_every_hook_sees_every_event_in_order() -> None:
    seen: list[tuple[str, int]] = []
    watch = Watch(
        hooks=[lambda e: seen.append(("a", e.step)), lambda e: seen.append(("b", e.step))]
    )
    events = watch.step(4, {"grad_norm": math.inf})
    assert [(e.detector, e.metric, e.severity) for e in events] == [
        ("divergence", "grad_norm", "HALT")
    ]
    assert seen == [("a", 4), ("b", 4)]


def test_a_spike_under_the_trainers_name_is_found() -> None:
    rng = random.Random(0)
    watch = Watch(names={"policy_loss": "loss"}, hooks=())
    for step in range(100):
        assert watch.step(step, {"policy_loss": 1.0 + 0.01 * rng.gauss(0, 1), "loss": 7.0}) == []
    events = watch.step(100, {"policy_loss": 2.0})
    assert [(e.step, e.detector, e.metric) for e in events] == [(100, "spike", "loss")]


def test_a_failing_hook_warns_and_the_next_hook_still_runs() -> None:
    seen: list[int] = []

    def broken(event: Event) -> None:
        raise ConnectionError("no network")

    watch = Watch(hooks=[broken, lambda e: seen.append(e.step)])
    with pytest.warns(RuntimeWarning, match="ConnectionError"):
        watch.step(0, {"loss": math.nan})
    assert seen == [0]


def test_the_default_hook_prints_and_no_hooks_print_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    Watch().step(0, {"loss": math.nan})
    assert capsys.readouterr().err == "rlvitals: step 0: HALT, divergence in loss (value nan)\n"
    Watch(hooks=()).step(0, {"loss": math.nan})
    assert capsys.readouterr().err == ""
