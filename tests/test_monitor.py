"""Monitor tests: severities by hand, quiet on clean runs, one event where each finding starts
(a lasting jump, repeated NaNs, a metric logged every other step), steps that must rise, and a
state that survives JSON and carries on exactly."""

import json
import math
import random

import pytest

from rlvitals.detectors import (
    ENTROPY,
    FLAT_GROUPS,
    GRAD_NORM,
    KL,
    LOSS,
    Collapse,
    Divergence,
    Spike,
    SpikeDetector,
)
from rlvitals.monitor import Event, Monitor, severity, to_event


def clean(steps: int, seed: int) -> list[dict[str, float]]:
    """A healthy run's metrics, one dict a step: noise around steady values."""
    rng = random.Random(seed)
    return [
        {
            LOSS: rng.gauss(0.0, 0.01),
            GRAD_NORM: math.exp(rng.gauss(0.0, 0.1)),
            KL: abs(rng.gauss(0.01, 0.001)),
            ENTROPY: rng.gauss(0.5, 0.05),
            FLAT_GROUPS: sum(rng.random() < 0.5 for _ in range(64)) / 64,
        }
        for _ in range(steps)
    ]


def run(monitor: Monitor, rows: list[dict[str, float]], first: int = 1) -> list[Event]:
    """Each row as one step, steps from `first`; every event."""
    return [event for step, row in enumerate(rows, first) for event in monitor.check(step, row)]


def test_severities_and_events_by_hand() -> None:
    assert severity(Divergence(1, LOSS, math.nan)) == "HALT"
    assert severity(Divergence(1, GRAD_NORM, math.inf)) == "HALT"
    assert severity(Divergence(1, GRAD_NORM, 0.02)) == "SNAPSHOT"
    assert severity(Spike(1, KL, 9.0, 1.0, 1.0, 5.4)) == "WARN"
    assert severity(Collapse(1, ENTROPY, 0.005, 0.01)) == "WARN"
    event = to_event("spike", Spike(7, KL, 9.0, 1.0, 0.5, 10.8))
    assert event == Event(
        7, "spike", KL, "WARN", {"value": 9.0, "median": 1.0, "mad": 0.5, "z": 10.8}
    )


@pytest.mark.parametrize("seed", range(10))
def test_a_clean_run_raises_no_event(seed: int) -> None:
    assert run(Monitor(), clean(1000, seed)) == []


def test_a_lasting_jump_is_one_event_where_it_starts() -> None:
    rows = clean(400, 0)
    for row in rows[300:]:
        row[LOSS] += 1.0  # 100 standard deviations of the noise, from step 301 on
    events = run(Monitor(), rows)
    assert [(e.step, e.detector, e.metric, e.severity) for e in events] == [
        (301, "spike", LOSS, "WARN")
    ]


def test_each_run_of_nans_is_one_halt() -> None:
    rows = clean(10, 0)
    for step in (3, 4, 5, 8):
        rows[step - 1][LOSS] = math.nan
    events = run(Monitor(), rows)
    assert [(e.step, e.detector, e.metric, e.severity) for e in events] == [
        (3, "divergence", LOSS, "HALT"),
        (8, "divergence", LOSS, "HALT"),
    ]
    assert math.isnan(events[0].details["value"])


def test_a_finding_stays_found_through_steps_without_its_metric() -> None:
    rows = clean(400, 0)
    for step, row in enumerate(rows, 1):
        if step % 2:
            del row[KL]  # kl logged on even steps only
        elif step > 300:
            row[KL] += 1.0
    events = run(Monitor(), rows)
    assert [(e.step, e.detector, e.metric) for e in events] == [(302, "spike", KL)]


def test_the_monitor_uses_the_detectors_it_is_given() -> None:
    rows = clean(400, 0)
    rows[300][LOSS] += 0.05  # about 5 standard deviations: under 6, over 3
    assert run(Monitor(), rows) == []
    events = run(Monitor(spike=SpikeDetector(threshold=3.0)), rows)
    assert (301, LOSS) in [(e.step, e.metric) for e in events]


def test_steps_must_rise() -> None:
    monitor = Monitor()
    monitor.check(5, {LOSS: 0.0})
    for step in (5, 4):
        with pytest.raises(ValueError, match="steps must rise"):
            monitor.check(step, {LOSS: 0.0})
    assert monitor.check(6, {LOSS: 0.0}) == []


def test_a_resumed_monitor_carries_on_exactly() -> None:
    rows = clean(400, 1)
    rows[250][GRAD_NORM] *= 10.0  # a spike after the resume, judged by the restored window
    whole = Monitor()
    before = run(whole, rows[:200])
    after = run(whole, rows[200:], first=201)
    resumed = Monitor()
    resumed.load_state(json.loads(json.dumps(state_after(rows[:200]))))
    assert before == []
    assert [(e.step, e.metric) for e in after] == [(251, GRAD_NORM)]
    assert run(resumed, rows[200:], first=201) == after
    assert resumed.state() == whole.state()


def state_after(rows: list[dict[str, float]]) -> dict[str, object]:
    """The state of a monitor that has seen `rows`."""
    monitor = Monitor()
    run(monitor, rows)
    return monitor.state()
