"""The monitor: one step's metrics in, events out. It runs the three detectors on each step's
values and turns what they find into events, one where each finding starts, each with a
severity that advises what to do about it. WARN: look at the run. SNAPSHOT: keep a checkpoint
now, while the weights are still sound. HALT: stop, the run is ruined. The monitor only
advises; it never touches the run."""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rlvitals.detectors import (
    Collapse,
    CollapseDetector,
    Divergence,
    DivergenceDetector,
    Spike,
    SpikeDetector,
)

Severity = Literal["WARN", "SNAPSHOT", "HALT"]  # each is graver than the one before it
Finding = Spike | Divergence | Collapse


@dataclass(frozen=True)
class Event:
    """A detector's finding, at the step it started. `metric` is what the detector watched;
    `details` are the numbers it judged by, such as a spike's value, median, mad and z."""

    step: int
    detector: str  # "spike", "divergence" or "collapse"
    metric: str
    severity: Severity
    details: dict[str, float]


def severity(finding: Finding) -> Severity:
    """How grave a finding is. A NaN or an infinity has ruined the run: HALT. A gradient norm
    growing steadily is on its way there: SNAPSHOT. A spike may pass, and a collapse wastes
    steps but breaks nothing: WARN."""
    if isinstance(finding, Divergence):
        return "SNAPSHOT" if math.isfinite(finding.value) else "HALT"
    return "WARN"


def to_event(detector: str, finding: Finding) -> Event:
    """The event for a finding: its numbers, less the step and the metric, are the details."""
    details = dataclasses.asdict(finding)
    del details["step"], details["metric"]
    return Event(finding.step, detector, finding.metric, severity(finding), details)


class Monitor:
    """Runs a spike, a divergence and a collapse detector (ScaleLab's numbers unless given
    others) on each step's metric values. A detector keeps finding a problem while it lasts (a
    lasting jump is a spike for 100 steps), so an event marks only where a finding starts: a
    (detector, metric) not found at the step before. A finding stays found through steps that
    lack its metric, so a metric logged every other step doesn't start it again each time."""

    def __init__(
        self,
        spike: SpikeDetector | None = None,
        divergence: DivergenceDetector | None = None,
        collapse: CollapseDetector | None = None,
    ) -> None:
        self.spike = spike or SpikeDetector()
        self.divergence = divergence or DivergenceDetector()
        self.collapse = collapse or CollapseDetector()
        self.found: set[tuple[str, str]] = set()  # (detector, metric) found at the last step
        self.last_step: int | None = None

    def check(self, step: int, values: Mapping[str, float]) -> list[Event]:
        """The events that start at `step`, given all its metric values. Steps must rise."""
        if self.last_step is not None and step <= self.last_step:
            raise ValueError(f"step {step} comes after step {self.last_step}: steps must rise")
        self.last_step = step
        findings: list[tuple[str, Finding]] = [
            *(("spike", finding) for finding in self.spike.check(step, values)),
            *(("divergence", finding) for finding in self.divergence.check(step, values)),
            *(("collapse", finding) for finding in self.collapse.check(step, values)),
        ]
        found = {(detector, finding.metric) for detector, finding in findings}
        events = [
            to_event(detector, finding)
            for detector, finding in findings
            if (detector, finding.metric) not in self.found
        ]
        self.found = found | {pair for pair in self.found if pair[1] not in values}
        return events

    def state(self) -> dict[str, Any]:
        """Everything the monitor remembers, as plain JSON: a resumed run's monitor, given it by
        load_state, carries on as if the run had never stopped."""
        return {
            "spike": {metric: list(window) for metric, window in self.spike.windows.items()},
            "divergence": [list(pair) for pair in self.divergence.window],
            "collapse": {metric: list(window) for metric, window in self.collapse.windows.items()},
            "found": sorted(list(pair) for pair in self.found),
            "last_step": self.last_step,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Carry on from `state` (what state() returned), in a new monitor."""
        for metric, values in state["spike"].items():
            self.spike.windows[metric].extend(values)
        self.divergence.window.extend((step, log) for step, log in state["divergence"])
        for metric, values in state["collapse"].items():
            self.collapse.windows[metric].extend(values)
        self.found = {(detector, metric) for detector, metric in state["found"]}
        self.last_step = state["last_step"]
