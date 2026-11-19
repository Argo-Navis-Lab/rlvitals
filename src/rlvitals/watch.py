"""Watching a run: a trainer's metrics in, under the trainer's own names, and the monitor's
events out to alert hooks. Each trainer names its metrics its own way ("actor/grad_norm",
"frac_reward_zero_std"); a Watch reads the ones it is told about, under rlvitals' names, and
ignores the rest. Hooks are plain functions of an event: print it, post it, stop the run."""

import sys
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import SupportsFloat

from rlvitals.detectors import ENTROPY, FLAT_GROUPS, GRAD_NORM, KL, LOSS
from rlvitals.monitor import Event, Monitor

METRICS = (LOSS, GRAD_NORM, KL, ENTROPY, FLAT_GROUPS)  # every name the detectors read

Hook = Callable[[Event], None]


def reading(names: Mapping[str, str] | None = None) -> dict[str, str]:
    """What to read: each trainer name, mapped to its rlvitals name. `names` maps a trainer's
    names onto rlvitals' own; an rlvitals name nothing maps onto is read under that name."""
    names = dict(names or {})
    for trainer_name, name in names.items():
        if name not in METRICS:
            raise ValueError(f"{trainer_name!r} maps onto {name!r}: not one of {METRICS}")
    targets = list(names.values())
    for name in METRICS:
        if targets.count(name) > 1:
            raise ValueError(f"more than one trainer name maps onto {name!r}")
    return {name: name for name in METRICS if name not in targets} | names


def rename(values: Mapping[str, object], read: Mapping[str, str]) -> dict[str, float]:
    """One step's values under rlvitals' names, as floats. Only the names in `read` are read; a
    None among them counts as absent, and anything float() can't take is an error."""
    renamed: dict[str, float] = {}
    for trainer_name, name in read.items():
        value = values.get(trainer_name)
        if value is None:
            continue
        if not isinstance(value, SupportsFloat):
            raise TypeError(f"{trainer_name} is {value!r}: not a number")
        renamed[name] = float(value)
    return renamed


def format_event(event: Event) -> str:
    """One line for a person: when, how grave, what was found, and its numbers."""
    numbers = ", ".join(f"{key} {value:.4g}" for key, value in event.details.items())
    return (
        f"rlvitals: step {event.step}: {event.severity}, {event.detector} in {event.metric}"
        f" ({numbers})"
    )


def print_event(event: Event) -> None:
    """The default hook: the event's line on standard error."""
    print(format_event(event), file=sys.stderr)


class Watch:
    """Watches one run. Call step(step, values) once a step with everything the trainer logged
    (a mapping from its names to its values); each event the monitor finds goes to every hook,
    in order, and is returned. A hook that fails is reported as a warning and the others still
    run: like the monitor, a watch never stops a run by itself."""

    def __init__(
        self,
        names: Mapping[str, str] | None = None,
        hooks: Iterable[Hook] = (print_event,),
        monitor: Monitor | None = None,
    ) -> None:
        self.read = reading(names)
        self.hooks = list(hooks)
        self.monitor = monitor or Monitor()

    def step(self, step: int, values: Mapping[str, object]) -> list[Event]:
        """The events that start at `step`, after every hook has seen each of them."""
        events = self.monitor.check(step, rename(values, self.read))
        for event in events:
            for hook in self.hooks:
                try:
                    hook(event)
                except Exception as error:
                    warnings.warn(
                        f"rlvitals: hook {hook!r} failed on {event}: {error!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return events
