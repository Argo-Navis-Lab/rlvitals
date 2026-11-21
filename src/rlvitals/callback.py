"""A Hugging Face callback: watches a Trainer's logs (TRL's GRPOTrainer included) while it
trains. It needs transformers (`pip install rlvitals[transformers]`); the rest of rlvitals
doesn't, so `import rlvitals` never imports this module."""

from collections.abc import Iterable, Mapping
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from rlvitals.monitor import Event, Monitor
from rlvitals.watch import TRL_NAMES, Hook, Watch, print_event


class RLVitalsCallback(TrainerCallback):
    """Feeds each training log to a watch, at the trainer's global step. Its steps are the
    logged ones (every `logging_steps`; TRL's GRPOConfig logs every 10), and the values are
    what the trainer logged there: loss and TRL's metrics averaged since the last log,
    grad_norm the last step's. A log with none of the watched names (an eval log, the summary
    at the end) is not a training step and is skipped.

    Advice only by default. stop_on_halt stops training at a HALT; save_on_snapshot saves a
    checkpoint at a SNAPSHOT. Every process runs the same watch on the same gathered values,
    so all of them stop together; only the main process calls the hooks. A resumed run starts
    a fresh monitor."""

    def __init__(
        self,
        names: Mapping[str, str] | None = TRL_NAMES,
        hooks: Iterable[Hook] = (print_event,),
        monitor: Monitor | None = None,
        *,
        stop_on_halt: bool = False,
        save_on_snapshot: bool = False,
    ) -> None:
        self.watch = Watch(names, hooks, monitor)
        self.stop_on_halt = stop_on_halt
        self.save_on_snapshot = save_on_snapshot
        self.events: list[Event] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            self.watch.hooks = []  # the other processes find the same events: stay silent

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs or not any(name in logs for name in self.watch.read):
            return
        events = self.watch.step(state.global_step, logs)
        self.events.extend(events)
        severities = {event.severity for event in events}
        if self.stop_on_halt and "HALT" in severities:
            control.should_training_stop = True
        if self.save_on_snapshot and "SNAPSHOT" in severities:
            control.should_save = True
