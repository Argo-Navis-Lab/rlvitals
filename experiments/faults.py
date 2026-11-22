"""P8's faults: each one switches on at a fixed step of a pilot run and stays on to the end.

lr          the learning rate jumps LR_FACTOR times
unclipped   gradient clipping off and the learning rate UNCLIPPED_LR_FACTOR times
reward      the reward function returns 0 for every completion (a broken answer parser)
temperature generation samples at LOW_TEMPERATURE; the training pass keeps 1, which TRL fixes
            when the trainer is built
fp16        a middle layer's MLP input grows GROWTH times a step in the training pass, until
            float16 overflows

The strengths are set on the development runs and fixed before any held-out run."""

from collections.abc import Callable
from typing import Any

import torch
from transformers import TrainerCallback

ONSET = 60
LR_FACTOR = 100.0
UNCLIPPED_LR_FACTOR = 10.0
LOW_TEMPERATURE = 0.1
GROWTH = 1.1
FAULTS = ("lr", "unclipped", "reward", "temperature", "fp16")

Reward = Callable[..., list[float]]


class Fault(TrainerCallback):
    """One fault, switched on at the start of step `onset` (the step TRL logs as `onset`)."""

    def __init__(self, name: str, onset: int = ONSET) -> None:
        if name not in FAULTS:
            raise ValueError(f"unknown fault {name!r}; known: {', '.join(FAULTS)}")
        if onset < 2:
            raise ValueError(f"onset must be 2 or more, not {onset}")
        self.name = name
        self.onset = onset
        self.on = False
        self.scale = 1.0
        self.trainer: Any = None  # set once the trainer exists (the temperature fault needs it)

    def reward(self, func: Reward) -> Reward:
        """The reward function, returning 0 for everything once the reward fault is on."""

        def wrapped(*args: Any, **kwargs: Any) -> list[float]:
            rewards = func(*args, **kwargs)
            return [0.0] * len(rewards) if self.on and self.name == "reward" else rewards

        wrapped.__name__ = func.__name__  # TRL logs each reward under its function's name
        return wrapped

    def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        # The Trainer logs the rate its schedule last set. Raised here, at the end of the step
        # before the onset (global_step doesn't count this step yet), the schedule sets it
        # right after: the onset is the first step to use it and to log it.
        if state.global_step + 2 == self.onset and self.name in ("lr", "unclipped"):
            factor = LR_FACTOR if self.name == "lr" else UNCLIPPED_LR_FACTOR
            raise_lr(kwargs["lr_scheduler"], factor)

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step + 1 < self.onset:  # global_step counts the steps already done
            return
        if not self.on:
            self.on = True
            self.start(args, **kwargs)
        if self.name == "fp16":
            self.scale *= GROWTH

    def start(self, args: Any, model: Any, **_: Any) -> None:
        if self.name == "unclipped":
            args.max_grad_norm = 0.0  # the Trainer clips only when it is above 0
        elif self.name == "temperature":
            self.trainer.generation_config.temperature = LOW_TEMPERATURE
        elif self.name == "fp16":
            layers = model.model.layers
            norm = layers[len(layers) // 2].post_attention_layernorm
            norm.register_forward_hook(self.grow)

    def grow(self, module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        """Scales the norm's output in passes that build a graph: training, not sampling."""
        return output * self.scale if torch.is_grad_enabled() else output


def raise_lr(lr_scheduler: Any, factor: float) -> None:
    """Multiplies the rate of every later step of the schedule."""
    lr_scheduler.base_lrs = [lr * factor for lr in lr_scheduler.base_lrs]
