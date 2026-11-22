"""P7's pilot: one GRPO run of a small instruct model on GSM8K, with rlvitals watching; with
--fault, one of P8's faults switches on at step --onset (see faults.py).

Writes three files into --out: metrics.jsonl (every training log: its step, the seconds since
training began, and what TRL logged), events.jsonl (rlvitals' events) and run.json (the
settings, package versions and GPU). `rlvitals replay OUT/metrics.jsonl --trl` reads the log
again. Needs Python 3.11+, trl 1.18.2 and rlvitals[transformers]; see experiments/README.md."""

import argparse
import json
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from faults import FAULTS, ONSET, Fault
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from rlvitals.callback import RLVitalsCallback
from rlvitals.hooks import event_body
from rlvitals.monitor import Event
from rlvitals.watch import print_event

SYSTEM = (
    "Solve the math problem. Think step by step, then write the final answer "
    "on the last line as: #### <number>"
)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> float | None:
    """The last number in a text, commas dropped; None if there is none."""
    numbers = NUMBER.findall(text)
    return float(numbers[-1].replace(",", "")) if numbers else None


def correct(completions: list[list[dict[str, str]]], answer: list[str], **_: Any) -> list[float]:
    """1 when a completion's last number equals the reference answer, else 0."""
    rewards: list[float] = []
    for completion, expected in zip(completions, answer, strict=True):
        got = final_number(completion[-1]["content"])
        rewards.append(1.0 if got is not None and got == float(expected) else 0.0)
    return rewards


def problems(count: int | None, seed: int) -> Any:
    """GSM8K's training problems as chat prompts, each with its reference answer."""
    data = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    if count is not None:
        data = data.select(range(count))
    return data.map(
        lambda row: {
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["question"]},
            ],
            "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
        },
        remove_columns=["question"],
    )


class LogFile(TrainerCallback):
    """Appends each training log to a JSON-lines file, with its step and elapsed seconds."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.start = time.monotonic()

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self.start = time.monotonic()  # model loading and the dataset aren't training time

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
        if logs and state.is_world_process_zero:
            record = {"step": state.global_step, "seconds": time.monotonic() - self.start}
            with self.path.open("a") as file:
                file.write(json.dumps(record | dict(logs)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("pilot"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--problems", type=int, default=None, help="default: all 7473")
    parser.add_argument("--completion-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fault", choices=FAULTS, default=None, help="default: a healthy run")
    parser.add_argument("--onset", type=int, default=ONSET)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    config = GRPOConfig(
        output_dir=str(args.out / "trainer"),
        max_steps=args.steps,
        learning_rate=1e-6,
        lr_scheduler_type="constant",  # TRL's default decays it linearly to 0 at max_steps
        beta=0.04,  # TRL's default is 0: no reference model, and no kl to watch
        num_generations=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,  # 32 completions a step: 4 prompts x 8
        max_completion_length=args.completion_tokens,
        temperature=1.0,
        logging_steps=1,  # rlvitals' windows count logs: one log a step
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        fp16=gpu,  # a T4 has no bf16: float32 weights, float16 autocast
        bf16=False,
        gradient_checkpointing=gpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # p8s1: recomputes with grad on
    )
    run = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "versions": {name: version(name) for name in ("trl", "transformers", "torch", "rlvitals")},
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if gpu else None,
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (args.out / "metrics.jsonl").write_text("")  # a rerun starts both files afresh
    (args.out / "events.jsonl").write_text("")

    def record(event: Event) -> None:
        with (args.out / "events.jsonl").open("a") as file:
            file.write(json.dumps(event_body(event)) + "\n")

    fault = Fault(args.fault, args.onset) if args.fault else None
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=fault.reward(correct) if fault else correct,
        args=config,
        train_dataset=problems(args.problems, args.seed),
        callbacks=[
            LogFile(args.out / "metrics.jsonl"),
            RLVitalsCallback(hooks=(print_event, record)),
            *([fault] if fault else []),
        ],
    )
    if fault:
        fault.trainer = trainer
    trainer.train()


if __name__ == "__main__":
    main()
