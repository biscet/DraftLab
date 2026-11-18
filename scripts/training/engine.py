"""Run one seeded pass over a shard with learning-rate warmup and AMP retries."""

import hashlib
import random
import time
from functools import partial

import torch

from scripts.coach_core.types.training import BatchResult
from scripts.training.amp_step import amp_train_step


def create_optimizer(model, learning_rate):
    fc = []
    body = []
    for name, param in model.named_parameters():
        if name.startswith(("fc.", "hidden_norm.")):
            fc.append(param)
        else:
            body.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": body, "lr": learning_rate, "base_lr": learning_rate},
            {"params": fc, "lr": learning_rate * 3, "base_lr": learning_rate * 3},
        ],
        weight_decay=0.01,
        foreach=True,
    )
    return optimizer


def train_batch(args, backend, batch, step, emit):
    model = backend.model
    objective = backend.objective
    optimizer = backend.optimizer
    scaler = backend.scaler
    model.train()
    width = 8 if step % 4 == 0 else 4
    anchors = [
        random.randint(sample["prompt_n"], len(sample["input_ids"]) - width - 1)
        for sample in batch
    ]
    scale = min(1.0, step / 100)
    for index, group in enumerate(optimizer.param_groups):
        group["base_lr"] = args.learning_rate * (3 if index == 1 else 1)
        group["lr"] = group["base_lr"] * scale
    def overflow_event(attempt, old_scale, new_scale, step=step, batch=batch):
        emit(
            "AMP_OVERFLOW_RETRY",
            step=step,
            attempt=attempt,
            scale_before=old_scale,
            scale_after=new_scale,
            batch_ids=[sample["id"] for sample in batch],
        )
    loss, metrics, retries = amp_train_step(
        partial(objective.compute, batch, anchors, width, training=True),
        model.parameters(),
        optimizer,
        scaler,
        on_overflow=overflow_event,
    )
    return loss, metrics, retries


def train_batches(args, backend, samples, start_step, emit, started):
    history = []
    amp_retries = 0
    def training_order(row):
        identity = args.stage_key + row["id"]
        return hashlib.sha256(identity.encode()).hexdigest()

    ordered = sorted(samples, key=training_order)
    batches = [
        ordered[start : start + args.batch_size]
        for start in range(0, len(ordered), args.batch_size)
    ]
    for local_step, batch in enumerate(batches):
        step = start_step + local_step
        loss, metrics, retries = train_batch(args, backend, batch, step, emit)
        amp_retries += retries
        row = {
            "step": step,
            "loss": float(loss.detach()),
            **metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        if step <= 20 or step % 50 == 0:
            history.append(row)
            elapsed = time.monotonic() - started
            seconds_per_step = elapsed / (step - start_step + 1)
            steps_left = max(0, args.steps - step)
            emit(
                "TRAIN_STEP",
                **row,
                epoch_step_estimate=args.steps,
                estimated_remaining_seconds=seconds_per_step * steps_left,
                peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
    return BatchResult(
        step=step,
        history=history,
        overflow_retries=amp_retries,
        accepted_ids=[sample["id"] for sample in ordered],
    )
