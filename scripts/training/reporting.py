"""Write progress events and assemble the trainer's stable JSON report."""

import json
import time

import torch


def emit_progress(run, event, **values):
    row = {"event": event, **values}
    print(json.dumps(row), flush=True)
    p = run / "evidence/training-progress.json"
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(row, indent=2))
    temp.replace(p)


def training_report(
    args, backend, progress, batches, before, last, validation_count, started,
):
    return {
        "stage_key": args.stage_key,
        "baseline": progress.baseline,
        "steps": batches.step,
        "best_step": progress.best_step,
        "before": before,
        "last": last,
        "best": progress.best_metrics,
        "seconds": time.monotonic() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "trainable_parameters": sum(p.numel() for p in backend.model.parameters()),
        "train_examples": len(batches.accepted_ids),
        "eval_examples": validation_count,
        "initial_draft": str(args.initial_draft),
        "learning_rate": args.learning_rate,
        "objective": args.objective,
        "amp_overflow_retries": batches.overflow_retries,
        "amp_final_scale": backend.scaler.get_scale(),
        "pool_field": args.pool_field,
        "accepted_training_ids": batches.accepted_ids,
        "method": "full draft fine-tuning, FP32 master weights, FP16 autocast; frozen Q5 target embedding and LM head",
        "history": batches.history,
    }
