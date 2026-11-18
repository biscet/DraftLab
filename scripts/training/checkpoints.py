"""Copy model state without retaining gradients or live GPU storage."""


import random

import torch

from scripts.coach_core.constants.recipe import UNIFORM_OBJECTIVE
from scripts.coach_core.types.training import TrainingProgress
from scripts.training.checkpoint_seed import validate_seed


def snapshot_model(model, *, half=False):
    state = {}
    for name, value in model.state_dict().items():
        value = value.detach()
        if half:
            value = value.half()
        state[name] = value.cpu().clone()
    return state


def restore_checkpoint(args, backend):
    checkpoint = args.state_dir / "last-state.pt"
    seeding = not checkpoint.exists() and args.seed_checkpoint is not None
    if not checkpoint.exists() and not seeding:
        return None

    source = args.seed_checkpoint if seeding else checkpoint
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if seeding:
        validate_seed(saved, args.seed_plan_hash, args.seed_step)
    elif saved["plan_hash"] != args.plan_hash:
        raise RuntimeError("Checkpoint belongs to another frozen plan")
    elif saved.get("objective", UNIFORM_OBJECTIVE) != args.objective:
        raise RuntimeError("Checkpoint belongs to a different training objective")

    backend.model.load_state_dict(saved["model"])
    backend.optimizer.load_state_dict(saved["optimizer"])
    backend.scaler.load_state_dict(saved["scaler"])
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for key in (
        "model", "optimizer", "scaler",
        "python_rng", "torch_rng", "cuda_rng",
    ):
        saved.pop(key, None)
    return saved


def initial_progress(saved, before, model):
    first_step = saved["step"] + 1 if saved else 1
    bad = saved.get("bad", 0) if saved else 0
    if first_step == 1:
        return TrainingProgress(
            first_step=1,
            best_step=0,
            best_metrics=before,
            best_model=snapshot_model(model, half=True),
            baseline=before,
            bad=bad,
        )

    return TrainingProgress(
        first_step=first_step,
        best_step=saved["best_step"],
        best_metrics=saved["best_metrics"],
        best_model=saved["best_model"],
        baseline=saved["baseline"],
        bad=bad,
    )


def select_best(progress, model, metrics, step):
    if metrics["loss"] < progress.best_metrics["loss"]:
        progress.best_step = step
        progress.best_metrics = metrics
        progress.best_model = snapshot_model(model, half=True)
        progress.bad = 0
    else:
        progress.bad += 1


def commit_checkpoint(args, backend, progress, current_state, result):
    record = {
        "step": result["steps"],
        "stage_key": args.stage_key,
        "plan_hash": args.plan_hash,
        "objective": args.objective,
        "model": current_state,
        "best_model": progress.best_model,
        "optimizer": backend.optimizer.state_dict(),
        "scaler": backend.scaler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "best_step": progress.best_step,
        "best_metrics": progress.best_metrics,
        "baseline": progress.baseline,
        "bad": progress.bad,
        "result": result,
    }
    temporary = args.state_dir / "last-state.tmp"
    torch.save(record, temporary)
    temporary.replace(args.state_dir / "last-state.pt")
