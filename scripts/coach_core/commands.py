"""Build explicit capture and training argument lists without launching them."""

from pathlib import Path

from scripts.coach_core.constants.paths import PROJECT_ROOT
from scripts.coach_core.constants.recipe import UNIFORM_OBJECTIVE
from scripts.coach_core.files import digest
from scripts.coach_core.types import Config, Record


def capture_command(cfg: Config, rows: list[Record]) -> list[str]:
    maximum = max(row["prompt_tokens"] for row in rows)
    assert maximum + cfg["max_new_tokens"] <= cfg["capture_context"]
    return [
        cfg["capture_binary"],
        "-m",
        cfg["target_model"],
        "--device",
        ",".join("CUDA" + str(i) for i in range(len(cfg["gpu_uuids"]))),
        "--split-mode",
        "none",
        "--gpu-layers",
        "all",
        "--override-tensor",
        "^per_layer_token_embd[.]weight$=CUDA1,.*=CUDA0" if len(cfg["gpu_uuids"]) > 1 else ".*=CUDA0",
        "--lazy-mode",
        "off",
        "--no-host",
        "--fit",
        "off",
        "-c",
        str(cfg["capture_context"]),
        "-b",
        "512",
        "-ub",
        "256",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "bf16",
        "--cache-type-v",
        "bf16",
    ]


def training_command(
    cfg: Config,
    plan: Record,
    run: Path,
    work: Path,
    shard: Record,
    epoch: int,
    learning_rate: float,
) -> list[str]:
    completed_epochs = epoch + 1 - cfg.get("initial_epoch", 0)
    estimated_steps = completed_epochs * plan["steps_per_epoch_estimate"]
    estimated_steps += cfg.get("seed_step", 0)
    cmd = [
        cfg["python"],
        str(PROJECT_ROOT / "coach.py"),
        "train-shard",
        "--run-dir",
        str(work),
        "--state-dir",
        str(run / "outputs"),
        "--plan-hash",
        digest(run / "plan.json"),
        "--stage-key",
        shard["key"],
        "--target-model",
        cfg["target_model"],
        "--initial-draft",
        cfg["initial_draft"],
        "--steps",
        str(estimated_steps),
        "--batch-size",
        str(cfg["batch_size"]),
        "--context",
        str(cfg["training_context"]),
        "--objective",
        cfg.get("training_objective", UNIFORM_OBJECTIVE),
        "--learning-rate",
        str(learning_rate),
        "--pool-field",
        "sampling_group",
        "--min-train",
        "1",
        "--min-eval",
        "16",
        "--min-total",
        "17",
        "--output-name",
        cfg["output_name"],
        "--model-name",
        cfg["model_name"],
    ]
    if cfg.get("seed_checkpoint"):
        cmd += [
            "--seed-checkpoint",
            cfg["seed_checkpoint"],
            "--seed-plan-hash",
            cfg["seed_plan_sha256"],
            "--seed-step",
            str(cfg["seed_step"]),
        ]
    return cmd
