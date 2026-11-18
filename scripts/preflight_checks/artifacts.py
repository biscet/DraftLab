"""Check the immutable configuration, weights, code and runtime binaries."""

from pathlib import Path

from scripts.coach_core.constants.paths import PROJECT_ROOT
from scripts.coach_core.constants.recipe import UNIFORM_OBJECTIVE
from scripts.coach_core.files import digest


def require_digest(path, expected, message):
    if digest(path) != expected:
        raise ValueError(message)


def check_configuration(cfg, plan):
    planned = {"training_objective": UNIFORM_OBJECTIVE, **plan["config"]}
    if {"training_objective": UNIFORM_OBJECTIVE, **cfg} != planned:
        raise ValueError("Configuration changed; prepare a new run_dir")

    for key in ("production_config", "production_launcher"):
        path = cfg.get(key)
        if path:
            require_digest(path, plan[key + "_sha256"], key + " changed")


def check_models(cfg, plan):
    weights = (
        ("target", cfg["target_model"]),
        ("initial_draft", cfg["initial_draft"]),
    )
    for name, path in weights:
        require_digest(
            path,
            plan[name + "_sha256"],
            name + " weights changed",
        )

    seed = cfg.get("seed_checkpoint")
    if seed:
        require_digest(
            seed,
            plan["seed_checkpoint_sha256"],
            "Seed checkpoint changed",
        )


def check_source_inputs(cfg, plan):
    dataset = Path(cfg["dataset"])
    for name, expected in plan["source_hashes"].items():
        require_digest(dataset / name, expected, "Input changed: " + name)


def check_runtime(plan):
    for name, expected in plan["code_hashes"].items():
        path = PROJECT_ROOT / name
        require_digest(path, expected, "Code changed: " + name)

    for name, expected in plan["binaries"].items():
        require_digest(name, expected, "Binary changed: " + name)
