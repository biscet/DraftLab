"""Run the stages of the patched SpecForge and llama.cpp Whittle recipe."""

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.coach_core.constants.eagle3 import (
    SPEC_FORGE_HELPER,
    SPEC_FORGE_RECIPE,
    VOLTA_COMPUTE_CAPABILITY,
    VOLTA_MIN_MEMORY_MIB,
)
from scripts.coach_core.types import PathLike


def volta_device() -> str:
    """Select a single visible V100 with 32 GiB of VRAM."""
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,compute_cap,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = {entry.strip() for entry in visible.split(",")} if visible is not None else None
    eligible = []
    for line in inventory.splitlines():
        uuid, compute, memory = (field.strip() for field in line.split(","))
        if (
            compute == VOLTA_COMPUTE_CAPABILITY
            and int(memory) >= VOLTA_MIN_MEMORY_MIB
            and (allowed is None or uuid in allowed)
        ):
            eligible.append(uuid)
    if len(eligible) != 1:
        raise ValueError("Select one 32 GB V100 using its UUID in CUDA_VISIBLE_DEVICES")
    return eligible[0]


def execute(
    args: argparse.Namespace,
    *command: PathLike,
    gpu: str | None = None,
    gguf: Path | None = None,
) -> None:
    """Invoke the selected Python environment against the patched source tree."""
    module_paths = [str(args.specforge)]
    if gguf is not None:
        module_paths.append(str(gguf))
    if inherited := os.environ.get("PYTHONPATH"):
        module_paths.append(inherited)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(module_paths), PYTHONNOUSERSITE="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run([args.python, *command], cwd=args.specforge, env=env, check=True)


def specforge_file(args: argparse.Namespace, relative_path: str) -> Path:
    """Locate an asset supplied by the SpecForge companion patch."""
    path = args.specforge / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {relative_path}. Apply the Whittle patch to SpecForge.")
    return path


def run_on_gpu(args: argparse.Namespace, command: list[PathLike]) -> None:
    """Prevent overlapping capture and training in the same job directory."""
    gpu = volta_device()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / ".eagle3.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Capture or training is already active in this work directory") from error
        execute(args, *command, gpu=gpu)


def prepare_target(args: argparse.Namespace) -> None:
    """Extract the frozen token embedding and LM head from the local teacher."""
    helper = specforge_file(args, SPEC_FORGE_HELPER)
    execute(args, helper, "prepare-target", "--model", args.model, "--target", args.work_dir / "target")


def capture(args: argparse.Namespace) -> None:
    """Record aligned teacher states in a resumable feature cache."""
    helper = specforge_file(args, SPEC_FORGE_HELPER)
    features = args.features or args.work_dir / "features"
    command: list[PathLike] = [
        helper,
        "capture",
        "--model",
        args.model,
        "--exporter",
        args.exporter,
        "--input",
        args.input,
        "--output",
        features,
        "--max-tokens",
        str(args.max_tokens),
        "--shard-size",
        str(args.shard_size),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    run_on_gpu(args, command)


def train(args: argparse.Namespace) -> None:
    """Resolve or run the offline recipe using the captured context limit."""
    recipe = specforge_file(args, SPEC_FORGE_RECIPE)
    target = args.work_dir / "target"
    if not (target / "config.json").is_file():
        raise FileNotFoundError(f"Place the Whittle config.json in {target}")
    features = args.features or args.work_dir / "features"
    command: list[PathLike] = ["-m", "specforge.cli", "train", "--config", recipe]
    if args.plan:
        command.append("--plan")
    command.extend(
        (
            f"model.target_model_path={target}",
            f"data.hidden_states_path={features}",
            f"data.cache_dir={args.work_dir / 'cache'}",
            f"output_dir={args.work_dir / 'runs'}",
        )
    )
    manifest = features / "manifest.json"
    if manifest.is_file():
        max_tokens = int(json.loads(manifest.read_text())["max_tokens"])
        command.append(f"data.max_length={max_tokens}")
    elif not args.plan:
        raise FileNotFoundError(f"Missing capture manifest at {manifest}. Run eagle3 capture first.")
    if args.max_steps is not None:
        command.append(f"training.max_steps={args.max_steps}")
    if args.resume is not None:
        command.extend(("initialization=null", f"resume={args.resume}"))
    if args.plan:
        execute(args, *command)
    else:
        run_on_gpu(args, command)


def export(args: argparse.Namespace) -> None:
    """Publish a complete F16 GGUF at a new path after conversion succeeds."""
    helper = specforge_file(args, SPEC_FORGE_HELPER)
    if args.output.suffix.lower() != ".gguf" or args.output.exists():
        raise ValueError("Select a new .gguf output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = args.work_dir / "target"
    with TemporaryDirectory(prefix=".whittle-eagle3-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "hf"
        result = Path(temporary) / "draft.gguf"
        execute(
            args,
            "-m",
            "specforge.cli",
            "export",
            "--converter",
            "sglang",
            "--checkpoint",
            args.checkpoint,
            "--target-model",
            target,
            "--output-dir",
            stage,
        )
        execute(args, helper, "tag-export", "--draft", stage)
        execute(
            args,
            args.runtime / "convert_hf_to_gguf.py",
            stage,
            "--target-model-dir",
            target,
            "--outtype",
            "f16",
            "--outfile",
            result,
            gguf=args.runtime / "gguf-py",
        )
        os.link(result, args.output)
