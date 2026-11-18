"""Capture, train and export a Whittle EAGLE-3 draft offline."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.coach_core.constants.eagle3 import (
    DEFAULT_CAPTURE_TOKENS,
    DEFAULT_SHARD_SIZE,
    MAX_CAPTURE_TOKENS,
)
from scripts.training import eagle3


def absolute_path(value: str) -> Path:
    """Keep the virtual environment's Python symlink intact."""
    return Path(value).expanduser().absolute()


def positive_int(value: str) -> int:
    """Read a positive CLI count."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    """Read the selected stage and its inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--specforge", type=absolute_path, required=True, help="Patched SpecForge checkout")
    common.add_argument("--work-dir", type=absolute_path, required=True, help="Job directory")
    common.add_argument("--python", type=absolute_path, default=sys.executable, help="SpecForge Python executable")

    commands = parser.add_subparsers(dest="action", required=True)

    target = commands.add_parser("target", parents=[common], help="Extract frozen embedding and LM head")
    target.add_argument("--model", type=absolute_path, required=True)
    target.set_defaults(run=eagle3.prepare_target)

    capture = commands.add_parser("capture", parents=[common], help="Capture GGUF teacher states")
    capture.add_argument("--model", type=absolute_path, required=True)
    capture.add_argument("--exporter", type=absolute_path, required=True)
    capture.add_argument("--input", type=absolute_path, required=True)
    capture.add_argument("--features", type=absolute_path)
    capture.add_argument("--max-tokens", type=positive_int, default=DEFAULT_CAPTURE_TOKENS)
    capture.add_argument("--shard-size", type=positive_int, default=DEFAULT_SHARD_SIZE)
    capture.add_argument("--limit", type=positive_int)
    capture.set_defaults(run=eagle3.capture)

    train = commands.add_parser("train", parents=[common], help="Train the draft from saved teacher states")
    train.add_argument("--features", type=absolute_path)
    train.add_argument("--resume", type=absolute_path)
    train.add_argument("--max-steps", type=positive_int)
    train.add_argument("--plan", action="store_true", help="Resolve the recipe without launching a CUDA worker")
    train.set_defaults(run=eagle3.train)

    export = commands.add_parser("export", parents=[common], help="Convert a checkpoint to FP16 GGUF")
    export.add_argument("--runtime", type=absolute_path, required=True, help="Patched llama.cpp checkout")
    export.add_argument("--checkpoint", type=absolute_path, required=True)
    export.add_argument("--output", type=absolute_path, required=True)
    export.set_defaults(run=eagle3.export)

    args = parser.parse_args()
    if args.action == "capture" and not 2 <= args.max_tokens <= MAX_CAPTURE_TOKENS:
        parser.error(f"--max-tokens must be between 2 and {MAX_CAPTURE_TOKENS}")
    return args


def main() -> None:
    """Dispatch the selected EAGLE-3 stage and preserve worker exit status."""
    args = parse_args()
    try:
        args.run(args)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
