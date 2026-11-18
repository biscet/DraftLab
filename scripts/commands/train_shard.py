#!/usr/bin/env python3
"""Parse one shard's training options and launch its backend."""

import argparse
from pathlib import Path

from scripts.coach_core.constants.recipe import TRAINING_OBJECTIVES, UNIFORM_OBJECTIVE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--plan-hash", required=True)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument("--seed-plan-hash")
    parser.add_argument("--seed-step", type=int)
    parser.add_argument("--stage-key", required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=384)
    parser.add_argument("--objective", choices=TRAINING_OBJECTIVES, default=UNIFORM_OBJECTIVE)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--initial-draft", type=Path, required=True)
    parser.add_argument("--output-name", default="draft-candidate-F16.gguf")
    parser.add_argument("--model-name", default="DFlash-Coach-candidate")
    parser.add_argument("--pool-field", default="project")
    parser.add_argument("--min-train", type=int, default=490)
    parser.add_argument("--min-eval", type=int, default=55)
    parser.add_argument("--min-total", type=int, default=550)
    parser.add_argument("--validate-data-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from scripts.training.pipeline import train

    train(args)


if __name__ == "__main__":
    main()
