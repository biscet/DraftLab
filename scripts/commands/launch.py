#!/usr/bin/env python3
"""Start an explicitly authorized Linux user service with a restoration hook."""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.coach_core.configuration import load_config
from scripts.coach_core.constants.configuration import DEFAULT_UNIT_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-authorized", action="store_true")
    args = parser.parse_args()
    if not args.start_authorized:
        raise SystemExit("Add --start-authorized to start training")
    if sys.platform != "linux":
        raise SystemExit("The background launcher requires Linux systemd")
    config = args.config.resolve()
    cfg = load_config(config)
    root = Path(__file__).resolve().parents[2]
    unit = cfg.get("unit_name", DEFAULT_UNIT_NAME)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", unit):
        raise ValueError("Invalid unit_name")
    restore = [
        cfg["python"], str(root / "coach.py"), "run",
        "--config", str(config), "--restore-only",
    ]
    command = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--collect",
        "--property=Type=exec",
        "--property=TimeoutStopSec=240",
        "--property=KillMode=control-group",
        "--property=UMask=0077",
        "--property=Nice=10",
        "--property=WorkingDirectory=" + str(root),
        "--property=ExecStopPost=" + shlex.join(restore),
        cfg["python"],
        str(root / "coach.py"),
        "run",
        "--config",
        str(config),
        "--start-authorized",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
