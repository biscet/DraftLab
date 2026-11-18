"""Resolve configuration paths and expand runtime arguments without a shell."""

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

from scripts.coach_core.constants.configuration import DEFAULT_CONFIG, PATH_FIELDS
from scripts.coach_core.constants.recipe import TRAINING_OBJECTIVES
from scripts.coach_core.types import Config, PathLike


def load_config(path: PathLike) -> Config:
    """Resolve filesystem settings relative to the configuration file."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg = {**deepcopy(DEFAULT_CONFIG), **cfg}
    resolve_paths(cfg, path.parent)
    cfg["python"] = resolve_python(cfg["python"], path.parent)
    cfg["mentor"] = resolve_mentor(cfg.get("mentor"), path.parent)
    if cfg["training_objective"] not in TRAINING_OBJECTIVES:
        raise ValueError("training_objective must be uniform or dpace")
    if Path(cfg["output_name"]).name != cfg["output_name"]:
        raise ValueError("output_name must be a filename")
    return cfg


def absolute_path(value: Path, config_dir: Path) -> str:
    if not value.is_absolute():
        value = config_dir / value
    return str(value.resolve())


def resolve_paths(cfg: Config, config_dir: Path) -> None:
    for name in PATH_FIELDS:
        if not cfg.get(name):
            continue
        expanded = os.path.expandvars(os.path.expanduser(cfg[name]))
        cfg[name] = absolute_path(Path(expanded), config_dir)


def resolve_python(executable: str, config_dir: Path) -> str:
    if "/" in executable:
        value = Path(executable).expanduser()
        return absolute_path(value, config_dir)
    return shutil.which(executable) or executable


def resolve_mentor(value: Config | None, config_dir: Path) -> Config:
    mentor = dict(value or {})
    if mentor.get("key_file"):
        key_path = Path(mentor["key_file"]).expanduser()
        mentor["key_file"] = absolute_path(key_path, config_dir)
    return mentor


def command_argv(cfg: Config) -> list[str]:
    """Expand a configured argument list without a shell."""
    values = {k: str(v) for k, v in cfg.items() if isinstance(v, (str, int, float))}
    return [str(value).format_map(values) for value in cfg["server_argv"]]
