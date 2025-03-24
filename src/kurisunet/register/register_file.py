from __future__ import annotations

import importlib.util
from pathlib import Path

from loguru import logger

from ..basic.utils import to_relative_path

from ..constants import CONFIG_SUFFIX, PYTHON_SUFFIX


def __exec_module(name: str, file_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec and spec.loader:
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    raise ValueError(f"Failed to execute module {name} from {file_path}")


def __register_from_path(path: Path) -> None:
    from .register_config import register_config

    if path.suffix in PYTHON_SUFFIX:
        logger.info(f"Registering module from {to_relative_path(path)}")
        __exec_module(path.stem, path)
    elif path.suffix in CONFIG_SUFFIX:
        logger.info(f"Registering config from {to_relative_path(path)}")
        register_config(path)
    else:
        logger.warning(f"Unsupported file type: {path.suffix}, skipping {path}")


def register_from_path_list(path_list: list[Path]) -> None:
    for path in filter(lambda x: x.is_file(), path_list):
        __register_from_path(path)
    for path in filter(lambda x: x.is_dir(), path_list):
        logger.info(f"Registering path {to_relative_path(path)}")
        register_from_path_list(list(path.iterdir()))
