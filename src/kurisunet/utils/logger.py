from pathlib import Path
import sys
from typing import Literal

from loguru import logger as base_logger

LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_NAMES = Literal[
    "KurisuNet", "Register", "Module", "SubModules", "Converter", "Layers", "Utils"
]
default_enabled = ("KurisuNet", "Register", "Module", "Converter", "Utils")

logger = base_logger.bind(name="KurisuNet")


def get_logger(name: LOG_NAMES):
    """Get a logger with the specified name."""
    if name == "KurisuNet":
        return logger
    return base_logger.bind(name=name)


def set_logger(
    level: LOG_LEVELS,
    names: tuple[LOG_NAMES, ...] = default_enabled,
    log_file: Path | str | None = None,
    log_file_rotation: str | None = None,
):
    """
    Set the logger level and names to display.
    Default enabled names: KurisuNet, Register, Module, Converter, Utils
    SubModules and Layers are disabled by default because they are too verbose.
    """
    name_len = max(len(name) for name in LOG_NAMES.__args__)
    level_len = max(len(level) for level in LOG_LEVELS.__args__)
    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<yellow>{{extra[name]: <{name_len}}}</yellow> | "
        f"<level>{{level: <{level_len}}}</level> | "
        "<level>{message}</level>"
    )

    def name_filter(record):
        if "name" not in record["extra"]:
            return False
        return record["extra"]["name"] in names

    def file_source(log_file, rotation):
        file_source = {"sink": log_file, "level": level.upper()}
        if rotation:
            file_source["rotation"] = rotation
        return file_source

    sources = [{"sink": sys.stderr, "level": level.upper()}]
    if log_file:
        sources.append(file_source(log_file, log_file_rotation))

    base_logger.remove()
    for source in [{**s, "format": format, "filter": name_filter} for s in sources]:
        base_logger.add(**source)
