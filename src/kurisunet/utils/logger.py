import sys
from typing import Literal

from loguru import logger as base_logger

LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_NAMES = Literal["KurisuNet", "Register", "Module", "Converter", "Parser", "Utils"]

logger = base_logger.bind(name="KurisuNet")


def get_logger(name: LOG_NAMES):
    """Get a logger with the specified name."""
    if name == "KurisuNet":
        return logger
    return base_logger.bind(name=name)


def set_logger(level: LOG_LEVELS, names: tuple[LOG_NAMES, ...] = LOG_NAMES.__args__):
    """Set the logger level and names to display."""
    name_len = max(len(name) for name in LOG_NAMES.__args__)
    level_len = max(len(level) for level in LOG_LEVELS.__args__)
    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<yellow>{{extra[name]: <{name_len}}}</yellow> | "
        f"<level>{{level: <{level_len}}}</level> | "
        "<level>{message}</level>"
    )

    def show(record):
        if "name" not in record["extra"]:
            return False
        return record["extra"]["name"] in names

    sources = [
        {"sink": sys.stderr, "level": level.upper()},
    ]

    base_logger.remove()
    for source in [{**s, "format": format, "filter": show} for s in sources]:
        base_logger.add(**source)
