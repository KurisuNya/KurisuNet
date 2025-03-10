import sys
from typing import Literal

from loguru import logger

LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def set_logger(level: LOG_LEVELS, show_line: bool = False):
    def dict_add(d1, d2):
        d1.update(d2)
        return d1

    hide_line = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    log_sources = [
        {"sink": sys.stderr, "level": level.upper()},
    ]

    logger.remove()
    if not show_line:
        log_sources = [dict_add(s, {"format": hide_line}) for s in log_sources]
    for source in log_sources:
        logger.add(**source)
