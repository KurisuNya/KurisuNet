from pathlib import Path

from kurisuinfo import summary
from loguru import logger

from kurisunet.module import get_main_module


def set_logger_level(level: str, show_line: bool = False):
    import sys

    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    logger.remove()
    if not show_line:
        logger.add(sys.stderr, level=level.upper(), format=format)
    else:
        logger.add(sys.stderr, level=level.upper())


if __name__ == "__main__":
    cfg = {
        "module": "SimpleCNN",
        "input_shape": (1, 3, 224, 224),
        "path": "./config.yaml",
    }

    set_logger_level("info")
    module = get_main_module(Path(__file__).parent / cfg["path"])
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['module']}:\n{module_summary}")
