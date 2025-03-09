from pathlib import Path

from kurisuinfo import summary
from loguru import logger
import yaml

from kurisunet.module import get_main_module


def set_logger_level(level: str):
    import sys

    logger.remove()
    logger.add(sys.stderr, level=level.upper())


if __name__ == "__main__":
    cfg = {
        "module": "SimpleCNN",
        "input_shape": (1, 3, 224, 224),
        "path": "./config.yaml",
    }

    set_logger_level("info")
    config = yaml.safe_load(open(Path(__file__).parent / cfg["path"]))
    module = get_main_module(config)
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['module']}:\n{module_summary}")
