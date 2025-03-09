from pathlib import Path

from kurisuinfo import summary
from loguru import logger
import yaml

from kurisunet.module import ModuleRegister


def set_logger_level(level: str):
    import sys

    logger.remove()
    logger.add(sys.stderr, level=level.upper())


if __name__ == "__main__":
    cfg = {
        "module": "SimpleCNN",
        "args": [],
        "kwargs": {"in_ch": 3, "class_num": 2, "width": 0.5},
        "input_shape": (1, 3, 224, 224),
        "path": "./config.yaml",
    }

    set_logger_level("info")
    config = yaml.safe_load(open(Path(__file__).parent / cfg["path"]))
    ModuleRegister.register_config(config)
    module = ModuleRegister.get(cfg["module"])(*cfg["args"], **cfg["kwargs"])
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['module']}:\n{module_summary}")
