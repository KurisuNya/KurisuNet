from pathlib import Path

from kurisuinfo import summary
from loguru import logger

from kurisunet.module import get_module


if __name__ == "__main__":
    dir = Path(__file__).parent
    cfg = {
        "path": "./config.yaml",
        "name": "SimpleCNN",
        "kwargs": {"in_ch": 1, "class_num": 2, "width": 0.5},
        "input_shape": (1, 1, 128, 128),
    }

    module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['name']}:\n{module_summary}")
