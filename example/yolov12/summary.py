from pathlib import Path

from kurisuinfo import summary

from kurisunet import get_module
from kurisunet.utils.logger import logger


if __name__ == "__main__":
    dir = Path(__file__).parent
    cfg = {
        "path": "./net.yaml",
        "name": "YOLOv12",
        "kwargs": {"in_ch": 3, "class_num": 80, "scale": "n"},
        "input_shape": (1, 3, 512, 512),
    }

    module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['name']}:\n{module_summary}")
