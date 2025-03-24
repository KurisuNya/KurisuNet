from pathlib import Path

from kurisuinfo import summary
from loguru import logger

from kurisunet import get_module

from kurisunet.utils.logger import set_logger
from kurisunet.utils.debug import register_log_shape_hook

import torch


if __name__ == "__main__":
    dir = Path(__file__).parent
    cfg = {
        "path": "./config.yaml",
        "name": "SimpleCNN",
        "kwargs": {"in_ch": 1, "class_num": 2},
        "input_shape": (1, 1, 128, 128),
    }

    set_logger("DEBUG")
    # register_log_shape_hook()

    module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['name']}:\n{module_summary}")

    # from test import Test
    # import cProfile
    # import torch
    #
    # old_module = Test(1)
    # module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])
    # input = torch.randn(*cfg["input_shape"])
    #
    # cProfile.run("old_module(input)", sort="cumtime")
    # cProfile.run("old_module(input)", sort="cumtime")
    # cProfile.run("module(input)", sort="cumtime")
