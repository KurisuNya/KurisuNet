from pathlib import Path

from kurisuinfo import summary

from kurisunet import get_module
from kurisunet.utils.logger import logger

if __name__ == "__main__":
    dir = Path(__file__).parent
    cfg = {
        "path": "./net.yaml",
        "name": "VAE",
        "kwargs": {
            "img_size": (1, 28, 28),
            "encoder_dims": [512, 256, 128],
            "decoder_dims": [128, 256, 512],
            "z_dim": 10,
        },
        "input_shape": (1, 1, 28, 28),
    }

    module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['name']}:\n{module_summary}")
