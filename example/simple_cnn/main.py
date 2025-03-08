from pathlib import Path

from kurisuinfo import summary
from loguru import logger
import yaml

from kurisunet.module import ModuleRegister, register_wrapper


def set_logger_level(level: str):
    import sys

    logger.remove()
    logger.add(sys.stderr, level=level.upper())


@register_wrapper
def resize_wrapper(width):
    def make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def is_backbone_module(module):
        if module in ["ConvBNReLU"]:
            return True
        return False

    def is_out_module(module):
        if module in ["Classifier"]:
            return True
        return False

    def resize_backbone(args, resize_in_ch=True):
        _in, out, *args = args
        if resize_in_ch:
            return make_divisible(_in * width, 8), make_divisible(out * width, 8), *args
        return _in, make_divisible(out * width, 8), *args

    def resize_out(args):
        _in, *args = args
        return make_divisible(_in * width, 8), *args

    def wrapper(layers):
        for i, (_, module, args, _) in enumerate(layers):
            if is_backbone_module(module) and i == 0:
                layers[i][2] = resize_backbone(args, resize_in_ch=False)
            elif is_backbone_module(module):
                layers[i][2] = resize_backbone(args)
            elif is_out_module(module):
                layers[i][2] = resize_out(args)
        return layers

    return wrapper


if __name__ == "__main__":
    cfg = {
        "module": "SimpleCNN",
        "args": [],
        "kwargs": {"in_ch": 3, "class_num": 2, "width": 1},
        "input_shape": (1, 3, 224, 224),
        "path": "./config.yaml",
    }

    set_logger_level("info")
    config = yaml.safe_load(open(Path(__file__).parent / cfg["path"]))
    ModuleRegister.register_config(config)
    module = ModuleRegister.get(cfg["module"])(*cfg["args"], **cfg["kwargs"])
    module_summary = str(summary(module, cfg["input_shape"], verbose=0))
    logger.info(f"Summary of {cfg['module']}:\n{module_summary}")
