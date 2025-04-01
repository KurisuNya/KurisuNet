from typing import Iterable, Literal

from kurisuinfo import CustomizedModuleName
from torch import Tensor
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger("Utils")


def __log_shape_hook(module, input, output):
    def get_shape(x):
        if isinstance(x, Tensor):
            return (
                str(x.shape)
                .replace("torch.Size", "")
                .replace("([", "(")
                .replace("])", ")")
            )
        if isinstance(x, tuple):
            return str([get_shape(i) for i in x]).replace("'", "")
        return str(x)

    def get_name(module: nn.Module | CustomizedModuleName):
        if isinstance(module, CustomizedModuleName):
            return module.get_module_name()
        if isinstance(module, nn.Module):
            return module.__class__.__name__

    input_shape = get_shape(input[0] if len(input) == 1 else input)
    output_shape = get_shape(output)
    module_name = get_name(module)
    info = "Module: {:<20} Shape: {} -> {}"
    logger.debug(info.format(module_name, input_shape, output_shape))


def register_log_shape_hook(modules: Iterable[nn.Module] | Literal["all"] = "all"):
    if modules == "all":
        nn.modules.module.register_module_forward_hook(__log_shape_hook)
        return
    for module in modules:
        module.register_forward_hook(__log_shape_hook)
