from copy import deepcopy
from typing import Callable

from kurisuinfo import CustomizedModuleName
import torch.nn as nn

from ..net.module import PipelineModule
from ..utils.logger import get_logger

logger = get_logger("Utils")


def get_module_name(module: nn.Module | CustomizedModuleName):
    """Get the name of a module."""
    if isinstance(module, CustomizedModuleName):
        return module.get_module_name()
    if isinstance(module, nn.Module):
        return module.__class__.__name__


def apply_module(
    module: nn.Module,
    func: Callable[[nn.Module], None],
    filter: Callable[[nn.Module], bool] | None = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Apply a function to all modules in a module.
    Use `filter` to filter modules to apply the function.
    """
    if inplace:
        logger.warning("The input module will be modified in-place")
    else:
        module = deepcopy(module)
    for m in module.modules():
        if filter is None or filter(m):
            func(m)
    return module


def drop_module(module: nn.Module, inplace: bool = False) -> nn.Module:
    """Drop all unused or set to dropped modules in PipelineModule."""
    return apply_module(
        module,
        lambda m: m.drop(resort=True),  # type: ignore
        filter=lambda m: isinstance(m, PipelineModule),
        inplace=inplace,
    )
