from types import FunctionType
from typing import Any

import torch.nn as nn

from ....basic.types import Env
from ...types import CustomModule, LayerModule, Module
from ...utils import eval_string


def __check_module(module: Any) -> None:
    if not isinstance(module, (str, CustomModule, type, FunctionType, nn.Module)):
        msg = f"Invalid module {module}, should be str/CustomModule/type/callable/nn.Module"
        raise ValueError(msg)


def __parse_module(module: LayerModule, env: Env) -> Module:
    if isinstance(module, str):
        module = eval_string(module, env)
    if isinstance(module, CustomModule):
        return lambda *a, **k: module.get_module(*a, **k)
    if isinstance(module, type):
        return module
    if isinstance(module, FunctionType):
        return lambda *a, **k: lambda *args: module(*args, *a, **k)
    if isinstance(module, nn.Module):
        return lambda: module
    msg = f"Invalid module {module}, should be str/CustomModule/type/callable/nn.Module"
    raise ValueError(msg)


def parse_module(module: LayerModule, env: Env | None) -> Module:
    """Parse the expressions in the module."""
    __check_module(module)
    return __parse_module(module, env or {})
