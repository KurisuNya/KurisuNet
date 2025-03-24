from __future__ import annotations

from typing import Any, Callable

from ....basic.types import Env
from ...types import CustomModule, LayerModule, Module
from ...utils import eval_string


def __check_module(module: Any) -> None:
    if not isinstance(module, (str, type, Callable, CustomModule)):
        msg = f"Invalid module {module}, should be str/type/callable/custom module"
        raise ValueError(msg)


def __parse_module(module: LayerModule, env: Env) -> Module:
    if isinstance(module, str):
        module = eval_string(module, env)
    if isinstance(module, CustomModule):
        return lambda *a, **k: module.get_module(*a, **k)
    if isinstance(module, type):
        return module
    if isinstance(module, Callable):
        return lambda *a, **k: lambda *args: module(*args, *a, **k)
    raise ValueError(f"Invalid module {module}, should be type/callable")


def parse_module(module: LayerModule, env: Env | None) -> Module:
    """Parse the expressions in the module."""
    __check_module(module)
    return __parse_module(module, env or {})
