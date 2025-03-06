from copy import deepcopy
from typing import Callable

from loguru import logger
import torch.nn as nn

from .config import parse_args, parse_layers
from .module import LambdaModule, OutputModule, StreamModule

ModuleLike = type[nn.Module] | Callable[..., nn.Module]


class Register:
    __modules = {}

    @staticmethod
    def register(name: str, module: ModuleLike):
        if name in Register.__modules:
            raise ValueError(f"Module {name} is already registered.")
        Register.__modules[name] = module
        logger.info(f"Module {name} registered successfully.")

    @staticmethod
    def __get_builtin_modules(name: str) -> ModuleLike | None:
        if "nn." in name:
            return getattr(nn, name.split(".")[-1])
        if name == "Output":
            return OutputModule  # type: ignore

    @staticmethod
    def get(name: str) -> ModuleLike:
        if module := Register.__get_builtin_modules(name):
            return module
        if name not in Register.__modules:
            raise ValueError(f"Module {name} is not registered.")
        return Register.__modules[name]

    @staticmethod
    def register_config(name: str, config: dict):
        def register_stream_module(name, config):
            args = lambda a, k: parse_args(config["args"], a, k)
            layers = lambda a, k: parse_layers(config["layers"], args(a, k))
            Register.register(name, lambda *a, **k: StreamModule(name, layers(a, k)))

        def register_lambda_module(name, forward):
            Register.register(name, lambda: LambdaModule(name, forward))

        if not isinstance(config, dict):
            logger.warning(f"{name} can't be recognized as a module.")
            return
        if "layers" in config:
            config["args"] = config.get("args", [])
            register_stream_module(name, config)
            return
        if "forward" in config:
            register_lambda_module(name, eval(config["forward"]))
            return
        logger.warning(f"{name} can't be recognized as a module.")

    @staticmethod
    def register_config_dict(config_dict: dict[str, dict]):
        for k, v in deepcopy(config_dict).items():
            Register.register_config(k, v)
