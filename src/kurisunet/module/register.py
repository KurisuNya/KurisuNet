from copy import deepcopy
from typing import Callable

from loguru import logger
import torch.nn as nn

from .config import parse_args, parse_layers, parse_wrapper
from .module import LambdaModule, OutputModule, StreamModule


def register_module(cls):
    ModuleRegister.register(cls.__name__, cls)
    return cls


def register_wrapper(fn):
    WrapperRegister.register(fn.__name__, fn)
    return fn


class WrapperRegister:
    __wrappers = {}

    @staticmethod
    def register(name: str, wrapper: Callable):
        if name in WrapperRegister.__wrappers:
            raise ValueError(f"Wrapper {name} is already registered.")
        WrapperRegister.__wrappers[name] = wrapper
        logger.info(f"Wrapper {name} registered successfully.")

    @staticmethod
    def get(name: str) -> Callable:
        if name not in WrapperRegister.__wrappers:
            raise ValueError(f"Wrapper {name} is not registered.")
        return WrapperRegister.__wrappers[name]


ModuleLike = type[nn.Module] | Callable[..., nn.Module]


class ModuleRegister:
    __modules = {}

    @staticmethod
    def register(name: str, module: ModuleLike):
        if name in ModuleRegister.__modules:
            raise ValueError(f"Module {name} is already registered.")
        ModuleRegister.__modules[name] = module
        logger.info(f"Module {name} registered successfully.")

    @staticmethod
    def get(name: str) -> ModuleLike:
        if module := ModuleRegister.__get_builtin_modules(name):
            return module
        if name not in ModuleRegister.__modules:
            raise ValueError(f"Module {name} is not registered.")
        return ModuleRegister.__modules[name]

    @staticmethod
    def __get_builtin_modules(name: str) -> ModuleLike | None:
        if "nn." in name:
            return getattr(nn, name.split(".")[-1])
        if name == "Output":
            return OutputModule  # type: ignore

    @staticmethod
    def register_config(config_dict: dict[str, dict]):
        for k, v in deepcopy(config_dict).items():
            ModuleRegister.__register_single_config(k, v)

    @staticmethod
    def __register_single_config(name: str, config: dict):
        def register_stream_module(name, config):
            arg_dict = lambda a, k: parse_args(config["args"], a, k)
            layers = lambda a, k: parse_layers(config["layers"], arg_dict(a, k))
            if "wrapper" not in config:
                module = lambda *a, **k: StreamModule(name, layers(a, k))
                ModuleRegister.register(name, module)
                return

            def wrapper(a, k):
                name, args, kwargs = parse_wrapper(config["wrapper"], arg_dict(a, k))
                return WrapperRegister.get(name)(*args, **kwargs)

            parsed_layers = lambda a, k: wrapper(a, k)(layers(a, k))
            module = lambda *a, **k: StreamModule(name, parsed_layers(a, k))
            ModuleRegister.register(name, module)

        def register_lambda_module(name, forward):
            ModuleRegister.register(name, lambda: LambdaModule(name, forward))

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
