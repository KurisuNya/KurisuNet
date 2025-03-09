from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from typing import Callable

from loguru import logger
import torch.nn as nn

from .config import parse_converter, parse_input, parse_layers, parse_main_module
from .module import LambdaModule, OutputModule, StreamModule
from .utils import get_except_key, get_except_keys

except_keys = ["auto_register", "main_module"]


def register_module(cls):
    ModuleRegister.register(cls.__name__, cls)
    return cls


def register_converter(fn):
    ConverterRegister.register(fn.__name__, fn)
    return fn


def get_main_module(config: dict) -> nn.Module:
    if "main_module" not in config:
        raise ValueError("No main module specified.")

    ModuleRegister.register_config(config)
    name, args, kwargs = parse_main_module(config["main_module"])
    return ModuleRegister.get(name)(*args, **kwargs)


class ConverterRegister:
    __converters = {}

    @staticmethod
    def register(name: str, converter: Callable):
        if name in ConverterRegister.__converters:
            raise ValueError(f"Converter {name} is already registered.")
        ConverterRegister.__converters[name] = converter
        logger.info(f"Converter {name} registered successfully.")

    @staticmethod
    def get(name: str) -> Callable:
        if name not in ConverterRegister.__converters:
            raise ValueError(f"Converter {name} is not registered.")
        return ConverterRegister.__converters[name]


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
        def import_from_path(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)  # type: ignore
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore

        def register(config: dict):
            for k, v in config.items():
                ModuleRegister.__register_single_config(k, v)

        if "auto_register" in config_dict:
            path_list = map(Path, config_dict["auto_register"])
            for path in path_list:
                import_from_path(path.stem, path)
        register(deepcopy(get_except_keys(config_dict, except_keys)))

    @staticmethod
    def __register_single_config(name: str, config: dict):
        def get_wrapped_config(args, kwargs, config):
            arg_dict = parse_input(config["args"], args, kwargs)
            name, args, kwargs = parse_converter(config["converter"], arg_dict)
            converter = ConverterRegister.get(name)(*args, **kwargs)
            return converter(get_except_key(config, "converter"))

        def stream_module(args, kwargs, config):
            arg_dict = parse_input(config["args"], args, kwargs)
            layers = parse_layers(config["layers"], arg_dict)
            return StreamModule(name, layers)

        def lambda_module(args, kwargs, config):
            return LambdaModule(name, eval(config["forward"]))

        def register_module(name, config, parser):
            def module(args, kwargs, config):
                if "converter" in config:
                    config = get_wrapped_config(args, kwargs, config)
                return parser(args, kwargs, config)

            ModuleRegister.register(name, lambda *a, **k: module(a, k, config))

        if not isinstance(config, dict):
            logger.warning(f"{name} can't be recognized as a module.")
            return
        config["args"] = config.get("args", [])
        if "layers" in config:
            register_module(name, config, stream_module)
            return
        if "forward" in config:
            register_module(name, config, lambda_module)
            return
        logger.warning(f"{name} can't be recognized as a module.")
