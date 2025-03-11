from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import Any, Callable, Iterable

from loguru import logger
import torch.nn as nn
import yaml

from .config import parse_converter, parse_input, parse_layers
from .module import LambdaModule, OutputModule, StreamModule
from .utils import get_except_key, get_except_keys, get_relative_path


def register_module(cls):
    ModuleRegister.register(cls.__name__, cls)
    return cls


def register_converter(fn):
    ConverterRegister.register(fn.__name__, fn)
    return fn


def get_module(
    name: str,
    args: Iterable[Any] = (),
    kwargs: dict[str, Any] = {},
    config: dict[str, Any] | Path | str | None = None,
) -> nn.Module:
    if config:
        register_config(config)
    return ModuleRegister.get(name)(*args, **kwargs)


def register_config(config: dict[str, Any] | Path | str):
    if isinstance(config, str):
        config = Path(config)
    if isinstance(config, Path):
        logger.info(f"Registering module from {get_relative_path(config)}")
        config = yaml.safe_load(open(config))
    if not isinstance(config, dict):
        raise ValueError("Invalid config type")

    def exec_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))  # type: ignore

    def register_path_list(path_list):
        py_suffix = [".py"]
        config_suffix = [".yaml", ".yml"]
        for path in filter(lambda x: x.suffix in py_suffix, path_list):
            logger.info(f"Registering module from {get_relative_path(path)}")
            exec_module(path.stem, path)
        for path in filter(lambda x: x.suffix in config_suffix, path_list):
            logger.info(f"Registering config from {get_relative_path(path)}")
            register_config(yaml.safe_load(open(path)))
        for path in filter(lambda x: x.is_dir(), path_list):
            logger.info(f"Registering path {get_relative_path(path)}")
            register_path_list(list(path.iterdir()))

    def register(config: dict):
        for k, v in config.items():
            __register_single_config(k, v)

    if "auto_register" in config:
        register_path_list([Path(x) for x in config["auto_register"]])
    except_keys = ["auto_register"]
    register(deepcopy(get_except_keys(config, except_keys)))


def __register_single_config(name: str, config: dict):
    def get_converted_config(args, kwargs, config):
        arg_dict = parse_input(config["args"], args, kwargs)
        converter_list = parse_converter(config["converter"], arg_dict)
        config = get_except_key(config, "converter")
        for name, args, kwargs in converter_list:
            logger.info(f"Converting {name} with args {args} and kwargs {kwargs}")
            config = ConverterRegister.get(name)(*args, **kwargs)(config)
        return config

    def stream_module(args, kwargs, config):
        arg_dict = parse_input(config["args"], args, kwargs)
        layers = parse_layers(config["layers"], arg_dict)
        logger.debug(f"Creating {name} with layers: {layers}")
        return StreamModule(name, layers)

    def lambda_module(args, kwargs, config):
        logger.debug(f"Creating {name} with forward: {config['forward']}")
        return LambdaModule(name, eval(config["forward"]))

    def register_module(name, config):
        def module(args, kwargs, config):
            if "converter" in config:
                logger.info(f"Converting {name} config")
                logger.debug(f"Config before conversion: {config}")
                config = get_converted_config(args, kwargs, config)
                logger.debug(f"Config after conversion: {config}")
                return module(args, kwargs, config)
            if "layers" in config:
                return stream_module(args, kwargs, config)
            if "forward" in config:
                return lambda_module(args, kwargs, config)
            raise ValueError(f"Converted config of {name} format is not recognized")

        ModuleRegister.register(name, lambda *a, **k: module(a, k, config))

    if not isinstance(config, dict):
        logger.warning(f"{name} can't be recognized as a module")
        return
    if all(k not in config for k in ["converter", "layers", "forward"]):
        logger.warning(f"{name} can't be recognized as a module")
        return
    config["args"] = config.get("args", [])
    register_module(name, config)


ModuleLike = type[nn.Module] | Callable[..., nn.Module]


class ConverterRegister:
    __converters = {}

    @staticmethod
    def register(name: str, converter: Callable):
        if name in ConverterRegister.__converters:
            raise ValueError(f"Converter {name} is already registered")
        ConverterRegister.__converters[name] = converter
        logger.debug(f"Converter {name} registered successfully")

    @staticmethod
    def get(name: str) -> Callable:
        if name not in ConverterRegister.__converters:
            raise ValueError(f"Converter {name} is not registered")
        return ConverterRegister.__converters[name]


class ModuleRegister:
    __modules = {}

    @staticmethod
    def register(name: str, module: ModuleLike):
        if name in ModuleRegister.__modules:
            raise ValueError(f"Module {name} is already registered")
        ModuleRegister.__modules[name] = module
        logger.debug(f"Module {name} registered successfully")

    @staticmethod
    def get(name: str) -> ModuleLike:
        if module := ModuleRegister.__get_builtin_modules(name):
            return module
        if name not in ModuleRegister.__modules:
            raise ValueError(f"Module {name} is not registered")
        return ModuleRegister.__modules[name]

    @staticmethod
    def __get_builtin_modules(name: str) -> ModuleLike | None:
        if "nn." in name:
            return getattr(nn, name.split(".")[-1])
        if name == "Output":
            return OutputModule  # type: ignore
