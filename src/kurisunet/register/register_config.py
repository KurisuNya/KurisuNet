from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from ..basic.types import Env
from ..basic.utils import (
    get_except_key,
    get_except_keys,
    is_env_conflict,
    merge_envs,
    to_path,
    to_relative_path,
)
from ..config.module import (
    exec_with_env,
    get_exec_env,
    get_imports_env,
    get_input_env,
    get_vars_env,
    parse_converters,
    parse_layers,
)
from ..constants import *
from ..net.module import PipelineModule
from ..utils.logger import get_logger
from .register import ConverterRegister, ModuleRegister
from .register_file import register_from_paths


EnvFunc = Callable[[Env], Env]


def _pipeline_merge_env(func_list: Iterable[EnvFunc], init_env: Env) -> Env:
    env = copy(init_env)
    for func in func_list:
        env = merge_envs((env, func(env)))
    return env


def get_module(
    name: str,
    args: Iterable[Any] = (),
    kwargs: dict[str, Any] = {},
    config: dict[str, Any] | Path | str | None = None,
):
    if config:
        register_config(config)
    module = ModuleRegister.get(name)
    return module(*args, **kwargs)


def register_config(config: dict[str, Any] | Path | str):
    logger = get_logger("Register")
    if isinstance(config, (str, Path)):
        config = to_path(config)
        logger.info(f"Registering config from {to_relative_path(config)}")
        config = yaml.safe_load(config.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format. Expected dict, got {type(config)}")

    def pipeline(config: dict[str, Any]):
        global_import = config.get(GLOBAL_IMPORTS_KEY, []) + BUILD_IN_IMPORT
        import_ = lambda _: get_imports_env(global_import)
        exec_ = lambda env: get_exec_env(config.get(GLOBAL_EXEC_KEY, ""), env)
        vars = lambda env: get_vars_env(config.get(GLOBAL_VARS_KEY, []), env)
        return [import_, exec_, vars]

    def register(config: dict[str, Any], env: Env):
        for k, v in config.items():
            if not isinstance(v, dict):
                logger.warning(f"{k} can't be recognized as a module, skipping")
                continue
            if CONVERTERS_KEY in v:
                v = __convert_single_config(v, env)
            __register_single_config(k, v, env)

    register_from_paths(Path(p) for p in config.get(AUTO_REGISTER_KEY, []))
    global_env = _pipeline_merge_env(pipeline(config), {})
    excepts = [AUTO_REGISTER_KEY, GLOBAL_IMPORTS_KEY, GLOBAL_EXEC_KEY, GLOBAL_VARS_KEY]
    register(deepcopy(get_except_keys(config, excepts)), global_env)


LazyConfig = dict[str, Any] | Callable[..., dict[str, Any]]


def __convert_single_config(config: dict[str, Any], env: Env) -> LazyConfig:
    logger = get_logger("Converter")

    def pipeline(*args: Any, **kwargs: Any):
        registered_modules = lambda _: ModuleRegister.get_env()
        registered_converters = lambda _: ConverterRegister.get_env()
        import_ = lambda _: get_imports_env(config.get(IMPORTS_KEY, []))
        input = lambda env: get_input_env(config.get(ARGS_KEY, []), args, kwargs, env)
        return [registered_modules, registered_converters, import_, input]

    def convert(*args: Any, **kwargs: Any) -> dict[str, Any]:
        local_env = _pipeline_merge_env(pipeline(*args, **kwargs), env)
        converters = parse_converters(config[CONVERTERS_KEY], local_env)
        converted = get_except_key(config, CONVERTERS_KEY)
        logger.debug(f"Converting {converted} with converters")
        logger.debug(f"Config before: {converted}")
        for c in converters:
            converted = c["converter"](deepcopy(converted), *c["args"], **c["kwargs"])
            logger.debug(f"Config after: {converted}")
        return converted

    return convert


def __register_single_config(name: str, config: LazyConfig, env: Env):
    logger = get_logger("Register")
    if isinstance(config, dict) and LAYERS_KEY not in config:
        logger.warning(f"{name} can't be recognized as a module")
        return
    ModuleRegister.register(name, LazyModule(name, config, env))


class LazyModule:
    def __init__(self, name: str, config: LazyConfig, env: Env | None):
        self.__name = name
        self.__config = config
        self.__global_env = env or {}

    def __prepare_config(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        config = self.__config
        config = config(*args, **kwargs) if callable(config) else config
        if not isinstance(config, dict):
            msg = f"Invalid config format. Expected dict, got {type(config)}"
            raise ValueError(msg)
        if LAYERS_KEY not in config:
            raise ValueError(f"Invalid config, missing {LAYERS_KEY} key")
        key_default_pairs = [
            (IMPORTS_KEY, []),
            (ARGS_KEY, []),
            (PRE_EXEC_KEY, ""),
            (BUFFERS_KEY, []),
            (PARAMS_KEY, []),
            (VARS_KEY, []),
            (POST_EXEC_KEY, ""),
        ]
        for key, default in key_default_pairs:
            if key not in config:
                config[key] = default
        return config

    def get_module(self, *args: Any, **kwargs: Any) -> Any:
        config = self.__prepare_config(*args, **kwargs)

        def pipeline_before():
            registered = lambda _: ModuleRegister.get_env()
            import_ = lambda _: get_imports_env(config[IMPORTS_KEY])
            input = lambda env: get_input_env(config[ARGS_KEY], args, kwargs, env)
            return [registered, import_, input]

        def pipeline_init():
            module = PipelineModule()
            init = lambda _: {"self": module}
            exec_ = lambda env: get_exec_env(config[PRE_EXEC_KEY], env)
            return module, [init, exec_]

        def pipeline_after():
            vars = lambda env: get_vars_env(config[VARS_KEY], env)
            return [vars]

        env = _pipeline_merge_env(pipeline_before(), self.__global_env)
        module, init_pipeline = pipeline_init()
        env = _pipeline_merge_env(init_pipeline, env)

        buffers = get_vars_env(config[BUFFERS_KEY], env)
        env = merge_envs((env, buffers))
        params = get_vars_env(config[PARAMS_KEY], env)
        if is_env_conflict(buffers, params):
            raise ValueError("Buffers and params should not have same key")
        env = merge_envs((env, buffers, params))
        env = _pipeline_merge_env(pipeline_after(), env)

        logger = get_logger("Layers")
        layers_str = "\n".join(str(layer) for layer in config[LAYERS_KEY])
        logger.debug(f"{self.__name} layers before parsing:\n{layers_str}")
        layers = parse_layers(config[LAYERS_KEY], env)
        layers_str = "\n".join(str(layer) for layer in layers)
        logger.debug(f"{self.__name} layers after parsing:\n{layers_str}")
        module.init(self.__name, layers, buffers=buffers, params=params)
        exec_with_env(config[POST_EXEC_KEY], env)
        return module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_module(*args, **kwargs)
