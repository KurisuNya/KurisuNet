from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from loguru import logger
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
    get_imports_env,
    get_input_env,
    get_vars_env,
    parse_converters,
    parse_layers,
)
from ..config.module.converters import parse_converters
from ..config.types import CustomModule
from ..constants import *
from ..net.module import PipelineModule
from .register import ModuleRegister
from .register_file import register_from_path_list


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
    if isinstance(module, CustomModule):
        return module.get_module(*args, **kwargs)
    return module(*args, **kwargs)


def register_config(config: dict[str, Any] | Path | str):
    if not isinstance(config, dict):
        config = to_path(config)
        logger.info(f"Registering config from {to_relative_path(config)}")
        config = yaml.safe_load(config.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format. Expected dict, got {type(config)}")

    def pipeline(config: dict[str, Any]):
        global_import = config.get(GLOBAL_IMPORTS_KEY, []) + BUILD_IN_IMPORT
        import_ = lambda env: get_imports_env(global_import)
        vars = lambda env: get_vars_env(config.get(GLOBAL_VARS_KEY, []), env)
        return [import_, vars]

    def register(config: dict[str, Any], env: Env):
        for k, v in config.items():
            if not isinstance(v, dict):
                logger.warning(f"{k} can't be recognized as a module, skipping")
                continue
            if CONVERTERS_KEY in v:
                v = __convert_single_config(v, env)
            __register_single_config(k, v, env)

    if AUTO_REGISTER_KEY in config:
        register_from_path_list([Path(p) for p in config[AUTO_REGISTER_KEY]])
    global_env = _pipeline_merge_env(pipeline(config), {})
    except_keys = [AUTO_REGISTER_KEY, GLOBAL_IMPORTS_KEY, GLOBAL_VARS_KEY]
    register(deepcopy(get_except_keys(config, except_keys)), global_env)


LazyConfig = dict[str, Any] | Callable[..., dict[str, Any]]


def __convert_single_config(config: dict[str, Any], env: Env) -> LazyConfig:
    def pipeline(*args: Any, **kwargs: Any):
        registered_modules = lambda env: ModuleRegister.get_env()
        registered_converters = lambda env: ModuleRegister.get_env()
        import_ = lambda env: get_imports_env(config.get(IMPORTS_KEY, []))
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
    module_keys = [LAYERS_KEY, CONVERTERS_KEY]
    if isinstance(config, dict) and all(k not in config for k in module_keys):
        logger.warning(f"{name} can't be recognized as a module")
        return
    ModuleRegister.register(name, LazyModule(name, config, env))


class LazyModule:
    def __init__(self, name: str, config: LazyConfig, env: Env | None):
        self.__name = name
        self.__config = config
        self.__global_env = env or {}

    def get_module(self, *args: Any, **kwargs: Any) -> Any:
        c = self.__config(*args, **kwargs) if callable(self.__config) else self.__config
        if LAYERS_KEY not in c:
            raise ValueError(f"Invalid config, missing {LAYERS_KEY} key")

        def pipeline_before():
            registered = lambda env: ModuleRegister.get_env()
            import_ = lambda env: get_imports_env(c.get(IMPORTS_KEY, []))
            input = lambda env: get_input_env(c.get(ARGS_KEY, []), args, kwargs, env)
            return [registered, import_, input]

        def pipeline_after():
            vars = lambda env: get_vars_env(c.get(VARS_KEY, []), env)
            return [vars]

        env = _pipeline_merge_env(pipeline_before(), self.__global_env)
        buffers = get_vars_env(c.get(BUFFERS_KEY, []), env)
        params = get_vars_env(c.get(PARAMS_KEY, []), env)
        if is_env_conflict(buffers, params):
            raise ValueError("Buffers and params should not have same key")
        env = _pipeline_merge_env(pipeline_after(), merge_envs((env, buffers, params)))

        layers_str = "\n".join(str(layer) for layer in c[LAYERS_KEY])
        logger.debug(f"{self.__name} layers before parsing:\n{layers_str}")
        layers = parse_layers(c[LAYERS_KEY], env)
        layers_str = "\n".join(str(layer) for layer in c[LAYERS_KEY])
        logger.debug(f"{self.__name} layers after parsing:\n{layers_str}")
        return PipelineModule(self.__name, layers, buffers=buffers, params=params)
