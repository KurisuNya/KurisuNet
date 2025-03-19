from copy import deepcopy
from string import ascii_letters
from typing import Any, Callable

from ..constants import *
from .register import ConverterRegister, ModuleRegister
from .types import ArgDict, Args, Converter, Former, Kwargs, Layer, Param
from .utils import get_except_keys, get_first_key, get_first_value


def __parse_str(
    string: str,
    arg_dict: ArgDict,
    import_list: list[str],
    force_eval: bool = False,
) -> Any:
    # INFO: A tricky way to import from config file & eval args
    # Use "__" to avoid conflict with imported modules
    __import_list = deepcopy(import_list)
    __import_list.extend(BUILD_IN_IMPORT)
    __arg_dict = arg_dict

    def parse(__str) -> Callable | type:
        for __k, __v in __arg_dict.items():
            exec(f"{__k} = __v")
        for __i in __import_list:
            exec(__i)
        __e = ["__str", "__k", "__v", "__arg_dict", "__i", "__import_list", "__e"]
        return eval(__str, get_except_keys(locals(), __e))

    def need_parse(string: str) -> bool:
        def get_base(string: str) -> str:
            if not string.startswith(tuple(ascii_letters)):
                return string
            for symbol in ["(", "[", "."]:
                if symbol in string:
                    string = string.split(symbol)[0]
            return string

        def get_package(import_str: str) -> str:
            return import_str.split(" ")[-1]

        if len(string) == 0:
            return False
        if string.startswith(("lambda ", "lambda:")):
            return True
        vars = (
            [get_package(i) for i in __import_list]
            + list(__arg_dict.keys())
            + BUILD_IN_KEYWORD
        )
        if get_base(string) in vars:
            return True
        return False

    string = string.strip()
    if string.startswith(EVAL_PREFIX):
        return parse(string[len(EVAL_PREFIX) :])
    if force_eval or need_parse(string):
        return parse(string)
    return string


def __parse_arg(arg: Any, arg_dict: ArgDict, import_list: list[str]) -> Any:
    if not isinstance(arg, str):
        return arg
    if arg.startswith(FORCE_STR_PREFIX):
        return arg[len(FORCE_STR_PREFIX) :]
    arg = __parse_str(arg, arg_dict, import_list)
    if not isinstance(arg, str):
        return arg
    if ModuleRegister.has(arg):
        return ModuleRegister.get(arg)
    return arg


def parse_input(
    params: list[Param],
    import_list: list[str],
    args: Args = [],
    kwargs: Kwargs = {},
) -> ArgDict:
    def get_param_name(param) -> str:
        return get_first_key(param) if isinstance(param, dict) else param

    def get_param_value(param) -> Any:
        value = get_first_value(param) if isinstance(param, dict) else None
        return __parse_arg(value, {}, import_list)

    def check_params_format(params):
        def check_type(param):
            if not isinstance(param, (str, dict)):
                raise ValueError(f"Invalid param type: {type(param)}")
            if isinstance(param, dict) and len(param) != 1:
                raise ValueError("Dict in param should have only one key")

        def get_first_index(lst, type):
            for i, item in enumerate(lst):
                if isinstance(item, type):
                    return i
            return len(lst)

        def get_last_index(lst, type):
            return len(lst) - 1 - get_first_index(list(reversed(lst)), type)

        for param in params:
            check_type(param)
        if get_first_index(params, dict) < get_last_index(params, str):
            raise ValueError("Non-default argument follows default argument")

    def check_args_format(params, args, kwargs):
        if len(args) > len(params):
            raise ValueError(f"Expected {len(params)} positional arguments")
        kw_list = params[len(args) :]
        valid_keys = [get_param_name(arg) for arg in kw_list]
        invalid_keys = set(kwargs.keys()) - set(valid_keys)
        if invalid_keys:
            raise ValueError(f"No parameter named {invalid_keys}")
        valid_keys = [arg for arg in kw_list if not isinstance(arg, dict)]
        invalid_keys = set(valid_keys) - set(kwargs.keys())
        if invalid_keys:
            raise ValueError(f"Argument missing for parameters {invalid_keys}")

    check_params_format(params)
    check_args_format(params, args, kwargs)

    parsed_input = {get_param_name(d): get_param_value(d) for d in params}
    for i, arg in enumerate(args):
        parsed_input[get_param_name(params[i])] = arg
    for key, value in kwargs.items():
        parsed_input[key] = value
    return parsed_input


def parse_vars(var_list: list[dict], arg_dict: ArgDict, import_list: list[str]):
    for var in var_list:
        if not isinstance(var, dict):
            raise ValueError(f"Invalid var type: {type(var)}")
        if len(var) != 1:
            raise ValueError("Dict in var should have only one key")

    return {
        k: __parse_arg(v, arg_dict, import_list)
        for var in var_list
        for k, v in var.items()
    }


def is_drop_former(former: Former | str) -> bool:
    if former == DROP_KEY:
        return True
    return False


def __regularize_layer_like_format(layer_like: list, prefix_len: int) -> list:
    if len(layer_like) < prefix_len or len(layer_like) > prefix_len + 2:
        raise ValueError(f"Invalid format: {layer_like}")
    if len(layer_like) == prefix_len:
        return layer_like + [[], {}]
    if len(layer_like) == prefix_len + 1 and isinstance(layer_like[-1], (list, tuple)):
        return layer_like[:prefix_len] + [layer_like[-1], {}]
    if len(layer_like) == prefix_len + 1 and isinstance(layer_like[-1], dict):
        return layer_like[:prefix_len] + [[], layer_like[-1]]
    return layer_like


def __parse_args(
    arg_dict: ArgDict, import_list: list[str], args: Args, kwargs: Kwargs
) -> tuple[Args, Kwargs]:
    args, kwargs = deepcopy(list(args)), deepcopy(kwargs)
    for i, arg in enumerate(args):
        args[i] = __parse_arg(arg, arg_dict, import_list)
    for key, value in kwargs.items():
        kwargs[key] = __parse_arg(value, arg_dict, import_list)
    return tuple(args), kwargs


def parse_layers(
    layers: list[list], arg_dict: ArgDict, import_list: list[str]
) -> list[Layer]:
    def check_former(former: Former):
        if isinstance(former, str) and not is_drop_former(former):
            raise ValueError(f"Invalid drop former {former}")

    def regularize_former(i: int, former: list[int | dict] | int | dict) -> Former:
        def to_list(x: Any | list[Any]) -> list[Any]:
            return x if isinstance(x, list) else [x]

        def get_f_key(f: int | dict) -> int:
            return get_first_key(f) if isinstance(f, dict) else f

        def get_f_value(f: int | dict) -> int | str:
            return get_first_value(f) if isinstance(f, dict) else "all"

        def regularize_f_key(i: int, f: int) -> int:
            if f >= i:
                raise ValueError(f"Former {f} should be less than itself {i}")
            if f >= 0:
                return f
            if i + f < 0:
                raise ValueError(f"Former {f} is out of range")
            return i + f

        f_dict = {get_f_key(f): get_f_value(f) for f in to_list(former)}
        return [{regularize_f_key(i, k): v} for k, v in f_dict.items()]

    def parse_module(name: str) -> type | Callable:
        module = __parse_str(name, arg_dict, import_list)
        if isinstance(module, str):
            return ModuleRegister.get(module)
        if isinstance(module, type):
            return module
        return lambda *a, **k: lambda *args: module(*args, *a, **k)

    def parse_layers(layers: list):
        def get_layers(layer):
            if isinstance(layer, str):
                return list(__parse_str(layer, arg_dict, import_list, force_eval=True))
            return [layer]

        layers_list = [get_layers(layer) for layer in layers]
        return [layer for layers in layers_list for layer in layers]

    layers = [
        __regularize_layer_like_format(layer, 2)
        for layer in parse_layers(deepcopy(layers))
    ]
    for i, (former, name, a, k) in enumerate(layers):
        check_former(former)
        layers[i][1] = parse_module(name) if isinstance(name, str) else name
        layers[i][2], layers[i][3] = __parse_args(arg_dict, import_list, a, k)
    used_layers = [l for l in layers if not is_drop_former(l[0])]
    for i, (former, _, _, _) in enumerate(used_layers):
        used_layers[i][0] = regularize_former(i + 1, former)
    return list(tuple(layer) for layer in layers)


def parse_converters(
    converters: list, arg_dict: ArgDict, import_list: list[str]
) -> list[Converter]:
    def parse_converter(name: str) -> Callable:
        converter = __parse_str(name, arg_dict, import_list)
        if isinstance(converter, str):
            return ConverterRegister.get(converter)
        return converter

    converters = [__regularize_layer_like_format(c, 1) for c in converters]
    for i, (name, a, k) in enumerate(converters):
        converters[i][0] = parse_converter(name) if isinstance(name, str) else name
        converters[i][1], converters[i][2] = __parse_args(arg_dict, import_list, a, k)
    return list(tuple(c) for c in converters)
