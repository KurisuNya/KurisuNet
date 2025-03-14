from copy import deepcopy
from typing import Any, Callable, Iterable

from .utils import get_except_keys, get_first_key, get_first_value

Param = str | dict[str, Any]
Args = Iterable[Any]
Kwargs = dict[str, Any]
ArgDict = dict[str, Any]

Former = list[dict[int, int | str]]
Converter = tuple[str, Args, Kwargs]
Layer = tuple[Former | str, str | Callable | type, Args, Kwargs]


def parse_input(params: list[Param], args: Args = [], kwargs: Kwargs = {}) -> ArgDict:
    def get_param_name(param) -> str:
        return get_first_key(param) if isinstance(param, dict) else param

    def get_param_value(param) -> Any:
        return get_first_value(param) if isinstance(param, dict) else None

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


def __parse_args(arg_dict: ArgDict, args: Args, kwargs: Kwargs) -> tuple[Args, Kwargs]:
    from .register import ModuleRegister

    def regularize_arg(arg):
        if isinstance(arg, str) and arg.startswith("args."):
            return arg_dict[arg[5:]]
        if isinstance(arg, str) and arg.startswith("module."):
            return ModuleRegister.get(arg[7:])
        return arg

    args, kwargs = deepcopy(list(args)), deepcopy(kwargs)
    for i, arg in enumerate(args):
        args[i] = regularize_arg(arg)
    for key, value in kwargs.items():
        kwargs[key] = regularize_arg(value)
    return tuple(args), kwargs


def is_drop_former(former: Former | str) -> bool:
    drop_formers = ["drop", "skip", "ignore"]
    if former in drop_formers:
        return True
    return False


def parse_layers(
    layers: list[list], arg_dict: ArgDict, import_list: list[str]
) -> list[Layer]:
    def check_former(former: Former):
        if isinstance(former, str) and not is_drop_former(former):
            raise ValueError(f"Invalid drop former {former}")

    def parse_name(name: str) -> str | Callable:
        __im_list = deepcopy(import_list)
        __im_list.append("import torch.nn as nn")

        # INFO: A tricky way to import from config file
        # Use "__" to avoid conflict with imported modules
        def module(__f) -> Callable | type:
            for __im in __im_list:
                exec(__im)
            __except = ["__f", "__im", "__im_list", "__except"]
            __f = eval(__f, get_except_keys(locals(), __except))
            if isinstance(__f, type):
                return __f
            return __f

        def get_package_name(name: str) -> str:
            if "." not in name:
                return name
            return ".".join(name.split(".")[:-1])

        def in_import_list(name: str) -> bool:
            for _im in __im_list:
                if name in _im:
                    return True
            return False

        if (
            name.startswith("lambda ")
            or name.startswith("lambda:")
            or in_import_list(get_package_name(name))
        ):
            return module(name)
        return name

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

    layers = [__regularize_layer_like_format(layer, 2) for layer in deepcopy(layers)]
    for i, (former, name, args, kwargs) in enumerate(layers):
        check_former(former)
        layers[i][1] = parse_name(name)
        layers[i][2], layers[i][3] = __parse_args(arg_dict, args, kwargs)
    used_layers = [l for l in layers if not is_drop_former(l[0])]
    for i, (former, _, _, _) in enumerate(used_layers):
        used_layers[i][0] = regularize_former(i + 1, former)
    return list(tuple(layer) for layer in layers)


def parse_converter(converter: list, arg_dict: ArgDict) -> list[Converter]:
    converter = [converter] if isinstance(converter[0], str) else converter
    converter = [__regularize_layer_like_format(c, 1) for c in converter]
    for i, (_, args, kwargs) in enumerate(converter):
        converter[i][1], converter[i][2] = __parse_args(arg_dict, args, kwargs)
    return list(tuple(c) for c in converter)
