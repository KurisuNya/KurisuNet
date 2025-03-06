from copy import deepcopy
from typing import Any

from .utils import get_first_key, get_first_value

Define = str | dict[str, Any]


def parse_args(
    define_list: list[Define],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] = {},
) -> dict[str, Any]:
    def get_define_name(define) -> str:
        return get_first_key(define) if isinstance(define, dict) else define

    def get_default_value(define) -> Any:
        return get_first_value(define) if isinstance(define, dict) else None

    def check_define_format(define_list):
        def check_type(define):
            if not isinstance(define, (str, dict)):
                raise ValueError(f"Invalid define type: {type(define)}")
            if isinstance(define, dict) and len(define) != 1:
                raise ValueError("Dict in define should have only one key")

        def get_first_index(lst, type):
            for i, item in enumerate(lst):
                if isinstance(item, type):
                    return i
            return len(lst)

        def get_last_index(lst, type):
            return len(lst) - 1 - get_first_index(list(reversed(lst)), type)

        for define in define_list:
            check_type(define)
        if get_first_index(define_list, dict) < get_last_index(define_list, str):
            raise ValueError("Non-default argument follows default argument.")

    def check_args_format(define_list, args, kwargs):
        if len(args) > len(define_list):
            raise ValueError(f"Expected {len(define_list)} positional arguments")
        valid_keys = [get_define_name(arg) for arg in define_list[len(args) :]]
        invalid_keys = set(kwargs.keys()) - set(valid_keys)
        if invalid_keys:
            raise ValueError(f"No parameter named {invalid_keys}")
        invalid_keys = set(valid_keys) - set(kwargs.keys())
        if invalid_keys:
            raise ValueError(f"Argument missing for parameters {invalid_keys}")

    check_define_format(define_list)
    check_args_format(define_list, args, kwargs)

    parsed_args = {get_define_name(d): get_default_value(d) for d in define_list}
    for i, arg in enumerate(args):
        parsed_args[get_define_name(define_list[i])] = arg
    for key, value in kwargs.items():
        parsed_args[key] = value
    return parsed_args


Former = list[dict[int, int | str]]
Layer = tuple[Former, str, tuple[Any, ...], dict[str, Any]]


def parse_layers(layers: list[list], arg_dict: dict[str, Any]) -> list[Layer]:
    def to_list(x: Any | list[Any]) -> list[Any]:
        return x if isinstance(x, list) else [x]

    def regularize_layer_len(layer):
        if len(layer) < 2:
            raise ValueError("Invalid layer format.")
        if len(layer) == 2:
            return layer + [[], {}]
        if len(layer) == 3 and isinstance(layer[2], list):
            return layer[:2] + [layer[2], {}]
        if len(layer) == 3 and isinstance(layer[2], dict):
            return layer[:2] + [[], layer[2]]
        return layer

    def regularize_former(i: int, former: list[int | dict]) -> Former:
        def get_key(f):
            return get_first_key(f) if isinstance(f, dict) else f

        def get_value(f):
            return get_first_value(f) if isinstance(f, dict) else "all"

        def convert_key(i: int, f: int) -> int:
            if f >= i:
                raise ValueError(f"Former {f} should be less than itself {i}.")
            if f >= 0:
                return f
            if i + f < 0:
                raise ValueError(f"Former {f} is out of range.")
            return i + f

        return [{convert_key(i, get_key(f)): get_value(f)} for f in former]

    def regularize_args_kwargs(args, kwargs):
        args, kwargs = deepcopy(args), deepcopy(kwargs)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg.startswith("args."):
                args[i] = arg_dict[arg[5:]]
        for key, value in kwargs.items():
            if isinstance(value, str) and value.startswith("args."):
                kwargs[key] = arg_dict[value[5:]]
        return tuple(args), kwargs

    layers = [regularize_layer_len(layer) for layer in deepcopy(layers)]
    for i, (former, _, args, kwargs) in enumerate(layers):
        layers[i][0] = regularize_former(i + 1, to_list(former))
        layers[i][2], layers[i][3] = regularize_args_kwargs(args, kwargs)

    return layers  # type: ignore
