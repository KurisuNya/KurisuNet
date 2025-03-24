from __future__ import annotations

from typing import Any

from ...basic.types import Env, ListTuple
from ...basic.utils import get_first_index_of, get_last_index_of, is_list_tuple_of
from ..types import ArgDict, Args, FormattedParam, Kwargs, Param
from ..utils import eval_string


def _check_params(params: Any) -> None:
    if not is_list_tuple_of(params, (str, dict, tuple)):
        error_msg = f"Invalid params {params}, should be list/tuple of str/dict/tuple"
        raise ValueError(error_msg)
    dict_default_params = [param for param in params if isinstance(param, dict)]
    tuple_default_params = [param for param in params if isinstance(param, tuple)]
    if any(len(param) != 1 for param in dict_default_params):
        raise ValueError("Dict default param should have one item")
    if any(len(param) != 2 for param in tuple_default_params):
        raise ValueError("Tuple default param should have two items")
    if any(not isinstance(list(param.keys())[0], str) for param in dict_default_params):
        raise ValueError("Dict default param should have str key")
    if any(not isinstance(param[0], str) for param in tuple_default_params):
        raise ValueError("Tuple default param should have str first item")
    first_default_index = get_first_index_of(params, (dict, tuple))
    last_non_default_index = get_last_index_of(params, str)
    if first_default_index is None or last_non_default_index is None:
        return
    if first_default_index < last_non_default_index:
        raise ValueError("Non-default argument follows default argument")


def _format_params(params: ListTuple[Param]) -> tuple[FormattedParam, ...]:
    def format_param(param: Param) -> FormattedParam:
        if isinstance(param, (str, tuple)):
            return param
        if isinstance(param, dict):  # only one key
            return tuple(param.items())[0]

    return tuple(format_param(param) for param in params)


def _get_input_arg_dict(
    params: ListTuple[FormattedParam], args: Args, kwargs: Kwargs
) -> ArgDict:
    def get_param_name(param: FormattedParam) -> str:
        return param[0] if isinstance(param, tuple) else param

    def get_param_value(param: FormattedParam) -> Any:
        return param[1] if isinstance(param, tuple) else None

    def check_args_kwargs(
        params: ListTuple[FormattedParam], args: Args, kwargs: Kwargs
    ) -> None:
        if len(args) > len(params):
            raise ValueError(f"Expected {len(params)} positional arguments")
        need_kwarg_params = params[len(args) :]
        need_kwargs_names = [get_param_name(param) for param in need_kwarg_params]
        if invalid_keys := set(kwargs.keys()) - set(need_kwargs_names):
            raise ValueError(f"No parameter named {invalid_keys} or already signed")
        no_default_names = [p for p in need_kwarg_params if not isinstance(p, tuple)]
        if invalid_keys := set(no_default_names) - set(kwargs.keys()):
            raise ValueError(f"Argument missing for parameters {invalid_keys}")

    check_args_kwargs(params, args, kwargs)
    vars = {get_param_name(param): get_param_value(param) for param in params}
    for i, arg in enumerate(args):
        vars[get_param_name(params[i])] = arg
    for key, value in kwargs.items():
        vars[key] = value
    return vars


def _get_arg_dict_env(arg_dict: ArgDict, env: Env) -> Env:
    """Get the environment from the argument dictionary."""
    return {
        key: eval_string(value, env) if isinstance(value, str) else value
        for key, value in arg_dict.items()
    }


def get_input_env(
    params: Any, args: Args, kwargs: Kwargs | None = None, env: Env | None = None
) -> Env:
    """Get the input environment from the params, args and kwargs."""
    _check_params(params)
    formatted_params = _format_params(params)
    arg_dict = _get_input_arg_dict(formatted_params, args, kwargs or {})
    return _get_arg_dict_env(arg_dict, env or {})
