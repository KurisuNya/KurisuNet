from copy import copy
from typing import Any

from ...basic.types import Env, ListTuple
from ...basic.utils import is_list_tuple_of
from ..types import FormattedVar, Var
from ..utils import eval_string


def _check_vars(vars: Any) -> None:
    if not is_list_tuple_of(vars, (dict, tuple)):
        raise ValueError(f"Invalid {vars}, should be list/tuple of dict/tuple")
    dict_vars = [var for var in vars if isinstance(var, dict)]
    tuple_vars = [var for var in vars if isinstance(var, tuple)]
    if any(len(var) != 1 for var in dict_vars):
        raise ValueError("Dict should have one item")
    if any(len(var) != 2 for var in tuple_vars):
        raise ValueError("Tuple should have two items")
    if any(not isinstance(list(var.keys())[0], str) for var in dict_vars):
        raise ValueError("Dict should have str key")
    if any(not isinstance(var[0], str) for var in tuple_vars):
        raise ValueError("Tuple should have str first item")


def _format_vars(vars: ListTuple[Var]) -> tuple[FormattedVar, ...]:
    def format_var(var: Var) -> FormattedVar:
        if isinstance(var, tuple):
            return var
        if isinstance(var, dict):  # only one key
            return tuple(var.items())[0]

    return tuple(format_var(var) for var in vars)


def _get_vars_env(vars: ListTuple[FormattedVar], env: Env) -> Env:
    used_env = copy(env)
    new_env = {}
    for key, value in vars:
        value = eval_string(value, used_env) if isinstance(value, str) else value
        used_env[key] = new_env[key] = value
    return new_env


def get_vars_env(vars: Any, env: Env | None = None) -> Env:
    """Get the variable environment from the vars."""
    _check_vars(vars)
    formatted_vars = _format_vars(vars)
    return _get_vars_env(formatted_vars, env or {})
