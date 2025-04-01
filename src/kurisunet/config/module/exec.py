from typing import Any

from ...basic.types import Env


def _check_exec(exec_: Any) -> None:
    if not isinstance(exec_, str):
        raise ValueError(f"Invalid {exec_}, should be str")


def _get_exec_env(exec_: str, env: Env) -> Env:
    if not exec_:
        return {}
    local_env = {}
    exec(exec_, env.copy(), local_env)
    return local_env


def _exec_with_env(exec_: str, env: Env) -> None:
    if not exec_:
        return
    exec(exec_, env.copy(), {})


def exec_with_env(exec_: str, env: Env | None = None) -> None:
    """Execute the given exec statement with the provided environment."""
    _check_exec(exec_)
    _exec_with_env(exec_, env or {})


def get_exec_env(exec_: str, env: Env | None = None) -> Env:
    """Execute the given exec statement and return the environment."""
    _check_exec(exec_)
    return _get_exec_env(exec_, env or {})
