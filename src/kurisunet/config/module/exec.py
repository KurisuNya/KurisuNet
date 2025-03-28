from typing import Any

from ...basic.types import Env


def _check_exec(exec_: Any) -> None:
    if not isinstance(exec_, str):
        raise ValueError(f"Invalid {exec_}, should be str")


def _get_exec_env(exec_: str, env: Env) -> Env:
    if not exec_:
        return {}
    local_env = {}
    exec(exec_, env, local_env)
    return local_env


def get_exec_env(exec_: str, env: Env | None = None) -> Env:
    """Get the environment from the given exec statement."""
    _check_exec(exec_)
    return _get_exec_env(exec_, env or {})
