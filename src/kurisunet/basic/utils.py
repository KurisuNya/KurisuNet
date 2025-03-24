from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, TypeVar

from .types import Env, ListTuple, OneOrMore

T = TypeVar("T")


def is_list_tuple_of(obj: Any, types: OneOrMore[type]) -> bool:
    """Check if the object is a list or tuple of the given type."""
    types = tuple(types) if isinstance(types, list) else types
    return isinstance(obj, (list, tuple)) and all(isinstance(i, types) for i in obj)


def get_first_index_of(iterable: Iterable[Any], types: OneOrMore[type]) -> int | None:
    """Get the first index of the given type in the list or tuple."""
    types = tuple(types) if isinstance(types, list) else types
    for i, item in enumerate(iterable):
        if isinstance(item, types):
            return i
    return None


def get_last_index_of(iterable: Iterable[Any], types: OneOrMore[type]) -> int | None:
    """Get the last index of the given type in the list or tuple."""
    iterable = list(iterable)
    reversed_index = get_first_index_of(reversed(iterable), types)
    if reversed_index is not None:
        return len(iterable) - 1 - reversed_index
    return None


def get_except_key(dic: dict[Any, T], key: Any) -> dict[Any, T]:
    """Return a new dictionary without the specified key."""
    return {k: v for k, v in dic.items() if k != key}


def get_except_keys(dic: dict[Any, T], keys: ListTuple[Any]) -> dict[Any, T]:
    """Return a new dictionary without the specified keys."""
    return {k: v for k, v in dic.items() if k not in keys}


def merge_envs(envs: ListTuple[Env]) -> Env:
    """Merge multiple environments into one, overwriting keys in order."""
    merged_env = {}
    for env in envs:
        merged_env.update(env)
    return merged_env


def is_env_conflict(env1: Env, env2: Env) -> bool:
    """Check if there is a conflict between two environments."""
    for key in env1.keys():
        if key in env2:
            return True
    return False


def to_path(path: str | Path) -> Path:
    """Convert a string or Path to a Path object."""
    return path if isinstance(path, Path) else Path(path)


def to_relative_path(path: str | Path, base_path: str | Path = os.getcwd()) -> Path:
    """Convert an absolute path to a relative path based on the base path."""
    path = to_path(path)
    return path.relative_to(base_path) if path.is_absolute() else path
