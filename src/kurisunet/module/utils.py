import os
from pathlib import Path
from typing import Any, Iterable


def get_first_key(dic: dict) -> Any:
    return next(iter(dic.keys()))


def get_first_value(dic: dict) -> Any:
    return next(iter(dic.values()))


def get_first_item(dic: dict) -> tuple[Any, Any]:
    return next(iter(dic.items()))


def get_except_key(dic: dict, key: Any) -> dict:
    return {k: v for k, v in dic.items() if k != key}


def get_except_keys(dic: dict, keys: Iterable[Any]) -> dict:
    return {k: v for k, v in dic.items() if k not in keys}


def get_relative_path(path: Path) -> Path:
    return path.relative_to(os.getcwd()) if path.is_absolute() else path
