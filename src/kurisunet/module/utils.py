from typing import Any


def get_first_key(dic: dict) -> Any:
    return next(iter(dic.keys()))


def get_first_value(dic: dict) -> Any:
    return next(iter(dic.values()))


def get_first_item(dic: dict) -> tuple[Any, Any]:
    return next(iter(dic.items()))
