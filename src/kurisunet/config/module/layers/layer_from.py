from typing import Any

from ....basic.types import Env
from ....constants import ALL_FROM, DROP_FROM
from ...types import FormattedFrom, FormattedLayerFrom, From, LayerFrom, ParsedLayerFrom
from ...utils import eval_string


def __check_layer_from(layer_from: Any) -> None:
    def check_from(from_: Any) -> None:
        if isinstance(from_, bool) or not isinstance(from_, (int, dict)):
            raise ValueError(f"Invalid from: {from_}, should be int or dict")
        if isinstance(from_, int):
            return
        if len(from_) != 1:
            raise ValueError(f"Invalid from: {from_}, dict should have only one key")
        key, value = list(from_.items())[0]
        if not isinstance(key, int):
            raise ValueError(f"Invalid from: {from_}, key should be int")
        if not isinstance(value, (int, str)):
            raise ValueError(
                f"Invalid from: {from_}, value should be int or {ALL_FROM}"
            )
        if isinstance(value, str) and value != ALL_FROM:
            raise ValueError(
                f"Invalid from: {from_}, value should be int or {ALL_FROM}"
            )

    if isinstance(layer_from, str):
        return
    if not isinstance(layer_from, (list, tuple)):
        check_from(layer_from)
        return
    if len(layer_from) == 0:
        raise ValueError("Layer from should not be empty")
    for from_ in layer_from:
        check_from(from_)


def __parse_layer_from(from_: LayerFrom, env: Env) -> ParsedLayerFrom:
    if not isinstance(from_, str) or from_ == DROP_FROM:
        return from_
    parsed = eval_string(from_, env)
    if isinstance(parsed, str) and not parsed == DROP_FROM:
        raise ValueError(f"Invalid drop key {parsed}")
    __check_layer_from(parsed)
    return parsed


def __format_layer_from(layer_from: ParsedLayerFrom) -> FormattedLayerFrom:
    def format_from(from_: From) -> FormattedFrom:
        if isinstance(from_, int):
            return (from_, ALL_FROM)
        if isinstance(from_, dict):
            return (list(from_.keys())[0], list(from_.values())[0])

    if isinstance(layer_from, str):
        return layer_from
    if isinstance(layer_from, (list, tuple)):
        return tuple(format_from(x) for x in layer_from)
    return (format_from(layer_from),)


def parse_layer_from(from_: Any, env: Env | None) -> FormattedLayerFrom:
    """Parse the expressions and format the layer from."""
    __check_layer_from(from_)
    parsed = __parse_layer_from(from_, env or {})
    return __format_layer_from(parsed)


def is_drop_key(layer_from: FormattedLayerFrom) -> bool:
    """Check if the layer from is a drop key."""
    if layer_from == DROP_FROM:
        return True
    return False
