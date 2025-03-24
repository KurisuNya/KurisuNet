from __future__ import annotations

from typing import Any, Callable

from ...basic.types import Env, ListTuple
from ...basic.utils import is_list_tuple_of
from ..types import Converter, ConverterLayer, FinalConverterLayer, ParsedConverter
from ..utils import eval_string
from .layers.args import parse_args, parse_kwargs


def __check_converters(converters: Any) -> None:
    def check_converter(converter: Any):
        if isinstance(converter, str):
            return
        if len(converter) < 1:
            raise ValueError(f"Converter should have at least two items {converter}")
        if len(converter) > 3:
            raise ValueError(f"Converter should have at most four items {converter}")
        if not isinstance(converter[0], (str, Callable)):
            msg = f"Converter should have str or callable as first item {converter}"
            raise ValueError(msg)
        if len(converter) == 2 and not isinstance(converter[1], (list, tuple, dict)):
            msg = f"Converter should have list/tuple/dict as second item {converter}"
            raise ValueError(msg)
        if len(converter) == 3 and not isinstance(converter[2], dict):
            raise ValueError(f"Converter should have dict as third item {converter}")

    if not is_list_tuple_of(converters, (list, tuple)):
        msg = f"Invalid converters {converters}, should be list/tuple of list/tuple"
        raise ValueError(msg)
    for converter in converters:
        check_converter(converter)


def __format_converters(converters: ListTuple) -> tuple[ConverterLayer, ...]:
    def format_converter(converter: list | tuple) -> ConverterLayer:
        if isinstance(converter, list):
            converter = tuple(converter)
        if len(converter) == 1:
            return (converter[0], (), {})
        if len(converter) == 2 and isinstance(converter[1], (list, tuple)):
            return (converter[0], tuple(converter[1]), {})
        if len(converter) == 2 and isinstance(converter[1], dict):
            return (converter[0], (), converter[1])
        if len(converter) == 3:
            return (converter[0], tuple(converter[1]), converter[2])
        raise ValueError(f"Invalid converter format {converter}")  # should never reach

    return tuple(format_converter(converter) for converter in converters)


def parse_converters(
    converters: ListTuple, env: Env | None = None
) -> tuple[FinalConverterLayer, ...]:
    """Parse the expressions in the converters."""

    def parse_converter(converter: Converter, env: Env) -> ParsedConverter:
        if not isinstance(converter, str):
            return converter
        parsed = eval_string(converter, env)
        if not isinstance(parsed, Callable):
            raise ValueError(f"Invalid converter {converter}, should be callable")
        return parsed

    def parse_converter_layer(layer: ConverterLayer) -> FinalConverterLayer:
        m, a, k = layer
        return {
            "converter": parse_converter(m, env or {}),
            "args": parse_args(a, env),
            "kwargs": parse_kwargs(k, env),
        }

    __check_converters(converters)
    converters = __format_converters(converters)
    return tuple(parse_converter_layer(layer) for layer in converters)
