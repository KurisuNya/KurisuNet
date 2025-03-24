from __future__ import annotations

from copy import copy
from typing import Any, Counter, Iterable, TypeVar, cast

from ..config.module import is_drop_key
from ..config.types import FinalLayer, FromTuple
from ..constants import LAYER_START_INDEX, MODULE_START_INDEX

T = TypeVar("T")


def auto_unpack(args: tuple[Any, ...]) -> tuple[Any, ...] | Any:
    """Unpack a tuple if it has only one element"""
    return args[0] if len(args) == 1 else args


def layer_enum(iterable: Iterable[T]) -> Iterable[tuple[int, T]]:
    """Enumerate an iterable starting from LAYER_START_INDEX."""
    return enumerate(iterable, start=LAYER_START_INDEX)


def module_enum(iterable: Iterable[T]) -> Iterable[tuple[int, T]]:
    """Enumerate an iterable starting from MODULE_START_INDEX."""
    return enumerate(iterable, start=MODULE_START_INDEX)


def get_except_indexes(iterable: Iterable[T], indexes: set[int]) -> tuple[T, ...]:
    """Get the elements of an iterable except those at the specified indexes with LAYER_START_INDEX as start."""
    return tuple(item for i, item in layer_enum(iterable) if i not in indexes)


def get_drop_layer_indexes(layers: Iterable[FinalLayer]) -> set[int]:
    """Get the indexes of the layers that are set to be dropped."""
    return {i for i, layer in layer_enum(layers) if is_drop_key(layer["from"])}


def regularize_layer_from(layers: Iterable[FinalLayer]) -> tuple[FinalLayer, ...]:
    """
    Regularize the layer from indexes to absolute indexes.
    Layers should be handled to exclude strings in the 'from' field before.
    """

    def to_absolute(index: int, from_: FromTuple) -> FromTuple:
        def regularize_key(i: int, f: int) -> int:
            if f >= i:
                raise ValueError(f"Layer from {f} should be less than index{i}")
            if f >= 0 and f < LAYER_START_INDEX - 1:
                raise ValueError(f"Layer from {f} out of range")
            if f >= 0:
                return f
            if i + f < LAYER_START_INDEX - 1:
                raise ValueError(f"Layer from {f} out of range")
            return i + f

        return tuple((regularize_key(index, k), v) for k, v in from_)

    if any(isinstance(l["from"], str) for l in layers):
        error_msg = "Layers should not contain string 'from' values when regularizing"
        raise ValueError(error_msg)

    layers = copy(list(layers))
    for i, layer in layer_enum(layers):
        from_ = cast(FromTuple, layer["from"])
        layer["from"] = to_absolute(i, from_)
    return tuple(layers)


def get_unused_layer_indexes(layers: Iterable[FinalLayer]) -> set[int]:
    """
    Get the indexes of the layers that are not used by other layers.
    Layers should be handled to exclude strings in the 'from' field and should be converted to absolute indexes before.
    """

    def get_used_indexes(from_list: list[FromTuple]) -> set[int]:
        key_list = [[k for k, _ in from_] for from_ in from_list]
        use_count = [Counter(f) for f in key_list]
        return set(sum(use_count, Counter()).keys())

    if any(isinstance(l["from"], str) for l in layers):
        error_msg = "Layers should not contain string 'from' values when calculating unused indexes"
        raise ValueError(error_msg)

    from_list = cast(list[FromTuple], [l["from"] for l in layers])
    used_indexes = get_used_indexes(from_list)
    all_indexes = set(range(LAYER_START_INDEX, len(from_list) + LAYER_START_INDEX - 1))
    return all_indexes.difference(used_indexes)
