from copy import copy
from itertools import combinations
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
    Layers with string "from" will be ignored.
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

    all_layers = copy(list(layers))
    layers = [l for l in all_layers if not isinstance(l["from"], str)]
    for i, layer in layer_enum(layers):
        from_ = cast(FromTuple, layer["from"])
        layer["from"] = to_absolute(i, from_)
    return tuple(all_layers)


def get_unused_layer_indexes(layers: Iterable[FinalLayer]) -> set[int]:
    """
    Get the indexes of the layers that are not used by other layers.
    Layers should be converted to absolute indexes before.
    Layers with string "from" will be ignored.
    """

    def get_used_indexes(from_list: list[FromTuple]) -> set[int]:
        key_list = [[k for k, _ in from_] for from_ in from_list]
        use_count = [Counter(f) for f in key_list]
        return set(sum(use_count, Counter()).keys())

    layer_range = lambda l: range(LAYER_START_INDEX, len(l) + LAYER_START_INDEX)

    indexed_all_layers = layer_enum(copy(list(layers)))
    not_str_layer = lambda p: not isinstance(p[1]["from"], str)
    indexed_layers = list(filter(not_str_layer, indexed_all_layers))
    from_list = cast(list[FromTuple], [l["from"] for _, l in indexed_layers])

    used_indexes = get_used_indexes(from_list)
    all_indexes = set(layer_range(from_list))
    last_index = len(from_list) + LAYER_START_INDEX - 1
    unused_indexes = all_indexes.difference(used_indexes).difference({last_index})

    index_dict = {i: p[0] for i, p in zip(layer_range(indexed_layers), indexed_layers)}
    return {index_dict[i] for i in unused_indexes}


def get_same_indexes(iterable: Iterable[T]) -> dict[int, set[int]]:
    """
    Get the indexes of the elements that are the same.
    It will return a dictionary where the keys are the indexes of the first element and the values are sets of indexes of the same elements.
    Indexes are starting from LAYER_START_INDEX.
    """

    same_dict: dict[int, set[int]] = {}
    sames = lambda d: {i for same in d.values() for i in same}
    for (i, item1), (j, item2) in combinations(layer_enum(iterable), 2):
        if item1 is not item2 or i in sames(same_dict):
            continue
        same_dict[i] = same_dict.get(i, set()) | {j}
    return same_dict
