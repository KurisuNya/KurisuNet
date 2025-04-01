from typing import Any

from ....basic.types import Env, ListTuple
from ....basic.utils import is_list_tuple_of
from ...types import FinalLayer, FormattedLayer, Layer
from ...utils import eval_string
from .args import parse_args, parse_kwargs
from .layer_from import parse_layer_from
from .module import parse_module


def __check_layers(layers: Any) -> None:
    def check_layer(layer: Any):
        if isinstance(layer, str):
            return
        if len(layer) < 2:
            raise ValueError(f"Layer should have at least two items {layer}")
        if len(layer) > 4:
            raise ValueError(f"Layer should have at most four items {layer}")
        if len(layer) == 3 and not isinstance(layer[2], (list, tuple, dict)):
            raise ValueError(f"Layer should have list/tuple/dict as third item {layer}")
        if len(layer) == 4 and not isinstance(layer[3], dict):
            raise ValueError(f"Layer should have dict as fourth item {layer}")

    if not is_list_tuple_of(layers, (str, list, tuple)):
        error_msg = f"Invalid layers {layers}, should be list/tuple of str/list/tuple"
        raise ValueError(error_msg)
    for layer in layers:
        check_layer(layer)


def __parse_layers(layers: ListTuple[Layer], env: Env) -> ListTuple:
    def parse_layer(layer: Layer, env: Env) -> ListTuple:
        if not isinstance(layer, str):
            return (layer,)
        parsed = eval_string(layer, env)
        if not is_list_tuple_of(parsed, (list, tuple)):
            parsed = (parsed,)
        __check_layers(parsed)
        return parsed

    parsed_layers = []
    for layer in layers:
        parsed_layers.extend(parse_layer(layer, env))
    return parsed_layers


def __format_layers(layers: ListTuple) -> tuple[FormattedLayer, ...]:
    def format_layer(layer: list | tuple) -> FormattedLayer:
        if isinstance(layer, list):
            layer = tuple(layer)
        if len(layer) == 2:
            return (layer[0], layer[1], (), {})
        if len(layer) == 3 and isinstance(layer[2], (list, tuple)):
            return (layer[0], layer[1], tuple(layer[2]), {})
        if len(layer) == 3 and isinstance(layer[2], dict):
            return (layer[0], layer[1], (), layer[2])
        if len(layer) == 4:
            return (layer[0], layer[1], tuple(layer[2]), layer[3])
        raise ValueError(f"Invalid layer format {layer}")  # should never reach

    return tuple(format_layer(layer) for layer in layers)


def parse_layers(layers: Any, env: Env | None = None) -> tuple[FinalLayer, ...]:
    """Parse the expressions between and inside the layers."""

    def parse_layer(layer: FormattedLayer) -> FinalLayer:
        f, m, a, k = layer
        return {
            "from": parse_layer_from(f, env),
            "module": parse_module(m, env),
            "args": parse_args(a, env),
            "kwargs": parse_kwargs(k, env),
        }

    __check_layers(layers)
    layers = __parse_layers(layers, env or {})
    layers = __format_layers(layers)
    return tuple(parse_layer(layer) for layer in layers)
