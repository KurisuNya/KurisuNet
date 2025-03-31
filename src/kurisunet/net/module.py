from typing import Any, Callable, Iterable, cast

from loguru import logger
import torch.nn as nn

from ..basic.types import Env
from ..config.types import FinalLayer, FromTuple
from ..constants import ALL_FROM
from .types import ModuleMeta
from .utils import auto_unpack, get_same_indexes, module_enum
from .utils import (
    get_drop_layer_indexes,
    get_except_indexes,
    get_unused_layer_indexes,
    layer_enum,
    regularize_layer_from,
)


def OutputModule(*args: Any) -> tuple[Any, ...] | Any:
    """Output module."""
    return auto_unpack(args)


class PipelineModule(nn.Module):
    """Pipeline module."""

    def __register_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self.register_parameter(key, value)

    def __register_buffers(self, buffers: dict[str, Any]) -> None:
        for key, value in buffers.items():
            self.register_buffer(key, value)

    def __register_modules(
        self,
        modules: Iterable[nn.Module | Callable[..., Any]],
        forward_drop: set[int] = set(),
    ) -> None:
        modules = list(modules)
        same_dict = get_same_indexes(modules)

        __import__("pprint").pprint(forward_drop)
        for i, same in same_dict.items():
            logger.info(
                f"modules with indexes {i} and {same} are the same "
                f"and will only be registered once"
            )
            all_indexes = {i} | same
            if all_indexes.issubset(forward_drop):
                forward_drop.difference_update(same)
            else:
                forward_drop.difference_update(all_indexes)
        __import__("pprint").pprint(forward_drop)

        is_module = lambda p: isinstance(p[1], nn.Module)
        not_same = lambda p: p[0] not in {i for s in same_dict.values() for i in s}
        indexed_module = filter(is_module, layer_enum(modules))
        indexed_module = filter(not_same, indexed_module)
        reindexed_module = list(module_enum(indexed_module))

        for module_index, (layer_index, module) in reindexed_module:
            module = cast(nn.Module, module)
            if layer_index in forward_drop:
                self.__meta["drop_set"].add(module_index)
            self.add_module(str(module_index), module)

    def __init__(self):
        """Lazy initialization of the pipeline module."""
        super().__init__()
        self.__meta: ModuleMeta = {"name": "PipelineModule", "drop_set": set()}

    def get_env(self) -> Env:
        return {"self": self}

    def init(
        self,
        name: str,
        layers: tuple[FinalLayer, ...],
        buffers: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ):
        """Real initialization of the pipeline module."""
        self.__meta["name"] = name
        self.__register_buffers(buffers or {})
        self.__register_params(params or {})

        layers = regularize_layer_from(layers)
        if forward_drop := get_drop_layer_indexes(layers):
            logger.info(
                f"layers with indexes {forward_drop} are set "
                f"to be dropped in forward pass"
            )
        if forward_unused := get_unused_layer_indexes(layers):
            logger.warning(
                f"layers with indexes {forward_unused} are not "
                f"connected to any other layers and will be dropped in forward pass"
            )
        # INFO: register all modules to load state_dict without drop
        all_modules = [l["module"](*l["args"], **l["kwargs"]) for l in layers]
        self.__register_modules(all_modules, forward_drop.union(forward_unused))

        layers = get_except_indexes(layers, forward_drop)
        from_list = cast(list[FromTuple], [l["from"] for l in layers])
        modules = get_except_indexes(all_modules, forward_drop)
        self.__modules = tuple(
            (i, (f, m))
            for i, (f, m) in layer_enum(zip(from_list, modules))
            if i not in get_unused_layer_indexes(layers)
        )

        if submodule_str := self.get_submodules_str():
            logger.debug(f"{name} is created with submodules:\n{submodule_str}")
        else:
            logger.debug(f"{name} is created without submodules")

    def forward(self, *x: Any) -> Any:
        """Forward pass through the pipeline module."""
        # INFO: because of torch graph will reference to the all tensors in forward pass,
        # save all results in a dict does not increase memory usage.
        results: dict[int, Any] = {0: x[0] if len(x) == 1 else x}
        for i, (f, m) in self.__modules:
            x = m(*(results[k] if v == ALL_FROM else results[k][v] for k, v in f))
            results[i] = x
        return x

    def add_drop(self, indexes: Iterable[int] | int):
        """Add submodules indexes to drop_set."""
        indexes = [indexes] if isinstance(indexes, int) else indexes
        self.__meta["drop_set"].update(indexes)

    def remove_drop(self, indexes: Iterable[int] | int):
        """Remove submodules indexes in drop_set."""
        indexes = [indexes] if isinstance(indexes, int) else indexes
        self.__meta["drop_set"].difference_update(indexes)

    def drop(self, resort: bool = False):
        """
        Drop submodules with indexes in drop_set.
        If resort is True, resort the submodules after dropping.
        """
        for i in self.__meta["drop_set"]:
            del self._modules[str(i)]
        self.__meta["drop_set"] = set()
        if resort:
            self.resort()

    def resort(self):
        """Resort the submodules."""
        modules = self._modules.copy()
        for k in modules.keys():
            del self._modules[k]
        for i, k in module_enum(sorted(modules.keys(), key=lambda x: int(x))):
            self.add_module(str(i), modules[k])
        del modules

    def get_module_name(self) -> str:
        """Get the module name."""
        return self.__meta["name"]

    def get_submodules_str(self) -> str:
        """Get the string representation of the submodules."""
        lines = str(self).split("\n")[1:-1]  # remove outermost brackets
        return "\n".join([s[2:] for s in lines])  # remove leading spaces

    def __repr__(self):
        """Get the string representation of the module."""
        lines = super().__repr__().split("\n")
        name = self.get_module_name()
        lines[0] = f"{name}(" if len(lines) > 1 else f"{name}()"
        return "\n".join(lines)
