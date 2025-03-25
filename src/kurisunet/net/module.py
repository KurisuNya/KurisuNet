from typing import Any, Callable, Iterable, cast

from loguru import logger
import torch.nn as nn

from ..basic.types import Env
from ..config.types import FinalLayer, FromTuple
from ..constants import ALL_FROM
from .types import ModuleMeta
from .utils import auto_unpack, module_enum
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
        drop_indexes: set[int] = set(),
    ) -> None:
        indexed_all = layer_enum(modules)
        indexed_module = filter(lambda p: isinstance(p[1], nn.Module), indexed_all)
        reindexed = list(module_enum(indexed_module))
        reindexed = cast(list[tuple[int, tuple[int, nn.Module]]], reindexed)
        for module_index, (layer_index, module) in reindexed:
            if layer_index in drop_indexes:
                self.__meta["drop_set"].add(module_index)
            self.add_module(str(module_index), module)

    def __init__(self):
        """Lazy initialization of the pipeline module."""
        super().__init__()

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
        super().__init__()
        self.__meta: ModuleMeta = {"name": name, "drop_set": set()}
        self.__register_buffers(buffers or {})
        self.__register_params(params or {})

        if drop_indexes := get_drop_layer_indexes(layers):
            logger.info(f"layers with indexes {drop_indexes} are set to be dropped")
        # INFO: register all modules to load state_dict without drop
        all_modules = [l["module"](*l["args"], **l["kwargs"]) for l in layers]
        self.__register_modules(all_modules, drop_indexes)

        layers = regularize_layer_from(get_except_indexes(layers, drop_indexes))
        if unused_indexes := get_unused_layer_indexes(layers):
            logger.warning(
                f"layers with indexes {unused_indexes} are not "
                f"connected to any other layers and will be dropped"
            )
        modules = get_except_indexes(all_modules, drop_indexes)
        from_list = cast(list[FromTuple], [l["from"] for l in layers])
        pairs = layer_enum(zip(from_list, modules))
        self.__modules = {i: (f, m) for i, (f, m) in pairs if i not in unused_indexes}

        if submodule_str := self.get_submodules_str():
            logger.debug(f"{name} is created with submodules:\n{submodule_str}")
        else:
            logger.debug(f"{name} is created without submodules")

    def forward(self, *x):
        """Forward pass through the pipeline module."""

        def get_input(from_: FromTuple, results: dict[int, Any]):
            return (results[k] if v == ALL_FROM else results[k][v] for k, v in from_)

        # INFO: because of torch graph will reference to the all tensors in forward pass,
        # save all results in a dict does not increase memory usage.
        results_dict = {0: auto_unpack(x)}
        index_pairs = zip(self.__modules.keys(), self.__modules.values())
        for i, (from_, module) in index_pairs:
            x = module(*get_input(from_, results_dict))
            results_dict[i] = x
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
