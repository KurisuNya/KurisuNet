from collections import Counter
from typing import Any, Callable

from kurisuinfo import CustomizedModuleName
from loguru import logger
import torch.nn as nn

from .config import Former, Layer, is_drop_former
from .utils import get_first_item, get_first_key


class OutputModule(nn.Module, CustomizedModuleName):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args

    def get_module_name(self) -> str:
        return "Output"


class StreamModule(nn.Module, CustomizedModuleName):
    def __init__(self, name: str, layers: list[Layer]):
        from .register import ModuleRegister

        super().__init__()
        self.__meta: dict[str, Any] = {}
        self.__meta["name"] = name

        def get_drop_set(former_list: list):
            return {i for i, f in enumerate(former_list, start=1) if is_drop_former(f)}

        def get_dropped(lst: list, drop_set: set[int]):
            return [x for i, x in enumerate(lst, start=1) if i not in drop_set]

        def get_unused_set(former_list: list[Former]):
            def count_former_list(former_list: list[Former]) -> dict[int, int]:
                f_list = [[get_first_key(f) for f in former] for former in former_list]
                results = [Counter(f) for f in f_list]
                return sum(results, Counter())

            count_result = count_former_list(former_list)
            indexes_except_last = set(range(len(former_list)))
            used_indexes = set(count_result.keys())
            drop_indexes = indexes_except_last.difference(used_indexes)
            return drop_indexes

        def get_modules(layers: list[Layer]):
            def get_module(module, a, k):
                if isinstance(module, str):
                    return ModuleRegister.get(module)(*a, **k)
                if isinstance(module, type):
                    return module(*a, **k)
                return lambda *args: module(*args, *a, **k)

            return [get_module(m, a, k) for _, m, a, k in layers]

        def add_modules(modules: list[nn.Module] | list[Callable]):
            nn_modules = list(filter(lambda m: isinstance(m, nn.Module), modules))
            self.__meta["module_count"] = len(nn_modules)
            for i, module in enumerate(nn_modules, start=1):
                self.add_module(str(i), module)  # type: ignore

        formers = [former for former, _, _, _ in layers]
        if drop_set := get_drop_set(formers):
            logger.info(f"layer(s) with index(es) {drop_set} is/are set to be dropped")
        self.__meta["drop_set"] = drop_set
        formers = get_dropped(formers, drop_set)
        modules = get_modules(layers)
        add_modules(modules)  # INFO: ensure state_dict can be loaded correctly
        modules = get_dropped(modules, drop_set)
        if unused_set := get_unused_set(formers):
            logger.warning(
                f"layer(s) with index(es) {unused_set} is/are not "
                f"connected to any other layer(s) and will be dropped"
            )
        index_pairs = enumerate(zip(formers, modules), start=1)
        self.__modules = {i: (f, m) for i, (f, m) in index_pairs if i not in unused_set}
        logger.debug(f"{name} is created:\n{self}")

    def __remove_modules(self, remove_set: set[int]):
        for i in remove_set:
            m = getattr(self, str(i))
            delattr(self, str(i))
            logger.debug(f"module {i} is removed:\n{m}")

    def __resort_modules(self):
        index_range = range(1, self.__meta["module_count"] + 1)
        indexes = [i for i in index_range if hasattr(self, str(i))]
        if len(indexes) == self.__meta["module_count"]:
            return
        modules = [getattr(self, str(i)) for i in indexes]
        for i, m in enumerate(modules, start=1):
            setattr(self, str(i), m)
        self.__meta["module_count"] = len(modules)
        index_range = range(1, len(modules) + 1)
        remove = set(indexes).difference(index_range)
        for i in remove:
            delattr(self, str(i))

    def remove_dropped(self):
        self.__remove_modules(self.__meta["drop_set"])
        self.__resort_modules()
        self.__meta["drop_set"] = set()

    def forward(self, *x):
        def get_input(former: Former, results: dict[int, Any]):
            items = [get_first_item(f) for f in former]
            return tuple(results[k] if v == "all" else results[k][v] for k, v in items)

        # INFO: because of torch graph will reference to the all tensors in forward pass,
        # save all results in a dict does not increase memory usage.
        results_dict = {0: x[0] if len(x) == 1 else x}
        index_pairs = zip(self.__modules.keys(), self.__modules.values())
        for i, (former, module) in index_pairs:
            x = module(*get_input(former, results_dict))
            results_dict[i] = x
        return x

    def get_module_name(self) -> str:
        return self.__meta["name"]
