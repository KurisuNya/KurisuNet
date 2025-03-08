from collections import Counter
from typing import Any, Callable

from kurisuinfo import CustomizedModuleName
from loguru import logger
import torch.nn as nn

from .config import Former, Layer
from .utils import get_first_item, get_first_key


class OutputModule(nn.Module, CustomizedModuleName):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args

    def get_module_name(self) -> str:
        return "Output"


class LambdaModule(nn.Module, CustomizedModuleName):
    def __init__(self, name: str, forward: Callable):
        super().__init__()
        self.__name = name
        self.__forward = forward

    def forward(self, x):
        return self.__forward(x)

    def get_module_name(self) -> str:
        return self.__name


class StreamModule(nn.Module, CustomizedModuleName):
    def __init__(self, name: str, layers: list[Layer]):
        from .register import ModuleRegister

        super().__init__()
        self.__name = name
        layers_info = "\n".join([f"{layer}" for layer in layers])
        logger.debug(f"Creating {name} with layers:\n{layers_info}")

        def get_drop_indexes(former_list: list[Former]) -> set[int]:
            def count_former_list(former_list: list[Former]) -> dict[int, int]:
                f_list = [[get_first_key(f) for f in former] for former in former_list]
                results = [Counter(f) for f in f_list]
                return sum(results, Counter())

            count_result = count_former_list(former_list)
            indexes_except_last = set(range(len(former_list)))
            used_indexes = set(count_result.keys())
            drop_indexes = indexes_except_last.difference(used_indexes)
            return drop_indexes

        self.__former_list = [former for former, _, _, _ in layers]
        self.__drop_indexes = get_drop_indexes(self.__former_list)
        if len(self.__drop_indexes) > 0:
            logger.warning(
                f"layer(s) with index(es) {self.__drop_indexes} "
                f"is/are not connected to any other layer(s) and will be dropped."
            )
        self.__modules = [ModuleRegister.get(m)(*a, **k) for _, m, a, k in layers]
        modules = filter(lambda m: isinstance(m, nn.Module), self.__modules)
        for i, module in enumerate(modules):
            self.add_module(str(i), module)  # type: ignore
        modules_info = "\n".join([f"{module}" for module in self.__modules])
        logger.debug(f"{name} is created with modules:\n{modules_info}")

    def forward(self, x):
        def get_input(former: Former, results: dict[int, Any]):
            items = [get_first_item(f) for f in former]
            return tuple(results[k] if v == "all" else results[k][v] for k, v in items)

        index_pairs = enumerate(zip(self.__former_list, self.__modules), start=1)
        index_pairs = filter(lambda p: p[0] not in self.__drop_indexes, index_pairs)

        # INFO: because of torch graph will reference to the all tensors in forward pass,
        # save all results in a dict does not increase memory usage.
        results_dict = {0: x}
        for i, (former, module) in index_pairs:
            x = module(*get_input(former, results_dict))
            results_dict[i] = x
        return x

    def get_module_name(self) -> str:
        return self.__name
