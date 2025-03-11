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

        def get_unused_indexes(former_list: list[Former]) -> set[int]:
            def count_former_list(former_list: list[Former]) -> dict[int, int]:
                f_list = [[get_first_key(f) for f in former] for former in former_list]
                results = [Counter(f) for f in f_list]
                return sum(results, Counter())

            count_result = count_former_list(former_list)
            indexes_except_last = set(range(len(former_list)))
            used_indexes = set(count_result.keys())
            drop_indexes = indexes_except_last.difference(used_indexes)
            return drop_indexes

        formers = [former for former, _, _, _ in layers]
        index_pairs = enumerate(formers, start=1)
        self.__drop_indexes = {i for i, f in index_pairs if is_drop_former(f)}
        if self.__drop_indexes:
            logger.info(
                f"layer(s) with index(es) {self.__drop_indexes} is/are set to be dropped"
            )

        self.__former_list: list[Former] = [f for f in formers if not is_drop_former(f)]  # type: ignore
        self.__unused_indexes = get_unused_indexes(self.__former_list)
        if self.__unused_indexes:
            logger.warning(
                f"layer(s) with index(es) {self.__unused_indexes} "
                f"is/are not connected to any other layer(s) and will be dropped"
            )

        modules = [ModuleRegister.get(m)(*a, **k) for _, m, a, k in layers]
        for i, module in enumerate(filter(lambda m: isinstance(m, nn.Module), modules)):
            self.add_module(str(i), module)
        modules_info = "\n".join([f"{module}" for module in modules])
        logger.debug(f"{name} is created with modules:\n{modules_info}")

        index_pairs = enumerate(modules, start=1)
        self.__modules = [m for i, m in index_pairs if i not in self.__drop_indexes]

    def forward(self, x):
        def get_input(former: Former, results: dict[int, Any]):
            items = [get_first_item(f) for f in former]
            return tuple(results[k] if v == "all" else results[k][v] for k, v in items)

        index_pairs = enumerate(zip(self.__former_list, self.__modules), start=1)
        index_pairs = filter(lambda p: p[0] not in self.__unused_indexes, index_pairs)

        # INFO: because of torch graph will reference to the all tensors in forward pass,
        # save all results in a dict does not increase memory usage.
        results_dict = {0: x}
        for i, (former, module) in index_pairs:
            x = module(*get_input(former, results_dict))
            results_dict[i] = x
        return x

    def get_module_name(self) -> str:
        return self.__name
