from typing import Any, Callable, Iterable

import torch.nn as nn

ModuleLike = type[nn.Module] | Callable[..., nn.Module]

Param = str | dict[str, Any]
Args = Iterable[Any]
Kwargs = dict[str, Any]
ArgDict = dict[str, Any]

Former = list[tuple[int, int | str]]
Converter = tuple[Callable, Args, Kwargs]
Layer = tuple[Former | str, type | Callable, Args, Kwargs]
