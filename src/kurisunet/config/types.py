from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TypeVar, TypedDict, Protocol, runtime_checkable

from ..basic.types import ListTuple, OneOrMore

T = TypeVar("T")
NeedEval = str | T

# input types
Args = ListTuple[Any]
Kwargs = dict[str, Any]

# args types
FormattedParam = str | tuple[str, NeedEval[Any]]
Param = FormattedParam | dict[str, NeedEval[Any]]
ArgDict = dict[str, NeedEval[Any]]

# vars types
FormattedVar = tuple[str, NeedEval[Any]]
Var = FormattedVar | dict[str, NeedEval[Any]]

# layers types
Drop = str
From = dict[int, int | str] | int
FormattedFrom = tuple[int, int | str]
FromTuple = tuple[FormattedFrom, ...]

LayerFrom = NeedEval[Drop | OneOrMore[From]]
ParsedLayerFrom = Drop | OneOrMore[From]
FormattedLayerFrom = Drop | FromTuple


@runtime_checkable
class CustomModule(Protocol):
    @abstractmethod
    def get_module(self, *args: Any, **kwargs: Any) -> Any:
        pass


Module = type | Callable[..., Any]
ParsedModule = Module | CustomModule
LayerModule = NeedEval[ParsedModule]

LayerArgs = tuple[NeedEval[Any], ...]
LayerKwargs = dict[str, NeedEval[Any]]

Layer = NeedEval[tuple[LayerFrom, LayerModule, LayerArgs, LayerKwargs]]
FormattedLayer = tuple[LayerFrom, LayerModule, LayerArgs, LayerKwargs]
FinalLayer = TypedDict(
    "FinalLayer",
    {
        "from": FormattedLayerFrom,
        "module": Module,
        "args": Args,
        "kwargs": Kwargs,
    },
)


# converters types
ParsedConverter = Callable[..., dict[str, Any]]
Converter = NeedEval[ParsedConverter]
ConverterLayer = tuple[Converter, LayerArgs, LayerKwargs]
FinalConverterLayer = TypedDict(
    "FinalConverterLayer",
    {
        "converter": ParsedConverter,
        "args": Args,
        "kwargs": Kwargs,
    },
)
