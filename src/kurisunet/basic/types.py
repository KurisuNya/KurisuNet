from typing import Any, TypeVar

T = TypeVar("T")
ListTuple = list[T] | tuple[T, ...]
OneOrMore = T | ListTuple[T]
Env = dict[str, Any]
