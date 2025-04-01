import ast
from ast import Import, ImportFrom
from typing import Any, cast

from ...basic.types import Env, ListTuple
from ...basic.utils import is_list_tuple_of

ImportType = Import | ImportFrom


def _check_imports(imports: Any) -> None:
    def check_import(import_: str):
        try:
            body = ast.parse(import_).body
        except SyntaxError:
            raise ValueError(f"Invalid import statement: {import_}")
        if len(body) != 1 or not isinstance(body[0], (Import, ImportFrom)):
            raise ValueError(f"Invalid import statement: {import_}")

    if not is_list_tuple_of(imports, str):
        raise ValueError(f"Invalid imports {imports}, should be list/tuple of str")
    for import_ in imports:
        check_import(import_)

    imports_ast = cast(list[ImportType], [ast.parse(i).body[0] for i in imports])
    names = [name.asname or name.name for ast in imports_ast for name in ast.names]
    unique_names = set(names)
    if len(unique_names) != len(names):
        raise ValueError(f"Duplicate import names found in {imports}")


def _get_imports_env(imports: ListTuple[str]) -> Env:
    modules = {}
    for import_ in imports:
        exec(import_, {}, modules)
    return modules


def get_imports_env(imports: Any) -> Env:
    """Get the imports environment from the given import statements."""
    _check_imports(imports)
    return _get_imports_env(imports)
