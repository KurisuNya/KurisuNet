from typing import Any

from ..basic.types import Env
from ..constants import STR_PREFIX


def eval_string(string: str, env: Env) -> Any:
    """Evaluate a string in the given environment."""
    if string.startswith(STR_PREFIX):
        return string[len(STR_PREFIX) :]
    return eval(string, env)
