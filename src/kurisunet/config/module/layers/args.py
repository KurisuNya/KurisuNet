from __future__ import annotations

from ....basic.types import Env
from ...types import Args, Kwargs, LayerArgs, LayerKwargs
from ...utils import eval_string

__eval = lambda x, env: eval_string(x, env) if isinstance(x, str) else x


def parse_args(args: LayerArgs, env: Env | None) -> Args:
    """Parse the expressions in args"""
    return tuple(__eval(x, env or {}) for x in args)


def parse_kwargs(kwargs: LayerKwargs, env: Env | None) -> Kwargs:
    """Parse the expressions in kwargs"""
    return {k: __eval(v, env or {}) for k, v in kwargs.items()}
