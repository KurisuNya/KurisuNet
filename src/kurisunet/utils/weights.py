from copy import deepcopy
from pathlib import Path
from typing import Dict, Literal

from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn as nn

from ..module.module import StreamModule


def save_state_dict(
    state_dict: Dict[str, torch.Tensor],
    path: str | Path,
    metadata: Dict[str, str] | None = None,
):
    save_file(state_dict, path, metadata=metadata)


def load_state_dict(path: str | Path, device="cpu") -> Dict[str, torch.Tensor]:
    with safe_open(path, "pt", device=device) as f:
        return {k: f.get_tensor(k) for k in f.keys()}


ConvertStrategy = Literal["instance_order"]


def convert_state_dict(
    old_state_dict: Dict[str, torch.Tensor],
    new_state_dict: Dict[str, torch.Tensor],
    strategy: ConvertStrategy = "instance_order",
) -> Dict[str, torch.Tensor]:

    def instance_order_key_map(old_state_dict, new_state_dict) -> Dict[str, str]:
        if len(old_state_dict) != len(new_state_dict):
            raise ValueError("Length of state dicts must be equal")
        old_keys = list(old_state_dict.keys())
        new_keys = list(new_state_dict.keys())
        key_map = {o: n for o, n in zip(old_keys, new_keys)}
        return key_map

    if strategy == "instance_order":
        key_map = instance_order_key_map(old_state_dict, new_state_dict)
    return {key_map[k]: v for k, v in old_state_dict.items()}


def get_dropped_state_dict(
    module: nn.Module, inplace: bool = False
) -> Dict[str, torch.Tensor]:
    if inplace:
        logger.warning("The input module will be modified in-place")
    else:
        module = deepcopy(module)
    for m in module.modules():
        if isinstance(m, StreamModule):
            m.remove_dropped()
    return module.state_dict()
