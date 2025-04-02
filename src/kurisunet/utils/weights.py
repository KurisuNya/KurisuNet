from pathlib import Path
from typing import Literal

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ..utils.logger import get_logger

logger = get_logger("Utils")


def save_state_dict(
    state_dict: dict[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
):
    """Save state dict to a file using safetensors."""
    save_file(state_dict, path, metadata=metadata)


def load_state_dict(path: str | Path, device="cpu") -> dict[str, torch.Tensor]:
    """Load state dict from a file using safetensors."""
    with safe_open(path, "pt", device=device) as f:
        return {k: f.get_tensor(k) for k in f.keys()}


CONVERT_STRATEGY = Literal["register_order"]


def convert_state_dict(
    old_state_dict: dict[str, torch.Tensor],
    new_state_dict: dict[str, torch.Tensor],
    strategy: CONVERT_STRATEGY = "register_order",
) -> dict[str, torch.Tensor]:
    """Convert old state dict to new state dict using a conversion strategy."""

    def register_order_key_map(old_state_dict, new_state_dict) -> dict[str, str]:
        if len(old_state_dict) != len(new_state_dict):
            raise ValueError("Length of state dicts must be equal")
        old_keys = list(old_state_dict.keys())
        new_keys = list(new_state_dict.keys())
        key_map = {o: n for o, n in zip(old_keys, new_keys)}
        return key_map

    if strategy == "register_order":
        key_map = register_order_key_map(old_state_dict, new_state_dict)
    return {key_map[k]: v for k, v in old_state_dict.items()}
