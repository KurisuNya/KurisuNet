from pathlib import Path

from kurisuinfo import summary
import torch

from kurisunet.module.register import get_module
from kurisunet.utils.weights import convert_state_dict, save_state_dict
from old_net import CNNClassifier


if __name__ == "__main__":
    dir = Path(__file__).parent
    cfg = {
        "path": "../module/config.yaml",
        "name": "SimpleCNN",
        "kwargs": {"in_ch": 1, "class_num": 2, "width": 0.5},
        "input_shape": (1, 1, 128, 128),
    }

    old_module = CNNClassifier(**cfg["kwargs"])
    new_module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])

    old_state_dict = torch.load(dir / "old_net_weights.pt", map_location="cpu")
    new_state_dict = convert_state_dict(old_state_dict, new_module.state_dict())
    save_state_dict(new_state_dict, dir / "new_net_weights.safetensors")

    new_module.load_state_dict(new_state_dict)
    summary(new_module, cfg["input_shape"])
