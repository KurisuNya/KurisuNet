import os
from pathlib import Path

from kurisuinfo import summary
import torch

from kurisunet import get_module
from kurisunet.utils.weights import convert_state_dict, load_state_dict, save_state_dict
from old_net import CNNClassifier


if __name__ == "__main__":
    dir = Path(__file__).parent
    cfg = {
        "path": "./net.yaml",
        "name": "SimpleCNN",
        "kwargs": {"in_ch": 1, "class_num": 2, "width": 0.5},
        "input_shape": (1, 1, 128, 128),
    }

    os.makedirs(dir / "weights", exist_ok=True)
    old_weights_path = dir / "weights" / "old_net_weights.pt"
    new_weights_path = dir / "weights" / "new_net_weights.pt"

    old_module = CNNClassifier(**cfg["kwargs"])
    new_module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])

    torch.save(old_module.state_dict(), old_weights_path)
    old_state_dict = torch.load(old_weights_path, map_location="cpu")
    new_state_dict = convert_state_dict(old_state_dict, new_module.state_dict())
    save_state_dict(new_state_dict, new_weights_path)

    new_state_dict = load_state_dict(new_weights_path)
    new_module.load_state_dict(new_state_dict)
    summary(new_module, cfg["input_shape"])
