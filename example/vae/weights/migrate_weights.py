from math import prod
from pathlib import Path

from kurisuinfo import summary
import torch

from kurisunet.module.register import get_module
from kurisunet.utils.weights import convert_state_dict, load_state_dict, save_state_dict
from old_net import VAE


if __name__ == "__main__":
    dir = Path(__file__).parent
    cfg = {
        "path": "../module/net.yaml",
        "name": "VAE",
        "kwargs": {
            "img_size": (1, 28, 28),
            "encoder_dims": [512, 256, 128],
            "decoder_dims": [128, 256, 512],
            "z_dim": 10,
        },
        "input_shape": (1, 1, 28, 28),
    }

    old_kwargs = {
        "in_dim": prod(cfg["kwargs"]["img_size"]),
        "z_dim": cfg["kwargs"]["z_dim"],
        "encoder_hid_dims": cfg["kwargs"]["encoder_dims"],
        "decoder_hid_dims": cfg["kwargs"]["decoder_dims"],
    }

    old_module = VAE(**old_kwargs)
    new_module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])

    torch.save(old_module.state_dict(), dir / "old_net_weights.pt")
    old_state_dict = torch.load(dir / "old_net_weights.pt", map_location="cpu")
    new_state_dict = convert_state_dict(old_state_dict, new_module.state_dict())
    save_state_dict(new_state_dict, dir / "new_net_weights.safetensors")

    new_state_dict = load_state_dict(dir / "new_net_weights.safetensors")
    new_module.load_state_dict(new_state_dict)
    summary(new_module, cfg["input_shape"])
