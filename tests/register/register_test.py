from pathlib import Path
import unittest

from kurisuinfo import summary
import torch
import yaml

from kurisunet import get_module

from kurisunet.constants import BUFFERS_KEY, CONVERTERS_KEY, PARAMS_KEY
from kurisunet.register import register_config, ModuleRegister


def invalid_converter(config, *args, **kwargs):
    return {"fake_config": "fake"}


class TestRegisterConfig(unittest.TestCase):
    def setUp(self):
        ModuleRegister.clear()

    def test_register_invalid_config(self):
        invalid = [(), [], 1, 1.0, True]
        invalid_path = ["fake.yaml"]
        warn = [
            {"test": "not a config"},
            {"test": {"fake_module": "fake"}},
        ]
        for config in invalid + invalid_path:
            with self.assertRaises((ValueError, FileNotFoundError)):
                register_config(config)
        for config in warn:
            register_config(config)

    def test_get_module(self):
        dir = Path(__file__).parent
        cfg = {
            "path": "../test_module/net.yaml",
            "name": "VAE",
            "kwargs": {
                "img_size": (1, 28, 28),
                "encoder_dims": [512, 256, 128],
                "decoder_dims": [128, 256, 512],
                "z_dim": 10,
            },
            "input_shape": (1, 1, 28, 28),
        }
        module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=dir / cfg["path"])
        summary(module, cfg["input_shape"], verbose=0)
        input = torch.rand(*cfg["input_shape"])
        output = module(input)
        self.assertEqual(output[0].shape, input.shape)

    def test_register_config(self):
        dir = Path(__file__).parent
        cfg = {
            "path": "../test_module/net.yaml",
            "name": "VAE",
            "kwargs": {
                "img_size": (1, 28, 28),
                "encoder_dims": [512, 256, 128],
                "decoder_dims": [128, 256, 512],
                "z_dim": 10,
            },
            "input_shape": (1, 1, 28, 28),
        }
        register_config(dir / cfg["path"])
        module = get_module(cfg["name"], kwargs=cfg["kwargs"])
        summary(module, cfg["input_shape"], verbose=0)
        input = torch.rand(*cfg["input_shape"])
        output = module(input)
        self.assertEqual(output[0].shape, input.shape)

    def test_invalid_converter(self):
        dir = Path(__file__).parent
        cfg = {
            "path": "../test_module/net.yaml",
            "name": "VAE",
            "kwargs": {
                "img_size": (1, 28, 28),
                "encoder_dims": [512, 256, 128],
                "decoder_dims": [128, 256, 512],
                "z_dim": 10,
            },
            "input_shape": (1, 1, 28, 28),
        }
        config = yaml.safe_load((dir / cfg["path"]).open())
        config["VAE"][CONVERTERS_KEY][0][0] = invalid_converter
        with self.assertRaises(ValueError):
            get_module(cfg["name"], kwargs=cfg["kwargs"], config=config)

    def test_params_buffers(self):
        dir = Path(__file__).parent
        cfg = {
            "path": "../test_module/net.yaml",
            "name": "VAE",
            "kwargs": {
                "img_size": (1, 28, 28),
                "encoder_dims": [512, 256, 128],
                "decoder_dims": [128, 256, 512],
                "z_dim": 10,
            },
            "input_shape": (1, 1, 28, 28),
        }
        config = yaml.safe_load((dir / cfg["path"]).open())
        config["VAE"][PARAMS_KEY] = [{"test_params": torch.nn.Parameter(torch.rand(1))}]
        config["VAE"][BUFFERS_KEY] = [{"test_buffer": torch.rand(1)}]
        module = get_module(cfg["name"], kwargs=cfg["kwargs"], config=config)
        summary(module, cfg["input_shape"], verbose=0)
        input = torch.rand(*cfg["input_shape"])
        output = module(input)
        self.assertEqual(output[0].shape, input.shape)

    def test_duplicate_params_buffers(self):
        dir = Path(__file__).parent
        cfg = {
            "path": "../test_module/net.yaml",
            "name": "VAE",
            "kwargs": {
                "img_size": (1, 28, 28),
                "encoder_dims": [512, 256, 128],
                "decoder_dims": [128, 256, 512],
                "z_dim": 10,
            },
            "input_shape": (1, 1, 28, 28),
        }
        config = yaml.safe_load((dir / cfg["path"]).open())
        config["VAE"][PARAMS_KEY] = [{"test": torch.nn.Parameter(torch.rand(1))}]
        config["VAE"][BUFFERS_KEY] = [{"test": torch.rand(1)}]
        with self.assertRaises(ValueError):
            get_module(cfg["name"], kwargs=cfg["kwargs"], config=config)


if __name__ == "__main__":
    unittest.main()
