import unittest

import torch
import torch.nn as nn

from kurisunet.config.types import FinalLayer
from kurisunet.constants import ALL_FROM, DROP_FROM
from kurisunet.net.module import OutputModule, PipelineModule
from kurisuinfo import CustomizedModuleName


class TestOutputModule(unittest.TestCase):
    def test_output_module(self):
        pairs = [
            ("test_input", "test_input"),
            ((1, 2, 3), (1, 2, 3)),
            ((1,), (1,)),
            ([], []),
            ([1, 2, 3], [1, 2, 3]),
            ([1], [1]),
        ]
        for input, expected in pairs:
            output = OutputModule(input)
            self.assertEqual(expected, output)


class TestPipelineModule(unittest.TestCase):
    def test_init(self):
        conv_bn_relu: tuple[FinalLayer, ...] = (
            {
                "args": (3, 16, 1, 1, 0, 1, 1),
                "from": ((-1, ALL_FROM),),
                "kwargs": {"bias": False},
                "module": nn.Conv2d,
            },
            {
                "args": (16,),
                "from": ((-1, ALL_FROM),),
                "kwargs": {},
                "module": nn.BatchNorm2d,
            },
            {
                "args": (),
                "from": ((-1, ALL_FROM),),
                "kwargs": {},
                "module": nn.SiLU,
            },
        )
        module = PipelineModule("ConvBNReLU", conv_bn_relu)
        module_str = (
            "ConvBNReLU(\n"
            "  (1): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)\n"
            "  (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n"
            "  (3): SiLU()\n"
            ")"
        )
        self.assertIsInstance(module, CustomizedModuleName)
        self.assertEqual(str(module), module_str)
        input = torch.randn(1, 3, 224, 224)
        self.assertEqual(module(input).shape, (1, 16, 224, 224))

    def test_init_with_drop(self):
        conv_bn_relu: tuple[FinalLayer, ...] = (
            {
                "args": (3, 16, 1, 1, 0, 1, 1),
                "from": ((-1, ALL_FROM),),
                "kwargs": {"bias": False},
                "module": nn.Conv2d,
            },
            {
                "args": (),
                "from": DROP_FROM,
                "kwargs": {},
                "module": lambda *a, **k: lambda x: None,
            },
            {
                "args": (16,),
                "from": DROP_FROM,
                "kwargs": {},
                "module": nn.BatchNorm2d,
            },
            {
                "args": (),
                "from": ((-1, ALL_FROM),),
                "kwargs": {},
                "module": lambda *a, **k: lambda x: x,
            },
            {
                "args": (),
                "from": ((-1, ALL_FROM),),
                "kwargs": {},
                "module": nn.SiLU,
            },
        )
        module = PipelineModule("ConvBNReLU", conv_bn_relu)
        module_str = (
            "ConvBNReLU(\n"
            "  (1): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)\n"
            "  (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n"
            "  (3): SiLU()\n"
            ")"
        )
        self.assertEqual(str(module), module_str)
        input = torch.randn(1, 3, 224, 224)
        self.assertEqual(module(input).shape, (1, 16, 224, 224))

        module.drop()
        module_str = (
            "ConvBNReLU(\n"
            "  (1): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)\n"
            "  (3): SiLU()\n"
            ")"
        )
        self.assertEqual(str(module), module_str)
        input = torch.randn(1, 3, 224, 224)
        self.assertEqual(module(input).shape, (1, 16, 224, 224))

        module.resort()
        module_str = (
            "ConvBNReLU(\n"
            "  (1): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)\n"
            "  (2): SiLU()\n"
            ")"
        )
        self.assertEqual(str(module), module_str)
        input = torch.randn(1, 3, 224, 224)
        self.assertEqual(module(input).shape, (1, 16, 224, 224))

        module = PipelineModule("ConvBNReLU", conv_bn_relu)
        module.drop(resort=True)
        module_str = (
            "ConvBNReLU(\n"
            "  (1): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)\n"
            "  (2): SiLU()\n"
            ")"
        )
        self.assertEqual(str(module), module_str)
        input = torch.randn(1, 3, 224, 224)
        self.assertEqual(module(input).shape, (1, 16, 224, 224))


if __name__ == "__main__":
    unittest.main()
