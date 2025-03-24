import unittest

import torch.nn as nn

from kurisunet.config.module.layers import parse_layers
from kurisunet.constants import ALL_FROM


class TestParseLayers(unittest.TestCase):
    def test_parse_conv_bn_silu(self):
        def autopad(k, p=None, d=1):  # kernel, padding, dilation
            """Pad to 'same' shape outputs."""
            if d > 1:
                k = (
                    d * (k - 1) + 1
                    if isinstance(k, int)
                    else [d * (x - 1) + 1 for x in k]
                )  # actual kernel-size
            if p is None:
                p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
            return p

        import_env = {"autopad": autopad, "nn": nn}
        args_env = {"c1": 3, "c2": 16, "k": 1, "s": 1, "p": None, "g": 1, "d": 1}
        layers = [
            [
                -1,
                "nn.Conv2d",
                ["c1", "c2", "k", "s", "autopad(k, p, d)", "d", "g"],
                {"bias": False},
            ],
            [-1, "nn.BatchNorm2d", ["c2"]],
            [-1, "nn.SiLU"],
        ]
        expected = (
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
        env = {}
        env.update(import_env)
        env.update(args_env)
        self.assertEqual(parse_layers(layers, env), expected)


if __name__ == "__main__":
    unittest.main()
