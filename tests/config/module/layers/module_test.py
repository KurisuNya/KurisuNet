import unittest

from kurisunet.config.module.layers.module import parse_module


class TestParseModule(unittest.TestCase):
    def test_parse_module(self):
        import torch

        env = {"torch": torch, "n": 1}

        module_type = torch.nn.Linear
        parsed_module = parse_module(module_type, env)
        self.assertEqual(parsed_module.__name__, "Linear")

        parsed_module = parse_module(lambda x: x, env)
        self.assertEqual(parsed_module()(1), 1)

        module_str = "torch.nn.Linear"
        parsed_module = parse_module(module_str, env)
        self.assertEqual(parsed_module.__name__, "Linear")

        module_str = "lambda x: x + n"
        parsed_module = parse_module(module_str, env)
        self.assertEqual(parsed_module()(1), 2)

        module_str = "lambda x, y: x - y + n"
        parsed_module = parse_module(module_str, env)
        self.assertEqual(parsed_module(2)(1), 0)

    def test_invalid_module(self):
        invalid_modules = [
            123,
            [1, 2, 3],
            {"key": "value"},
            None,
            "invalid_module",
            "lambda x: x + n + m",
        ]
        for module in invalid_modules:
            with self.assertRaises((ValueError, NameError)):
                parse_module(module, None)()(1)


if __name__ == "__main__":
    unittest.main()
