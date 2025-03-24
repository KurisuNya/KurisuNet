import unittest

from kurisunet.config.module.converters import parse_converters


class TestParseLayers(unittest.TestCase):
    def test_parse_converters(self):
        def converter1(x):
            return x + 2

        class converter2:
            def __call__(self, x):
                return x + 3

        env = {"z": 2, "converter1": converter1, "converter2": converter2()}
        converters = [
            ["lambda x, y: x + y", [1]],
            [lambda x, y: x + y, {"y": "z"}],
            ["converter1"],
            ["converter2"],
        ]

        input = 1
        expected = (2, 3, 3, 4)
        converters = parse_converters(converters, env)

        for c, e in zip(converters, expected):
            self.assertEqual(c["converter"](input, *c["args"], **c["kwargs"]), e)


if __name__ == "__main__":
    unittest.main()
