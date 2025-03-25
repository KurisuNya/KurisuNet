import unittest

from kurisunet.config.module.converters import parse_converters


class TestParseLayers(unittest.TestCase):
    def test_invalid_converters(self):
        env = {"converter": lambda x: x + 1}
        invalid_types = [1, 1.0, True, None, {}]
        converters = [
            ["converter", (), {}, "too_long"],
            [],
            ["converter", "invalid_args"],
            ["converter", (), "invalid_kwargs"],
            ["1+1"],
            [True],
        ]
        for converter in converters + invalid_types:
            with self.assertRaises((ValueError, NameError)):
                parse_converters([converter], env)

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
            ["converter2", (), {}],
        ]

        input = 1
        expected = (2, 3, 3, 4)
        converters = parse_converters(converters, env)

        for c, e in zip(converters, expected):
            self.assertEqual(c["converter"](input, *c["args"], **c["kwargs"]), e)


if __name__ == "__main__":
    unittest.main()
