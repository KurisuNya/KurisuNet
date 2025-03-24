import unittest

from kurisunet.config.module.layers.args import parse_args, parse_kwargs
from kurisunet.constants import STR_PREFIX


class TestParseArgs(unittest.TestCase):
    def test_parse_args(self):
        args_list = [
            (1, 2, 3),
            (),
            ("1+2", "3"),
            (STR_PREFIX + "1+2", "3"),
            ("prod((n, 2))", "3"),
        ]
        expected_results = [
            (1, 2, 3),
            (),
            (3, 3),
            ("1+2", 3),
            (2, 3),
        ]
        env = {"n": 1, "prod": __import__("math").prod}
        for args, expected in zip(args_list, expected_results):
            with self.subTest(args=args):
                result = parse_args(args, env)
                self.assertEqual(result, expected)

    def test_parse_kwargs(self):
        kwargs_list = [
            {"a": 1, "b": 2},
            {},
            {"a": "1+2", "b": "3"},
            {"a": STR_PREFIX + "1+2", "b": "3"},
            {"a": "prod((n, 2))", "b": "3"},
        ]
        expected_results = [
            {"a": 1, "b": 2},
            {},
            {"a": 3, "b": 3},
            {"a": "1+2", "b": 3},
            {"a": 2, "b": 3},
        ]
        env = {"n": 1, "prod": __import__("math").prod}
        for kwargs, expected in zip(kwargs_list, expected_results):
            with self.subTest(kwargs=kwargs):
                result = parse_kwargs(kwargs, env)
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
