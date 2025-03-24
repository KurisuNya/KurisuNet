import unittest

from kurisunet.config.module.layers.layer_from import is_drop_key, parse_layer_from
from kurisunet.constants import ALL_FROM, DROP_FROM, STR_PREFIX


class TestIsDropKey(unittest.TestCase):
    def test_is_drop_key(self):
        self.assertTrue(is_drop_key(DROP_FROM))
        self.assertFalse(is_drop_key("not_a_drop_key"))
        self.assertFalse(is_drop_key(123))  # type: ignore
        self.assertFalse(is_drop_key([1, 2, 3]))  # type: ignore
        self.assertFalse(is_drop_key({"key": "value"}))  # type: ignore


class TestParseLayerFrom(unittest.TestCase):
    def test_parse_layer_from(self):
        layer_from = [
            DROP_FROM,
            STR_PREFIX + DROP_FROM,
            1,
            {1: 0},
            {1: ALL_FROM},
            [1, {2: 3}],
            (1, {2: ALL_FROM}),
        ]
        expected = [
            DROP_FROM,
            DROP_FROM,
            ((1, ALL_FROM),),
            ((1, 0),),
            ((1, ALL_FROM),),
            ((1, ALL_FROM), (2, 3)),
            ((1, ALL_FROM), (2, ALL_FROM)),
        ]
        __import__("pprint").pprint(expected)
        for layer_from, expected in zip(layer_from, expected):
            self.assertEqual(parse_layer_from(layer_from, None), expected)

    def test_invalid_format(self):
        invalid_layer_from = [
            {},
            {1: "invalid"},
            {"a": 1},
            {1: 2, 3: 4},
            [],
            [{}],
            [{1: 2, 3: 4}],
            [{"a": 1}],
            (1, {2: "invalid"}),
        ]
        for layer_from in invalid_layer_from:
            with self.assertRaises(ValueError):
                parse_layer_from(layer_from, None)

    def test_expressions(self):
        layer_from = [
            "[1, {2: 3}]",
            "list(range(2))",
            "list(range(n))",
            "list(range(2)) + [{4: 5}]",
        ]
        expected = [
            ((1, ALL_FROM), (2, 3)),
            ((0, ALL_FROM), (1, ALL_FROM)),
            ((0, ALL_FROM), (1, ALL_FROM)),
            ((0, ALL_FROM), (1, ALL_FROM), (4, 5)),
        ]
        env = {"n": 2}
        for layer_from, expected in zip(layer_from, expected):
            self.assertEqual(parse_layer_from(layer_from, env), expected)

    def test_invalid_expressions(self):
        invalid_layer_from = [
            "invalid_expression",
            "list(range(n)) + [{4: 5}] + [6]",
            "list(range(2)) + [invalid_expression]",
            "list(range(n)) + [{4: 5}] + ",
        ]
        for layer_from in invalid_layer_from:
            with self.assertRaises((NameError, SyntaxError)):
                parse_layer_from(layer_from, None)

    def test_expressions_with_invalid_format(self):
        invalid_layer_from = [
            "{}",
            "{1: 2, 3: 4}",
            "list(range(2)) + [{4: 'invalid'}]",
            "list(range(n)) + ['test']",
        ]
        env = {"n": 2}
        for layer_from in invalid_layer_from:
            with self.assertRaises(ValueError):
                parse_layer_from(layer_from, env)


if __name__ == "__main__":
    unittest.main()
