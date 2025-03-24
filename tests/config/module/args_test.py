import unittest

from kurisunet.config.module.args import (
    _check_params,
    _format_params,
    _get_arg_dict_env,
    _get_input_arg_dict,
)
from kurisunet.constants import STR_PREFIX


class TestCheckParams(unittest.TestCase):
    def test_invalid_params_type(self):
        invalid_params = [1, 1.0, False, None, {}, set()]
        for params in invalid_params:
            with self.assertRaises(ValueError):
                _check_params(params)

    def test_invalid_default_param(self):
        invalid_params = [
            ["a", {"b": 2, "c": 3}],
            ["a", {1: 2}],
            ["a", ("a", 2, 3)],
            ["a", (1, 2)],
        ]
        for params in invalid_params:
            with self.assertRaises(ValueError):
                _check_params(params)

    def test_non_default_argument_follows_default_argument(self):
        invalid_params = ["a", {"b": 2}, "c"]
        with self.assertRaises(ValueError):
            _check_params(invalid_params)

    def test_valid_params(self):
        valid_params = [
            ["a", "b", {"c": 3}],
            ["a", "b", ("c", 3)],
            ["a", {"b": 2}, ("c", 3)],
            ["a", ("b", 2), {"c": 3}],
        ]
        for params in valid_params:
            _check_params(params)


class TestFormatParams(unittest.TestCase):
    def test_format_params(self):
        params = ["a", "b", {"c": 3}, ("d", 4)]
        expected = ("a", "b", ("c", 3), ("d", 4))
        self.assertEqual(_format_params(params), expected)

    def test_format_params_empty(self):
        params = []
        expected = ()
        self.assertEqual(_format_params(params), expected)


class TestGetArgDictEnv(unittest.TestCase):
    def test_get_arg_dict_env(self):
        arg_dict = {"a": 1, "b": 2}
        expected = {"a": 1, "b": 2}
        self.assertEqual(_get_arg_dict_env(arg_dict, {}), expected)

    def test_eval_string(self):
        arg_dict = {"a": "1", "b": "2"}
        expected = {"a": 1, "b": 2}
        self.assertEqual(_get_arg_dict_env(arg_dict, {}), expected)

    def test_str_prefix(self):
        arg_dict = {"a": "1", "b": STR_PREFIX + "2"}
        expected = {"a": 1, "b": "2"}
        self.assertEqual(_get_arg_dict_env(arg_dict, {}), expected)

    def test_use_custom_env(self):
        arg_dict = {"a": 1, "c": "b + 1"}
        env = {"b": 2}
        expected = {"a": 1, "c": 3}
        self.assertEqual(_get_arg_dict_env(arg_dict, env), expected)


class TestGetInputArgDict(unittest.TestCase):
    def test_get_input_arg_dict(self):
        params = ("a", "b", ("c", 3), ("d", 4))
        args = (1, "arg2")
        kwargs = {"c": 1, "d": "kwarg2"}
        expected = {"a": 1, "b": "arg2", "c": 1, "d": "kwarg2"}
        self.assertEqual(_get_input_arg_dict(params, args, kwargs), expected)

    def test_more_args_than_params(self):
        params = ("a", "b", ("c", 3))
        args = (1, "arg2", "arg3", "arg4")
        with self.assertRaises(ValueError):
            _get_input_arg_dict(params, args, {})

    def test_missing_kwargs(self):
        params = ("a", "b", ("c", 3))
        args = (1,)
        kwargs = {"c": 1}
        with self.assertRaises(ValueError):
            _get_input_arg_dict(params, args, kwargs)

    def test_already_signed_kwargs(self):
        params = ("a", "b", ("c", 3))
        args = (1, "arg2")
        kwargs = {"b": 2}
        with self.assertRaises(ValueError):
            _get_input_arg_dict(params, args, kwargs)

    def test_invalid_kwargs(self):
        params = ("a", "b", ("c", 3))
        args = (1, "arg2")
        kwargs = {"d": 1}
        with self.assertRaises(ValueError):
            _get_input_arg_dict(params, args, kwargs)


if __name__ == "__main__":
    unittest.main()
