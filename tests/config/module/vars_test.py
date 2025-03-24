import unittest

from kurisunet.config.module.vars import _check_vars, _format_vars, _get_vars_env
from kurisunet.constants import STR_PREFIX


class TestCheckVars(unittest.TestCase):
    def test_invalid_vars_type(self):
        invalid_vars = [1, 1.0, False, None, {}, set()]
        for vars in invalid_vars:
            with self.assertRaises(ValueError):
                _check_vars(vars)

    def test_invalid_var(self):
        invalid_vars = ["a", {"b": 2, "c": 3}, {1: 2}, ("a", 2, 3), (1, 2)]
        for vars in invalid_vars:
            with self.assertRaises(ValueError):
                _check_vars([vars])

    def test_valid_vars(self):
        valid_vars = [
            [{"a": 1}],
            [{"a": 1}, {"b": "2"}],
            [("a", 1), ("b", 2)],
            [{"a": 1}, ("b", 2)],
        ]
        for vars in valid_vars:
            _check_vars(vars)


class TestFormatVars(unittest.TestCase):
    def test_format_vars(self):
        vars = [{"a": 1}, ("b", 2)]
        expected = (("a", 1), ("b", 2))
        self.assertEqual(_format_vars(vars), expected)

    def test_format_vars_empty(self):
        vars = []
        expected = ()
        self.assertEqual(_format_vars(vars), expected)


class TestGetVarsEnv(unittest.TestCase):
    def test_get_vars_env(self):
        vars = [("a", 1), ("b", 2)]
        expected = {"a": 1, "b": 2}
        self.assertEqual(_get_vars_env(vars, {}), expected)

    def test_eval_string(self):
        vars = [("a", "1"), ("b", "2")]
        expected = {"a": 1, "b": 2}
        self.assertEqual(_get_vars_env(vars, {}), expected)

    def test_str_prefix(self):
        vars = [("a", "1"), ("b", STR_PREFIX + "2")]
        expected = {"a": 1, "b": "2"}
        self.assertEqual(_get_vars_env(vars, {}), expected)

    def test_use_former_vars(self):
        vars = [("a", 1), ("b", "a + 1")]
        expected = {"a": 1, "b": 2}
        self.assertEqual(_get_vars_env(vars, {}), expected)

    def test_use_custom_env(self):
        vars = [("a", 1), ("c", "b + 1")]
        env = {"b": 2}
        expected = {"a": 1, "c": 3}
        self.assertEqual(_get_vars_env(vars, env), expected)

    def test_cover_env(self):
        vars = [("a", 1), ("b", 2), ("c", "b + 1")]
        env = {"b": 3}
        expected = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(_get_vars_env(vars, env), expected)


if __name__ == "__main__":
    unittest.main()
