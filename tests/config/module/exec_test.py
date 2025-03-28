import unittest

from kurisunet.config.module.exec import _check_exec, _get_exec_env, get_exec_env


class TestCheckExec(unittest.TestCase):
    def test_invalid_exec_type(self):
        invalid_execs = [1, 1.0, False, None, {}, set(), b"abc"]
        for exec_ in invalid_execs:
            with self.assertRaises(ValueError):
                _check_exec(exec_)

    def test_valid_exec_type(self):
        valid_execs = ["1 + 1", "if name == '__main__':", "print('Hello, World!')"]
        for exec_ in valid_execs:
            self.assertIsNone(_check_exec(exec_))


class TestGetExecEnv(unittest.TestCase):
    def test_valid_exec_env(self):
        env = {"a": 1}
        exec_ = "b = a + 1"
        local_env = _get_exec_env(exec_, env)
        self.assertEqual(local_env, {"b": 2})
        local_env = get_exec_env(exec_, env)
        self.assertEqual(local_env, {"b": 2})

    def test_cover_exec_env(self):
        env = {"a": 1}
        exec_ = "a = a + 1"
        local_env = _get_exec_env(exec_, env)
        self.assertEqual(local_env, {"a": 2})
        self.assertEqual(env, {"a": 1})
        local_env = get_exec_env(exec_, env)
        self.assertEqual(local_env, {"a": 2})
        self.assertEqual(env, {"a": 1})

    def test_invalid_exec_env(self):
        env = {"a": 1}
        exec_ = "b = a + '1'"
        with self.assertRaises(TypeError):
            _get_exec_env(exec_, env)
        with self.assertRaises(TypeError):
            get_exec_env(exec_, env)

    def test_empty_exec(self):
        env = {"a": 1}
        exec_ = ""
        local_env = _get_exec_env(exec_, env)
        self.assertEqual(local_env, {})
        local_env = get_exec_env(exec_, env)
        self.assertEqual(local_env, {})


if __name__ == "__main__":
    unittest.main()
