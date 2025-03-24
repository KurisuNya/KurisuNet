from pathlib import Path
import unittest

from kurisunet.basic.utils import (
    get_except_key,
    get_except_keys,
    get_first_index_of,
    get_last_index_of,
    is_env_conflict,
    is_list_tuple_of,
    merge_envs,
    to_path,
    to_relative_path,
)


class TestUtils(unittest.TestCase):
    def test_get_except_key(self):
        self.assertEqual(get_except_key({"a": 1, "b": 2}, "a"), {"b": 2})
        self.assertEqual(get_except_key({"a": 1, "b": 2}, "c"), {"a": 1, "b": 2})

    def test_get_except_keys(self):
        self.assertEqual(get_except_keys({"a": 1, "b": 2}, ["a"]), {"b": 2})
        self.assertEqual(get_except_keys({"a": 1, "b": 2}, ["c"]), {"a": 1, "b": 2})
        self.assertEqual(get_except_keys({"a": 1, "b": 2}, ["a", "b"]), {})

    def test_get_first_index_of(self):
        self.assertEqual(get_first_index_of([1, "2", 3], int), 0)
        self.assertEqual(get_first_index_of([1, 2, "3", 4], str), 2)
        self.assertEqual(get_first_index_of(["a", "b", "c"], int), None)
        self.assertEqual(get_first_index_of([], int), None)

    def test_get_last_index_of(self):
        self.assertEqual(get_last_index_of([1, "2", 3], int), 2)
        self.assertEqual(get_last_index_of([1, 2, "3", 4], str), 2)
        self.assertEqual(get_last_index_of(["a", "b", "c"], int), None)
        self.assertEqual(get_last_index_of([], int), None)

    def test_is_env_conflict(self):
        env1 = {"a": 1, "b": 2}
        env2 = {"b": 3, "c": 4}
        self.assertTrue(is_env_conflict(env1, env2))
        self.assertTrue(is_env_conflict(env2, env1))
        self.assertFalse(is_env_conflict(env1, {"c": 5}))

    def test_is_list_tuple_of(self):
        self.assertTrue(is_list_tuple_of([1, 2, 3], int))
        self.assertTrue(is_list_tuple_of((1, 2, 3), int))
        self.assertFalse(is_list_tuple_of([1, "2", 3], int))
        self.assertFalse(is_list_tuple_of((1, "2", 3), int))
        self.assertFalse(is_list_tuple_of("123", int))

    def test_merge_envs(self):
        old_env = {"a": 1, "b": 2}
        new_env = {"b": 3, "c": 4}
        merged_env = merge_envs([old_env, new_env])
        self.assertEqual(merged_env, {"a": 1, "b": 3, "c": 4})

    def test_to_path(self):
        self.assertEqual(to_path("test.txt"), Path("test.txt"))
        self.assertEqual(to_path(Path("test.txt")), Path("test.txt"))

    def test_to_relative_path(self):
        self.assertEqual(
            to_relative_path("/base/test.txt", base_path=Path("/base")),
            Path("test.txt"),
        )


if __name__ == "__main__":
    unittest.main()
