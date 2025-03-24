import unittest

from kurisunet.config.utils import eval_string
from kurisunet.constants import STR_PREFIX


class TestUtils(unittest.TestCase):
    def test_eval_string(self):
        string = "a + b"
        env = {"a": 1, "b": 2}
        self.assertEqual(eval_string(string, env), 3)
        string = STR_PREFIX + string
        self.assertEqual(eval_string(string, env), "a + b")


if __name__ == "__main__":
    unittest.main()
