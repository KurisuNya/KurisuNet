import unittest

from kurisunet.config.module.imports import (
    _check_imports,
    _get_imports_env,
    get_imports_env,
)

valid_imports = [
    # basic import
    ("import os", {"os"}),
    ("import os, collections", {"os", "collections"}),
    ("import os.path", {"os"}),
    ("import os.path, collections.abc", {"os", "collections"}),
    # import submodules
    ("from os import path", {"path"}),
    ("from os import path, open", {"path", "open"}),
    ("from os.path import join", {"join"}),
    ("from os.path import join, exists", {"join", "exists"}),
    # import basic with alias
    ("import os as operating_system", {"operating_system"}),
    ("import os as operating_system, collections as col", {"operating_system", "col"}),
    ("import os.path as path", {"path"}),
    ("import os.path as path, collections.abc as col", {"path", "col"}),
    # import submodules with alias
    ("from os import path as p", {"p"}),
    ("from os import path as p, open as o", {"p", "o"}),
    ("from os.path import join as j", {"j"}),
    ("from os.path import join as j, exists as e", {"j", "e"}),
]


class TestCheckImports(unittest.TestCase):
    def test_invalid_imports_type(self):
        invalid_imports = [1, 1.0, False, None, {}, set(), "abc", b"abc"]
        for imports in invalid_imports:
            with self.assertRaises(ValueError):
                _check_imports(imports)

    def test_valid_imports_format(self):
        imports = [k for k, _ in valid_imports]
        for import_ in imports:
            self.assertIsNone(_check_imports([import_]))

    def test_invalid_imports_format(self):
        imports = [
            "import os,",
            "form os import path, open",
            "1 + 1",
            "if name == '__main__':",
        ]
        for import_ in imports:
            with self.assertRaises(ValueError):
                _check_imports([import_])

    def test_duplicate_imports(self):
        imports = ["import os.path as path", "from os.path import path"]
        with self.assertRaises(ValueError):
            _check_imports(imports)


class TestGetImportsEnv(unittest.TestCase):
    def test_get_imports_env(self):
        for import_, expected in valid_imports:
            names = {name for name in _get_imports_env([import_]).keys()}
            self.assertEqual(names, expected)
        for import_, expected in valid_imports:
            names = {name for name in get_imports_env([import_]).keys()}
            self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()
