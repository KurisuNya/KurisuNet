import unittest

from kurisunet.config.types import FinalLayer
from kurisunet.constants import (
    ALL_FROM,
    DROP_FROM,
    LAYER_START_INDEX,
    MODULE_START_INDEX,
)
from kurisunet.net.utils import (
    auto_unpack,
    get_drop_layer_indexes,
    get_except_indexes,
    get_unused_layer_indexes,
    layer_enum,
    module_enum,
    regularize_layer_from,
)


Module = lambda x: x


class TestAutoUnpack(unittest.TestCase):
    def test_auto_unpack_single_element(self):
        result = auto_unpack((42,))
        self.assertEqual(result, 42)

    def test_auto_unpack_multiple_elements(self):
        result = auto_unpack((1, 2, 3))
        self.assertEqual(result, (1, 2, 3))

    def test_auto_unpack_empty_tuple(self):
        result = auto_unpack(())
        self.assertEqual(result, ())

    def test_auto_unpack_single_element_empty_tuple(self):
        with self.assertRaises(TypeError):
            auto_unpack(None)  # type: ignore


class TestGetExceptIndexes(unittest.TestCase):
    def test_get_except_indexes(self):
        seq = [1, 2, 3, 4, 5]
        drop_set = {0 + LAYER_START_INDEX, 2 + LAYER_START_INDEX}
        result = list(get_except_indexes(seq, drop_set))
        self.assertEqual(result, [2, 4, 5])

        seq = []
        drop_set = {1, 2, 3}
        result = list(get_except_indexes(seq, drop_set))
        self.assertEqual(result, [])


class TestGetDropLayerIndexes(unittest.TestCase):
    def test_get_drop_layer_indexes(self):
        layers: list[FinalLayer] = [
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": DROP_FROM, "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": DROP_FROM, "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        ]
        drop_indexes = {2 + LAYER_START_INDEX, 4 + LAYER_START_INDEX}
        result = get_drop_layer_indexes(layers)
        self.assertEqual(result, drop_indexes)


class TestGetUnusedLayerIndexes(unittest.TestCase):
    def test_get_unused_layer_indexes(self):
        i = [i + LAYER_START_INDEX - 1 for i in range(6)]
        layers: list[FinalLayer] = [
            {"from": ((i[0], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[1], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[2], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[3], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[4], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[5], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        ]
        unused_indexes = set()
        result = get_unused_layer_indexes(layers)
        self.assertEqual(result, unused_indexes)

        layers: list[FinalLayer] = [
            {"from": ((i[0], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[1], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[1], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[3], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[3], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[5], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        ]
        unused_indexes = {i[2], i[4]}
        result = get_unused_layer_indexes(layers)
        self.assertEqual(result, unused_indexes)

    def test_invalid_layer_indexes(self):
        layers: list[FinalLayer] = [
            {"from": ((0, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": DROP_FROM, "module": Module, "args": (), "kwargs": {}},
            {"from": ((1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": DROP_FROM, "module": Module, "args": (), "kwargs": {}},
            {"from": ((3, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((3, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((5, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        ]
        with self.assertRaises(ValueError):
            get_unused_layer_indexes(layers)


class TestLayerEnum(unittest.TestCase):
    def test_layer_enum(self):
        seq = [1, 2, 3]
        result = list(layer_enum(seq))
        indexes = [i + LAYER_START_INDEX for i in range(len(seq))]
        expected = [(index, value) for index, value in zip(indexes, seq)]
        self.assertEqual(result, expected)

    def test_layer_enum_empty(self):
        result = list(layer_enum([]))
        self.assertEqual(result, [])


class TestModuleEnum(unittest.TestCase):
    def test_module_enum(self):
        seq = [1, 2, 3]
        result = list(module_enum(seq))
        indexes = [i + MODULE_START_INDEX for i in range(len(seq))]
        expected = [(index, value) for index, value in zip(indexes, seq)]
        self.assertEqual(result, expected)

    def test_module_enum_empty(self):
        result = list(module_enum([]))
        self.assertEqual(result, [])


class TestRegularizeLayerFrom(unittest.TestCase):
    def test_regularize_layer_from(self):
        i = [i + LAYER_START_INDEX - 1 for i in range(6)]
        layers: list[FinalLayer] = [
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        ]
        result = regularize_layer_from(layers)
        expected = (
            {"from": ((i[0], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[1], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[2], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[3], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[4], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[5], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        )
        self.assertEqual(result, expected)

        layers: list[FinalLayer] = [
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-2, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-2, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((-3, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        ]
        result = regularize_layer_from(layers)
        expected = (
            {"from": ((i[0], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[0], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[2], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[3], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[3], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            {"from": ((i[3], ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
        )
        self.assertEqual(result, expected)

        invalid: list[list[FinalLayer]] = [
            [
                {"from": ((-2, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
                {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            ],
            [
                {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
                {"from": ((2, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
            ],
            [
                {"from": ((-1, ALL_FROM),), "module": Module, "args": (), "kwargs": {}},
                {"from": DROP_FROM, "module": Module, "args": (), "kwargs": {}},
            ],
        ]
        for layers in invalid:
            with self.assertRaises(ValueError):
                regularize_layer_from(layers)


if __name__ == "__main__":
    unittest.main()
