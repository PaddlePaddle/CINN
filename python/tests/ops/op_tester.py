import unittest
from op_test import OpTest, OpTestTool


class Test(OpTest):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shapes = [{1}, {1, 2}, {1, 2, 3}]
        self.y_shapes = [{3}, {3, 4}, {3, 3, 3}]
        self.dtypes = ["float32", "float64", "int32", "int64"]
        self.attrs = {
            "attr1": ["xxx", "yyy"],
            "attrx": ["aaa", "bbb"],
        }

    def test_all(self):
        self.run_test_cases()


if __name__ == "__main__":
    unittest.main()
