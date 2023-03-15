import unittest
from op_test import OpTest, OpTestTool


class Test(OpTest):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.inputs = [
            {"x": {1}, "y": {3}},
            {"x": {1, 2}, "y": {3, 4}},
        ]
        self.dtypes = [
            {"x_dtype": "int32", "y_dtype": "int32"},
            {"x_dtype": "fp32", "y_dtype": "fp32"},
        ]
        self.attrs = {
            "attr1": ["xxx", "yyy"],
            "attr2": ["aaa", "bbb"],
            "attr3": ["ccc", "ddd"],
        }

    def test_all(self):
        self.run_test_cases()


if __name__ == "__main__":
    unittest.main()
