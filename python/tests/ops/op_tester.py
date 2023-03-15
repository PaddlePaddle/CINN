# Copyright (c) 2023 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest, OpTestTool


class Test(OpTest):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.inputs = [
            {
                "x_shape": [3],
                "y_shape": [3]
            },
            {
                "x_shape": [3, 3],
                "y_shape": [3, 3]
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "int32",
                "y_dtype": "int32"
            },
            {
                "x_dtype": "float32",
                "y_dtype": "float32"
            },
        ]
        self.attrs = {
            "attr1": ["xxx", "yyy"],
            "attr2": ["aaa", "bbb"],
            "attr3": ["ccc", "ddd"],
        }

    def prepare_input(self):
        self.x_np = np.random.random(self.case["x_shape"]).astype(
            self.case["x_dtype"])
        self.y_np = np.random.random(self.case["y_shape"]).astype(
            self.case["y_dtype"])

    def test_all(self):
        self.run_all_cases()


if __name__ == "__main__":
    unittest.main()
