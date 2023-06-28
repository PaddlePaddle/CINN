#!/usr/bin/env python3

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
from op_mapper_test import OpMapperTest, logger
import paddle


class TestMatmulOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            "X": self.random([16, 32], "float32"),
            "Y": self.random([32, 16], "float32")
        }

    def set_op_type(self):
        return "matmul"

    def set_op_inputs(self):
        X = paddle.static.data('X', self.feed_data["X"].shape,
                               self.feed_data["X"].dtype)
        Y = paddle.static.data('Y', self.feed_data["Y"].shape,
                               self.feed_data["Y"].dtype)
        return {'X': [X], 'Y': [Y]}

    def set_op_attrs(self):
        return {"trans_x": False, "trans_y": False}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['X'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
