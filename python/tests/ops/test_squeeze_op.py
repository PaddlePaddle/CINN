# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import cinn
import numpy as np
import paddle
import unittest

from cinn.frontend import *
from cinn.common import *
from op_test import OpTest, OpTestTool


class TestSqueezeOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": np.random.random([2, 3, 1, 1]).astype("float32")}
        self.axes = [2, 3]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.squeeze(x, self.axes)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("squeeze_test")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        out = builder.squeeze(x, self.axes)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]],
                                   [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSqueezeCase1(TestSqueezeOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([2, 3, 1, 1]).astype("float32")}
        self.axes = [2]


class TestSqueezeCase2(TestSqueezeOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([2, 3, 1, 1]).astype("float32")}
        self.axes = []


class TestSqueezeCase3(TestSqueezeOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([2, 3, 1, 1]).astype("float32")}
        self.axes = [-2]


if __name__ == "__main__":
    unittest.main()
