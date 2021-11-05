#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
import paddle
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *
import sys

enable_cudnn = sys.argv.pop()


class TestMulOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random((16, 64)).astype("float32"),
            "y": np.random.random((64, 16)).astype("float32")
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = paddle.matmul(x, y)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("mul")

        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")

        if enable_cudnn == "ON":
            tran_y = self.inputs["y"].reshape(-1)
        else:
            tran_y = self.inputs["y"].transpose().reshape(-1)

        y = builder.create_input(
            Float(32), [self.inputs["y"].shape[1], self.inputs["y"].shape[0]],
            "y")
        out = builder.mul(x, y)
        prog = builder.build()
        forward_res = self.get_cinn_output(prog, target, [x, y],
                                           [self.inputs["x"], tran_y], [out])

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
