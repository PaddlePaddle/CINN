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
import cinn
import numpy as np
import paddle
import unittest

from cinn.frontend import *
from cinn.common import *
from cinn.runtime import *
from op_test import OpTest, OpTestTool

set_cinn_cudnn_deterministic(True)
paddle.fluid.set_flags({'FLAGS_cudnn_deterministic': 1})


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestConv2dOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": self.random([3, 16, 32, 32], "float32"),
            "weight": self.random([16, 16, 3, 3], "float32"),
            "dy": self.random([3, 16, 30, 30], "float32")
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        weight = paddle.to_tensor(self.inputs["weight"], stop_gradient=False)

        y = paddle.nn.functional.conv2d(x, weight)

        self.paddle_outputs = [y.numpy()]
        self.paddle_grads = self.get_paddle_grads([y], [x, weight],
                                                  [self.inputs["dy"]])

    def build_cinn_program(self, target):
        builder = NetBuilder("conv2d")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        weight = builder.create_input(
            self.nptype2cinntype(self.inputs["weight"].dtype),
            self.inputs["weight"].shape, "weight")
        dy = builder.create_input(
            self.nptype2cinntype(self.inputs["dy"].dtype),
            self.inputs["dy"].shape, "dy")

        y = builder.conv2d(x, weight)
        x_grad, weight_grad = builder.conv2d_grad(dy, x, weight)
        prog = builder.build()

        res = self.get_cinn_output(
            prog,
            target, [x, weight, dy],
            [self.inputs["x"], self.inputs["weight"], self.inputs["dy"]],
            [y, x_grad, weight_grad],
            passes=[])

        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1], res[2]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestConv2dOpFP16(TestConv2dOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([3, 16, 32, 32], "float16"),
            "weight": self.random([16, 16, 3, 3], "float16"),
            "dy": self.random([3, 16, 30, 30], "float16")
        }

    def test_check_results(self):
        self.check_outputs_and_grads(1e-3)


if __name__ == "__main__":
    unittest.main()
