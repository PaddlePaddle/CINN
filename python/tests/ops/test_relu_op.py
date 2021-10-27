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
from op_test import OpTest
import paddle
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *


class TestReluOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random([
                32,
                64,
            ]).astype("float32"),
            "dout": np.random.random((32, 64)).astype("float32")
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = F.relu(x)

        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads([out], [x],
                                                  [self.inputs["dout"]])

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("relu")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        out = builder.relu(x)
        prog = builder.build()
        forward_res = self.get_cinn_output(prog, target, [x],
                                           [self.inputs["x"]], [out])

        builder = NetBuilder("relu_grad")
        shape = self.inputs["dout"].shape
        dout = builder.create_input(Float(32), shape, "dout")
        out = builder.create_input(Float(32), shape, "out")
        x_grad = builder.relu_grad(dout, out)
        prog = builder.build()
        backward_res = self.get_cinn_output(
            prog, target, [dout, out], [self.inputs["dout"], forward_res[0]],
            [x_grad])

        self.cinn_outputs = forward_res
        self.cinn_grads = backward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
