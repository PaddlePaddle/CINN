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

from cinn.frontend import *
from cinn import Target
from cinn.framework import *
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *


class TestReluOp(OpTest):
    def setUp(self):
        self.init_results()
        self.init_case()
        self.init_target()

    def init_case(self):
        self.inputs = {
            "x": np.random.random([4, 4]).astype("float32"),
            "dout": np.ones((4, 4)).astype("float32")
        }

    def build_paddle_program(self):
        x = paddle.to_tensor(self.inputs["x"])
        x.stop_gradient = False
        out = F.relu(x)

        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads([out], [x])

    def build_cinn_program(self):
        builder = NetBuilder("relu")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        out = builder.relu(x)

        dout = builder.create_input(
            Float(32), self.inputs["dout"].shape, "dout")
        x_grad = builder.relu_grad(dout, out)

        prog = builder.build()
        results = self.get_cinn_output(prog, [x, dout],
                                       [self.inputs["x"], self.inputs["dout"]],
                                       [out, x_grad])

        self.cinn_outputs = [results[0]]
        self.cinn_grads = [results[1]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
