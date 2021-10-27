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


class TestSumOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x1": np.random.random([
                32,
                64,
            ]).astype("float32"),
            "x2": np.random.random([
                32,
                64,
            ]).astype("float32"),
            "dout": np.random.random((32, 64)).astype("float32")
        }

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.inputs["x1"], stop_gradient=False)
        x2 = paddle.to_tensor(self.inputs["x2"], stop_gradient=False)
        out = paddle.add(x1, x2)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sum")
        x1 = builder.create_input(Float(32), self.inputs["x1"].shape, "x1")
        x2 = builder.create_input(Float(32), self.inputs["x2"].shape, "x2")
        out = builder.sum([x1, x2])
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x1, x2], [self.inputs["x1"], self.inputs["x2"]],
            [out])

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
