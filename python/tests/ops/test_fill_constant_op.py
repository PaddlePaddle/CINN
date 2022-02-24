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


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestAddOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.shape = [32]
        self.value = float(1.0)
        self.cinn_type = Float(32)
        self.inputs = {"y": np.random.random(self.shape).astype("float32")}

    def build_paddle_program(self, target):
        x = paddle.full(self.shape, self.value)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=True)

        out = paddle.add(x, y)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("add")
        x = builder.fill_constant(self.shape, self.value, "x")
        y = builder.create_input(self.cinn_type, self.inputs["y"].shape, "y")
        out = builder.add(x, y)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [y], [self.inputs["y"]],
                                   [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestAddCase1(TestAddOp):
    def init_case(self):
        self.shape = [10, 32, 4]
        self.value = float(-100.0)
        self.cinn_type = Float(32)
        self.inputs = {"y": np.random.random(self.shape).astype("float32")}


if __name__ == "__main__":
    unittest.main()
