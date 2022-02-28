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
        self.inputs = {
            "x": np.random.random([10, 5, 5]).astype("float32"),
            "y": np.random.random([3, 5, 5]).astype("float32"),
            "index": np.random.randint(0, 10, size=3)
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=True)

        out = x
        for i in range(len(self.inputs["index"])):
            out[self.inputs["index"][i]] = y[i]

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("index_assign")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        index = builder.create_input(
            Int(32), self.inputs["index"].shape, "index")
        out = builder.index_assign(x, y, index)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y, index],
            [self.inputs["x"], self.inputs["y"], self.inputs["index"]], [out])

        print(res[0])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
