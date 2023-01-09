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
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestCholeskyOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"shape": np.array([2, 3]).astype(np.int32)}
        self.outputs = {
            "out":
            np.array([[0.98147416, 0., 0.], [0.89824611, 0.76365221,
                                             0.]]).astype(np.float32)
        }
        self.mean = 0.0
        self.std = 1.0
        self.seed = 10
        self.dtype = "float32"

    def build_paddle_program(self, target):
        out = paddle.to_tensor(self.outputs["out"], stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("gaussian_random")
        shape = builder.create_input(
            self.nptype2cinntype(self.inputs["shape"].dtype),
            self.inputs["shape"].shape, "shape")
        out = builder.gaussian_random(shape, self.mean, self.std, self.seed,
                                      self.dtype)
        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target, [shape], [self.inputs["shape"]], [out],
            passes=[])
        print(res)
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
