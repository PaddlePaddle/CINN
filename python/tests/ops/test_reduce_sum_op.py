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
from op_test import OpTest, OpTestTool, random
import paddle
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestReduceSumOp(OpTest):
    def setUp(self):
        self.config()
        self.inputs = {
            "x": random(self.x_shape, self.dtype),
        }

    def config(self):
        self.dtype = "float32"
        self.x_shape = [16, 32, 16, 16]
        self.dim = [0, 2, 3]
        self.keep_dim = False
        self.out_shape = [self.x_shape[1]]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.sum(x, axis=self.dim, keepdim=self.keep_dim)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reduce_sum")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        out = builder.reduce_sum(x, self.dim, self.keep_dim)
        prog = builder.build()
        forward_res = self.get_cinn_output(prog, target, [x],
                                           [self.inputs["x"]], [out])
        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
