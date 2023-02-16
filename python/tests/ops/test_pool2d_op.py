#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cudnn(),
                    "x86 test will be skipped due to timeout.")
class TestPool2dOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([1, 3, 32, 32], "float32")}
        self.kernel_size = [2, 2]
        self.data_format = "NCHW"
        self.strides = [2, 2]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)

        out = paddle.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            data_format=self.data_format,
            stride=self.strides)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("pow")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        out = builder.pool2d(
            x,
            polling_type="max",
            kernel_size=self.kernel_size,
            data_format=self.data_format,
            strides=self.strides)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out], passes=[])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestPool2dNHWC(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 32, 32, 3], "float32")}
        self.kernel_size = [2, 2]
        self.data_format = "NHWC"
        self.strides = [2, 2]


class TestPool2dFP16(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 3, 32, 32], "float16")}
        self.kernel_size = [2, 2]
        self.data_format = "NCHW"
        self.strides = [2, 2]


if __name__ == "__main__":
    unittest.main()
