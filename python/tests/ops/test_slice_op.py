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
class TestSliceOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"inputs": np.random.random([10, 12]).astype("float32")}
        self.axes = [0, 1]
        self.starts = [2, 2]
        self.ends = [5, 5]
        self.strides = [1, 1]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["inputs"], stop_gradient=True)
        self.strides = [1] * len(self.axes)
        res = paddle.strided_slice(x, self.axes, self.starts, self.ends,
                                   self.strides)
        print("Paddle:", res)
        self.paddle_outputs = [res]

    def build_cinn_program(self, target):
        builder = NetBuilder("slice")
        inputs = builder.create_input(
            Float(32), self.inputs["inputs"].shape, "inputs")
        out = builder.slice(
            inputs,
            axes=self.axes,
            starts=self.starts,
            ends=self.ends,
            strides=self.strides)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [inputs],
                                   [self.inputs["inputs"]], [out])
        print("CINN:", res)
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestSliceCase1(TestSliceOp):
    def init_case(self):
        self.inputs = {"inputs": np.random.random([10, 12]).astype("float32")}
        self.axes = [0, 1]
        self.starts = [1, 2]
        self.ends = [6, 1000]
        self.strides = [1, 2]


class TestSliceCase2(TestSliceOp):
    def init_case(self):
        self.inputs = {"inputs": np.random.random([10, 12]).astype("float32")}
        self.axes = [0, 1]
        self.starts = [2, 1]
        self.ends = [-1, 7]
        self.strides = [3, 2]


class TestSliceCase3(TestSliceOp):
    def init_case(self):
        self.inputs = {"inputs": np.random.random([10, 12]).astype("float32")}
        self.axes = [0, 1]
        self.starts = [2, 1000]
        self.ends = [8, 1]
        self.strides = [1, -2]


class TestSliceCase4(TestSliceOp):
    def init_case(self):
        self.inputs = {"inputs": np.random.random([10, 12]).astype("float32")}
        self.axes = [0, 1]
        self.starts = [-1, -2]
        self.ends = [-5, -8]
        self.strides = [-1, -2]


if __name__ == "__main__":
    unittest.main()
