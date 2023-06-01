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

import copy
import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestScatterOp(OpTest):
    def setUp(self):
        self.init_case()
        self.target = DefaultNVGPUTarget()

    def init_case(self):
        self.axis = -1
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([5, 5]).astype("float32"),
            "index": np.array([0, 5, 0, 9, 0]).astype("int32"),
            "src": self.random([2, 5], "float32"),
            "out": self.random([3, 5], "float32"),
            "index0": np.array([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]).astype("int32"),
        }
        print(self.inputs["src"], '\n', self.inputs["index0"], '\n', self.inputs["out"])

    def build_paddle_program(self, target):
        # x = paddle.to_tensor(self.inputs["src"], stop_gradient=True)
        # y = paddle.to_tensor(self.inputs["out"], stop_gradient=True)
        # index = paddle.to_tensor(self.inputs["index0"], stop_gradient=True)
        x = self.inputs["src"]
        y = self.inputs["out"]
        index = self.inputs["index0"]

        pos_axis = self.axis
        if pos_axis < 0:
            pos_axis += len(x.shape)

        dim = len(x.shape)
        res = copy.deepcopy(y)
        if dim == 1:
            pass
        elif dim == 2:
            if pos_axis == 0:
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        res[index[i, j], j] = x[i, j]
            elif pos_axis == 1:
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        res[i, index[i, j]] = x[i, j]
        elif dim == 3:
            pass
        elif dim == 4:
            pass
        else:
            raise NotImplementedError
        
        self.paddle_outputs = [paddle.to_tensor(res)]
        print("paddle res:\n", res)

    def build_cinn_program(self, target):
        builder = NetBuilder("scatter")
        src = builder.create_input(
            OpTest.nptype2cinntype(self.inputs["src"].dtype),
            self.inputs["src"].shape, "src")
        index = builder.create_input(
            OpTest.nptype2cinntype(self.inputs["index0"].dtype),
            self.inputs["index0"].shape, "index0")
        out = builder.create_input(
            OpTest.nptype2cinntype(self.inputs["out"].dtype),
            self.inputs["out"].shape, "out")
        out1 = builder.scatter(src, index, out, 0)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [src, index, out],
            [self.inputs["src"], self.inputs["index0"], self.inputs["out"]],
            [out1])

        self.cinn_outputs = res
        print("cinn res:\n", res)

    def test_check_results(self):
        self.check_outputs_and_grads()


# class TestScatterOp1(TestScatterOp):
#     def setUp(self):
#         self.init_case()
#         self.target = DefaultNVGPUTarget()

#     def init_case(self):
#         self.axis = 0
#         self.inputs = {
#             "x": np.random.random([10, 5]).astype("int32"),
#             "y": np.random.random([5, 5]).astype("int32"),
#             "index": np.array([0, 5, 0, 9, 0]).astype("int32")
#         }

#     def test_check_results(self):
#         self.check_outputs_and_grads()

if __name__ == "__main__":
    unittest.main()
