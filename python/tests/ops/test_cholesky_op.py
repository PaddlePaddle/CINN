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


class TestCholeskyOp(OpTest):
    def setUp(self):
        # self.target = DefaultHostTarget()
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x":
            np.array([[0.96329159, 0.88160539, 0.40593964],
                      [0.88160539, 1.39001071, 0.48823422],
                      [0.40593964, 0.48823422, 0.19755946]]).astype(np.float32)
        }
        self.outputs = {
            "y":
            np.array([[0.98147416, 0., 0.], [0.89824611, 0.76365221, 0.],
                      [0.41360193, 0.15284170, 0.05596709]]).astype(np.float32)
        }
        self.upper = False

    def build_paddle_program(self, target):
        y = paddle.to_tensor(self.outputs["y"], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("cholesky")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        out = builder.cholesky(x, self.upper)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out], passes=[])
        print(res)
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        # print(self.inputs["x"])
        # print(self.cinn_outputs)
        self.check_outputs_and_grads()


# class TestCholeskyCase1(TestCholeskyOp):
#     def init_case(self):
#         self.inputs = {
#             "x": np.array([[
#                 [0.96329159, 0.88160539, 0.40593964],
#                 [0.88160539, 1.39001071, 0.48823422],
#                 [0.40593964, 0.48823422, 0.19755946]
#             ], [
#                 [0.96329159, 0.88160539, 0.40593964],
#                 [0.88160539, 1.39001071, 0.48823422],
#                 [0.40593964, 0.48823422, 0.19755946]
#             ]]).astype(np.float32)
#         }
#         self.outputs = {
#             "y": np.array([[
#                 [0.98147416, 0., 0.],
#                 [0.89824611, 0.76365221, 0.],
#                 [0.41360193, 0.15284170, 0.05596709]
#             ], [
#                 [0.98147416, 0., 0.],
#                 [0.89824611, 0.76365221, 0.],
#                 [0.41360193, 0.15284170, 0.05596709]
#             ]]).astype(np.float32)
#         }
#         self.upper = False

# class TestCholeskyCase2(TestCholeskyOp):
#     def init_case(self):
#         self.inputs = {
#             "x": np.array([[
#                 [0.96329159, 0.88160539, 0.40593964],
#                 [0.88160539, 1.39001071, 0.48823422],
#                 [0.40593964, 0.48823422, 0.19755946]
#             ], [
#                 [0.96329159, 0.88160539, 0.40593964],
#                 [0.88160539, 1.39001071, 0.48823422],
#                 [0.40593964, 0.48823422, 0.19755946]
#             ]]).astype(np.float32)
#         }
#         self.outputs = {
#             "y": np.array([[
#                 [0.98147416, 0.89824611, 0.41360193],
#                 [0., 0.76365221, 0.15284170],
#                 [0., 0., 0.05596709]
#             ], [
#                 [0.98147416, 0.89824611, 0.41360193],
#                 [0., 0.76365221, 0.15284170],
#                 [0., 0., 0.05596709]
#             ]]).astype(np.float32)
#         }
#         self.upper = True

if __name__ == "__main__":
    unittest.main()
