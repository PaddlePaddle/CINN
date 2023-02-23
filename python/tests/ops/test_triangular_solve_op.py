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
                    "triangular solve op support GPU only now.")
class TestTriangularSolveOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "input1": np.array([[1.0, 1.0, 1.0],
                                [0.0, 2.0, 1.0],
                                [0.0, 0.0, -1.0]]).astype(np.float32),
            "input2":  np.array([[0.0], [-9.0], [5.0]]).astype(np.float32),
        }
        self.outputs = {
            "solution": np.array([[7.0], [2.0], [-5.0]]).astype(np.float32)
        }
        self.left_side = False
        self.upper = False
        self.transpose_a = False
        self.unit_diagonal = False

    def build_paddle_program(self, target):
        solution = paddle.to_tensor(self.outputs["solution"], stop_gradient=False)
        self.paddle_outputs = [solution]

    def build_cinn_program(self, target):
        builder = NetBuilder("triangular_solve")
        input1 = builder.create_input(
            self.nptype2cinntype(self.inputs["input1"].dtype),
            self.inputs["input1"].shape, "input1")
        input2 = builder.create_input(
            self.nptype2cinntype(self.inputs["input2"].dtype),
            self.inputs["input2"].shape, "input2")
        out = builder.triangular_solve(input1, input2, self.left_side, self.upper, self.transpose_a, self.unit_diagonal)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [input1, input2], [self.inputs["input1"], self.inputs["input2"]], [out], passes=[])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
