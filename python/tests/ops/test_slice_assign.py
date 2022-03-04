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
import cinn
from cinn.frontend import *
from cinn.common import *


class TestSliceAssignOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "inputs": np.zeros((64, 64)).astype("float32"),
            "assign": np.ones((32, 32)).astype("float32")
        }
        self.axis = [0, 1]
        self.starts = [16, 16]
        self.strides = [32, 32]

    def build_paddle_program(self, target):
        self.paddle_outputs = self.inputs["inputs"].copy()
        for idx in range(32):
            self.paddle_outputs[16 + idx][16:48] = self.inputs["assign"][idx]
        self.paddle_outputs = [self.paddle_outputs]

    def build_cinn_program(self, target):
        builder = NetBuilder("slice_assign")
        inputs = builder.create_input(
            Float(32), self.inputs["inputs"].shape, "inputs")
        assign = builder.create_input(
            Float(32), self.inputs["assign"].shape, "assign")
        out = builder.slice_assign(
            inputs,
            assign,
            axis=self.axis,
            starts=self.starts,
            strides=self.strides)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [inputs, assign],
            [self.inputs["inputs"], self.inputs["assign"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
