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
from op_test_helper import TestCaseHelper
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestNormOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"])

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.linalg.norm(
            x,
            p=self.case["p"],
            axis=self.case["axis"],
            keepdim=self.case["keepdim"])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("norm")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        out = builder.norm(
            x, axis=self.case["axis"], epsilon=self.case["epsilon"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestNormOpCase(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestNormOpCase"
        self.cls = TestNormOp
        self.inputs = [
            {
                "x_shape": [1],
                "p": "fro",
                "axis": -1,
                "keepdim": False,
                "epsilon": 1.0e-10,
            },
            {
                "x_shape": [32],
                "p": "inf",
                "axis": None,
                "keepdim": False,
                "epsilon": 1.0e-10,
            },
            {
                "x_shape": [1024],
                "p": "-inf",
                "axis": -1,
                "keepdim": True,
                "epsilon": 1.0e-10,
            },
            {
                "x_shape": [32, 64],
                "p": "0",
                "axis": None,
                "keepdim": False,
                "epsilon": 1.0e-10,
            },
            {
                "x_shape": [2, 3, 4],
                "p": "1",
                "axis": -1,
                "keepdim": True,
                "epsilon": 1.0e-10,
            },
            {
                "x_shape": [16, 8, 4, 2],
                "p": "2",
                "axis": -1,
                "keepdim": False,
                "epsilon": 1.0e-10,
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "p": "fro",
                "axis": None,
                "keepdim": True,
                "epsilon": 1.0e-10,
            },
        ]

        self.dtypes = [
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]

        self.attrs = []


if __name__ == "__main__":
    TestNormOpCase().run()
